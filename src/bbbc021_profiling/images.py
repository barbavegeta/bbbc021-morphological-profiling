"""Download the selected TIFFs from the plate ZIPs, one range request per TIFF.

For each plate the ZIP's central directory is read and cached. Every needed file must be listed
there under <plate>/<file name from BBBC021_v1_image.csv>. Each TIFF is fetched on its own,
decompressed, checked against the directory's CRC-32 and size, written atomically and recorded in a
manifest with its SHA-256. A rerun skips TIFFs that are already in the manifest and on disk.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from .metadata import CHANNELS, ZIP_URL
from .remote_zip import ArchiveChanged, Member, MemberError, RangeReader, extract, read_directory

TIFF_COLUMNS = ["plate", "path", "zip_member", "compressed_bytes", "bytes", "crc32", "sha256", "downloaded_utc"]
PLATE_COLUMNS = ["plate", "zip_url", "zip_bytes", "etag", "last_modified", "members", "needed",
                 "needed_compressed_bytes", "needed_bytes", "done", "state"]
MEMBER_COLUMNS = [f.name for f in dataclasses.fields(Member)]


def needed_files(selected: pd.DataFrame) -> pd.DataFrame:
    """One row per TIFF: plate, relative output path (Week/Plate/File) and ZIP member name."""
    rows = []
    for ch in CHANNELS:
        rows.append(pd.DataFrame({"plate": selected["plate"], "path": selected[f"path_{ch}"]}))
    out = pd.concat(rows, ignore_index=True)
    parts = out["path"].str.split("/")
    if not (parts.str.len() == 3).all() or not (parts.str[1] == out["plate"]).all():
        raise ValueError("image paths are not of the form <week>/<plate>/<file>")
    out["zip_member"] = out["plate"] + "/" + parts.str[2]
    if out["path"].duplicated().any():
        raise ValueError("duplicate image paths")
    return out.sort_values(["plate", "path"]).reset_index(drop=True)


def safe_path(root: Path, rel: str) -> Path:
    root = root.resolve()
    p = (root / rel).resolve()
    if not p.is_relative_to(root) or p == root:
        raise ValueError(f"path escapes the image folder: {rel}")
    return p


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.exists() and path.stat().st_size:
        return pd.read_csv(path, keep_default_na=False)
    return pd.DataFrame(columns=columns)


class _Appender:
    """Thread-safe, flushed appends to a CSV with a fixed header."""

    def __init__(self, path: Path, columns: list[str]):
        new = not path.exists() or path.stat().st_size == 0
        self._fh = open(path, "a", newline="")
        self._w = csv.writer(self._fh)
        self._lock = threading.Lock()
        if new:
            self._w.writerow(columns)
            self._fh.flush()

    def write(self, row: list) -> None:
        with self._lock:
            self._w.writerow(row)
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        self._fh.close()


def plate_directory(plate: str, reader: RangeReader, cache_dir: Path, known: dict | None) -> list[Member]:
    """The plate ZIP's member list, read once and cached as CSV.

    The cache is used only while the server reports the same archive size and ETag as when it was
    read.
    """
    cache = cache_dir / f"{plate}.csv"
    same = known is not None and int(known["zip_bytes"]) == reader.size and (known["etag"] or None) == getattr(reader, "etag", None)
    if cache.exists() and same:
        d = pd.read_csv(cache, keep_default_na=False)
        return [Member(**{k: (str(v) if k == "name" else int(v)) for k, v in r.items()}) for r in d.to_dict("records")]
    if cache.exists() and known is not None and not same:
        raise ArchiveChanged(f"{plate}: archive size or ETag differs from the cached directory")
    members = read_directory(reader)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dataclasses.asdict(m) for m in members], columns=MEMBER_COLUMNS).to_csv(cache, index=False)
    return members


def download_images(
    needed: pd.DataFrame,
    image_root: Path,
    manifest_dir: Path,
    open_reader: Callable[[str], RangeReader],
    workers: int = 4,
    directories_only: bool = False,
    log: Callable[[str], None] = print,
) -> dict:
    """Fetch every TIFF in needed (see needed_files) into image_root, recording it in manifest_dir."""
    image_root.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    plates_csv, tiffs_csv = manifest_dir / "plates.csv", manifest_dir / "tiffs.csv"
    plates_prev = _read_csv(plates_csv, PLATE_COLUMNS).set_index("plate").to_dict("index") if plates_csv.exists() else {}

    done_rows = _read_csv(tiffs_csv, TIFF_COLUMNS)
    done = {r["path"] for r in done_rows.to_dict("records")
            if safe_path(image_root, r["path"]).exists()
            and safe_path(image_root, r["path"]).stat().st_size == int(r["bytes"])}
    appender = None if directories_only else _Appender(tiffs_csv, TIFF_COLUMNS)

    plate_rows, totals = [], {"tiffs_fetched": 0, "bytes_transferred": 0}
    t0, lock = time.monotonic(), threading.Lock()

    def fetch_one(reader: RangeReader, plate: str, m: Member, rel: str) -> None:
        for attempt in range(2):
            raw = reader.read(m.header_offset, m.end - 1)
            try:
                data = extract(raw, m)
                break
            except MemberError:
                if attempt == 1:
                    raise
        out = safe_path(image_root, rel)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, out)
        appender.write([plate, rel, m.name, m.compressed_size, m.size, m.crc32,
                        hashlib.sha256(data).hexdigest(),
                        datetime.now(timezone.utc).isoformat(timespec="seconds")])
        with lock:
            totals["tiffs_fetched"] += 1
            totals["bytes_transferred"] += len(raw)
            n = totals["tiffs_fetched"]
            if n % 250 == 0:
                mb = totals["bytes_transferred"] / 1e6
                log(f"  {n} TIFFs fetched, {mb:,.0f} MB, {mb / (time.monotonic() - t0):.1f} MB/s")

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for plate, want in needed.groupby("plate", sort=True):
                reader = open_reader(plate)
                members = plate_directory(plate, reader, manifest_dir / "zip_directories", plates_prev.get(plate))
                by_name = {m.name: m for m in members}
                missing = sorted(set(want["zip_member"]) - set(by_name))
                if missing:
                    raise MemberError(f"{plate}: {len(missing)} files from the image table are not in the ZIP, e.g. {missing[:3]}")
                want_members = [by_name[n] for n in want["zip_member"]]
                todo = [(by_name[z], p) for z, p in zip(want["zip_member"], want["path"]) if p not in done]
                if not directories_only and todo:
                    futures = [pool.submit(fetch_one, reader, plate, m, rel) for m, rel in todo]
                    try:
                        for f in futures:
                            f.result()
                    except BaseException:
                        for f in futures:
                            f.cancel()
                        raise
                n_done = len(want) - (len(todo) if directories_only else 0)
                plate_rows.append({
                    "plate": plate, "zip_url": getattr(reader, "url", ZIP_URL.format(plate=plate)),
                    "zip_bytes": reader.size, "etag": getattr(reader, "etag", None) or "",
                    "last_modified": getattr(reader, "last_modified", None) or "",
                    "members": sum(not m.name.endswith("/") for m in members), "needed": len(want),
                    "needed_compressed_bytes": sum(m.compressed_size for m in want_members),
                    "needed_bytes": sum(m.size for m in want_members), "done": n_done,
                    "state": "done" if n_done == len(want) else "directory",
                })
                merged = {**plates_prev, **{r["plate"]: {k: v for k, v in r.items() if k != "plate"} for r in plate_rows}}
                pd.DataFrame([{"plate": k, **v} for k, v in sorted(merged.items())], columns=PLATE_COLUMNS).to_csv(plates_csv, index=False)
                if not directories_only:
                    log(f"{plate}: {len(want)} TIFFs present ({len(todo)} fetched now)")
    finally:
        if appender is not None:
            appender.close()

    p = pd.DataFrame(plate_rows)
    return {
        "plates": len(p),
        "tiffs_needed": int(p["needed"].sum()),
        "tiffs_done": int(p["done"].sum()),
        "needed_compressed_bytes": int(p["needed_compressed_bytes"].sum()),
        "needed_bytes": int(p["needed_bytes"].sum()),
        "tiffs_fetched_now": totals["tiffs_fetched"],
        "bytes_transferred_now": totals["bytes_transferred"],
    }
