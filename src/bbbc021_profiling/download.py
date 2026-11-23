"""Download BBBC021 files with size checks and a SHA-256 manifest."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = "https://data.broadinstitute.org/bbbc/BBBC021"
METADATA_FILES = ("BBBC021_v1_image.csv", "BBBC021_v1_compound.csv", "BBBC021_v1_moa.csv")
USER_AGENT = "bbbc021-profiling"
CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> dict:
    """Download url to dest via a temporary file and return a manifest record.

    The download is rejected if the byte count differs from Content-Length.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
        expected = resp.headers.get("Content-Length")
        n = 0
        for block in iter(lambda: resp.read(CHUNK), b""):
            out.write(block)
            n += len(block)
    if expected is not None and int(expected) != n:
        tmp.unlink()
        raise OSError(f"{url}: received {n} bytes, expected {expected}")
    tmp.replace(dest)
    return {
        "file": dest.name,
        "url": url,
        "bytes": n,
        "sha256": sha256_file(dest),
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def download_metadata(dest_dir: Path, base_url: str = BASE_URL, force: bool = False) -> list[dict]:
    """Download the three metadata CSVs and write manifest.json beside them."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dest_dir / "manifest.json"
    old = {r["file"]: r for r in json.loads(manifest_path.read_text())} if manifest_path.exists() else {}
    records = []
    for name in METADATA_FILES:
        path = dest_dir / name
        if not force and name in old and path.exists() and sha256_file(path) == old[name]["sha256"]:
            records.append(old[name])
            continue
        records.append(fetch(f"{base_url}/{name}", path))
    manifest_path.write_text(json.dumps(records, indent=1) + "\n")
    return records


def head_sizes(urls: list[str]) -> dict[str, dict]:
    """Content-Length and Accept-Ranges for each URL from HEAD requests (no body is fetched)."""
    out = {}
    for url in urls:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as resp:
            out[url] = {
                "bytes": int(resp.headers["Content-Length"]),
                "accept_ranges": resp.headers.get("Accept-Ranges"),
                "last_modified": resp.headers.get("Last-Modified"),
            }
    return out
