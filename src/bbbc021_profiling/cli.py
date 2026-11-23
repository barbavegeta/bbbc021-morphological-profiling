"""Command-line entry point: one subcommand per pipeline stage."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from . import download, images, metadata, remote_zip


def _data_dir(args) -> Path:
    return Path(args.data_dir or os.environ.get("BBBC021_DATA_DIR", "data"))


def cmd_metadata(args) -> None:
    for r in download.download_metadata(_data_dir(args) / "metadata", force=args.force):
        print(f"{r['file']}\t{r['bytes']} bytes\tsha256 {r['sha256']}")


def cmd_zip_sizes(args) -> None:
    meta = _data_dir(args) / "metadata"
    plates = sorted(metadata.load_fields(meta)["plate"].unique())
    urls = [metadata.ZIP_URL.format(plate=p) for p in plates]
    sizes = download.head_sizes(urls)
    df = pd.DataFrame([{"plate": p, "zip_url": u, **sizes[u]} for p, u in zip(plates, urls)])
    df.to_csv(meta / "zip_sizes.csv", index=False)
    print(f"{len(df)} ZIPs, {df['bytes'].sum() / 1e9:.2f} GB, range support: {sorted(df['accept_ranges'].dropna().unique())}")


def cmd_scope(args) -> None:
    data = _data_dir(args)
    fields = metadata.load_fields(data / "metadata")
    selected = metadata.select_fields(fields, include_plate_controls=args.include_plate_controls)
    sizes_csv = data / "metadata" / "zip_sizes.csv"
    zip_bytes = dict(pd.read_csv(sizes_csv)[["zip_url", "bytes"]].values) if sizes_csv.exists() else None
    plates = metadata.plate_table(fields, selected, zip_bytes)
    summary = metadata.scope_summary(fields, selected, plates)
    summary["include_plate_controls"] = args.include_plate_controls

    out = data / "derived" / ("scope_with_controls" if args.include_plate_controls else "scope")
    out.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out / "fields_selected.csv", index=False)
    plates.to_csv(out / "plates.csv", index=False)
    metadata.treatment_summary(fields).to_csv(out / "treatments.csv", index=False)
    fields.drop_duplicates(["plate", "well"])[["plate", "week", "well", "role", "compound", "concentration", "moa"]] \
        .sort_values(["plate", "well"]).to_csv(out / "wells.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


def cmd_images(args) -> None:
    data = _data_dir(args)
    fields = metadata.load_fields(data / "metadata")
    needed = images.needed_files(metadata.select_fields(fields, include_plate_controls=args.include_plate_controls))
    if args.plate:
        unknown = set(args.plate) - set(needed["plate"])
        if unknown:
            raise SystemExit(f"no selected TIFFs on plate(s): {sorted(unknown)}")
        needed = needed[needed["plate"].isin(args.plate)]
    session = remote_zip.HttpSession(download.USER_AGENT)
    summary = images.download_images(
        needed, data / "images", data / "manifests" / "images",
        lambda plate: remote_zip.HttpRangeReader(metadata.ZIP_URL.format(plate=plate), session),
        workers=args.workers, directories_only=args.directories_only, log=lambda s: print(s, flush=True),
    )
    print(json.dumps(summary, indent=1))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="bbbc021", description=__doc__)
    p.add_argument("--data-dir", help="data root (default: $BBBC021_DATA_DIR or ./data)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("metadata", help="download the three metadata CSVs with a SHA-256 manifest")
    s.add_argument("--force", action="store_true", help="download again even if the files match the manifest")
    s.set_defaults(func=cmd_metadata)

    s = sub.add_parser("zip-sizes", help="record each plate ZIP's size from HTTP HEAD requests")
    s.set_defaults(func=cmd_zip_sizes)

    s = sub.add_parser("scope", help="join metadata and count the plates, wells, fields and TIFFs in scope")
    s.add_argument("--include-plate-controls", action="store_true",
                   help="also keep the plate-control wells on the selected plates")
    s.set_defaults(func=cmd_scope)

    s = sub.add_parser("images", help="fetch the selected TIFFs from the plate ZIPs with range requests")
    s.add_argument("--include-plate-controls", action="store_true",
                   help="also fetch the plate-control wells on the selected plates")
    s.add_argument("--plate", action="append", help="limit to this plate (repeatable)")
    s.add_argument("--workers", type=int, default=4, help="parallel connections (default 4)")
    s.add_argument("--directories-only", action="store_true",
                   help="read and cache each ZIP's file list and report exact sizes, without fetching TIFFs")
    s.set_defaults(func=cmd_images)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
