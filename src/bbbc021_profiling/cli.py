"""Command-line entry point: one subcommand per pipeline stage."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from . import download, illumination, images, metadata, qc, remote_zip


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
    if args.verify:
        report = images.verify_images(needed, data / "images", data / "manifests" / "images")
        print(json.dumps(report, indent=1))
        if report["verified"] != report["tiffs_needed"] or report["failed"]:
            raise SystemExit(1)
        return
    session = remote_zip.HttpSession(download.USER_AGENT)
    summary = images.download_images(
        needed, data / "images", data / "manifests" / "images",
        lambda plate: remote_zip.HttpRangeReader(metadata.ZIP_URL.format(plate=plate), session),
        workers=args.workers, directories_only=args.directories_only, log=lambda s: print(s, flush=True),
    )
    print(json.dumps(summary, indent=1))


def _selected(args) -> pd.DataFrame:
    """The scope's fields, limited to --plate if given, after checking that their TIFFs are on disk."""
    data = _data_dir(args)
    selected = metadata.select_fields(metadata.load_fields(data / "metadata"), include_plate_controls=True)
    if args.plate:
        selected = selected[selected["plate"].isin(args.plate)]
    missing = [p for p in images.needed_files(selected)["path"] if not (data / "images" / p).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} TIFFs are missing, e.g. {missing[0]}; run the images stage first")
    return selected


def cmd_qc(args) -> None:
    data = _data_dir(args)
    out = data / "derived" / "qc"
    m = qc.run_qc(_selected(args), data / "images", out, workers=args.workers)
    s = qc.format_summary(m)
    s.to_csv(out / "format_summary.csv", index=False)
    print(s.to_string(index=False))
    flags = m["flag"].str.get_dummies(";").groupby(m["channel"]).sum()
    print(flags.to_string())
    print(f"flagged field-channels: {(m['flag'] != '').sum()} of {len(m)}")


def cmd_illumination_compare(args) -> None:
    df = illumination.run_compare(_selected(args), _data_dir(args) / "derived" / "qc")
    t = df.groupby(["channel", "estimator", "scope"]).agg(
        half_plates=("cv", "size"), fields_fit=("fields_fit", "median"), cv=("cv", "median"), max_over_min=("max_over_min", "median"))
    print(t.round(3).to_string())


def cmd_illumination_fit(args) -> None:
    df = illumination.run_fit(_selected(args), _data_dir(args) / "derived" / "qc", args.estimator, args.scope)
    print(df.groupby("channel")[["offset", "gain_min", "gain_max"]].describe().T.round(3).to_string())


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
    s.add_argument("--verify", action="store_true",
                   help="recompute every SHA-256 against the manifest instead of fetching")
    s.set_defaults(func=cmd_images)

    s = sub.add_parser("qc", help="per-field QC metrics, outlier flags and block summaries")
    s.add_argument("--plate", action="append", help="limit to this plate (repeatable)")
    s.add_argument("--workers", type=int, default=4, help="parallel processes (default 4)")
    s.set_defaults(func=cmd_qc)

    s = sub.add_parser("illumination-compare", help="held-out flatness of every illumination estimator and scope")
    s.add_argument("--plate", action="append", help="limit to this plate (repeatable)")
    s.set_defaults(func=cmd_illumination_compare)

    s = sub.add_parser("illumination-fit", help="fit and save the illumination functions")
    s.add_argument("--plate", action="append", help="limit to this plate (repeatable)")
    s.add_argument("--estimator", choices=illumination.ESTIMATORS, required=True)
    s.add_argument("--scope", choices=illumination.SCOPES, required=True)
    s.set_defaults(func=cmd_illumination_fit)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
