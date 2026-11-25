"""Command-line entry point: one subcommand per pipeline stage."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from . import download, figures, illumination, images, metadata, qc, remote_zip, segmentation


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


def cmd_figures(args) -> None:
    data = _data_dir(args)
    for path in figures.run(_selected(args), data / "images", data / "derived" / "qc", args.scope):
        print(path)


def _bbbc039_dir(args) -> Path:
    return Path(args.bbbc039_dir or os.environ.get("BBBC039_DATA_DIR", _data_dir(args).parent / "bbbc039"))


def cmd_segment_validate(args) -> None:
    bbbc039 = _bbbc039_dir(args)
    out = _data_dir(args) / "derived" / "segmentation"
    out.mkdir(parents=True, exist_ok=True)
    diameter = segmentation.training_diameter(bbbc039)
    print(f"median training diameter {diameter:.1f} px")
    frames = []
    if args.method == "classical":
        spacing, tuning = segmentation.tune_seed_spacing(bbbc039, diameter)
        tuning.to_csv(out / "classical_seed_spacing.csv", index=False)
        print(tuning.round(4).to_string(index=False), f"\nseed spacing {spacing}")
        methods = {"classical": segmentation.ClassicalNuclei(diameter, spacing)}
    elif args.method == "stardist":
        methods = {"stardist": segmentation.nuclei_method("stardist")}
    else:
        methods = {"cellpose_auto": segmentation.nuclei_method("cellpose_auto", gpu=not args.cpu),
                   "cellpose_fixed": segmentation.nuclei_method("cellpose_fixed", gpu=not args.cpu, diameter=diameter),
                   "cellpose_fixed_flow08": segmentation.nuclei_method("cellpose_fixed_flow08", gpu=not args.cpu, diameter=diameter)}
    for name, m in methods.items():
        frames.append(segmentation.validate(bbbc039, m, name, args.split))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out / f"validation_{args.method}_{args.split}.csv", index=False)
    print(segmentation.summarise(df).round(4).T.to_string())


def cmd_segment_benchmark(args) -> None:
    data, out = _data_dir(args), _data_dir(args) / "derived" / "segmentation"
    selected = _selected(args)
    selected = selected[selected["role"] != "plate_control"] if not args.include_plate_controls else selected
    sample = segmentation.sample_fields(selected, seed=args.seed)
    (out / "local_only").mkdir(parents=True, exist_ok=True)
    sample.fields.to_csv(out / "sample_fields.csv", index=False)
    corrector = illumination.Corrector(data / "derived" / "qc")
    load = lambda r: segmentation.load_channels(r, data / "images", corrector)
    labels = out / "local_only" / "labels"
    d = args.diameter
    nuclei, cells, models, calibration = {}, {}, {}, []
    if args.method == "classical":
        nuclei = {"classical": segmentation.ClassicalNuclei(d, args.seed_spacing)}
        cells = {"watershed_classical": segmentation.cells_from("watershed", {}, "classical")}
    elif args.method == "stardist":
        nuclei = {"stardist": segmentation.nuclei_method("stardist")}
        cells = {"watershed_stardist": segmentation.cells_from("watershed", {}, "stardist")}
    else:
        gpu = not args.cpu
        auto = {k: segmentation.CellposeSegmenter("nuclei" if k == "nuclei" else "cyto3", None, gpu) for k in ("nuclei", *segmentation.CELL_INPUTS)}
        fixed_at = {}
        for k, m in auto.items():
            diams = []
            for _, r in sample.calibration.iterrows():
                img = load(r)
                body = img["DAPI"] if k == "nuclei" else segmentation.cells_body(k, img)
                m(body, None if k == "nuclei" else img["DAPI"])
                diams.append(m.last_diameter)
            fixed_at[k] = float(pd.Series(diams).median())
            calibration += [{"method": k, "field_diameter_px": x, "field": segmentation.field_id(r)} for x, (_, r) in zip(diams, sample.calibration.iterrows())]
        print("fixed diameters (median auto over DMSO calibration fields):", {k: round(v, 1) for k, v in fixed_at.items()})
        pd.DataFrame(calibration).to_csv(out / "cellpose_calibration.csv", index=False)
        nuclei = {"cellpose_auto": auto["nuclei"],
                  "cellpose_fixed": segmentation.CellposeSegmenter("nuclei", fixed_at["nuclei"], gpu),
                  "cellpose_fixed_flow08": segmentation.CellposeSegmenter("nuclei", fixed_at["nuclei"], gpu, 0.8)}
        for k in segmentation.CELL_INPUTS:
            models[f"{k}_auto"], models[f"{k}_fixed"] = auto[k], segmentation.CellposeSegmenter("cyto3", fixed_at[k], gpu)
            for mode in ("auto", "fixed"):
                cells[f"{k}_{mode}"] = segmentation.cells_from(f"{k}_{mode}", models, "cellpose_fixed")
        cells["watershed_cellpose"] = segmentation.cells_from("watershed", {}, "cellpose_fixed")
    df = segmentation.run_benchmark(sample, nuclei, cells, load, labels)
    df.to_csv(out / f"benchmark_{args.method}.csv", index=False)
    pairs = {name: "cellpose_fixed" if name.startswith("cyto3") or name == "watershed_cellpose" else name.rsplit("_", 1)[1]
             for name in cells}
    segmentation.checks(sample, pairs, labels).to_csv(out / f"checks_{args.method}.csv", index=False)
    print(df.groupby(["kind", "method"])["seconds"].describe()[["count", "mean", "50%", "max"]].round(2).to_string())


def cmd_segment_report(args) -> None:
    out = _data_dir(args) / "derived" / "segmentation"
    val = [pd.read_csv(f) for f in sorted(out.glob(f"validation_*_{args.split}.csv")) if not f.name.startswith("validation_summary")]
    if val:
        s = segmentation.summarise(pd.concat(val, ignore_index=True))
        s.to_csv(out / f"validation_summary_{args.split}.csv", index=False)
        print(s.drop(columns=["truth", "pred", "tp"]).round(4).to_string(index=False), "\n")
    bench = pd.concat([pd.read_csv(f) for f in sorted(out.glob("benchmark_*.csv"))], ignore_index=True)
    methods, pipes = segmentation.runtime_tables(bench)
    methods.to_csv(out / "runtime_methods.csv", index=False)
    pipes.to_csv(out / "runtime_pipelines.csv", index=False)
    print(methods.round(3).to_string(index=False), "\n\n", pipes.round(3).to_string(index=False), "\n")
    counts = segmentation.nuclei_counts(pd.read_csv(out / "sample_fields.csv"), segmentation.NUCLEI_METHODS, out / "local_only" / "labels")
    counts.to_csv(out / "nuclei_counts.csv", index=False)
    print(counts.drop(columns="field").to_string(index=False), "\n")
    checks = pd.concat([pd.read_csv(f) for f in sorted(out.glob("checks_*.csv"))], ignore_index=True)
    t = segmentation.check_table(checks)
    t.to_csv(out / "checks_summary.csv", index=False)
    print(t.round(3).to_string(index=False))
    corrector = illumination.Corrector(_data_dir(args) / "derived" / "qc")
    load = lambda r: segmentation.load_channels(r, _data_dir(args) / "images", corrector)
    fields = pd.read_csv(out / "sample_fields.csv")
    for path in figures.segmentation_overlays(fields, out / "local_only" / "labels", load, out / "local_only" / "overlays",
                                              ["classical", "stardist", "cellpose_fixed", "cellpose_fixed_flow08"],
                                              ["watershed_classical", "watershed_stardist", "cyto3_actin_fixed", "cyto3_both_fixed"]):
        print(path)


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

    s = sub.add_parser("figures", help="QC and flatness figures, and local-only examples of flagged fields")
    s.add_argument("--plate", action="append", help="limit to this plate (repeatable)")
    s.add_argument("--scope", choices=illumination.SCOPES, default="plate", help="scope shown in the flatness maps (default plate)")
    s.set_defaults(func=cmd_figures)

    for name, fn, help_ in (("segment-validate", cmd_segment_validate, "score nuclear methods on the BBBC039 ground truth"),
                            ("segment-benchmark", cmd_segment_benchmark, "time every method on sampled BBBC021 fields and save labels")):
        s = sub.add_parser(name, help=help_)
        s.add_argument("--method", choices=("classical", "stardist", "cellpose"), required=True)
        s.add_argument("--bbbc039-dir", help="BBBC039 folder (default: $BBBC039_DATA_DIR or ../bbbc039 beside the data root)")
        s.add_argument("--split", choices=("training", "validation", "test"), default="test", help="BBBC039 split to score (default test)")
        s.add_argument("--cpu", action="store_true", help="run Cellpose on the CPU")
        s.add_argument("--plate", action="append", help=argparse.SUPPRESS)
        s.add_argument("--include-plate-controls", action="store_true", help=argparse.SUPPRESS)
        s.add_argument("--diameter", type=float, default=segmentation.MCF7_NUCLEUS_PX, help="nuclear diameter in px for the classical method (default 36)")
        s.add_argument("--seed-spacing", type=float, default=0.3, help="seed distance in diameters for the classical method")
        s.add_argument("--seed", type=int, default=0, help="random seed for the field sample")
        s.set_defaults(func=fn)

    s = sub.add_parser("segment-report", help="combine the validation, runtime and check tables and draw local-only overlays")
    s.add_argument("--split", default="test")
    s.set_defaults(func=cmd_segment_report)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
