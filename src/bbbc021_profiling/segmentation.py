"""Nuclear and cell segmentation: candidate methods, object-level scoring, and checks that need no truth.

Nuclei
  classical   Gaussian smoothing, Otsu threshold, watershed of the distance transform from its peaks
  stardist    StarDist 2D_versatile_fluo with its default thresholds (1 to 99.8 percentile normalisation)
  cellpose    Cellpose nuclei model; the diameter is estimated per image (auto) or fixed for all images,
              at the default flow threshold 0.4 or a permissive 0.8 (flow08)
Cells
  watershed   Otsu foreground of log Actin, each pixel given to the nearest nucleus, so every cell has
              exactly one nucleus
  cyto3_*     Cellpose cyto3 on Actin, Tubulin or their mean, with DAPI as the nuclear channel
The deep-learning packages are optional and are imported only when a method is built, so scoring, the
classical method and the tests do not need them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import skimage.io
import tifffile
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops_table
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.segmentation import watershed

from .illumination import Corrector
from .metadata import CHANNELS

IOU = 0.5                       # objects match above this intersection over union
MIN_AREA = 25                   # px; smaller objects are ignored on both sides of a score (unet4nuclei min_nucleus_size)
CLIP_PERCENTILE = 99.8          # brightest pixels left out of an Otsu threshold, as in the illumination stage
SEED_SPACING = (0.2, 0.3, 0.4, 0.5)     # candidate seed distances for the classical method, in nuclear diameters
CELL_INPUTS = {"cyto3_actin": ("Actin",), "cyto3_tubulin": ("Tubulin",), "cyto3_both": ("Actin", "Tubulin")}
MCF7_NUCLEUS_PX = 36            # median equivalent diameter of DMSO nuclei, from the illumination check (36.3 px)


# Scoring -------------------------------------------------------------------------------------------

def overlaps(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersection counts of two label images, and the pixel counts of each label (background 0 included)."""
    na, nb = int(a.max()) + 1, int(b.max()) + 1
    inter = np.bincount(a.ravel().astype(np.int64) * nb + b.ravel(), minlength=na * nb).reshape(na, nb)
    return inter, inter.sum(axis=1), inter.sum(axis=0)


def ignore_small(labels: np.ndarray, min_area: int = MIN_AREA) -> np.ndarray:
    """Labels with fewer than min_area pixels set to background."""
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return np.where(sizes[labels] < min_area, 0, labels)


def match(truth: np.ndarray, pred: np.ndarray, iou: float = IOU) -> dict:
    """Object-level counts. Above IoU 0.5 a truth object can match at most one prediction and the
    other way round, so the pairs are the matrix entries above the threshold."""
    if iou < 0.5:
        raise ValueError("below IoU 0.5 a match is not unique")
    inter, a, b = overlaps(truth, pred)
    inter, a, b = inter[1:, 1:], a[1:], b[1:]
    with np.errstate(invalid="ignore"):
        ratio = inter / np.maximum(a[:, None] + b[None, :] - inter, 1)
    tp = int((ratio > iou).sum()) if ratio.size else 0
    n_truth, n_pred = int((a > 0).sum()), int((b > 0).sum())
    matched = ratio[ratio > iou] if ratio.size else np.array([])
    return {"truth": n_truth, "pred": n_pred, "tp": tp, "fp": n_pred - tp, "fn": n_truth - tp,
            "matched_iou": float(matched.mean()) if tp else float("nan")}


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    return p, r, 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")


def summarise(per_image: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Pooled precision, recall and F1 per method, with bootstrap intervals over images."""
    rows = []
    for (method, split), g in per_image.groupby(["method", "split"], sort=False):
        rng = np.random.default_rng(seed)       # the same resampled images for every method
        c = g[["tp", "fp", "fn"]].to_numpy()
        p, r, f = prf(*c.sum(axis=0))
        boots = [prf(*c[rng.integers(len(c), size=len(c))].sum(axis=0)) for _ in range(n_boot)]
        lo, hi = np.nanpercentile(np.array(boots), [2.5, 97.5], axis=0)
        c75 = g[["tp75", "fp75", "fn75"]].to_numpy().sum(axis=0)
        rows.append({"method": method, "split": split, "images": len(g), "truth": int(g["truth"].sum()),
                     "pred": int(g["pred"].sum()), "tp": int(c[:, 0].sum()),
                     "precision": p, "recall": r, "f1": f, "f1_low": lo[2], "f1_high": hi[2],
                     "precision_low": lo[0], "precision_high": hi[0], "recall_low": lo[1], "recall_high": hi[1],
                     "f1_iou75": prf(*c75)[2], "matched_iou": float((g["matched_iou"] * g["tp"]).sum() / max(g["tp"].sum(), 1)),
                     "seconds_per_image": g["seconds"].median()})
    return pd.DataFrame(rows)


# Methods -------------------------------------------------------------------------------------------

def otsu_foreground(img: np.ndarray, sigma: float) -> np.ndarray:
    """Smoothed image above its Otsu threshold, with the brightest pixels left out of the threshold."""
    s = gaussian(img.astype(np.float32), sigma, preserve_range=True)
    return s > threshold_otsu(np.minimum(s, np.percentile(s, CLIP_PERCENTILE)))


def cell_foreground(actin: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Otsu threshold of the smoothed log intensity. Actin brightness varies by an order of magnitude
    between cells, so a linear-intensity threshold would drop the dim cells of a confluent field."""
    s = gaussian(actin.astype(np.float32), sigma, preserve_range=True)
    logged = np.log1p(np.maximum(s - np.percentile(s, 0.5), 0))
    return logged > threshold_otsu(np.minimum(logged, np.percentile(logged, CLIP_PERCENTILE)))


def classical_nuclei(img: np.ndarray, diameter: float, seed_spacing: float) -> np.ndarray:
    """Otsu foreground split by a watershed of the distance transform."""
    mask = otsu_foreground(img, max(1.0, diameter / 30))
    area = np.pi * diameter ** 2 / 4
    mask = remove_small_holes(remove_small_objects(mask, max_size=int(0.1 * area)), max_size=int(0.2 * area))
    distance = ndimage.gaussian_filter(ndimage.distance_transform_edt(mask), diameter / 20)
    seeds = peak_local_max(distance, min_distance=max(1, round(seed_spacing * diameter)), labels=label(mask), exclude_border=False)
    markers = np.zeros(img.shape, np.int32)
    markers[tuple(seeds.T)] = np.arange(1, len(seeds) + 1)
    return watershed(-distance, markers, mask=mask).astype(np.int32)


def watershed_cells(actin: np.ndarray, nuclei: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """One cell per nucleus: log-Otsu foreground of Actin, each pixel given to the nearest nucleus.

    The flood runs over the distance to the nuclei, so touching cells split halfway between them. A
    flood over the Actin intensity instead leaks along the dim cell rims into neighbouring cells.
    """
    mask = cell_foreground(actin, sigma) | (nuclei > 0)
    distance = ndimage.distance_transform_edt(nuclei == 0)
    return watershed(distance, nuclei.astype(np.int32), mask=mask).astype(np.int32)


class ClassicalNuclei:
    """Otsu and distance watershed; diameter in pixels, seed_spacing in diameters."""

    def __init__(self, diameter: float, seed_spacing: float):
        self.diameter, self.seed_spacing = diameter, seed_spacing

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return classical_nuclei(img, self.diameter, self.seed_spacing)


class StarDistNuclei:
    def __init__(self):
        from stardist.models import StarDist2D
        self.model = StarDist2D.from_pretrained("2D_versatile_fluo")

    def __call__(self, img: np.ndarray) -> np.ndarray:
        from csbdeep.utils import normalize
        return self.model.predict_instances(normalize(img.astype(np.float32), 1, 99.8))[0].astype(np.int32)


class CellposeSegmenter:
    """Cellpose model with a fixed diameter, or with the size model's per-image estimate (diameter=None).

    A higher flow_threshold keeps masks whose shape disagrees more with the predicted flows; 0.4 is the
    Cellpose default.
    """

    def __init__(self, model_type: str, diameter: float | None, gpu: bool = True, flow_threshold: float = 0.4):
        from cellpose import models
        self.model = models.Cellpose(gpu=gpu, model_type=model_type)
        self.diameter = diameter
        self.flow_threshold = flow_threshold
        self.last_diameter = float("nan")

    def __call__(self, img: np.ndarray, nucleus: np.ndarray | None = None) -> np.ndarray:
        x, channels = (img, [0, 0]) if nucleus is None else (np.stack([img, nucleus], axis=-1), [1, 2])
        masks, _, _, diams = self.model.eval(x.astype(np.float32), diameter=self.diameter, channels=channels,
                                             flow_threshold=self.flow_threshold)
        self.last_diameter = float(diams)
        return masks.astype(np.int32)


def nuclei_method(name: str, gpu: bool = True, diameter: float | None = None, seed_spacing: float = 0.3) -> Callable:
    """Build one of the nuclear methods: classical, stardist, cellpose_auto or cellpose_fixed."""
    if name == "classical":
        return ClassicalNuclei(diameter, seed_spacing)
    if name == "stardist":
        return StarDistNuclei()
    if name in ("cellpose_auto", "cellpose_fixed", "cellpose_fixed_flow08"):
        return CellposeSegmenter("nuclei", None if name == "cellpose_auto" else diameter, gpu,
                                 0.8 if name.endswith("flow08") else 0.4)
    raise ValueError(name)


NUCLEI_METHODS = ("classical", "stardist", "cellpose_auto", "cellpose_fixed", "cellpose_fixed_flow08")


# Validation on BBBC039 -------------------------------------------------------------------------------

def bbbc039_truth(path: Path) -> np.ndarray:
    """Object labels of a BBBC039 mask: connected regions of the first channel (the dataset's decoding gist)."""
    m = skimage.io.imread(path)
    return ignore_small(label(m[..., 0] if m.ndim == 3 else m, connectivity=1).astype(np.int32))


def equivalent_diameter(labels: np.ndarray) -> np.ndarray:
    return 2 * np.sqrt(np.bincount(labels.ravel())[1:][np.bincount(labels.ravel())[1:] > 0] / np.pi)


def validate(bbbc039: Path, method: Callable, name: str, split: str) -> pd.DataFrame:
    """Counts of matched and unmatched nuclei on every image of one split."""
    rows = []
    for n in (bbbc039 / "metadata" / f"{split}.txt").read_text().split():
        img = tifffile.imread(bbbc039 / "images" / n.replace(".png", ".tif")).astype(np.float32)
        truth = bbbc039_truth(bbbc039 / "masks" / n)
        t = time.perf_counter()
        pred = method(img)
        seconds = time.perf_counter() - t
        pred = ignore_small(pred)
        m75 = match(truth, pred, 0.75)
        rows.append({"method": name, "split": split, "image": n, **match(truth, pred), "tp75": m75["tp"],
                     "fp75": m75["fp"], "fn75": m75["fn"], "seconds": seconds,
                     "diameter_px": getattr(method, "last_diameter", np.nan)})
    return pd.DataFrame(rows)


def training_diameter(bbbc039: Path) -> float:
    """Median equivalent diameter of the training nuclei, used as the fixed diameter."""
    names = (bbbc039 / "metadata" / "training.txt").read_text().split()
    return float(np.median(np.concatenate([equivalent_diameter(bbbc039_truth(bbbc039 / "masks" / n)) for n in names])))


def tune_seed_spacing(bbbc039: Path, diameter: float, spacings: tuple = SEED_SPACING) -> tuple[float, pd.DataFrame]:
    """The classical seed spacing with the best pooled F1 on the training images, and the scores of all."""
    rows = []
    for s in spacings:
        c = validate(bbbc039, ClassicalNuclei(diameter, s), "classical", "training")[["tp", "fp", "fn"]].sum()
        p, r, f = prf(*c)
        rows.append({"seed_spacing": s, "precision": p, "recall": r, "f1": f})
    t = pd.DataFrame(rows)
    return float(t.loc[t["f1"].idxmax(), "seed_spacing"]), t


# Benchmark on MCF-7 fields ---------------------------------------------------------------------------

@dataclass
class Sample:
    """Fields for the runtime benchmark and visual checks: one per mechanism of action, plus DMSO."""

    fields: pd.DataFrame
    calibration: pd.DataFrame       # DMSO fields used only to set the fixed diameters


def sample_fields(selected: pd.DataFrame, seed: int = 0, calibration: int = 8) -> Sample:
    treatments = selected[selected["role"] == "treatment"]
    picked = treatments.groupby("moa", sort=True).sample(n=1, random_state=seed)
    dmso = selected[selected["role"] == "dmso"]
    one = dmso.sample(n=1, random_state=seed)
    rest = dmso.drop(one.index).groupby("week", sort=True).sample(n=1, random_state=seed).sample(
        n=min(calibration, dmso["week"].nunique()), random_state=seed)
    return Sample(pd.concat([picked, one]).reset_index(drop=True), rest.reset_index(drop=True))


def field_id(r) -> str:
    return f"{r['plate']}_{r['well']}_{r['image_number']}"


def load_channels(r, image_root: Path, corrector: Corrector) -> dict[str, np.ndarray]:
    return {c: corrector(tifffile.imread(image_root / r[f"path_{c}"]), r["plate"], c) for c in CHANNELS}


def field_checks(nuclei: np.ndarray, cells: np.ndarray) -> dict:
    """Counts and ratios that need no ground truth, for one field."""
    nuc_area, cell_area = np.bincount(nuclei.ravel())[1:], np.bincount(cells.ravel())[1:]
    n_nuc, n_cells = int((nuc_area > 0).sum()), int((cell_area > 0).sum())
    edge = np.concatenate([cells[0], cells[-1], cells[:, 0], cells[:, -1]])
    border = np.setdiff1d(np.unique(edge), [0])
    inter, _, _ = overlaps(nuclei, cells)
    inter = inter[1:, :]
    present = inter.sum(axis=1) > 0
    owner = np.where(present, inter[:, 1:].argmax(axis=1) + 1 if inter.shape[1] > 1 else 0, 0)
    owner = np.where(inter[:, 1:].max(axis=1) > 0, owner, 0) if inter.shape[1] > 1 else np.zeros(len(inter), int)
    per_cell = np.bincount(owner[present], minlength=len(cell_area) + 1)[1:]
    cells_present = cell_area > 0
    single = np.flatnonzero((per_cell == 1) & cells_present) + 1
    nuc_of = {c: i + 1 for i, c in enumerate(owner) if present[i]}
    ratios = [nuc_area[nuc_of[c] - 1] / cell_area[c - 1] for c in single]
    return {
        "nuclei": n_nuc, "cells": n_cells,
        "cells_touching_border": int(len(border)),
        "nuclei_without_cell": int((present & (owner == 0)).sum()),
        "cells_without_nucleus": int((cells_present & (per_cell == 0)).sum()),
        "cells_with_2plus_nuclei": int((cells_present & (per_cell >= 2)).sum()),
        "median_nucleus_area_px": float(np.median(nuc_area[nuc_area > 0])) if n_nuc else np.nan,
        "median_cell_area_px": float(np.median(cell_area[cells_present])) if n_cells else np.nan,
        "median_nucleus_cell_ratio": float(np.median(ratios)) if ratios else np.nan,
    }


def cells_body(kind: str, img: dict[str, np.ndarray]) -> np.ndarray:
    """The cyto3 cell-body image for a kind: one channel, or the mean of its channels."""
    chans = CELL_INPUTS[kind]
    return np.mean([img[c] for c in chans], axis=0) if len(chans) > 1 else img[chans[0]]


def cells_from(kind: str, models: dict, nuclei_name: str) -> Callable:
    """A cell method (image channels, nuclear labels by method) -> labels, for run_benchmark.

    kind is watershed, or one of CELL_INPUTS for cyto3 (models holds the Cellpose segmenters by kind).
    """
    if kind == "watershed":
        return lambda img, nuc: watershed_cells(img["Actin"], nuc[nuclei_name])
    model = models[kind]

    def run(img, nuc):
        return model(cells_body(kind.rsplit("_", 1)[0], img), img["DAPI"])
    run.model = model
    return run


def run_benchmark(sample: Sample, nuclei: dict[str, Callable], cells: dict[str, Callable], load: Callable,
                  out: Path) -> pd.DataFrame:
    """Labels and times for every field. Nuclear methods run first; the cell methods reuse their labels.

    Labels are saved in out as <kind>_<method>_<field>.npz. The first field is run once beforehand so
    that model warm-up is not timed.
    """
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    first = load(sample.fields.iloc[0])
    warm = {}
    for name, m in nuclei.items():
        warm[name] = m(first["DAPI"])
    for m in cells.values():
        m(first, warm)
    for _, r in sample.fields.iterrows():
        img = load(r)
        nuc = {}
        for name, m in nuclei.items():
            t = time.perf_counter()
            nuc[name] = m(img["DAPI"])
            rows.append({"kind": "nuclei", "method": name, "field": field_id(r), "seconds": time.perf_counter() - t,
                         "diameter_px": getattr(m, "last_diameter", np.nan)})
            np.savez_compressed(out / f"nuclei_{name}_{field_id(r)}.npz", labels=nuc[name])
        for name, m in cells.items():
            t = time.perf_counter()
            lab = m(img, nuc)
            rows.append({"kind": "cells", "method": name, "field": field_id(r), "seconds": time.perf_counter() - t,
                         "diameter_px": getattr(getattr(m, "model", None), "last_diameter", np.nan)})
            np.savez_compressed(out / f"cells_{name}_{field_id(r)}.npz", labels=lab)
    return pd.DataFrame(rows)


def checks(sample: Sample, pairs: dict[str, str], labels: Path) -> pd.DataFrame:
    """field_checks of every cell method (pairs maps it to the nuclear method it belongs with)."""
    rows = []
    for _, r in sample.fields.iterrows():
        fid = field_id(r)
        for cell_name, nuc_name in pairs.items():
            nuc = np.load(labels / f"nuclei_{nuc_name}_{fid}.npz")["labels"]
            lab = np.load(labels / f"cells_{cell_name}_{fid}.npz")["labels"]
            rows.append({"cell_method": cell_name, "nuclei_method": nuc_name, "field": fid, "moa": r["moa"], **field_checks(nuc, lab)})
    return pd.DataFrame(rows)


def weight_files(*roots: Path) -> list[dict]:
    """Name, size and SHA-256 of every downloaded model file, for the record."""
    from .download import sha256_file
    return [{"file": f.name, "bytes": f.stat().st_size, "sha256": sha256_file(f)}
            for root in roots for f in sorted(root.rglob("*")) if f.is_file()]


def nuclei_counts(fields: pd.DataFrame, methods: tuple[str, ...], labels: Path) -> pd.DataFrame:
    """Nuclei found per sampled field by every method (from the saved labels)."""
    rows = []
    for _, r in fields.iterrows():
        row = {"moa": r["moa"] if isinstance(r["moa"], str) else "DMSO", "field": field_id(r)}
        for m in methods:
            path = labels / f"nuclei_{m}_{field_id(r)}.npz"
            row[m] = int(np.load(path)["labels"].max()) if path.exists() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# Reports ---------------------------------------------------------------------------------------------

TOTAL_FIELDS = 3560             # fields of the selected scope (treatments, DMSO and plate controls)
PIPELINES = {"classical": ("classical", "watershed_classical"), "stardist": ("stardist", "watershed_stardist"),
             "cellpose nuclei + cyto3 (actin)": ("cellpose_fixed", "cyto3_actin_fixed"),
             "cellpose nuclei + watershed": ("cellpose_fixed", "watershed_cellpose")}


def runtime_tables(bench: pd.DataFrame, fields: int = TOTAL_FIELDS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seconds per field for every method, and for the whole pipelines with their projected total."""
    t = bench.groupby(["kind", "method"])["seconds"].agg(["median", "mean", "max"]).reset_index()
    t["projected_hours_for_all_fields"] = t["mean"] * fields / 3600
    per = bench.pivot_table(index="field", columns=["kind", "method"], values="seconds")
    rows = []
    for name, (nuc, cell) in PIPELINES.items():
        if ("nuclei", nuc) in per and ("cells", cell) in per:
            total = per[("nuclei", nuc)] + per[("cells", cell)]
            rows.append({"pipeline": name, "median_s_per_field": total.median(), "mean_s_per_field": total.mean(),
                         "max_s_per_field": total.max(), "projected_hours": total.mean() * fields / 3600})
    return t, pd.DataFrame(rows)


def check_table(checks_df: pd.DataFrame) -> pd.DataFrame:
    """Means over the sampled fields, per cell method."""
    cols = ["nuclei", "cells", "cells_touching_border", "nuclei_without_cell", "cells_without_nucleus",
            "cells_with_2plus_nuclei", "median_nucleus_area_px", "median_cell_area_px", "median_nucleus_cell_ratio"]
    return checks_df.groupby("cell_method", sort=False)[cols].mean().reset_index()
