"""Per-field image QC: one pass over every selected TIFF, then outlier flags.

For each field and channel the pass records the TIFF's format (dimensions, bit depth, exposure, pixel
size) and
  focus          slope of the radially averaged power spectrum on log-log axes over wavelengths of
                 8 to 64 px; blur makes it more negative. Shorter wavelengths are left out because
                 sensor noise dominates them, and a blurred noisy image would look sharper there.
  saturated_pct  percentage of pixels at the image's own maximum, when that maximum lies in the clipping
                 range. The images clip at a level that differs between plates and lies below 4095, so
                 a clipped image is recognised by its pixels piling up on one top value.
  background     1st percentile of pixel intensities
It also saves block summaries per plate and channel (see foreground_blocks) for the illumination
correction, so that stage does not read the images again.
"""

from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.filters import threshold_otsu

from .metadata import CHANNELS

CLIP_MIN = 3900 * 16        # lowest clip level; 12-bit camera values are stored multiplied by 16 in the 16-bit TIFFs
BLOCK = 16                  # block size in pixels for the illumination summaries
FOCUS_WAVELENGTHS = (8, 64) # pixels; the band of the power spectrum used for the focus slope
FLAG_Z = 5.0                # |robust z| above which a metric is flagged within its plate and channel
FLAG_SATURATED_PCT = 0.1    # percentage of saturated pixels above which a field is flagged


def read_tiff(path: Path) -> tuple[np.ndarray, dict]:
    """Pixels of a single-page TIFF, and its format as recorded in the tags."""
    with tifffile.TiffFile(path) as t:
        page = t.pages[0]
        img = page.asarray()
        uic = page.tags.get("UIC1tag")
        uic = uic.value if uic is not None else {}
        exposure = re.search(r"Exposure:\s*([\d.]+)\s*ms", page.description or "")
        return img, {
            "dtype": str(img.dtype), "height": img.shape[0], "width": img.shape[1],
            "exposure_ms": float(exposure.group(1)) if exposure else float("nan"),
            "pixel_um": float(uic.get("XCalibration", np.nan)),
        }


@lru_cache(maxsize=4)
def _radial_index(shape: tuple[int, int]) -> tuple[np.ndarray, int]:
    h, w = shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    n = min(h, w)
    return np.rint(np.sqrt(fx ** 2 + fy ** 2) * n).astype(np.int64), n // 2


def power_log_log_slope(img: np.ndarray, wavelengths: tuple[int, int] = FOCUS_WAVELENGTHS) -> float:
    """Least-squares slope of log power against log spatial frequency over a band of wavelengths."""
    a = img.astype(np.float64)
    a -= a.mean()
    power = np.abs(np.fft.rfft2(a)) ** 2
    k, kmax = _radial_index(a.shape)
    radial = np.bincount(k.ravel(), weights=power.ravel(), minlength=kmax + 1) / np.bincount(k.ravel(), minlength=kmax + 1)
    n = min(a.shape)
    radii = np.arange(max(1, n // wavelengths[1]), n // wavelengths[0] + 1)
    radii = radii[radial[radii] > 0]
    if len(radii) < 2:
        return float("nan")
    return float(np.polyfit(np.log(radii), np.log(radial[radii]), 1)[0])


def field_metrics(img: np.ndarray) -> dict:
    p1, p50, p99 = np.percentile(img, [1, 50, 99])
    at_max = int((img == img.max()).sum())
    return {
        "focus": power_log_log_slope(img),
        "saturated_pct": 100 * at_max / img.size if img.max() >= CLIP_MIN else 0.0,
        "background": float(p1), "median": float(p50), "p99": float(p99),
        "min": int(img.min()), "max": int(img.max()), "pixels_at_max": at_max,
        "multiple_of_16": bool((img % 16 == 0).all()),
    }


def block_sum(a: np.ndarray, block: int = BLOCK) -> np.ndarray:
    h, w = a.shape
    if h % block or w % block:
        raise ValueError(f"image shape {a.shape} is not a multiple of {block}")
    return a.reshape(h // block, block, w // block, block).sum(axis=(1, 3))


def foreground_blocks(img: np.ndarray, block: int = BLOCK) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Block means of the image, and per block the summed intensity and pixel count of foreground.

    Foreground is above the Otsu threshold of the 2x2-binned image, with its brightest 0.2% clipped
    so that debris does not set the threshold.
    """
    a = img.astype(np.float32)
    h, w = a.shape
    small = a.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    mask = a > threshold_otsu(np.minimum(small, np.percentile(small, 99.8)))
    return (block_sum(a, block) / block ** 2, block_sum(np.where(mask, a, 0), block),
            block_sum(mask.astype(np.float32), block))


def process_plate(plate: str, fields: pd.DataFrame, image_root: Path, block_dir: Path) -> pd.DataFrame:
    """Metrics for every field and channel of one plate; writes one block-summary file per channel."""
    block_dir.mkdir(parents=True, exist_ok=True)
    fields = fields.sort_values(["well", "image_number"])
    rows = []
    for ch in CHANNELS:
        mean, fg_sum, fg_count, background = [], [], [], []
        for r in fields.itertuples(index=False):
            img, fmt = read_tiff(image_root / getattr(r, f"path_{ch}"))
            m = field_metrics(img)
            b = foreground_blocks(img)
            rows.append({"plate": plate, "week": r.week, "well": r.well, "role": r.role,
                         "image_number": r.image_number, "channel": ch, **fmt, **m,
                         "foreground_pct": float(100 * b[2].sum() / img.size)})
            for store, v in zip((mean, fg_sum, fg_count, background), (*b, m["background"])):
                store.append(v)
        np.savez(block_dir / f"{plate}_{ch}.npz", mean=np.stack(mean), fg_sum=np.stack(fg_sum),
                 fg_count=np.stack(fg_count), background=np.array(background),
                 well=fields["well"].to_numpy().astype(str))
    return pd.DataFrame(rows)


def _process(args):
    return process_plate(*args)


def run_qc(selected: pd.DataFrame, image_root: Path, out_dir: Path, workers: int = 4) -> pd.DataFrame:
    """Metrics and block summaries for every plate in selected; writes field_metrics.csv."""
    jobs = [(p, f, image_root, out_dir / "blocks") for p, f in selected.groupby("plate", sort=True)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        metrics = flag_outliers(pd.concat(list(pool.map(_process, jobs)), ignore_index=True))
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "field_metrics.csv", index=False)
    return metrics


def robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = 1.4826 * (x - med).abs().median()
    return (x - med) / mad if mad > 0 else pd.Series(0.0, index=x.index)


def flag_outliers(m: pd.DataFrame, z: float = FLAG_Z) -> pd.DataFrame:
    """Robust z of focus and background within each plate and channel, and a flag with its reasons.

    blur            focus z below -z (steeper power spectrum than the plate's other fields)
    flat-spectrum   focus z above z (shallower spectrum: noise, or bright fine texture)
    high-background background z above z
    saturated       more than FLAG_SATURATED_PCT of pixels clipped
    """
    m = m.copy()
    g = m.groupby(["plate", "channel"])
    m["focus_z"] = g["focus"].transform(robust_z)
    m["background_z"] = g["background"].transform(robust_z)
    reasons = pd.Series("", index=m.index)
    reasons[m["focus_z"] < -z] += "blur;"
    reasons[m["focus_z"] > z] += "flat-spectrum;"
    reasons[m["background_z"] > z] += "high-background;"
    reasons[m["saturated_pct"] > FLAG_SATURATED_PCT] += "saturated;"
    m["flag"] = reasons.str.rstrip(";")
    return m


def format_summary(m: pd.DataFrame) -> pd.DataFrame:
    """Format and intensity ranges per channel and exposure time."""
    return m.groupby(["channel", "exposure_ms"]).agg(
        fields=("focus", "size"), plates=("plate", "nunique"),
        weeks=("week", lambda s: " ".join(sorted(set(s), key=lambda x: int(x.removeprefix("Week"))))),
        dtypes=("dtype", lambda s: " ".join(sorted(set(s)))),
        height=("height", "median"), width=("width", "median"),
        all_multiple_of_16=("multiple_of_16", "all"),
        min=("min", "min"), background=("background", "median"), median=("median", "median"),
        p99=("p99", "median"), max=("max", "max"),
    ).reset_index()
