"""Illumination correction functions from the block summaries written by the QC stage.

A function f is a smooth multiplicative gain over the field, scaled to a mean of 1. An image is
corrected as (image - offset) / f + offset, where offset is the plate and channel's background level,
so the background stays level while the signal above it is flattened.

Two estimators:
  median      smoothed pixelwise median of the block means of all fields
  foreground  mean foreground intensity above the offset at each position, as a ratio of smoothed
              sums, with every field scaled to its own foreground mean; cell density cancels out
              because each position is averaged over foreground pixels only
Three scopes: the fields of one plate, of one week, or of the whole dataset.

Flatness is scored on fields that were not used to fit f: how much the mean foreground intensity
above the offset varies over a grid of tiles, before and after correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from .metadata import CHANNELS

SIGMA = 4                   # Gaussian smoothing in blocks (16 px each)
TILES = 8                   # flatness is scored on a TILES x TILES grid
OFFSET_PERCENTILE = 10      # of the per-field backgrounds on a plate
ESTIMATORS = ("median", "foreground")
SCOPES = ("plate", "week", "all")


@dataclass
class Blocks:
    """Block summaries of one plate and channel (arrays indexed by field)."""

    mean: np.ndarray
    fg_sum: np.ndarray
    fg_count: np.ndarray
    background: np.ndarray
    well: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "Blocks":
        with np.load(path) as z:
            return cls(**{k: z[k] for k in ("mean", "fg_sum", "fg_count", "background", "well")})

    def take(self, keep: np.ndarray) -> "Blocks":
        return Blocks(self.mean[keep], self.fg_sum[keep], self.fg_count[keep], self.background[keep], self.well[keep])

    @property
    def offset(self) -> float:
        return float(np.percentile(self.background, OFFSET_PERCENTILE))

    def scaled_foreground(self) -> tuple[np.ndarray, np.ndarray]:
        """Per block, summed over fields: foreground intensity above the offset in units of each
        field's own foreground mean, and the foreground pixel count."""
        count = self.fg_count.sum(axis=(1, 2))
        signal = self.fg_sum - self.offset * self.fg_count
        level = signal.sum(axis=(1, 2)) / np.maximum(count, 1)
        ok = level > 0
        return (signal[ok] / level[ok, None, None]).sum(axis=0), self.fg_count[ok].sum(axis=0)


def _smooth(a: np.ndarray, sigma: float = SIGMA) -> np.ndarray:
    return ndimage.gaussian_filter(a.astype(np.float64), sigma, mode="nearest")


def median_function(parts: list[Blocks]) -> np.ndarray:
    f = _smooth(np.median(np.concatenate([p.mean - p.offset for p in parts]), axis=0))
    return f / f.mean()


def foreground_function(parts: list[Blocks]) -> np.ndarray:
    sums = [p.scaled_foreground() for p in parts]
    num, den = (_smooth(np.sum([s[i] for s in sums], axis=0)) for i in (0, 1))
    f = num / np.maximum(den, 1e-9)
    return f / f.mean()


def fit(parts: list[Blocks], estimator: str) -> np.ndarray:
    f = {"median": median_function, "foreground": foreground_function}[estimator](parts)
    return np.maximum(f, 0.05)      # a gain near zero would blow up the corrected image


def _tiles(a: np.ndarray, tiles: int = TILES) -> np.ndarray:
    h, w = a.shape
    return a.reshape(tiles, h // tiles, tiles, w // tiles).sum(axis=(1, 3))


def foreground_map(b: Blocks, f: np.ndarray | None = None) -> np.ndarray:
    """Mean foreground intensity above the offset per tile, scaled to a mean of 1."""
    if f is not None:
        b = Blocks(b.mean, (b.fg_sum - b.offset * b.fg_count) / f + b.offset * b.fg_count, b.fg_count, b.background, b.well)
    num, den = b.scaled_foreground()
    m = _tiles(num) / np.maximum(_tiles(den), 1e-9)
    return m / m.mean()


def flatness(m: np.ndarray) -> dict:
    low = m.min()
    return {"cv": float(m.std() / m.mean()), "max_over_min": float(m.max() / low) if low > 0 else float("inf")}


def noise_floor(b: Blocks, rng: np.random.Generator, n: int = 20) -> float:
    """Median flatness cv after shifting every field by a random offset (with wrap-around).

    Shifting keeps each field's cells but removes any pattern tied to position, so what is left is
    the variation expected from this number of fields with perfectly even illumination.
    """
    h, w = b.fg_sum.shape[1:]
    out = []
    for _ in range(n):
        s = [(int(rng.integers(h)), int(rng.integers(w))) for _ in range(len(b.fg_sum))]
        roll = lambda x: np.stack([np.roll(a, d, (0, 1)) for a, d in zip(x, s)])
        out.append(flatness(foreground_map(Blocks(b.mean, roll(b.fg_sum), roll(b.fg_count), b.background, b.well)))["cv"])
    return float(np.median(out))


def well_halves(wells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two boolean masks that split the fields by alternate wells (all fields of a well together)."""
    first = np.isin(wells, np.unique(wells)[0::2])
    return first, ~first


def compare(blocks: dict[str, Blocks], week: dict[str, str], seed: int = 0) -> pd.DataFrame:
    """Held-out flatness of every estimator and scope on every plate of one channel.

    Plate scope fits on half of the plate's wells and scores the other half, both ways. Week and
    all scopes fit on the other plates. Every row scores the same half-plates, so the cv values and
    the noise floor are comparable across rows.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for plate, b in sorted(blocks.items()):
        halves = well_halves(b.well)
        others = {"week": [x for p, x in blocks.items() if p != plate and week[p] == week[plate]],
                  "all": [x for p, x in blocks.items() if p != plate]}
        shared = {(e, s): fit(parts, e) for s, parts in others.items() if parts for e in ESTIMATORS}
        for i, test in enumerate(halves):
            t = b.take(test)
            base = {"plate": plate, "week": week[plate], "half": i, "fields_scored": len(t.mean)}
            rows.append({**base, "estimator": "none", "scope": "none", "fields_fit": 0, **flatness(foreground_map(t))})
            rows.append({**base, "estimator": "noise floor", "scope": "none", "fields_fit": 0,
                         "cv": noise_floor(t, rng), "max_over_min": np.nan})
            for e in ESTIMATORS:
                rows.append({**base, "estimator": e, "scope": "plate", "fields_fit": int((~test).sum()),
                             **flatness(foreground_map(t, fit([b.take(~test)], e)))})
                for s, parts in others.items():
                    if parts:
                        rows.append({**base, "estimator": e, "scope": s, "fields_fit": sum(len(p.mean) for p in parts),
                                     **flatness(foreground_map(t, shared[e, s]))})
    return pd.DataFrame(rows)


def load_blocks(block_dir: Path, plates: list[str], channel: str) -> dict[str, Blocks]:
    return {p: Blocks.load(block_dir / f"{p}_{channel}.npz") for p in plates}


def run_compare(selected: pd.DataFrame, qc_dir: Path) -> pd.DataFrame:
    week = dict(selected.drop_duplicates("plate")[["plate", "week"]].values)
    out = []
    for ch in CHANNELS:
        out.append(compare(load_blocks(qc_dir / "blocks", sorted(week), ch), week).assign(channel=ch))
    df = pd.concat(out, ignore_index=True)
    df.to_csv(qc_dir / "flatness.csv", index=False)
    return df


def run_fit(selected: pd.DataFrame, qc_dir: Path, estimator: str, scope: str) -> pd.DataFrame:
    """Fit and save one function per plate and channel (shared within the scope) and its offset."""
    week = dict(selected.drop_duplicates("plate")[["plate", "week"]].values)
    out_dir = qc_dir / "illumination"
    out_dir.mkdir(parents=True, exist_ok=True)
    key = {"plate": lambda p: p, "week": lambda p: week[p], "all": lambda p: "all"}[scope]
    rows = []
    for ch in CHANNELS:
        blocks = load_blocks(qc_dir / "blocks", sorted(week), ch)
        fitted = {}
        for p, b in blocks.items():
            k = key(p)
            if k not in fitted:
                fitted[k] = fit([x for q, x in blocks.items() if key(q) == k], estimator).astype(np.float32)
            np.save(out_dir / f"{p}_{ch}.npy", fitted[k])
            rows.append({"plate": p, "channel": ch, "estimator": estimator, "scope": scope, "group": k,
                         "offset": b.offset, "gain_min": float(fitted[k].min()), "gain_max": float(fitted[k].max())})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "functions.csv", index=False)
    return df


def upsample(f: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinear upsampling of a block-resolution function to full image size."""
    zoom = (shape[0] / f.shape[0], shape[1] / f.shape[1])
    return ndimage.zoom(f, zoom, order=1, mode="nearest", grid_mode=True).astype(np.float32)


def correct(img: np.ndarray, f: np.ndarray, offset: float) -> np.ndarray:
    """Illumination-corrected image as float32; f may be at block resolution."""
    if f.shape != img.shape:
        f = upsample(f, img.shape)
    return (img.astype(np.float32) - offset) / f + offset
