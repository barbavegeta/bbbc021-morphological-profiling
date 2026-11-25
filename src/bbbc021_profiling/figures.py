"""Figures of the QC and illumination stages.

flatness.png        held-out flatness of every estimator and scope, per channel
flatness_maps.png   mean held-out foreground intensity across the field, before and after correction
qc_metrics.png      focus and background of every field by plate, flagged fields highlighted
flag_examples_*.png flagged fields beside typical fields. These contain image data, so they go to
                    local_only/ and are never committed.
segmentation/overlays_*.png  nuclear and cell outlines of each method on sampled fields (local only)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

from . import illumination
from .metadata import CHANNELS
from .qc import read_tiff

INK, MUTED, GRID, SURFACE = "#191919", "#666666", "#e4e4e4", "#ffffff"
ESTIMATOR_STYLE = {"median": ("#257de0", "o", "Smoothed median"), "foreground": ("#f47230", "s", "Foreground mean"),
                   "cellprofiler": ("#00b894", "D", "CellProfiler reference")}
SCOPE_LABEL = {"plate": "Plate", "week": "Week", "all": "All plates"}
DIVERGING = LinearSegmentedColormap.from_list("blue_red", ["#1757ac", "#6aabf1", "#efefef", "#f0908f", "#b32a2b"])
EXAMPLES = 6                # fields per row of the flag example figures

STYLE = {"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "font.size": 9,
         "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID,
         "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10, "axes.titlelocation": "left"}


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def flatness(df: pd.DataFrame, path: Path) -> Path:
    """Median and interquartile range over half-plates of the held-out variation across the field."""
    scopes = [s for s in SCOPE_LABEL if (df["scope"] == s).any()]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(CHANNELS), figsize=(11, 3.4), sharey=True)
        for ax, ch in zip(axes, CHANNELS):
            d = df[df["channel"] == ch]
            for name, ls, label, side in (("none", "-", "uncorrected", 1), ("noise floor", ":", "noise\nfloor", -1)):
                x = 100 * d.loc[d["estimator"] == name, "cv"].median()
                ax.axvline(x, color=MUTED, lw=1.2, ls=ls, zorder=1)
                ax.annotate(f"{label}\n{x:.1f}", (x, 1), xycoords=("data", "axes fraction"), xytext=(3 * side, -2),
                            textcoords="offset points", va="top", ha="left" if side > 0 else "right", fontsize=7.5, color=MUTED)
            for i, s in enumerate(scopes):
                for j, (e, (c, mk, label)) in enumerate(ESTIMATOR_STYLE.items()):
                    q = 100 * d.loc[(d["estimator"] == e) & (d["scope"] == s), "cv"].quantile([0.25, 0.5, 0.75])
                    y = i + (j - 1) * 0.22
                    ax.plot(q.iloc[[0, 2]], [y, y], color=c, lw=2, solid_capstyle="round", zorder=2)
                    ax.plot(q.iloc[1], y, mk, color=c, ms=8, mec=SURFACE, mew=1.5, zorder=3, label=label if i == 0 else None)
            ax.set_title(ch, color=INK, pad=8)
            ax.set_xlabel("Variation across the field, CV (%, log scale)")
            ax.set_xscale("log")
            ax.set_xlim(1.5, 60)
            ax.set_xticks([2, 3, 5, 10, 20, 40], ["2", "3", "5", "10", "20", "40"])
            ax.set_xticks([], minor=True)
            ax.grid(axis="x", color=GRID, lw=0.6)
            ax.tick_params(length=0)
            ax.set_axisbelow(True)
        axes[0].set_yticks(range(len(scopes)), [SCOPE_LABEL[s] for s in scopes])
        axes[0].set_ylim(len(scopes) - 0.4, -0.95)
        axes[0].set_ylabel("Function fitted per")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.1))
        fig.tight_layout()
        return _save(fig, path)


def flatness_maps(maps: dict[str, dict[str, np.ndarray]], path: Path, scope: str) -> Path:
    """maps: channel -> "none" or estimator -> mean held-out tile map (see illumination.heldout_maps)."""
    cols = ["none", *ESTIMATOR_STYLE]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(CHANNELS), len(cols), figsize=(10.5, 6.4))
        for row, ch in zip(axes, CHANNELS):
            lim = max(np.abs(m - 1).max() for m in maps[ch].values())
            for ax, c in zip(row, cols):
                m = maps[ch][c]
                im = ax.imshow(m, cmap=DIVERGING, norm=TwoSlopeNorm(1, 1 - lim, 1 + lim), aspect=0.8)
                ax.set_xticks([]), ax.set_yticks([])
                ax.spines[:].set_visible(False)
                ax.set_xlabel(f"CV {100 * m.std() / m.mean():.1f}%, range {m.min():.2f} to {m.max():.2f}", fontsize=8)
                if ch == CHANNELS[0]:
                    ax.set_title("Uncorrected" if c == "none" else ESTIMATOR_STYLE[c][2], color=INK)
            row[0].set_ylabel(ch, color=INK, fontsize=10)
            cb = fig.colorbar(im, ax=list(row), fraction=0.02, pad=0.015)
            cb.outline.set_visible(False)
            cb.ax.tick_params(length=0, labelsize=7.5)
        fig.suptitle(f"Foreground intensity across the field, held-out wells, mean over plates (function fitted per {scope}; 1 = field mean)",
                     x=0.125, ha="left", fontsize=9.5, color=INK)
        return _save(fig, path)


def qc_metrics(m: pd.DataFrame, path: Path) -> Path:
    plates = m.drop_duplicates("plate").sort_values(["week", "plate"], key=lambda s: s.str.extract(r"(\d+)$")[0].astype(int) if s.name == "week" else s)
    pos = {p: i for i, p in enumerate(plates["plate"])}
    starts = plates.reset_index(drop=True).groupby("week", sort=False).apply(lambda g: g.index[0], include_groups=False)
    rng = np.random.default_rng(0)
    metrics = (("focus", "Focus (power-spectrum slope)", ("blur", "flat-spectrum"), False),
               ("background", "Background (1st percentile)", ("high-background",), True))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(metrics), len(CHANNELS), figsize=(13, 5.6), sharex=True)
        for row, (col, label, reasons, log) in zip(axes, metrics):
            for ax, ch in zip(row, CHANNELS):
                d = m[m["channel"] == ch]
                hit = d["flag"].str.contains("|".join(reasons)).to_numpy()
                x = d["plate"].map(pos).to_numpy() + rng.uniform(-0.3, 0.3, len(d))
                ax.scatter(x[~hit], d.loc[~hit, col], s=3, color="#b9b9b9", lw=0, rasterized=True, label="Field")
                ax.scatter(x[hit], d.loc[hit, col], s=16, color="#257de0", ec=SURFACE, lw=0.5, zorder=3, label="Flagged field")
                for w, s in starts.items():
                    ax.axvline(s - 0.5, color=GRID, lw=0.8, zorder=0)
                ax.set_xticks([s for s in starts], [w.removeprefix("Week") for w in starts.index], ha="left", fontsize=8)
                ax.tick_params(length=0)
                if log:
                    ax.set_yscale("log")
                    ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 4, 6)))
                    ax.yaxis.set_major_formatter(ScalarFormatter())
                    ax.yaxis.set_minor_formatter(NullFormatter())
                ax.set_title(f"{ch}: {int(hit.sum())} of {len(d)} flagged", color=INK)
            row[0].set_ylabel(label)
        for ax in axes[-1]:
            ax.set_xlabel("Week (one column of points per plate)")
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04), markerscale=1.5)
        fig.tight_layout()
        return _save(fig, path)


def flag_examples(m: pd.DataFrame, selected: pd.DataFrame, image_root: Path, out_dir: Path) -> list[Path]:
    """Local-only: the most extreme flagged fields per reason, above typical fields of the same plates."""
    rules = {"blur": ("focus_z", True), "flat-spectrum": ("focus_z", False), "high-background": ("background_z", False),
             "saturated": ("saturated_pct", False)}
    paths = selected.set_index(["plate", "well", "image_number"])
    out = []
    for ch in CHANNELS:
        d = m[m["channel"] == ch]
        scale = d.groupby("plate").agg(lo=("background", "median"), hi=("p99", "median"))
        for reason, (col, ascending) in rules.items():
            hits = d[d["flag"].str.split(";").map(lambda r: reason in r)].sort_values(col, ascending=ascending).head(EXAMPLES)
            if hits.empty:
                continue
            normal = d[d["flag"] == ""].assign(distance=lambda g: g["focus_z"].abs() + g["background_z"].abs())
            typical = [normal[normal["plate"] == p].nsmallest(k + 1, "distance").iloc[[-1]]
                       for p, k in zip(hits["plate"], hits.groupby("plate").cumcount())]
            with plt.rc_context(STYLE):
                fig, axes = plt.subplots(2, len(hits), figsize=(max(2.6 * len(hits), 5), 5.4), squeeze=False, layout="constrained")
                for k, rows in enumerate((hits, pd.concat(typical))):
                    for ax, r in zip(axes[k], rows.itertuples(index=False)):
                        img, _ = read_tiff(image_root / paths.loc[(r.plate, r.well, r.image_number), f"path_{ch}"])
                        ax.imshow(img, cmap="gray", vmin=scale.lo[r.plate] * 0.8, vmax=scale.hi[r.plate] * 1.3)
                        ax.set_axis_off()
                        ax.set_title(f"{r.plate} {r.well}\n{col} {getattr(r, col):.2f}", fontsize=7.5, color=INK)
                fig.suptitle(f"{ch}, {reason}: flagged (top) and a typical field of the same plate (bottom)", x=0.02, ha="left", fontsize=9.5)
                out.append(_save(fig, out_dir / f"flag_examples_{ch}_{reason}.png"))
    return out


def run(selected: pd.DataFrame, image_root: Path, qc_dir: Path, scope: str = "plate") -> list[Path]:
    m = pd.read_csv(qc_dir / "field_metrics.csv", keep_default_na=False)
    plates = sorted(selected["plate"].unique())
    week = dict(selected.drop_duplicates("plate")[["plate", "week"]].values)
    maps = {ch: illumination.heldout_maps(illumination.load_blocks(qc_dir / "blocks", plates, ch), week, scope) for ch in CHANNELS}
    fig_dir = qc_dir / "figures"
    return [flatness(pd.read_csv(qc_dir / "flatness.csv"), fig_dir / "flatness.png"),
            flatness_maps(maps, fig_dir / "flatness_maps.png", SCOPE_LABEL[scope].lower()),
            qc_metrics(m, fig_dir / "qc_metrics.png"),
            *flag_examples(m, selected, image_root, qc_dir / "local_only")]


CROP = 480                  # px; centre of the field shown in the segmentation overlays
OUTLINE = {"DAPI": "#ff9f1c", "Actin": "#3fd0ff"}


def _outline(ax, img: np.ndarray, labels: np.ndarray, colour: str, title: str) -> None:
    from skimage.segmentation import find_boundaries
    edge = find_boundaries(labels, mode="inner")
    overlay = np.zeros((*edge.shape, 4))
    overlay[edge] = matplotlib.colors.to_rgba(colour)
    ax.imshow(img, cmap="gray", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.7))
    ax.imshow(overlay)
    ax.set_title(title, fontsize=7.5, color=INK)
    ax.set_axis_off()


def segmentation_overlays(fields: pd.DataFrame, labels_dir: Path, load, out_dir: Path, nuclei: list[str],
                          cells: list[str], crop: int = CROP) -> list[Path]:
    """Local-only: outlines of every method on the sampled fields, one figure per field."""
    out = []
    for _, r in fields.iterrows():
        fid = f"{r['plate']}_{r['well']}_{r['image_number']}"
        img = load(r)
        h, w = img["DAPI"].shape
        y, x = (h - crop) // 2, (w - crop) // 2
        view = (slice(y, y + crop), slice(x, x + crop))
        n = max(len(nuclei), len(cells))
        with plt.rc_context(STYLE):
            fig, axes = plt.subplots(2, n, figsize=(2.9 * n, 6.2), squeeze=False, layout="constrained")
            for a in axes.flat:
                a.set_axis_off()
            for row, kind, names, ch in ((0, "nuclei", nuclei, "DAPI"), (1, "cells", cells, "Actin")):
                for ax, name in zip(axes[row], names):
                    path = labels_dir / f"{kind}_{name}_{fid}.npz"
                    if not path.exists():
                        continue
                    lab = np.load(path)["labels"]
                    _outline(ax, img[ch][view], lab[view], OUTLINE[ch], f"{kind}: {name}\n{int(lab.max())} in field")
            fig.suptitle(f"{r['moa'] if isinstance(r['moa'], str) else 'DMSO'}, {r['plate']} {r['well']} (centre {crop} px)", x=0.02, ha="left", fontsize=9)
            out.append(_save(fig, out_dir / f"overlays_{fid}.png"))
    return out
