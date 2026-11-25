"""QC metrics and illumination correction on synthetic images (no BBBC021 data)."""

import numpy as np
import pandas as pd
import pytest
import tifffile
from scipy import ndimage

from bbbc021_profiling import figures, illumination, qc
from bbbc021_profiling.metadata import CHANNELS

SHAPE = (256, 256)
OFFSET = 600


def vignette(shape=SHAPE, depth=0.3):
    y, x = (np.linspace(-1, 1, n) for n in shape)
    v = 1 - depth * (y[:, None] ** 2 + x[None, :] ** 2) / 2
    return v / v.mean()


def cells(rng, n=60, shape=SHAPE, density=None):
    """Disc-shaped cells with varying brightness on a flat background."""
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    img = np.zeros(shape)
    for _ in range(n):
        cy, cx = rng.uniform(0, shape[0]), rng.uniform(0, shape[1])
        if density is not None:     # pull cells toward the centre
            cy, cx = shape[0] / 2 + (cy - shape[0] / 2) * density, shape[1] / 2 + (cx - shape[1] / 2) * density
        img = np.maximum(img, np.where((yy - cy) ** 2 + (xx - cx) ** 2 < 36, rng.uniform(2500, 3500), 0))
    return img


def image(rng, gain, **kw):
    return np.clip(OFFSET + cells(rng, **kw) * gain + rng.normal(0, 10, SHAPE), 0, 65520).astype(np.uint16)


def blocks(rng, gain, n=60, **kw):
    imgs = [image(rng, gain, **kw) for _ in range(n)]
    b = [qc.foreground_blocks(i, block=8) for i in imgs]
    return illumination.Blocks(*(np.stack([x[k] for x in b]) for k in range(3)),
                               np.array([np.percentile(i, 1) for i in imgs]),
                               np.array([f"W{i // 4:02d}" for i in range(n)]))


def test_focus_slope_falls_with_blur():
    rng = np.random.default_rng(0)
    c = OFFSET + cells(rng) * ndimage.gaussian_filter(rng.normal(1, 1.5, SHAPE), 2)  # cells with inner texture
    noise = rng.normal(0, 10, SHAPE)        # sensor noise is added after the optics blur the cells
    slopes = [qc.power_log_log_slope(ndimage.gaussian_filter(c, s) + noise) for s in (0.5, 1, 2, 3)]
    assert slopes == sorted(slopes, reverse=True)


def test_field_metrics():
    img = np.full(SHAPE, 800, dtype=np.uint16)
    img[:8, :16] = 3957 * 16            # a clip level below the 12-bit maximum
    m = qc.field_metrics(img)
    assert m["saturated_pct"] == pytest.approx(100 * 128 / img.size) and m["pixels_at_max"] == 128
    assert m["background"] == 800 and m["max"] == 63312 and m["multiple_of_16"]
    img[:8, :16] = 3000 * 16            # bright, but below the clipping range
    assert qc.field_metrics(img)["saturated_pct"] == 0


def test_foreground_blocks_count_the_cells():
    rng = np.random.default_rng(1)
    c = cells(rng)
    img = (OFFSET + c).astype(np.uint16)
    mean, fg_sum, fg_count = qc.foreground_blocks(img, block=8)
    assert mean.shape == (32, 32)
    assert fg_count.sum() == (c > 0).sum()
    assert fg_sum.sum() == pytest.approx(img[c > 0].sum(), rel=1e-5)


def test_flag_outliers_states_its_reasons():
    rng = np.random.default_rng(2)
    m = pd.DataFrame({"plate": "P1", "channel": "DAPI", "focus": rng.normal(-3, 0.05, 40),
                      "background": rng.normal(700, 5, 40), "saturated_pct": 0.0})
    m.loc[0, "focus"], m.loc[1, "background"], m.loc[2, "saturated_pct"] = -4, 900, 0.5
    f = qc.flag_outliers(m)
    assert f["flag"].iloc[:3].tolist() == ["blur", "high-background", "saturated"]
    assert (f["flag"].iloc[3:] == "").all()


def test_foreground_estimator_recovers_the_gain_and_ignores_density():
    rng = np.random.default_rng(3)
    true = illumination._tiles(qc.block_sum(vignette(), 8))
    central = slice(1, 7), slice(1, 7)      # the crowded layout leaves the border nearly empty
    true = true[central] / true[central].mean()
    for density in (None, 0.7):             # even, and crowded toward the centre
        got = illumination._tiles(illumination.fit([blocks(rng, vignette(), n=150, density=density)], "foreground"))[central]
        assert np.abs(got / got.mean() - true).max() < 0.06


def test_cellprofiler_estimator_has_no_border_falloff_and_divides_without_an_offset():
    n = SHAPE[0] // 8
    count = np.full((4, n, n), 16.0)
    rest = count, np.full(4, float(OFFSET)), np.array(["B02"] * 4)
    flat = illumination.Blocks(np.full((4, n, n), 2000.0), 2000 * count, *rest)
    assert illumination.fit([flat], "cellprofiler") == pytest.approx(np.ones((n, n)))
    gain = qc.block_sum(vignette(), 8) / 64
    shaded = illumination.Blocks(2000 * gain * np.ones((4, 1, 1)), 2000 * gain * count, *rest)
    cv = {e: illumination.flatness(illumination.foreground_map(shaded, gain, e))["cv"] for e in ("cellprofiler", "foreground")}
    assert cv["cellprofiler"] < 1e-6 < 1e-3 < cv["foreground"]


def test_correction_flattens_held_out_fields():
    rng = np.random.default_rng(4)
    b = {f"P{i}": blocks(rng, vignette(), n=48) for i in range(3)}
    df = illumination.compare(b, {p: "Week1" for p in b})
    cv = df.groupby(["estimator", "scope"])["cv"].median()
    assert cv["foreground", "all"] < 0.6 * cv["none", "none"]
    assert cv["noise floor", "none"] < cv["none", "none"]


def test_halves_keep_wells_together():
    wells = np.array(["B02"] * 4 + ["C03"] * 4 + ["D04"] * 4)
    a, b = illumination.well_halves(wells)
    assert (a ^ b).all() and not set(wells[a]) & set(wells[b])


def test_correct_keeps_background_and_restores_signal():
    gain = vignette()
    flat = np.full(SHAPE, 3000.0)
    shaded = (OFFSET + flat * gain).astype(np.float32)
    out = illumination.correct(shaded, gain.astype(np.float32), OFFSET)
    assert out == pytest.approx(OFFSET + flat, rel=1e-4)
    assert illumination.correct(np.full(SHAPE, OFFSET, np.float32), gain, OFFSET) == pytest.approx(np.full(SHAPE, OFFSET))
    up = illumination.upsample(qc.block_sum(gain, 8) / 64, SHAPE)
    assert up.shape == SHAPE and np.abs(up - gain).max() < 0.02


def test_qc_and_illumination_stages(tmp_path):
    rng = np.random.default_rng(5)
    rows = []
    for plate in ("P1", "P2", "P3"):
        for well in ("B02", "B03", "C02", "C03"):
            for site in (1, 2):
                r = {"plate": plate, "week": "Week1", "well": well, "role": "dmso", "image_number": len(rows)}
                for ch in CHANNELS:
                    rel = f"Week1/{plate}/{well}_s{site}_{ch}.tif"
                    (tmp_path / "images" / rel).parent.mkdir(parents=True, exist_ok=True)
                    tifffile.imwrite(tmp_path / "images" / rel, image(rng, vignette()) // 16 * 16, description="Exposure: 10 ms")
                    r[f"path_{ch}"] = rel
                rows.append(r)
    selected = pd.DataFrame(rows)
    out = tmp_path / "qc"
    m = qc.run_qc(selected, tmp_path / "images", out, workers=2)
    assert len(m) == 24 * 3 and m["multiple_of_16"].all() and (m["exposure_ms"] == 10).all()
    assert len(qc.format_summary(m)) == 3
    df = illumination.run_compare(selected, out)
    assert set(df["scope"]) == {"none", "plate", "week", "all"}
    written = figures.run(selected, tmp_path / "images", out)
    assert {"flatness.png", "flatness_maps.png", "qc_metrics.png"} <= {p.name for p in written}
    assert all(p.parent.name == ("local_only" if p.name.startswith("flag_examples") else "figures") for p in written)
    fn = illumination.run_fit(selected, out, "foreground", "week")
    assert len(fn) == 9 and np.load(out / "illumination" / "P2_Actin.npy").shape == (16, 16)
