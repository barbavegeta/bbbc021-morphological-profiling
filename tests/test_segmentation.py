"""Object scoring, the classical methods and the benchmark bookkeeping on synthetic images (no BBBC data)."""

import numpy as np
import pandas as pd
import pytest
import skimage.io
import tifffile

from bbbc021_profiling import segmentation as seg
from bbbc021_profiling import illumination
from bbbc021_profiling.metadata import CHANNELS

SHAPE = (200, 200)


def discs(centres, radius, shape=SHAPE):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    lab = np.zeros(shape, np.int32)
    for i, (cy, cx) in enumerate(centres, 1):
        lab[(yy - cy) ** 2 + (xx - cx) ** 2 < radius ** 2] = i
    return lab


GRID = [(30 + 40 * i, 30 + 40 * j) for i in range(4) for j in range(4)]


def test_match_counts_true_and_missed_objects():
    truth = discs(GRID, 12)
    assert seg.match(truth, truth) == pytest.approx(
        {"truth": 16, "pred": 16, "tp": 16, "fp": 0, "fn": 0, "matched_iou": 1.0})
    pred = np.roll(truth, 3, axis=1)                    # small shift: every object still matches
    m = seg.match(truth, pred)
    assert m["tp"] == 16 and m["matched_iou"] < 1
    assert seg.match(truth, np.roll(truth, 20, axis=1))["tp"] == 0      # half the diameter away
    merged = np.where(np.isin(truth, [1, 2]), 1, truth)                 # one prediction covers two objects
    m = seg.match(truth, merged)
    assert m["tp"] == 14 and m["fp"] == 1 and m["fn"] == 2
    with pytest.raises(ValueError):
        seg.match(truth, truth, 0.4)


def test_ignore_small_objects_drops_them_from_both_sides():
    truth = discs(GRID, 12)
    small = np.zeros(SHAPE, np.int32)
    small[5, 5:9] = 1
    both = np.where(small > 0, 17, truth)
    assert seg.ignore_small(both).max() == 16 and seg.match(truth, seg.ignore_small(both))["fp"] == 0
    assert seg.match(seg.ignore_small(truth), seg.ignore_small(small))["tp"] == 0


def test_summary_pools_counts_and_has_intervals():
    rows = []
    for i in range(6):
        for t, f, n in ((8, 1, 2), (4, 4, 0)):
            rows.append({"method": "m", "split": "test", "image": i, "truth": t + n, "pred": t + f, "tp": t, "fp": f,
                         "fn": n, "tp75": t - 1, "fp75": f + 1, "fn75": n + 1, "matched_iou": 0.8, "seconds": 0.1})
    s = seg.summarise(pd.DataFrame(rows)).iloc[0]
    assert s["images"] == 12 and s["tp"] == 72
    assert s["precision"] == pytest.approx(72 / 102) and s["f1_low"] <= s["f1"] <= s["f1_high"]
    assert s["f1_iou75"] < s["f1"]


def test_classical_nuclei_split_touching_nuclei():
    lab = discs([(60, 60), (60, 88), (150, 150)], 16)       # two nuclei whose edges overlap
    rng = np.random.default_rng(0)
    img = np.where(lab > 0, 2000, 300) + rng.normal(0, 30, SHAPE)
    got = seg.classical_nuclei(img, diameter=32, seed_spacing=0.3)
    m = seg.match(lab, got)
    assert got.max() == 3 and m["tp"] == 3
    assert seg.classical_nuclei(img, 32, 5.0).max() == 2        # seeds too far apart leave the pair joined


def test_watershed_cells_give_one_cell_per_nucleus():
    nuc = discs([(60, 60), (60, 140), (140, 100)], 14)
    actin = np.full(SHAPE, 300.0)
    yy, xx = np.mgrid[:200, :200]
    actin[(yy - 60) ** 2 + (xx - 60) ** 2 < 40 ** 2] = 1500
    cells = seg.watershed_cells(actin, nuc)
    assert set(np.unique(cells)) == {0, 1, 2, 3}
    assert (cells[nuc == 1] == 1).all() and (cells > 0)[nuc > 0].all()
    assert (cells == 1).sum() > (nuc == 1).sum()


def test_field_checks_find_the_problems():
    nuclei = discs([(50, 50), (50, 130), (140, 50)], 12)
    cells = np.zeros(SHAPE, np.int32)
    cells[30:75, 30:75] = 1                                  # a normal cell with nucleus 1
    cells[30:75, 100:160] = 2                                # holds nucleus 2, touches nothing else
    cells[120:160, 0:80] = 3                                 # touches the border, holds nucleus 3
    cells[170:195, 90:110] = 4                               # no nucleus
    c = seg.field_checks(nuclei, cells)
    assert c["nuclei"] == 3 and c["cells"] == 4
    assert c["cells_touching_border"] == 1
    assert c["cells_without_nucleus"] == 1 and c["nuclei_without_cell"] == 0 and c["cells_with_2plus_nuclei"] == 0
    assert c["median_nucleus_cell_ratio"] == pytest.approx((nuclei == 1).sum() / (cells == 1).sum(), rel=0.35)
    c2 = seg.field_checks(nuclei, np.where(cells == 4, 0, cells))
    assert c2["cells_without_nucleus"] == 0
    c3 = seg.field_checks(nuclei, (cells == 2).astype(np.int32) + 0)
    assert c3["nuclei_without_cell"] == 2


def test_bbbc039_decoding_and_validation(tmp_path):
    rng = np.random.default_rng(1)
    for d in ("images", "masks", "metadata"):
        (tmp_path / d).mkdir()
    names = {"training": ["a.png", "b.png"], "test": ["c.png"]}
    for split, ns in names.items():
        (tmp_path / "metadata" / f"{split}.txt").write_text("\n".join(ns) + "\n")
        for n in ns:
            lab = discs(GRID, 12)
            mask = np.zeros((*SHAPE, 3), np.uint8)
            mask[..., 0] = lab * 10                          # each nucleus its own value, as in BBBC039
            skimage.io.imsave(tmp_path / "masks" / n, mask, check_contrast=False)
            img = (lab > 0) * 1500 + 300 + rng.normal(0, 20, SHAPE)
            tifffile.imwrite(tmp_path / "images" / n.replace(".png", ".tif"), img.astype(np.uint16))
    assert seg.bbbc039_truth(tmp_path / "masks" / "a.png").max() == 16
    d = seg.training_diameter(tmp_path)
    assert d == pytest.approx(24, abs=1)
    spacing, tuning = seg.tune_seed_spacing(tmp_path, d, (0.3, 0.5))
    assert spacing in (0.3, 0.5) and len(tuning) == 2
    df = seg.validate(tmp_path, seg.ClassicalNuclei(d, spacing), "classical", "test")
    s = seg.summarise(df).iloc[0]
    assert s["images"] == 1 and s["f1"] > 0.95


class FakeSegmenter:
    """Stands in for a deep-learning model: the diameter it reports and thresholds of the input."""

    def __init__(self, diameter=20.0):
        self.last_diameter = diameter

    def __call__(self, img, nucleus=None):
        return seg.label(img > img.mean()).astype(np.int32)


def test_benchmark_runs_every_method_and_saves_labels(tmp_path):
    rng = np.random.default_rng(2)
    nuclei = discs(GRID, 10)
    fields = pd.DataFrame([{"plate": "P1", "well": f"B0{i}", "image_number": i, "moa": m, "role": "treatment"}
                           for i, m in enumerate(("A", "B"))])
    sample = seg.Sample(fields, fields.iloc[:0])

    def load(r):
        img = {c: (nuclei > 0) * 1000.0 + 100 + rng.normal(0, 5, SHAPE) for c in CHANNELS}
        img["Actin"] = img["Actin"] + 500
        return img

    models = {"cyto3_actin_auto": FakeSegmenter()}
    nuc = {"classical": seg.ClassicalNuclei(20, 0.3), "cellpose_fixed": FakeSegmenter()}
    cells = {"watershed_classical": seg.cells_from("watershed", {}, "classical"),
             "cyto3_actin_auto": seg.cells_from("cyto3_actin_auto", models, "cellpose_fixed")}
    df = seg.run_benchmark(sample, nuc, cells, load, tmp_path)
    assert len(df) == 2 * 4 and (df["seconds"] >= 0).all()
    assert len(list(tmp_path.glob("*.npz"))) == 8
    c = seg.checks(sample, {"watershed_classical": "classical", "cyto3_actin_auto": "cellpose_fixed"}, tmp_path)
    assert len(c) == 4 and set(c["cell_method"]) == {"watershed_classical", "cyto3_actin_auto"}


def test_sample_has_every_moa_and_dmso(tmp_path):
    rows = [{"plate": p, "week": w, "well": f"C{i:02d}", "image_number": i, "role": "treatment", "moa": m}
            for w, p in (("W1", "P1"), ("W2", "P2")) for i, m in enumerate(("A", "B"))]
    rows += [{"plate": p, "week": w, "well": "B02", "image_number": 99 + k, "role": "dmso", "moa": np.nan}
             for k, (w, p) in enumerate((("W1", "P1"), ("W2", "P2"), ("W3", "P3")))]
    s = seg.sample_fields(pd.DataFrame(rows), calibration=2)
    assert sorted(s.fields["moa"].dropna()) == ["A", "B"] and (s.fields["role"] == "dmso").sum() == 1
    assert (s.calibration["role"] == "dmso").all()
    assert set(s.calibration["well"]).isdisjoint({}) and len(s.calibration) <= 2


def test_corrector_uses_the_saved_functions(tmp_path):
    (tmp_path / "illumination").mkdir()
    f = np.ones((2, 2), np.float32) * 2
    np.save(tmp_path / "illumination" / "P1_DAPI.npy", f)
    pd.DataFrame([{"plate": "P1", "channel": "DAPI", "estimator": "foreground", "scope": "all", "group": "all",
                   "offset": 100.0, "gain_min": 1, "gain_max": 2}]).to_csv(tmp_path / "illumination" / "functions.csv", index=False)
    c = illumination.Corrector(tmp_path)
    out = c(np.full((4, 4), 300, np.uint16), "P1", "DAPI")
    assert out == pytest.approx(np.full((4, 4), 200.0)) and out.dtype == np.float32


def test_runtime_tables_and_check_table():
    bench = pd.DataFrame([{"kind": k, "method": m, "field": f, "seconds": s}
                          for f in ("a", "b") for k, m, s in (("nuclei", "classical", 1.0), ("cells", "watershed_classical", 2.0))])
    methods, pipes = seg.runtime_tables(bench, fields=3600)
    assert set(methods["method"]) == {"classical", "watershed_classical"}
    assert methods.set_index("method").loc["classical", "projected_hours_for_all_fields"] == pytest.approx(1.0)
    assert pipes["pipeline"].tolist() == ["classical"] and pipes["projected_hours"].iloc[0] == pytest.approx(3.0)
    checks = pd.DataFrame([{"cell_method": "m", **seg.field_checks(discs(GRID, 12), discs(GRID, 15))} for _ in range(2)])
    t = seg.check_table(checks)
    assert len(t) == 1 and t["cells"].iloc[0] == 16 and 0 < t["median_nucleus_cell_ratio"].iloc[0] < 1


def test_overlays_are_written(tmp_path):
    from bbbc021_profiling import figures
    lab = discs(GRID, 12)
    labels = tmp_path / "labels"
    labels.mkdir()
    for kind, name in (("nuclei", "n1"), ("cells", "c1")):
        np.savez_compressed(labels / f"{kind}_{name}_P1_A01_1.npz", labels=lab)
    fields = pd.DataFrame([{"plate": "P1", "well": "A01", "image_number": 1, "moa": np.nan}])
    img = {c: np.random.default_rng(0).normal(500, 50, SHAPE) for c in CHANNELS}
    out = figures.segmentation_overlays(fields, labels, lambda r: img, tmp_path / "o", ["n1", "missing"], ["c1"], crop=100)
    assert len(out) == 1 and out[0].exists()


def test_cell_foreground_keeps_dim_cells_in_a_confluent_field():
    yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]]
    actin = np.full(SHAPE, 100.0)
    actin[:, 20:] = 600                                     # a dim sheet of cells
    actin[(yy - 100) ** 2 + (xx - 120) ** 2 < 30 ** 2] = 9000    # one bright cell among them
    actin += np.random.default_rng(3).normal(0, 10, SHAPE)
    mask = seg.cell_foreground(actin)
    assert mask[:, 40:].mean() > 0.95 and mask[:, :10].mean() < 0.2


def test_bootstrap_intervals_do_not_depend_on_the_other_methods():
    rng = np.random.default_rng(4)
    def rows(name):
        return [{"method": name, "split": "test", "image": i, "truth": 20, "pred": 20, "tp": int(t), "fp": 20 - int(t), "fn": 20 - int(t),
                 "tp75": int(t) - 1, "fp75": 21 - int(t), "fn75": 21 - int(t), "matched_iou": 0.8, "seconds": 0.1}
                for i, t in enumerate(rng.integers(10, 20, 15))]
    a, b = rows("a"), rows("b")
    alone = seg.summarise(pd.DataFrame(a)).iloc[0]
    both = seg.summarise(pd.DataFrame(b + a)).set_index("method").loc["a"]
    assert alone["f1_low"] == both["f1_low"] and alone["f1_high"] == both["f1_high"]


def test_nuclei_counts_table(tmp_path):
    lab = discs(GRID, 12)
    np.savez_compressed(tmp_path / "nuclei_m1_P1_A01_1.npz", labels=lab)
    fields = pd.DataFrame([{"plate": "P1", "well": "A01", "image_number": 1, "moa": np.nan}])
    c = seg.nuclei_counts(fields, ("m1", "absent"), tmp_path)
    assert c.loc[0, "moa"] == "DMSO" and c.loc[0, "m1"] == 16 and np.isnan(c.loc[0, "absent"])


def test_watershed_cells_keep_the_cytoplasm_of_a_bright_cell_beside_a_dim_one():
    # cell B fades outward from its nucleus, so a flood over intensity would let A enter it from the rim
    yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]]
    nuc = discs([(100, 50), (100, 140)], 12)
    r_b = np.hypot(yy - 100, xx - 140)
    actin = np.full(SHAPE, 100.0)
    actin[np.hypot(yy - 100, xx - 50) < 45] = 250.0
    body_b = r_b < 45
    actin[body_b] = np.maximum(250, 900 - 14 * r_b)[body_b]
    cells = seg.watershed_cells(actin, nuc)
    own = (cells == 2)[body_b & (xx > 100)].mean()
    assert set(np.unique(cells)) == {0, 1, 2} and own > 0.95
