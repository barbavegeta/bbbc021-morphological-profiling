"""Metadata join and scope selection on small synthetic tables (no BBBC021 data)."""

import hashlib
import json

import pandas as pd
import pytest

from bbbc021_profiling import download, metadata

FIELDS_PER_WELL = 4
# (plate, week, well, compound, concentration, replicate)
WELLS = [
    # every plate: DMSO at B02, a plate control at B11
    *[(p, w, "B02", "DMSO", 0.0, 1) for p, w in [("Week1_1", "Week1"), ("Week1_2", "Week1"), ("Week2_1", "Week2")]],
    *[(p, w, "B11", "ctrl", 1.0, 1) for p, w in [("Week1_1", "Week1"), ("Week1_2", "Week1"), ("Week2_1", "Week2")]],
    # the control compound-concentration also used as an ordinary treatment on one plate
    ("Week1_1", "Week1", "D05", "ctrl", 1.0, 1),
    # annotated treatments: A in three wells, B in two
    ("Week1_1", "Week1", "C03", "A", 0.3, 1),
    ("Week1_2", "Week1", "C03", "A", 0.3, 2),
    ("Week1_2", "Week1", "C04", "A", 0.3, 3),
    ("Week1_1", "Week1", "C05", "B", 3.0, 1),
    ("Week1_2", "Week1", "C05", "B", 3.0, 2),
    # unannotated compound, and a plate without annotated treatments (Week2_1)
    ("Week1_1", "Week1", "E07", "U", 1.0, 1),
    ("Week2_1", "Week2", "E07", "U", 1.0, 1),
]


@pytest.fixture
def meta_dir(tmp_path):
    rows, n = [], 0
    for plate, week, well, compound, conc, rep in WELLS:
        for s in range(1, FIELDS_PER_WELL + 1):
            n += 1
            row = {"TableNumber": int(week[4:]) - 1, "ImageNumber": n}
            for ch, tag in [("DAPI", "w1"), ("Tubulin", "w2"), ("Actin", "w4")]:
                row[f"Image_FileName_{ch}"] = f"{well}_s{s}_{tag}X.tif"
                row[f"Image_PathName_{ch}"] = f"{week}/{plate}"
            row |= {"Image_Metadata_Plate_DAPI": plate, "Image_Metadata_Well_DAPI": well, "Replicate": rep,
                    "Image_Metadata_Compound": compound, "Image_Metadata_Concentration": conc}
            rows.append(row)
    pd.DataFrame(rows).to_csv(tmp_path / "BBBC021_v1_image.csv", index=False)
    # concentrations written with six decimals, as in the real MoA file
    (tmp_path / "BBBC021_v1_moa.csv").write_text(
        '"compound",concentration,"moa"\n"A",0.300000,"M1"\n"B",3.000000,"M2"\n'
        '"ctrl",1.000000,"M1"\n"DMSO",0.000000,"DMSO"\n'
    )
    (tmp_path / "BBBC021_v1_compound.csv").write_text('"compound","smiles"\n"A","C"\n')
    return tmp_path


def test_roles(meta_dir):
    f = metadata.load_fields(meta_dir)
    roles = f.drop_duplicates(["plate", "well"]).set_index(["plate", "well"])["role"]
    assert roles[("Week1_1", "B02")] == "dmso"
    assert roles[("Week2_1", "B11")] == "plate_control"
    assert roles[("Week1_1", "D05")] == "treatment"
    assert roles[("Week1_1", "C03")] == "treatment"
    assert roles[("Week1_1", "E07")] == "unannotated"
    assert f.loc[f["compound"] == "A", "moa"].eq("M1").all()


def test_selection_skips_plates_without_treatments(meta_dir):
    f = metadata.load_fields(meta_dir)
    sel = metadata.select_fields(f, include_plate_controls=False)
    assert set(sel["plate"]) == {"Week1_1", "Week1_2"}
    wells = sel.drop_duplicates(["plate", "well"])["role"].value_counts().to_dict()
    assert wells == {"treatment": 6, "dmso": 2}
    sel_c = metadata.select_fields(f, include_plate_controls=True)
    assert sel_c.drop_duplicates(["plate", "well"])["role"].value_counts()["plate_control"] == 2
    assert len(sel_c) == len(sel) + 2 * FIELDS_PER_WELL


def test_treatment_summary(meta_dir):
    t = metadata.treatment_summary(metadata.load_fields(meta_dir)).set_index("treatment")
    assert t.loc["A|0.300000", "wells"] == 3
    assert t.loc["B|3.000000", "wells"] == 2
    assert t.loc["ctrl|1.000000", "wells"] == 1  # plate-control wells excluded
    assert (t["weeks"] == 1).all()


def test_plate_table_and_summary(meta_dir):
    f = metadata.load_fields(meta_dir)
    sel = metadata.select_fields(f, include_plate_controls=False)
    sizes = {metadata.ZIP_URL.format(plate=p): 1000 for p in f["plate"].unique()}
    p = metadata.plate_table(f, sel, sizes).set_index("plate")
    # Week1_1 has 6 wells, of which C03, C05, D05 and B02 are selected
    assert p.loc["Week1_1", "fields_total"] == 6 * FIELDS_PER_WELL
    assert p.loc["Week1_1", "fields_selected"] == 4 * FIELDS_PER_WELL
    assert p.loc["Week1_1", "est_bytes_selected"] == round(1000 * 4 / 6)
    assert p.loc["Week2_1", "fields_selected"] == 0
    s = metadata.scope_summary(f, sel, p.reset_index())
    assert s["annotated_treatments"] == 3 and s["annotated_compounds"] == 3 and s["moa_classes"] == 2
    assert s["plates_selected"] == 2 and s["tiffs_selected"] == 3 * len(sel)
    assert s["zip_bytes_to_download"] == 2000 and s["largest_zip_bytes"] == 1000


def test_duplicate_moa_rows_rejected(meta_dir):
    with open(meta_dir / "BBBC021_v1_moa.csv", "a") as fh:
        fh.write('"A",0.3,"M2"\n')
    with pytest.raises(ValueError):
        metadata.load_fields(meta_dir)


def test_download_metadata_manifest(meta_dir, tmp_path):
    dest = tmp_path / "out"
    recs = download.download_metadata(dest, base_url=meta_dir.as_uri())
    assert [r["file"] for r in recs] == list(download.METADATA_FILES)
    for r in recs:
        data = (dest / r["file"]).read_bytes()
        assert r["bytes"] == len(data)
        assert r["sha256"] == hashlib.sha256(data).hexdigest()
    first = json.loads((dest / "manifest.json").read_text())
    download.download_metadata(dest, base_url=meta_dir.as_uri())  # unchanged files are not fetched again
    assert json.loads((dest / "manifest.json").read_text()) == first
