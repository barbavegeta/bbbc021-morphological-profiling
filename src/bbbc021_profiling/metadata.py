"""Join the BBBC021 metadata tables and work out which plates, wells and fields are in scope."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CHANNELS = ("DAPI", "Tubulin", "Actin")
ZIP_URL = "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_{plate}.zip"


def treatment_key(compound: pd.Series, concentration: pd.Series) -> pd.Series:
    """Compound|concentration with the concentration fixed to 6 decimals, as written in the CSVs."""
    return compound.astype(str) + "|" + concentration.astype(float).round(6).map("{:.6f}".format)


def load_fields(meta_dir: Path) -> pd.DataFrame:
    """One row per field of view, with plate, week, well, treatment, MoA and role.

    role is one of:
      dmso           DMSO vehicle wells
      plate_control  an MoA-annotated treatment placed at the same well position on every plate
      treatment      any other MoA-annotated well
      unannotated    compound-concentrations without an MoA label
    """
    img = pd.read_csv(meta_dir / "BBBC021_v1_image.csv")
    moa = pd.read_csv(meta_dir / "BBBC021_v1_moa.csv")

    f = pd.DataFrame({
        "table_number": img["TableNumber"],
        "image_number": img["ImageNumber"],
        "plate": img["Image_Metadata_Plate_DAPI"],
        "week": img["Image_PathName_DAPI"].str.split("/").str[0],
        "well": img["Image_Metadata_Well_DAPI"],
        "replicate": img["Replicate"],
        "compound": img["Image_Metadata_Compound"],
        "concentration": img["Image_Metadata_Concentration"].astype(float),
    })
    for ch in CHANNELS:
        f[f"path_{ch}"] = img[f"Image_PathName_{ch}"] + "/" + img[f"Image_FileName_{ch}"]
    f["treatment"] = treatment_key(f["compound"], f["concentration"])

    moa = moa.assign(treatment=treatment_key(moa["compound"], moa["concentration"]))
    if moa["treatment"].duplicated().any():
        raise ValueError("duplicate compound-concentrations in the MoA table")
    f = f.merge(moa[["treatment", "moa"]], on="treatment", how="left", validate="many_to_one")

    n_plates = f["plate"].nunique()
    annotated = f["moa"].notna() & (f["compound"] != "DMSO")
    wells = f.loc[annotated, ["plate", "well", "treatment"]].drop_duplicates()
    per_position = wells.groupby(["treatment", "well"])["plate"].nunique()
    control_positions = set(per_position[per_position == n_plates].index)
    is_control = pd.Series(
        [(t, w) in control_positions for t, w in zip(f["treatment"], f["well"])], index=f.index
    )

    f["role"] = "unannotated"
    f.loc[annotated, "role"] = "treatment"
    f.loc[annotated & is_control, "role"] = "plate_control"
    f.loc[f["compound"] == "DMSO", "role"] = "dmso"
    return f


def select_fields(fields: pd.DataFrame, include_plate_controls: bool) -> pd.DataFrame:
    """Treatment wells, plus DMSO (and optionally plate-control) wells on the plates that hold them."""
    plates = set(fields.loc[fields["role"] == "treatment", "plate"])
    roles = {"treatment", "dmso"} | ({"plate_control"} if include_plate_controls else set())
    return fields[fields["plate"].isin(plates) & fields["role"].isin(roles)]


def treatment_summary(fields: pd.DataFrame) -> pd.DataFrame:
    """One row per MoA-annotated treatment (wells in the plate-control role are excluded)."""
    t = fields[fields["role"] == "treatment"].drop_duplicates(["plate", "well"])
    return (
        t.groupby("treatment")
        .agg(compound=("compound", "first"), concentration=("concentration", "first"),
             moa=("moa", "first"), wells=("well", "size"), plates=("plate", "nunique"),
             weeks=("week", "nunique"), week=("week", "first"))
        .reset_index()
        .sort_values(["moa", "compound", "concentration"])
    )


def plate_table(fields: pd.DataFrame, selected: pd.DataFrame, zip_bytes: dict[str, int] | None = None) -> pd.DataFrame:
    """Per plate: wells by role, fields and TIFFs in scope, and the estimated bytes kept."""
    wells = fields.drop_duplicates(["plate", "well"])
    counts = wells.pivot_table(index="plate", columns="role", values="well", aggfunc="size", fill_value=0)
    counts.columns = [f"wells_{c}" for c in counts.columns]
    p = fields.groupby("plate").agg(week=("week", "first"), fields_total=("well", "size")).join(counts)
    p["fields_selected"] = selected.groupby("plate").size().reindex(p.index, fill_value=0)
    p["tiffs_selected"] = p["fields_selected"] * len(CHANNELS)
    p["fraction_selected"] = p["fields_selected"] / p["fields_total"]
    p["zip_url"] = [ZIP_URL.format(plate=pl) for pl in p.index]
    if zip_bytes is not None:
        p["zip_bytes"] = p["zip_url"].map(zip_bytes)
        p["est_bytes_selected"] = (p["zip_bytes"] * p["fraction_selected"]).round().astype("int64")
    return p.reset_index()


def scope_summary(fields: pd.DataFrame, selected: pd.DataFrame, plates: pd.DataFrame) -> dict:
    t = treatment_summary(fields)
    used = plates[plates["fields_selected"] > 0]
    out = {
        "fields_total": int(len(fields)),
        "plates_total": int(fields["plate"].nunique()),
        "weeks_total": int(fields["week"].nunique()),
        "annotated_treatments": int(fields.loc[fields["moa"].notna() & (fields["compound"] != "DMSO"), "treatment"].nunique()),
        "annotated_compounds": int(fields.loc[fields["moa"].notna() & (fields["compound"] != "DMSO"), "compound"].nunique()),
        "moa_classes": int(fields.loc[fields["compound"] != "DMSO", "moa"].dropna().nunique()),
        "treatments_with_n_wells": {int(k): int(v) for k, v in t["wells"].value_counts().sort_index().items()},
        "weeks_per_compound": {int(k): int(v) for k, v in t.groupby("compound")["week"].nunique().value_counts().items()},
        "plates_selected": int(len(used)),
        "wells_selected": {r: int(n) for r, n in selected.drop_duplicates(["plate", "well"])["role"].value_counts().items()},
        "fields_selected": int(len(selected)),
        "tiffs_selected": int(len(selected) * len(CHANNELS)),
    }
    if "zip_bytes" in plates:
        out["zip_bytes_to_download"] = int(used["zip_bytes"].sum())
        out["est_bytes_kept"] = int(used["est_bytes_selected"].sum())
        out["largest_zip_bytes"] = int(used["zip_bytes"].max())
    return out
