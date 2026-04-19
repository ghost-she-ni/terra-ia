from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def filter_parcelles(
    df: pd.DataFrame,
    parcelles: gpd.GeoDataFrame,
    *,
    all_features: list[str],
    plu_constructible_zones: list[str],
    seuil_nan: float,
    seuil_surface: float,
    seuil_surf_min: float,
) -> pd.DataFrame:
    lidar_cols = [col for col in all_features if col in df.columns]
    df["nan_ratio"] = df[lidar_cols].isna().mean(axis=1) if lidar_cols else 0.0
    df["surface_m2"] = parcelles.geometry.area.values
    plu_constructible = (
        df["zone_plu"].isin(plu_constructible_zones + ["inconnu"])
        if "zone_plu" in df.columns
        else pd.Series(True, index=df.index)
    )

    df["is_valid"] = (
        (df["nan_ratio"] <= seuil_nan)
        & (df["surface_m2"] <= seuil_surface)
        & (df["surface_m2"] >= seuil_surf_min)
        & plu_constructible
    )

    n_tot = len(df)
    n_valid = int(df["is_valid"].sum())
    n_nan = int((df["nan_ratio"] > seuil_nan).sum())
    n_big = int((df["surface_m2"] > seuil_surface).sum())
    n_small = int((df["surface_m2"] < seuil_surf_min).sum())
    n_plu = int((~plu_constructible).sum())

    print(f"    Total               : {n_tot:,}")
    print(f"    Valides             : {n_valid:,} ({n_valid / max(n_tot, 1) * 100:.1f}%)")
    print(f"    NaN > {seuil_nan * 100:.0f}%          : {n_nan:,} ({n_nan / max(n_tot, 1) * 100:.1f}%)")
    print(f"    > {seuil_surface:.0f}m2          : {n_big:,}")
    print(f"    < {seuil_surf_min:.0f}m2            : {n_small:,}")
    print(f"    Zone N/A (PLU)      : {n_plu:,} ({n_plu / max(n_tot, 1) * 100:.1f}%)")
    return df


def calibrate_tau(group_scores_df: pd.DataFrame, target_std: float = 18.0) -> float:
    groups = group_scores_df.values

    def score_std(tau: float) -> float:
        values = groups.astype(float)
        shifted = values - np.max(values, axis=1, keepdims=True)
        exp_neg = np.exp(-shifted / tau)
        weights = exp_neg / exp_neg.sum(axis=1, keepdims=True)
        soft = (weights * values).sum(axis=1)
        combined = 0.70 * groups.mean(axis=1) + 0.30 * soft
        return float(combined.std())

    lo, hi = 0.5, 100.0
    for _ in range(50):
        mid = (lo + hi) / 2
        current = score_std(mid)
        if current < target_std:
            hi = mid
        else:
            lo = mid

    tau_opt = (lo + hi) / 2
    print(f"    Tau calibre : {tau_opt:.2f}  (std visee={target_std}, obtenue={score_std(tau_opt):.2f})")
    return tau_opt


def softmin(values: np.ndarray, tau: float) -> np.ndarray:
    vals = np.array(values, dtype=float)
    if vals.ndim == 1:
        vals = vals.reshape(1, -1)
    shifted = vals - np.max(vals, axis=1, keepdims=True)
    exp_neg = np.exp(-shifted / max(tau, 1e-9))
    weights = exp_neg / exp_neg.sum(axis=1, keepdims=True)
    return (weights * vals).sum(axis=1)


def normalize(
    series: pd.Series,
    invert: bool = False,
    pct_low: float = 1.0,
    pct_high: float = 99.0,
    stretch: bool = False,
) -> pd.Series:
    valid = series.dropna()
    mn = np.percentile(valid, pct_low) if len(valid) > 0 else 0
    mx = np.percentile(valid, pct_high) if len(valid) > 0 else 1
    if mx == mn:
        return pd.Series(0.5, index=series.index)

    normalized = (series.fillna(mn).clip(mn, mx) - mn) / (mx - mn)
    if stretch:
        normalized = 1 / (1 + np.exp(-6 * (normalized - 0.5)))
        normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-9)
    return 1 - normalized if invert else normalized


def precompute_group_scores_for_tau(df: pd.DataFrame) -> pd.DataFrame:
    def get(col: str, default: float = 0.5) -> pd.Series:
        if col in df.columns:
            return df[col].fillna(df[col].median() if df[col].notna().any() else default)
        return pd.Series(default, index=df.index)

    score_pente = (
        0.40 * normalize(get("slope_p50"), invert=True)
        + 0.35 * normalize(get("slope_p90"), invert=True)
        + 0.25 * normalize(get("slope_std"), invert=True)
    )
    score_hydro = normalize(get("twi_mean"), invert=True)
    score_morpho = 0.55 * normalize(get("profile_curvature_mean"), invert=False) + 0.45 * normalize(
        get("tri_mean"), invert=True
    )
    score_soleil = (
        0.40 * normalize(get("aspect_south_ratio_mean"))
        + 0.35 * normalize(get("hillshade_winter_mean"))
        + 0.25 * normalize(get("svf_mean"))
    )

    return pd.DataFrame(
        {
            "p": score_pente * 100,
            "h": score_hydro * 100,
            "m": score_morpho * 100,
            "s": score_soleil * 100,
        }
    )


def compute_parcel_compactness(parcelles: gpd.GeoDataFrame) -> dict:
    compactness = []
    elongation = []

    for geom in parcelles.geometry:
        try:
            if geom is None or geom.is_empty:
                compactness.append(np.nan)
                elongation.append(np.nan)
                continue
            area = geom.area
            perimeter = geom.length
            compact = (4 * math.pi * area) / (perimeter**2) if perimeter > 0 and area > 0 else np.nan
            minx, miny, maxx, maxy = geom.bounds
            length = max(maxx - minx, maxy - miny)
            width = min(maxx - minx, maxy - miny)
            elong = (length / width) if width > 0 else np.nan
        except Exception:
            compact = np.nan
            elong = np.nan
        compactness.append(compact)
        elongation.append(elong)

    med_c = np.nanmedian(compactness) if compactness else np.nan
    n_prob = int(np.sum(np.array(compactness) < 0.20))
    pct_prob = n_prob / len(compactness) * 100 if compactness else 0.0
    print(f"    Compacite parcelles - mediane={med_c:.3f}  ratio_problemes (<0.20)={n_prob} ({pct_prob:.1f}%)")
    return {
        "compactness_ratio": compactness,
        "elongation_ratio": elongation,
    }


def compute_ces_residuel(
    parcelles: gpd.GeoDataFrame,
    bd_topo_path: Path,
    *,
    plu_ces_max: float = 0.40,
) -> dict:
    n = len(parcelles)
    try:
        bati = gpd.read_file(bd_topo_path)
        if bati.empty:
            raise ValueError("BD TOPO vide")
        if bati.crs is None or str(bati.crs.to_epsg()) != "2154":
            bati = bati.to_crs(parcelles.crs)

        parc = parcelles.reset_index().rename(columns={"index": "parcel_idx"})
        inter = gpd.overlay(parc[["parcel_idx", "geometry"]], bati[["geometry"]], how="intersection")
        inter["built_area"] = inter.geometry.area
        built_area = inter.groupby("parcel_idx")["built_area"].sum()
        built_series = built_area.reindex(range(n), fill_value=0.0)
        surface_parcelle = parcelles.geometry.area

        ces_existant = (built_series / surface_parcelle).clip(0, 1)
        ces_residuel = np.maximum(0, plu_ces_max - ces_existant)
        emprise_residuelle = ces_residuel * surface_parcelle

        med_res = float(np.nanmedian(ces_residuel)) if n else np.nan
        n_viable = int((emprise_residuelle >= 80).sum())
        n_sature = int((ces_residuel < 0.05).sum())
        pct_viable = n_viable / n * 100 if n else 0.0
        pct_sat = n_sature / n * 100 if n else 0.0

        print(f"    CES residuel - mediane={med_res:.2f}")
        print(f"    Parcelles avec emprise residuelle >= 80m2: {n_viable} ({pct_viable:.1f}%)")
        print(f"    Parcelles saturees (CES_residuel < 0.05): {n_sature} ({pct_sat:.1f}%)")

        return {
            "ces_existant": ces_existant.values.tolist(),
            "ces_residuel": ces_residuel.values.tolist(),
            "emprise_residuelle_m2": emprise_residuelle.values.tolist(),
        }
    except Exception as exc:
        print(f"    Warning: CES residuel erreur : {exc}")
        nan_list = [np.nan] * n
        return {
            "ces_existant": nan_list,
            "ces_residuel": nan_list,
            "emprise_residuelle_m2": nan_list,
        }


def compute_cpi_v3(
    df: pd.DataFrame,
    tau: float,
    *,
    group_weights: dict[str, float],
    group_weights_technique: dict[str, float],
) -> pd.DataFrame:
    def get(col: str, default: float = 0.5) -> pd.Series:
        if col in df.columns:
            return df[col].fillna(df[col].median() if df[col].notna().any() else default)
        return pd.Series(default, index=df.index)

    score_pente = (
        0.40 * normalize(get("slope_p50"), invert=True, stretch=True)
        + 0.35 * normalize(get("slope_p90"), invert=True, stretch=True)
        + 0.25 * normalize(get("slope_std"), invert=True)
    )

    score_hydro = normalize(get("twi_mean"), invert=True, stretch=True)
    if "has_thalweg_mean" in df.columns:
        thalweg = df["has_thalweg_mean"].fillna(0).clip(0, 1)
        score_hydro = score_hydro * (1 - 0.4 * thalweg)

    compactness_norm = normalize(get("compactness_ratio"), invert=False)
    score_morpho = (
        0.44 * normalize(get("profile_curvature_mean"), invert=False)
        + 0.36 * normalize(get("tri_mean"), invert=True)
        + 0.20 * compactness_norm
    )

    has_svf = "svf_mean" in df.columns
    if has_svf:
        score_soleil = (
            0.40 * normalize(get("aspect_south_ratio_mean"))
            + 0.35 * normalize(get("hillshade_winter_mean"))
            + 0.25 * normalize(get("svf_mean"))
        )
    else:
        score_soleil = 0.57 * normalize(get("aspect_south_ratio_mean")) + 0.43 * normalize(
            get("hillshade_winter_mean")
        )

    df["score_pente"] = (score_pente * 100).round(1)
    df["score_hydro"] = (score_hydro * 100).round(1)
    df["score_morpho"] = (score_morpho * 100).round(1)
    df["score_soleil"] = (score_soleil * 100).round(1)

    df["score_continu"] = (
        group_weights["SLOPE"] * df["score_pente"]
        + group_weights["HYDROLOGY"] * df["score_hydro"]
        + group_weights["MORPHOLOGY"] * df["score_morpho"]
        + group_weights["SUNLIGHT"] * df["score_soleil"]
    ).round(1)

    group_scores = df[["score_pente", "score_hydro", "score_morpho", "score_soleil"]].values
    df["score_softmin"] = softmin(group_scores, tau).round(1)
    df["cpi_brut"] = (0.70 * df["score_continu"] + 0.30 * df["score_softmin"]).round(1)

    df["gate_factor"] = 1.0
    df["gate_reason"] = ""

    if "has_thalweg_mean" in df.columns:
        mask = df["has_thalweg_mean"].fillna(0) > 0.20
        df.loc[mask, "gate_factor"] = np.minimum(
            df.loc[mask, "gate_factor"],
            30.0 / df.loc[mask, "cpi_brut"].clip(lower=1),
        )
        df.loc[mask, "gate_reason"] = "talweg_central"

    if "slope_p90" in df.columns:
        mask = df["slope_p90"].fillna(0) > 25.0
        df.loc[mask, "gate_factor"] = np.minimum(
            df.loc[mask, "gate_factor"],
            25.0 / df.loc[mask, "cpi_brut"].clip(lower=1),
        )
        df.loc[mask & (df["gate_reason"] == ""), "gate_reason"] = "pente_extreme"
        df.loc[mask & (df["gate_reason"] != ""), "gate_reason"] = "talweg+pente"

    if "ces_existant" in df.columns:
        saturated = df["ces_existant"].fillna(0) > 0.80
        df.loc[saturated, "gate_factor"] = np.minimum(
            df.loc[saturated, "gate_factor"],
            40.0 / df.loc[saturated, "cpi_brut"].clip(lower=1),
        )
        df.loc[saturated & (df["gate_reason"] == ""), "gate_reason"] = "parcelle_saturee_CES"
        print(f"    Gate CES: {int(saturated.sum())} parcelles saturees (>80% bati) -> CPI <= 40")

    df["CPI_v3"] = (df["cpi_brut"] * df["gate_factor"]).clip(0, 100).round(1)

    def interpret(score: float) -> str:
        if score < 20:
            return "Eliminatoire"
        if score < 40:
            return "Faible"
        if score < 65:
            return "Moyen"
        if score < 82:
            return "Bon"
        return "Excellent"

    df["CPI_v3_label"] = df["CPI_v3"].apply(interpret)
    df["shape_warning"] = ""
    if ("compactness_ratio" in df.columns) or ("elongation_ratio" in df.columns):
        compactness = df.get("compactness_ratio", pd.Series(np.nan, index=df.index))
        elongation = df.get("elongation_ratio", pd.Series(np.nan, index=df.index))
        df["shape_warning"] = ((compactness < 0.20) | (elongation > 3.0)).map(
            {True: "Parcelle allongee - emprise reelle limitee", False: ""}
        )

    score_cont_tech = (
        group_weights_technique["SLOPE"] * df["score_pente"]
        + group_weights_technique["HYDROLOGY"] * df["score_hydro"]
        + group_weights_technique["MORPHOLOGY"] * df["score_morpho"]
    ).round(1)
    group_scores_tech = df[["score_pente", "score_hydro", "score_morpho"]].values
    score_softmin_tech = softmin(group_scores_tech, tau).round(1)
    cpi_tech_brut = (0.70 * score_cont_tech + 0.30 * score_softmin_tech).round(1)
    df["CPI_technique"] = (cpi_tech_brut * df["gate_factor"]).clip(0, 100).round(1)

    def interpret_technique(score: float) -> str:
        if score < 20:
            return "Eliminatoire"
        if score < 45:
            return "Contraint"
        if score < 65:
            return "Faisable"
        if score < 82:
            return "Favorable"
        return "Optimal"

    df["CPI_technique_label"] = df["CPI_technique"].apply(interpret_technique)

    if has_svf:
        cpi_valeur_raw = (
            0.40 * normalize(get("aspect_south_ratio_mean"))
            + 0.35 * normalize(get("hillshade_winter_mean"))
            + 0.25 * normalize(get("svf_mean"))
        )
    else:
        cpi_valeur_raw = 0.57 * normalize(get("aspect_south_ratio_mean")) + 0.43 * normalize(
            get("hillshade_winter_mean")
        )
    df["CPI_valeur"] = (cpi_valeur_raw * 100).clip(0, 100).round(1)

    def interpret_valeur(score: float) -> str:
        if score < 33:
            return "Faible attractivite"
        if score < 67:
            return "Attractivite moyenne"
        return "Forte attractivite"

    df["CPI_valeur_label"] = df["CPI_valeur"].apply(interpret_valeur)

    df["ces_warning"] = ""
    if "emprise_residuelle_m2" in df.columns:
        low_emprise = df["emprise_residuelle_m2"].fillna(999) < 80
        df.loc[low_emprise, "ces_warning"] = "Emprise residuelle < 80m2 - constructibilite tres limitee"

    valid_cpi = df.loc[df.get("is_valid", pd.Series(True, index=df.index)), "CPI_v3"]
    print(
        f"    CPI_v3 (parcelles valides) - moy={valid_cpi.mean():.1f}  std={valid_cpi.std():.1f}  "
        f"min={valid_cpi.min():.1f}  max={valid_cpi.max():.1f}"
    )
    print(f"    Gates : {(df['gate_factor'] < 1).sum()} parcelles plafonnees")
    print("    Distribution :")
    for label in ["Eliminatoire", "Faible", "Moyen", "Bon", "Excellent"]:
        count = int((df["CPI_v3_label"] == label).sum())
        print(f"      {label:<15}: {count:>5} ({count / max(len(df), 1) * 100:.1f}%)")

    return df


__all__ = [
    "calibrate_tau",
    "compute_ces_residuel",
    "compute_cpi_v3",
    "compute_parcel_compactness",
    "filter_parcelles",
    "normalize",
    "precompute_group_scores_for_tau",
    "softmin",
]
