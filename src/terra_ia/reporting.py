from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

from .exports import build_ml_dataset_readme


def export_results(
    df: pd.DataFrame,
    parcelles: gpd.GeoDataFrame,
    stats_report: dict,
    *,
    output_csv_v6: Path,
    output_csv_ml_v6: Path,
    output_report: Path,
    output_readme_ml_v6: Path,
    output_shap_parcelle: Path,
    all_features: list[str],
    feature_groups: dict[str, list[str]],
    grid_size_m: int,
    commune_code: str,
    tau_softmin: float | None,
) -> None:
    score_cols = [
        "CPI_v3",
        "CPI_v3_label",
        "CPI_ML_v3",
        "CPI_technique",
        "CPI_technique_label",
        "score_pente",
        "score_hydro",
        "score_morpho",
        "score_soleil",
        "score_continu",
        "score_softmin",
        "gate_factor",
        "gate_reason",
        "zone_plu",
        "compactness_ratio",
        "elongation_ratio",
        "shape_warning",
        "ces_existant",
        "ces_residuel",
        "emprise_residuelle_m2",
        "ces_warning",
        "cluster_score",
        "cpi_ml_ci_low",
        "cpi_ml_ci_high",
        "cpi_ml_std",
        "consensus_score",
        "consensus_confidence",
        "n_approaches_used",
    ]
    brgm_cols = [col for col in df.columns if col.startswith("brgm_")]
    feat_cols = [col for col in all_features if col in df.columns]
    meta_cols = ["commune", "surface_m2", "nan_ratio", "is_valid", "proxy_label", "block_id"]
    internal_val_cols = [col for col in ["CPI_valeur", "CPI_valeur_label"] if col in df.columns]
    all_cols = [col for col in meta_cols + feat_cols + score_cols + brgm_cols + internal_val_cols if col in df.columns]

    df_export = df.copy()
    if "id" in parcelles.columns:
        df_export.insert(0, "id_parcelle", parcelles["id"].values)

    df_export[all_cols].to_csv(output_csv_v6, index=True)
    print(f"\n  Dataset complet : {output_csv_v6}")
    print(f"    {len(df_export):,} parcelles x {len(all_cols)} colonnes")

    valid_labeled = (
        (df["is_valid"] & (df["proxy_label"] != -1))
        if "proxy_label" in df.columns and "is_valid" in df.columns
        else pd.Series(False, index=df.index)
    )

    ml_cols = (["id_parcelle"] if "id_parcelle" in df_export.columns else []) + ["block_id"] + feat_cols + [
        "proxy_label",
        "CPI_v3",
    ]
    ml_cols = [col for col in ml_cols if col in df_export.columns]
    df_ml = df_export[valid_labeled][ml_cols].copy() if valid_labeled.any() else pd.DataFrame(columns=ml_cols)

    df_ml.to_csv(output_csv_ml_v6, index=False)
    n_pos = int((df_ml.get("proxy_label", pd.Series()) == 1).sum())
    n_neg = int((df_ml.get("proxy_label", pd.Series()) == 0).sum())
    print(f"\n  Dataset ML : {output_csv_ml_v6}")
    print(f"    {len(df_ml):,} parcelles x {len(df_ml.columns)} colonnes")
    print(f"    Labels : {n_pos} pos / {n_neg} neg")

    readme = build_ml_dataset_readme(
        output_csv_ml_v6=output_csv_ml_v6,
        feat_cols=feat_cols,
        feature_groups=feature_groups,
        grid_size_m=grid_size_m,
        shap_filename=output_shap_parcelle.name,
        generated_from="python pipeline.py",
    )
    with open(output_readme_ml_v6, "w", encoding="utf-8") as handle:
        handle.write(readme)
    print(f"\n  README coequipier : {output_readme_ml_v6}")

    valid = df[df.get("is_valid", pd.Series(True, index=df.index))]
    stats_report.update(
        {
            "version": "6.0",
            "commune": commune_code,
            "n_total": int(len(df)),
            "n_valid": int(df.get("is_valid", pd.Series(True, index=df.index)).sum()),
            "n_labeled": int(len(df_ml)),
            "n_pos": int(n_pos),
            "n_neg": int(n_neg),
            "n_blocks": int(df.get("block_id", pd.Series()).nunique()),
            "cpi_v3_stats": {
                "mean": float(valid["CPI_v3"].mean()) if "CPI_v3" in valid else None,
                "std": float(valid["CPI_v3"].std()) if "CPI_v3" in valid else None,
                "min": float(valid["CPI_v3"].min()) if "CPI_v3" in valid else None,
                "max": float(valid["CPI_v3"].max()) if "CPI_v3" in valid else None,
            },
            "cpi_technique_stats": {
                "mean": float(valid["CPI_technique"].mean()) if "CPI_technique" in valid.columns else None,
                "std": float(valid["CPI_technique"].std()) if "CPI_technique" in valid.columns else None,
                "min": float(valid["CPI_technique"].min()) if "CPI_technique" in valid.columns else None,
                "max": float(valid["CPI_technique"].max()) if "CPI_technique" in valid.columns else None,
            },
            "brgm_flags": {col: int(df[col].sum()) for col in brgm_cols if col in df.columns},
            "features": feat_cols,
            "all_features": all_features,
            "grid_size_m": grid_size_m,
            "tau_softmin": float(tau_softmin) if tau_softmin else None,
            "plu_distribution": _plu_distribution(df),
            "shape_metrics": {
                "median_compactness": float(df["compactness_ratio"].median()) if "compactness_ratio" in df.columns else None,
                "n_elongated": int((df.get("compactness_ratio", pd.Series()) < 0.20).sum())
                if "compactness_ratio" in df.columns
                else 0,
            },
            "ces_metrics": {
                "median_residuel": float(df["ces_residuel"].median()) if "ces_residuel" in df.columns else None,
                "n_viable_80m2": int((df.get("emprise_residuelle_m2", pd.Series()) >= 80).sum())
                if "emprise_residuelle_m2" in df.columns
                else 0,
                "n_saturated": int((df.get("ces_existant", pd.Series()) > 0.80).sum())
                if "ces_existant" in df.columns
                else 0,
            },
            "cluster_label_agreement": stats_report.get("cluster_label_agreement"),
        }
    )

    with open(output_report, "w", encoding="utf-8") as handle:
        json.dump(stats_report, handle, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Rapport stats JSON : {output_report}")

    _print_console_summary(df, brgm_cols)


def _plu_distribution(df: pd.DataFrame) -> dict:
    if "zone_plu" not in df.columns:
        return {"U": 0, "AU": 0, "N": 0, "A": 0, "inconnu": 0}
    series = df["zone_plu"]
    return {
        "U": int((series.str.upper().str.startswith("U") & ~series.str.upper().str.startswith("AU")).sum()),
        "AU": int(series.str.upper().str.startswith("AU").sum()),
        "N": int((series == "N").sum()),
        "A": int((series == "A").sum()),
        "inconnu": int((series == "inconnu").sum()),
    }


def _print_console_summary(df: pd.DataFrame, brgm_cols: list[str]) -> None:
    print("\n  Distribution CPI_v3 (parcelles valides)")
    for label in ["Eliminatoire", "Faible", "Moyen", "Bon", "Excellent"]:
        count = int((df["CPI_v3_label"] == label).sum()) if "CPI_v3_label" in df.columns else 0
        pct = count / max(len(df), 1) * 100
        print(f"    {label:<15} : {count:>6,} ({pct:>5.1f}%)")

    if "CPI_technique_label" in df.columns:
        print("\n  Distribution CPI_technique")
        for label in ["Eliminatoire", "Contraint", "Faisable", "Favorable", "Optimal"]:
            count = int((df["CPI_technique_label"] == label).sum())
            pct = count / max(len(df), 1) * 100
            print(f"    {label:<15} : {count:>6,} ({pct:>5.1f}%)")

    print("\n  Analyse PLU")
    if "zone_plu" in df.columns:
        counts = _plu_distribution(df)
        for zone in ["U", "AU", "N", "A", "inconnu"]:
            print(f"    {zone:<10}: {counts[zone]:>6,} parcelles")
    else:
        print("    zone_plu indisponible")

    print("\n  Forme des parcelles")
    if "compactness_ratio" in df.columns:
        n_elong = int((df["compactness_ratio"] < 0.20).sum())
        print(f"    Parcelles allongees (<0.20) : {n_elong:,} ({n_elong / max(len(df), 1) * 100:.1f}%)")
    else:
        print("    Compacite non calculee")

    print("\n  CES residuel")
    if "emprise_residuelle_m2" in df.columns:
        n_viable = int((df["emprise_residuelle_m2"] >= 80).sum())
        n_saturated = (
            int(df.get("ces_existant", pd.Series()).fillna(0).gt(0.80).sum()) if "ces_existant" in df.columns else 0
        )
        print(f"    Emprise >= 80m2  : {n_viable:,} parcelles")
        print(f"    Parcelles saturees (>80% bati) : {n_saturated:,}")
    else:
        print("    CES residuel non calcule")

    print("\n  Correlations croisees des scores")
    score_cols = ["CPI_v3", "CPI_technique", "CPI_ML_v3", "cluster_score"]
    available = [col for col in score_cols if col in df.columns]
    if len(available) >= 2:
        corr_matrix = df[available].corr(method="pearson").round(3)
        print(corr_matrix.to_string())
    else:
        print("    Correlation non calculable (scores manquants)")

    if brgm_cols:
        print("\n  Alertes BRGM")
        for col in brgm_cols:
            if col in df.columns:
                count = int(df[col].sum())
                pct = count / max(len(df), 1) * 100
                print(f"    {col:<30} : {count:>6,} parcelles ({pct:>5.1f}%)")

    if "CPI_ML_v3" in df.columns:
        print("\n  Top 5 parcelles CPI_ML_v3")
        show_cols = [col for col in ["CPI_ML_v3", "CPI_v3", "slope_p50", "slope_p90", "svf_mean", "twi_mean", "surface_m2"] if col in df.columns]
        valid_sc = df[df.get("is_valid", pd.Series(True, index=df.index))]
        print(valid_sc.nlargest(5, "CPI_ML_v3")[show_cols].round(2).to_string())


__all__ = ["export_results"]
