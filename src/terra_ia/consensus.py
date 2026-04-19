from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_parcelle_ids(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(" ", "", regex=False)
        .str.replace("-", "", regex=False)
        .str.ljust(14)
        .str[:14]
    )


def compute_consensus_score(
    df: pd.DataFrame,
    *,
    approach3_path: Path,
    approach4_path: Path,
    load_cluster_scores: Callable[[pd.Index], pd.Series],
) -> tuple[pd.DataFrame, dict]:
    scores: dict[str, pd.Series] = {}
    meta: dict[str, object] = {}

    if "CPI_v3" in df.columns:
        scores["CPI_v3"] = df["CPI_v3"]
        meta["CPI_v3"] = True
    else:
        meta["CPI_v3"] = False

    cluster_series = df["cluster_score"] if "cluster_score" in df.columns else load_cluster_scores(df.index)
    if cluster_series.notna().any():
        scores["cluster_score"] = cluster_series
        meta["cluster_score"] = True
    else:
        meta["cluster_score"] = False

    cpi_market = None
    if approach3_path.exists():
        try:
            approach3_df = pd.read_csv(approach3_path)
            if "CPI_market" in approach3_df.columns:
                if "id_parcelle" in approach3_df.columns and "id" in df.columns:
                    score_map = approach3_df.set_index(normalize_parcelle_ids(approach3_df["id_parcelle"]))["CPI_market"]
                    cpi_market = normalize_parcelle_ids(df["id"]).map(score_map)
                elif len(approach3_df) == len(df):
                    cpi_market = approach3_df["CPI_market"]
                    cpi_market.index = df.index
        except Exception as exc:
            print(f"  WARNING Approach3 lecture : {exc}")

    meta["CPI_market"] = cpi_market is not None and pd.Series(cpi_market).notna().any()
    if meta["CPI_market"]:
        scores["CPI_market"] = pd.Series(cpi_market, index=df.index)
    else:
        print("  Approach3 CPI_market : absent")

    cpi_preference = None
    if approach4_path.exists():
        try:
            approach4_df = pd.read_csv(approach4_path)
            preference_col = "CPI_preference" if "CPI_preference" in approach4_df.columns else None
            id_col = None
            for candidate in ["id_parcelle_norm", "id_parcelle"]:
                if candidate in approach4_df.columns:
                    id_col = candidate
                    break
            if preference_col:
                if id_col and "id" in df.columns:
                    ids_norm = normalize_parcelle_ids(approach4_df[id_col])
                    score_map = pd.Series(approach4_df[preference_col].values, index=ids_norm)
                    cpi_preference = normalize_parcelle_ids(df["id"]).map(score_map)
                elif len(approach4_df) == len(df):
                    cpi_preference = approach4_df[preference_col]
                    cpi_preference.index = df.index
        except Exception as exc:
            print(f"  WARNING Approach4 lecture : {exc}")

    meta["CPI_preference"] = cpi_preference is not None and pd.Series(cpi_preference).notna().any()
    if meta["CPI_preference"]:
        scores["CPI_preference"] = pd.Series(cpi_preference, index=df.index)
    else:
        print("  Approach4 CPI_preference : absent")

    normalized_scores: dict[str, pd.Series] = {}
    for name, series in scores.items():
        normalized_scores[name] = pd.Series(series, index=df.index).rank(pct=True) * 100

    if not normalized_scores:
        print("  WARNING Aucun score disponible pour le consensus")
        empty = pd.DataFrame(
            np.nan,
            index=df.index,
            columns=["consensus_score", "consensus_confidence", "n_approaches_used"],
        )
        return empty, meta

    normalized_df = pd.concat(normalized_scores.values(), axis=1)
    normalized_df.columns = list(normalized_scores.keys())
    consensus_score = normalized_df.mean(axis=1, skipna=True)
    consensus_confidence = 1 - normalized_df.std(axis=1, skipna=True).fillna(0) / 100
    n_used = normalized_df.notna().sum(axis=1)

    result = pd.DataFrame(
        {
            "consensus_score": consensus_score,
            "consensus_confidence": consensus_confidence.clip(0, 1),
            "n_approaches_used": n_used,
        },
        index=df.index,
    )

    available_count = int(sum(bool(value) for value in meta.values()))
    meta["available_count"] = available_count
    meta["median_consensus"] = float(consensus_score.median()) if not consensus_score.empty else np.nan
    meta["median_confidence"] = float(consensus_confidence.median()) if not consensus_confidence.empty else np.nan
    meta["high_conf"] = int((consensus_confidence > 0.80).sum())
    meta["low_conf"] = int((consensus_confidence < 0.50).sum())
    meta["n_total"] = len(df)

    print(f"Consensus score - {available_count} approches disponibles sur 4")
    print(f"  CPI_v3          : {'disponible' if meta.get('CPI_v3') else 'absent'}")
    print(f"  cluster_score   : {'disponible' if meta.get('cluster_score') else 'absent'}")
    print(f"  CPI_market      : {'disponible' if meta.get('CPI_market') else 'absent'}")
    print(f"  CPI_preference  : {'disponible' if meta.get('CPI_preference') else 'absent'}")
    if meta["median_consensus"] is not None:
        print(f"Consensus median : {meta['median_consensus']:.1f}")
    if meta["median_confidence"] is not None:
        print(f"Confiance mediane : {meta['median_confidence']:.2f}")
    if meta["n_total"] > 0:
        print(
            f"Parcelles haute confiance (>0.80) : {meta['high_conf']} "
            f"({meta['high_conf'] / meta['n_total'] * 100:.1f}%)"
        )
        print(
            f"Parcelles faible confiance (<0.50) : {meta['low_conf']} "
            f"({meta['low_conf'] / meta['n_total'] * 100:.1f}%)"
        )

    return result, meta
