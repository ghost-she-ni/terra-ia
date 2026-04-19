from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def prepare_dvf(dvf_path: Path, *, commune_code: str) -> pd.DataFrame | None:
    if not dvf_path.exists():
        return None

    try:
        df = pd.read_csv(dvf_path, low_memory=False)
        print(f"    DVF charge : {len(df)} lignes")

        rename: dict[str, str] = {}
        already_renamed: set[str] = set()
        for col in df.columns:
            col_key = col.lower().replace(" ", "_")
            if "valeur" in col_key and "fonci" in col_key and "valeur_fonciere" not in already_renamed:
                rename[col] = "valeur_fonciere"
                already_renamed.add("valeur_fonciere")
            elif "surface" in col_key and "terrain" in col_key and "surface_terrain" not in already_renamed:
                rename[col] = "surface_terrain"
                already_renamed.add("surface_terrain")
            elif ("id_parcelle" in col_key or "code_parcelle" in col_key) and "id_parcelle" not in already_renamed:
                rename[col] = "id_parcelle"
                already_renamed.add("id_parcelle")
            elif "nature" in col_key and "mutation" in col_key and "nature_mutation" not in already_renamed:
                rename[col] = "nature_mutation"
                already_renamed.add("nature_mutation")
        df = df.rename(columns=rename)

        if "nature_mutation" in df.columns:
            df = df[df["nature_mutation"].str.contains("Vente", na=False)]

        if "valeur_fonciere" not in df.columns:
            print("    Warning: DVF colonne valeur_fonciere introuvable")
            return None

        df["valeur_fonciere"] = pd.to_numeric(df["valeur_fonciere"], errors="coerce")
        df["surface_terrain"] = pd.to_numeric(df.get("surface_terrain", pd.Series()), errors="coerce")
        df = df.dropna(subset=["valeur_fonciere"])
        df = df[df["valeur_fonciere"] > 0]

        if "surface_terrain" in df.columns and df["surface_terrain"].notna().any():
            df = df[df["surface_terrain"] > 0]
            df["prix_m2"] = df["valeur_fonciere"] / df["surface_terrain"]
            df = df[df["prix_m2"].between(10, 10000)]

        commune_col = None
        for candidate in ["code_commune", "codecom", "code_dep_commune", "l_codinsee", "commune", "code_insee"]:
            if candidate in df.columns:
                commune_col = candidate
                break

        if commune_col is None and "id_parcelle" in df.columns:
            df["_code_commune_from_id"] = df["id_parcelle"].astype(str).str[:5]
            commune_col = "_code_commune_from_id"
            print("    Code commune extrait depuis id_parcelle")

        if commune_col is not None:
            df_commune = df[df[commune_col].astype(str).str.strip() == commune_code]
            print(f"    Filtre commune ({commune_col}={commune_code}) : {len(df_commune)} transactions")
        else:
            df_commune = pd.DataFrame()
            print(f"    Warning: colonne commune introuvable - colonnes disponibles : {list(df.columns[:15])}")

        if len(df_commune) >= 50:
            df = df_commune
            print(f"    DVF commune : {len(df)} transactions")
        else:
            print(f"    Warning: fallback departement ({len(df)} transactions) - commune trouvee : {len(df_commune)}")

        if "id_parcelle" in df.columns:
            sample = df["id_parcelle"].dropna().head(3).tolist()
            print(f"    Sample id_parcelle DVF : {sample}")

        p20 = df["prix_m2"].quantile(0.20) if "prix_m2" in df.columns else None
        p80 = df["prix_m2"].quantile(0.80) if "prix_m2" in df.columns else None
        p20_str = f"{p20:.0f}" if p20 is not None else "N/A"
        p80_str = f"{p80:.0f}" if p80 is not None else "N/A"
        med = df.get("prix_m2", pd.Series()).median()
        med_str = f"{med:.0f}" if pd.notna(med) else "N/A"
        print(f"    DVF propre : {len(df)} ventes | prix median={med_str}EUR/m2 | P20={p20_str}EUR P80={p80_str}EUR")
        return df
    except Exception as exc:
        print(f"    Warning: DVF erreur : {exc}")
        return None


def load_cluster_scores(cluster_scores_path: Path, parcelles_index) -> pd.Series:
    if not cluster_scores_path.exists():
        print(f"    H5 CLUSTERING : {cluster_scores_path.name} absent - ignore")
        return pd.Series(np.nan, index=parcelles_index)

    cluster_scores = pd.read_csv(cluster_scores_path, index_col=0)
    if "cluster_score" not in cluster_scores.columns:
        print("    H5 CLUSTERING : colonne cluster_score absente - ignore")
        return pd.Series(np.nan, index=parcelles_index)

    aligned = cluster_scores["cluster_score"].reindex(parcelles_index)
    n_matched = aligned.notna().sum()
    print(f"    H5 CLUSTERING : {n_matched:,} cluster_scores charges")
    return aligned


def create_snorkel_labels_v6(
    df: pd.DataFrame,
    *,
    dvf: pd.DataFrame | None = None,
    cluster_scores: pd.Series | None = None,
) -> pd.Series:
    idx = df.index
    h1 = pd.Series(0.0, index=idx)
    h2 = pd.Series(0.0, index=idx)
    h3 = pd.Series(0.0, index=idx)
    h4 = pd.Series(0.0, index=idx)

    slope_p90 = df.get("slope_p90", pd.Series(12.0, index=idx)).fillna(12.0)
    slope_std = df.get("slope_std", pd.Series(6.0, index=idx)).fillna(6.0)
    thalweg = df.get("has_thalweg_mean", pd.Series(0.0, index=idx)).fillna(0.0)
    surface = df.get("surface_m2", pd.Series(300.0, index=idx)).fillna(300.0)

    h1[(slope_p90 < 10.0) & (slope_std < 5.0) & (thalweg < 0.10) & (surface > 150.0)] = 1.0
    h2[(slope_p90 > 25.0) | (thalweg > 0.30) | (slope_std > 15.0)] = 1.0

    if dvf is not None and "prix_m2" in dvf.columns and len(dvf) > 0:
        p80 = dvf["prix_m2"].quantile(0.80)
        p20 = dvf["prix_m2"].quantile(0.20)
        if "id_parcelle" in dvf.columns and "id" in df.columns:
            dvf_clean = dvf.copy()
            if isinstance(dvf_clean["id_parcelle"], pd.DataFrame):
                dvf_clean["id_parcelle"] = dvf_clean["id_parcelle"].iloc[:, 0]
            df_id_norm = df["id"].astype(str).str.strip().str.upper()
            dvf_clean["id_parcelle"] = dvf_clean["id_parcelle"].astype(str).str.strip().str.upper()
            prix_parcel = dvf_clean.groupby("id_parcelle")["prix_m2"].median()
            df_prix = df_id_norm.map(prix_parcel)
            h3[df_prix > p80] = 1.0
            h4[df_prix < p20] = 1.0
        else:
            print("    Warning: DVF sans jointure id_parcelle - H3/H4 ignorees")

    print(f"    H1 actif (slope_p90<10 AND std<5 AND thalweg<0.10) : {h1.sum():.0f} parcelles")
    print(f"    H2 actif : {h2.sum():.0f} parcelles")
    print(f"    H3 DVF   : {h3.sum():.0f} parcelles")
    print(f"    H4 DVF   : {h4.sum():.0f} parcelles")

    if cluster_scores is None:
        cluster_scores = pd.Series(np.nan, index=idx)
    else:
        cluster_scores = cluster_scores.reindex(idx)

    h5_available = cluster_scores.notna().any()
    h5_pos = (cluster_scores > 80).astype(float).fillna(0.0)
    h5_neg = (cluster_scores < 20).astype(float).fillna(0.0)
    print(f"    H5 cluster>80 : {h5_pos.sum():.0f} positifs | cluster<20 : {h5_neg.sum():.0f} negatifs")

    if h5_available:
        pos_score = 0.40 * h1 + 0.45 * h3 + 0.15 * h5_pos
        neg_score = 0.75 * h2 + 0.20 * h4 + 0.05 * h5_neg
    else:
        pos_score = 0.50 * h1 + 0.50 * h3
        neg_score = 0.75 * h2 + 0.25 * h4

    threshold = 0.35
    labels = pd.Series(-1, index=idx)
    labels[(pos_score >= neg_score) & (pos_score >= threshold)] = 1
    labels[(neg_score > pos_score) & (neg_score >= threshold)] = 0

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    n_abs = int((labels == -1).sum())
    print(f"    Labels finaux V6 : {n_pos} pos / {n_neg} neg / {n_abs} abstain")
    print(f"    Ratio pos/neg : {n_pos / (n_neg + 1e-9):.2f}")
    return labels


def _empty_cluster_validation(index: pd.Index) -> dict:
    nan_series = pd.Series(np.nan, index=index, name="cluster_score")
    return {
        "cluster_score": nan_series,
        "cluster_label_agreement": np.nan,
        "pure_pos_in_best": np.nan,
        "pure_neg_in_worst": np.nan,
    }


def _score_cluster_profiles(stat_df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=stat_df.index, dtype=float)
    components = 0

    def add_component(column: str, *, higher_is_better: bool) -> None:
        nonlocal score, components
        if column not in stat_df.columns:
            return

        values = stat_df[column]
        if not values.notna().any():
            return

        min_val = values.min(skipna=True)
        max_val = values.max(skipna=True)
        if pd.isna(min_val) or pd.isna(max_val):
            return

        if max_val > min_val:
            normalized = (values - min_val) / (max_val - min_val)
        else:
            normalized = pd.Series(0.5, index=values.index, dtype=float)

        if higher_is_better:
            score = score + normalized.fillna(0.5)
        else:
            score = score + (1.0 - normalized.fillna(0.5))
        components += 1

    for column in ["slope_p90", "slope_std", "tri_mean", "has_thalweg_mean", "twi_mean"]:
        add_component(column, higher_is_better=False)
    for column in ["svf_mean", "hillshade_winter_mean", "aspect_south_ratio_mean"]:
        add_component(column, higher_is_better=True)

    if components == 0 and "label_rate" in stat_df.columns and stat_df["label_rate"].notna().any():
        return stat_df["label_rate"].fillna(stat_df["label_rate"].median())

    if components == 0:
        return pd.Series(0.5, index=stat_df.index, dtype=float)

    return score / components


def validate_labels_with_clustering(df: pd.DataFrame, labels: pd.Series, feat_cols: list[str]) -> dict:
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("  Warning: scikit-learn manquant - validation clustering ignoree")
        return _empty_cluster_validation(df.index)

    mask = labels != -1
    if not mask.any() or not feat_cols:
        print("  Warning: validation clustering impossible (donnees insuffisantes)")
        return _empty_cluster_validation(df.index)

    usable_cols = [col for col in feat_cols if col in df.columns and not df[col].isna().all()]
    ignored_cols = [col for col in feat_cols if col in df.columns and col not in usable_cols]
    if ignored_cols:
        print(f"  Validation clustering: colonnes ignorees (tout NaN) : {ignored_cols}")
    if not usable_cols:
        print("  Warning: validation clustering impossible (aucune feature exploitable)")
        return _empty_cluster_validation(df.index)

    x = df[usable_cols].copy()
    for col in usable_cols:
        median_val = x[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        x[col] = x[col].fillna(median_val)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    try:
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(x_scaled)
    except Exception as exc:
        print(f"  Warning: KMeans erreur : {exc}")
        return _empty_cluster_validation(df.index)

    cluster_labels = pd.Series(clusters, index=df.index, name="cluster_id")

    stats = []
    for cluster_id in np.unique(clusters):
        cluster_mask = cluster_labels == cluster_id
        row = {"cluster": cluster_id}
        for column in [
            "slope_p90",
            "slope_std",
            "tri_mean",
            "has_thalweg_mean",
            "twi_mean",
            "svf_mean",
            "hillshade_winter_mean",
            "aspect_south_ratio_mean",
        ]:
            row[column] = df.loc[cluster_mask, column].median() if column in df.columns else np.nan
        row["label_rate"] = labels.loc[cluster_mask].replace(-1, np.nan).mean()
        stats.append(row)
    stat_df = pd.DataFrame(stats)
    stat_df["constructibility_score"] = _score_cluster_profiles(stat_df)

    ranked = stat_df.sort_values(["constructibility_score", "cluster"], ascending=[False, True]).reset_index(drop=True)
    best_cluster = int(ranked.iloc[0]["cluster"])
    worst_cluster = int(ranked.iloc[-1]["cluster"])
    if best_cluster == worst_cluster and len(ranked) > 1:
        worst_cluster = int(ranked.iloc[1]["cluster"])

    pos_mask = labels == 1
    neg_mask = labels == 0
    pure_pos = (cluster_labels[pos_mask] == best_cluster).mean() * 100 if pos_mask.any() else np.nan
    pure_neg = (cluster_labels[neg_mask] == worst_cluster).mean() * 100 if neg_mask.any() else np.nan

    agreements = []
    if pos_mask.any():
        agreements.append(cluster_labels[pos_mask] == best_cluster)
    if neg_mask.any():
        agreements.append(cluster_labels[neg_mask] == worst_cluster)
    agreement_pct = pd.concat(agreements).mean() * 100 if agreements else np.nan

    distances = kmeans.transform(x_scaled)
    dist_best = distances[:, best_cluster]
    dist_worst = distances[:, worst_cluster]
    cluster_score = 1 - (dist_best / (dist_best + dist_worst + 1e-9))
    cluster_score = (cluster_score * 100).clip(0, 100)
    cluster_score_series = pd.Series(cluster_score, index=df.index, name="cluster_score")

    print("  Validation clustering des labels Snorkel V6:")
    if not np.isnan(agreement_pct):
        print(f"    Accord global labels/clusters : {agreement_pct:.1f}%")
    if not np.isnan(pure_pos):
        print(f"    Labels positifs dans best cluster : {pure_pos:.1f}%")
    if not np.isnan(pure_neg):
        print(f"    Labels negatifs dans worst cluster : {pure_neg:.1f}%")
    print(f"    Clusters retenus : best={best_cluster} worst={worst_cluster} | features={usable_cols}")
    print("    -> Si accord > 75%: labels coherents avec structure naturelle des donnees")
    print("    -> Si accord < 60%: dependance circulaire significative detectee")

    return {
        "cluster_score": cluster_score_series,
        "cluster_label_agreement": float(agreement_pct) if pd.notna(agreement_pct) else np.nan,
        "pure_pos_in_best": float(pure_pos) if pd.notna(pure_pos) else np.nan,
        "pure_neg_in_worst": float(pure_neg) if pd.notna(pure_neg) else np.nan,
    }


def create_spatial_blocks(parcelles: gpd.GeoDataFrame, *, grid_size: int) -> pd.Series:
    centroids = parcelles.geometry.centroid
    block_x = (centroids.x / grid_size).astype(int)
    block_y = (centroids.y / grid_size).astype(int)
    return (block_x.astype(str) + "_" + block_y.astype(str)).rename("block_id")


def analyze_blocks(df: pd.DataFrame, labels: pd.Series, *, grid_size_m: int) -> dict:
    mask = labels != -1
    valid = df[mask].copy()
    valid["label"] = labels[mask].values

    if "block_id" not in valid.columns:
        return {}

    stats = valid.groupby("block_id")["label"].agg(["sum", "count"])
    stats["ratio"] = stats["sum"] / stats["count"]

    pure_pos = (stats["ratio"] > 0.90).sum()
    pure_neg = (stats["ratio"] < 0.10).sum()
    mixed = ((stats["ratio"] >= 0.10) & (stats["ratio"] <= 0.90)).sum()

    print(f"    Analyse blocs {grid_size_m}m :")
    print(f"      Blocs mixtes (10-90%)    : {mixed}")
    print(f"      Blocs purs positifs >90% : {pure_pos}")
    print(f"      Blocs purs negatifs <10% : {pure_neg}")
    if pure_pos + pure_neg > 0.3 * len(stats):
        print(f"    Warning: beaucoup de blocs monoclasses - essayer {grid_size_m + 100}m ou plus")

    return {
        "mixed": int(mixed),
        "pure_pos": int(pure_pos),
        "pure_neg": int(pure_neg),
        "total": int(len(stats)),
    }


__all__ = [
    "analyze_blocks",
    "create_snorkel_labels_v6",
    "create_spatial_blocks",
    "load_cluster_scores",
    "prepare_dvf",
    "validate_labels_with_clustering",
]
