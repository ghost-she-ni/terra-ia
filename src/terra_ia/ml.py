from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 20) -> float:
    k_actual = min(k, len(y_true))
    if k_actual <= 0:
        return 0.0
    top_idx = np.argsort(y_score)[-k_actual:]
    return float(y_true[top_idx].mean())


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 20) -> float:
    k_actual = min(k, len(y_true))
    if k_actual <= 0:
        return 0.0

    order = np.argsort(y_score)[::-1][:k_actual]
    gains = y_true[order].astype(float)
    discounts = np.log2(np.arange(2, k_actual + 2))
    dcg = np.sum(gains / discounts)

    ideal = np.sort(y_true.astype(float))[::-1][:k_actual]
    idcg = np.sum(ideal / discounts)
    return float(dcg / idcg) if idcg > 0 else 0.0


def _prepare_feature_frame(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    prepared = df[feat_cols].copy()
    for col in feat_cols:
        median_val = prepared[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        prepared[col] = prepared[col].fillna(median_val)

    nan_cols = prepared.columns[prepared.isna().any()].tolist()
    if nan_cols:
        print(f"    Warning: persistent NaN after imputation: {nan_cols}")
        prepared = prepared.fillna(0.0)
    return prepared


def _empty_bootstrap_frame(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cpi_ml_mean": np.nan,
            "cpi_ml_std": np.nan,
            "cpi_ml_ci_low": np.nan,
            "cpi_ml_ci_high": np.nan,
        },
        index=index,
    )


def compare_models(
    df: pd.DataFrame,
    labels: pd.Series,
    *,
    all_features: list[str],
    grid_size_m: int,
    n_folds: int,
) -> dict:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import confusion_matrix, roc_auc_score
        from sklearn.model_selection import GroupKFold, ParameterGrid
        from sklearn.preprocessing import StandardScaler
        import xgboost as xgb
    except ImportError as exc:
        print(f"  x Missing import: {exc}")
        return {}

    mask = (labels != -1) & df.get("is_valid", pd.Series(True, index=df.index))
    feat_cols = [col for col in all_features if col in df.columns]
    if not feat_cols or mask.sum() < 100:
        print(f"  x Insufficient data ({mask.sum()} labeled examples)")
        return {}

    x_full = _prepare_feature_frame(df, feat_cols)
    x = x_full.loc[mask].values
    y = labels.loc[mask].values
    blocks = (
        df.loc[mask, "block_id"].values
        if "block_id" in df.columns
        else np.zeros(mask.sum(), dtype=int)
    )
    cpi_ref = df.loc[mask, "CPI_v3"].values if "CPI_v3" in df.columns else None

    unique_blocks = np.unique(blocks)
    n_splits = min(n_folds, len(unique_blocks))
    if n_splits < 2:
        print("  x Insufficient spatial blocks for GroupKFold")
        return {}

    print(f"    Dataset : {len(x):,} parcels ({y.sum()} pos / {(y == 0).sum()} neg)")
    print(f"    Features : {feat_cols}")
    print(f"    Blocks : {len(unique_blocks)} blocks of {grid_size_m}m")

    gkf = GroupKFold(n_splits=n_splits)

    print("\n    GridSearch XGBRanker ...", flush=True)
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.03, 0.05, 0.1],
    }
    best_ndcg = 0.0
    best_params: dict = {}

    for params in ParameterGrid(param_grid):
        ndcgs = []
        model = xgb.XGBRanker(
            objective="rank:ndcg",
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            **params,
        )
        for tr_idx, val_idx in gkf.split(x, y, groups=blocks):
            x_tr, x_val = x[tr_idx], x[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            blocks_tr = blocks[tr_idx]
            _, qid_tr = np.unique(blocks_tr, return_inverse=True)
            sort_idx = np.argsort(qid_tr)
            x_tr_sorted = x_tr[sort_idx]
            y_tr_sorted = y_tr[sort_idx]
            qid_sorted = qid_tr[sort_idx]
            _, counts = np.unique(qid_sorted, return_counts=True)
            model.fit(x_tr_sorted, y_tr_sorted, group=counts)
            scores = model.predict(x_val)
            ndcgs.append(ndcg_at_k(y_val, scores, k=min(20, len(y_val))))

        mean_ndcg = float(np.mean(ndcgs))
        if mean_ndcg > best_ndcg:
            best_ndcg = mean_ndcg
            best_params = params

    print(f"    XGBRanker best params : {best_params}  NDCG@20={best_ndcg:.4f}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost rank:ndcg": xgb.XGBRanker(
            objective="rank:ndcg",
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            **best_params,
        ),
    }

    results: dict = {}
    rf_cms = []

    for name, model in models.items():
        aucs = []
        p20s = []
        ndcgs = []
        spearmans = []

        for tr_idx, val_idx in gkf.split(x, y, groups=blocks):
            x_tr, x_val = x[tr_idx], x[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            if name == "Logistic Regression":
                scaler = StandardScaler()
                x_tr = scaler.fit_transform(x_tr)
                x_val = scaler.transform(x_val)
                model.fit(x_tr, y_tr)
                scores = model.predict_proba(x_val)[:, 1]
            elif name == "Random Forest":
                model.fit(x_tr, y_tr)
                scores = model.predict_proba(x_val)[:, 1]
                preds = model.predict(x_val)
                rf_cms.append(confusion_matrix(y_val, preds, labels=[0, 1]))
            else:
                blocks_tr = blocks[tr_idx]
                _, qid_tr = np.unique(blocks_tr, return_inverse=True)
                sort_idx = np.argsort(qid_tr)
                x_tr_sorted = x_tr[sort_idx]
                y_tr_sorted = y_tr[sort_idx]
                _, counts = np.unique(qid_tr[sort_idx], return_counts=True)
                model.fit(x_tr_sorted, y_tr_sorted, group=counts)
                scores = model.predict(x_val)

            if len(np.unique(y_val)) > 1:
                try:
                    aucs.append(float(roc_auc_score(y_val, scores)))
                except Exception:
                    pass

            p20s.append(precision_at_k(y_val, scores, k=min(20, len(y_val))))
            ndcgs.append(ndcg_at_k(y_val, scores, k=min(20, len(y_val))))

            if cpi_ref is not None:
                cpi_fold = cpi_ref[val_idx]
                valid_corr = np.isfinite(scores) & np.isfinite(cpi_fold)
                if valid_corr.sum() > 1:
                    corr, _ = spearmanr(scores[valid_corr], cpi_fold[valid_corr])
                    if pd.notna(corr):
                        spearmans.append(float(corr))

        results[name] = {
            "AUC": f"{np.mean(aucs):.3f} +/- {np.std(aucs):.3f}" if aucs else "N/A",
            "Precision@20": f"{np.mean(p20s):.3f} +/- {np.std(p20s):.3f}",
            "NDCG@20": f"{np.mean(ndcgs):.3f} +/- {np.std(ndcgs):.3f}",
            "Spearman_CPI": f"{np.mean(spearmans):.3f}" if spearmans else "N/A",
            "_ndcg_mean": float(np.mean(ndcgs)),
            "_auc_mean": float(np.mean(aucs)) if aucs else 0.0,
        }
        if name == "XGBoost rank:ndcg":
            results[name]["_best_params"] = best_params

    if rf_cms:
        cm_avg = np.array(rf_cms).mean(axis=0).astype(int)
        tn, fp, fn, tp = cm_avg.ravel()
        print("\n    Random Forest confusion matrix (avg folds):")
        print(f"      True negatives : {tn:>5}  |  False positives : {fp:>5}")
        print(f"      False negatives: {fn:>5}  |  True positives  : {tp:>5}")
        print(f"      Precision: {tp / (tp + fp + 1e-9):.3f}  Recall: {tp / (tp + fn + 1e-9):.3f}")
        results["RF_confusion_matrix"] = cm_avg.tolist()

    best_model = max(
        results,
        key=lambda key: results[key].get("_ndcg_mean", 0.0)
        if key != "RF_confusion_matrix"
        else -1.0,
    )

    print(f"\n    {'Model':<25} {'AUC':>14} {'P@20':>16} {'NDCG@20':>16} {'Spearman':>12}")
    print(f"    {'-' * 88}")
    for name, result in results.items():
        if name == "RF_confusion_matrix":
            continue
        marker = "  <- best NDCG" if name == best_model else ""
        print(
            f"    {name:<25} {result['AUC']:>14} {result['Precision@20']:>16} "
            f"{result['NDCG@20']:>16} {result['Spearman_CPI']:>12}{marker}"
        )

    print(f"\n    -> Selected model     : {best_model}")
    print("    -> Main criterion     : NDCG@20")
    if any(
        isinstance(result, dict) and result.get("Spearman_CPI") != "N/A"
        for result in results.values()
        if isinstance(result, dict)
    ):
        print("    -> External check     : Spearman vs CPI_v3")

    return results


def train_and_explain(
    df: pd.DataFrame,
    labels: pd.Series,
    *,
    all_features: list[str],
    feature_groups: dict[str, list[str]],
    group_weights: dict[str, float],
    data_dir: Path,
    output_shap_parcelle: Path,
    shap_importance_path: Path,
    shap_group_path: Path,
    best_params: dict | None = None,
) -> None:
    try:
        import shap
        import xgboost as xgb
    except ImportError:
        print("  x xgboost or shap missing")
        return

    mask = (labels != -1) & df.get("is_valid", pd.Series(True, index=df.index))
    feat_cols = [col for col in all_features if col in df.columns]
    if not feat_cols or mask.sum() < 50:
        print("  x Insufficient data")
        return

    x_full = _prepare_feature_frame(df, feat_cols)
    x_labeled = x_full.loc[mask]
    y_labeled = labels.loc[mask].values
    blocks = (
        df.loc[mask, "block_id"].values
        if "block_id" in df.columns
        else np.zeros(len(y_labeled), dtype=int)
    )

    _, qid = np.unique(blocks, return_inverse=True)
    sort_idx = np.argsort(qid)
    x_sorted = x_labeled.values[sort_idx]
    y_sorted = y_labeled[sort_idx]
    _, counts = np.unique(qid[sort_idx], return_counts=True)

    params = best_params or {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
    }
    model = xgb.XGBRanker(
        objective="rank:ndcg",
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        **params,
    )
    model.fit(x_sorted, y_sorted, group=counts)

    valid_mask = df.get("is_valid", pd.Series(True, index=df.index))
    scores = model.predict(x_full.loc[valid_mask].values)
    score_min = scores.min()
    score_max = scores.max()
    if score_max > score_min:
        scores_norm = ((scores - score_min) / (score_max - score_min + 1e-9) * 100).round(1)
    else:
        scores_norm = np.full_like(scores, 50.0)

    df["CPI_ML_v3"] = np.nan
    df.loc[valid_mask, "CPI_ML_v3"] = scores_norm

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_labeled.values)

    def feature_group_name(feature: str) -> str:
        for group_name, features in feature_groups.items():
            if feature in features:
                return group_name
        return "?"

    shap_imp = pd.DataFrame(
        {
            "feature": feat_cols,
            "mean_|SHAP|": np.abs(shap_values).mean(axis=0),
            "groupe": [feature_group_name(feature) for feature in feat_cols],
        }
    ).sort_values("mean_|SHAP|", ascending=False)
    group_imp = shap_imp.groupby("groupe")["mean_|SHAP|"].sum().sort_values(ascending=False)

    print("\n    SHAP global feature importance")
    print(f"    {'Feature':<35} {'Group':<15} {'|SHAP|':>10}  Bar")
    print(f"    {'-' * 75}")
    max_shap = float(shap_imp["mean_|SHAP|"].max()) if not shap_imp.empty else 0.0
    for _, row in shap_imp.iterrows():
        bar = "#" * max(1, int(row["mean_|SHAP|"] / max(max_shap, 1e-9) * 25))
        print(
            f"    {row['feature']:<35} {row['groupe']:<15} "
            f"{row['mean_|SHAP|']:>10.4f}  {bar}"
        )

    print("\n    SHAP thematic group importance")
    max_group_imp = float(group_imp.max()) if len(group_imp) else 0.0
    for group_name, value in group_imp.items():
        weight = group_weights.get(group_name, 0.0)
        bar = "#" * max(1, int(value / max(max_group_imp, 1e-9) * 20))
        print(f"    {group_name:<20} SHAP={value:.4f}  Spec_weight={weight:.0%}  {bar}")

    best_idx = df["CPI_ML_v3"].dropna().idxmax()
    best_score = df.loc[best_idx, "CPI_ML_v3"]
    labeled_indices = df.loc[mask].index.tolist()
    if best_idx in labeled_indices:
        best_position = labeled_indices.index(best_idx)
        best_shap = pd.Series(shap_values[best_position], index=feat_cols).sort_values()
        print(f"\n    Best parcel SHAP decomposition (CPI_ML={best_score:.1f})")
        for feature, value in best_shap.items():
            group_name = feature_group_name(feature)
            sign = "+" if value >= 0 else ""
            direction = "POS" if value >= 0 else "NEG"
            print(f"    {direction:<3} {feature:<35} {sign}{value:>8.4f}  [{group_name}]")

    if "CPI_v3" in df.columns:
        common = df[["CPI_v3", "CPI_ML_v3"]].dropna()
        if len(common) > 10:
            corr_pearson = common["CPI_v3"].corr(common["CPI_ML_v3"])
            corr_spearman = spearmanr(common["CPI_v3"], common["CPI_ML_v3"])[0]
            diff = (common["CPI_ML_v3"] - common["CPI_v3"]).abs()
            print("\n    CPI_v3 vs CPI_ML_v3")
            print(f"    Pearson  : {corr_pearson:.3f}")
            print(f"    Spearman : {corr_spearman:.3f}")
            print(f"    Mean gap : {diff.mean():.1f} pts  max={diff.max():.1f} pts")

    data_dir.mkdir(parents=True, exist_ok=True)
    shap_importance_path.parent.mkdir(parents=True, exist_ok=True)
    shap_group_path.parent.mkdir(parents=True, exist_ok=True)
    output_shap_parcelle.parent.mkdir(parents=True, exist_ok=True)

    shap_imp.to_csv(shap_importance_path, index=False)
    group_imp.to_csv(shap_group_path, header=True)
    print(f"\n    SHAP saved -> {shap_importance_path}")

    shap_df = pd.DataFrame(
        shap_values,
        columns=[f"shap_{feature}" for feature in feat_cols],
        index=x_labeled.index,
    )
    if "id_parcelle" in df.columns:
        shap_df.insert(0, "id_parcelle", df.loc[x_labeled.index, "id_parcelle"])
    elif "id" in df.columns:
        shap_df.insert(0, "id_parcelle", df.loc[x_labeled.index, "id"])

    shap_df["CPI_ML_v6"] = df.loc[x_labeled.index, "CPI_ML_v3"] if "CPI_ML_v3" in df.columns else np.nan
    shap_df["CPI_v6"] = df.loc[x_labeled.index, "CPI_v3"] if "CPI_v3" in df.columns else np.nan
    shap_df["CPI_technique"] = (
        df.loc[x_labeled.index, "CPI_technique"] if "CPI_technique" in df.columns else np.nan
    )
    extra_cols = [
        "zone_plu",
        "compactness_ratio",
        "elongation_ratio",
        "shape_warning",
        "ces_existant",
        "ces_residuel",
        "emprise_residuelle_m2",
        "ces_warning",
        "cluster_score",
        "brgm_argiles_flag",
        "brgm_mvt_terrain_flag",
        "cpi_ml_ci_low",
        "cpi_ml_ci_high",
        "cpi_ml_std",
    ]
    for col in extra_cols:
        if col in df.columns:
            shap_df[col] = df.loc[x_labeled.index, col]

    for group_name, features in feature_groups.items():
        shap_cols = [f"shap_{feature}" for feature in features if f"shap_{feature}" in shap_df.columns]
        if shap_cols:
            shap_df[f"shap_groupe_{group_name}"] = shap_df[shap_cols].sum(axis=1)

    shap_df.to_csv(output_shap_parcelle, index=False)
    print(f"\n    Parcel SHAP saved -> {output_shap_parcelle}")
    print(f"    {len(shap_df)} parcels x {len(shap_df.columns)} SHAP columns")


def compute_cpi_bootstrap(
    df: pd.DataFrame,
    feat_cols: list[str],
    *,
    n_bootstrap: int = 100,
    confidence: float = 0.95,
    random_seed: int = 42,
) -> pd.DataFrame:
    try:
        import xgboost as xgb
    except ImportError:
        print("  Warning: xgboost missing - bootstrap skipped")
        return _empty_bootstrap_frame(df.index)

    if not feat_cols:
        print("  Warning: bootstrap skipped (no features)")
        return _empty_bootstrap_frame(df.index)

    mask_valid = df.get("is_valid", pd.Series(True, index=df.index))
    mask_labeled = mask_valid & (df.get("proxy_label", pd.Series(-1, index=df.index)) != -1)
    if mask_labeled.sum() < 10:
        print("  Warning: bootstrap impossible (insufficient labels)")
        return _empty_bootstrap_frame(df.index)

    df_valid = df.loc[mask_valid].copy()
    x_full = _prepare_feature_frame(df_valid, feat_cols)

    df_labeled = df.loc[mask_labeled, feat_cols].copy()
    for col in feat_cols:
        median_val = df_labeled[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        df_labeled[col] = df_labeled[col].fillna(median_val)
    y_labeled = df.loc[mask_labeled, "proxy_label"].values

    rng = np.random.default_rng(random_seed)
    preds = []

    for run_idx in range(n_bootstrap):
        if (run_idx + 1) % 10 == 0 or run_idx == 0:
            print(f"  Bootstrap run {run_idx + 1}/{n_bootstrap}...")

        sample_idx = rng.choice(len(df_labeled), size=max(1, int(0.8 * len(df_labeled))), replace=True)
        x_boot = df_labeled.iloc[sample_idx].values
        y_boot = y_labeled[sample_idx]

        model = xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_seed + run_idx,
            n_jobs=-1,
        )
        model.fit(x_boot, y_boot)
        proba = model.predict_proba(x_full.values)[:, 1]
        proba = (proba - proba.min()) / (proba.max() - proba.min() + 1e-9) * 100
        preds.append(proba)

    preds_array = np.vstack(preds)
    mean = preds_array.mean(axis=0)
    std = preds_array.std(axis=0)
    lower = np.percentile(preds_array, (1 - confidence) / 2 * 100, axis=0)
    upper = np.percentile(preds_array, (1 + confidence) / 2 * 100, axis=0)

    half_width = (upper - lower) / 2
    stable = int((upper - lower < 5).sum())
    unstable = int((upper - lower > 15).sum())
    n_valid = len(df_valid)

    print("\nBootstrap CPI_ML:")
    print(f"  Mean interval : +/- {half_width.mean():.1f} pts")
    print(f"  Very stable parcels (CI < 5pts) : {stable} ({stable / max(n_valid, 1) * 100:.1f}%)")
    print(f"  Unstable parcels (CI > 15pts)   : {unstable} ({unstable / max(n_valid, 1) * 100:.1f}%)")

    return pd.DataFrame(
        {
            "cpi_ml_mean": mean,
            "cpi_ml_std": std,
            "cpi_ml_ci_low": lower,
            "cpi_ml_ci_high": upper,
        },
        index=df_valid.index,
    )


__all__ = [
    "compare_models",
    "compute_cpi_bootstrap",
    "ndcg_at_k",
    "precision_at_k",
    "train_and_explain",
]
