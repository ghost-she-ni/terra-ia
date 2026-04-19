from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence


def build_ml_dataset_readme(
    *,
    output_csv_ml_v6: Path,
    feat_cols: Sequence[str],
    feature_groups: Mapping[str, Sequence[str]],
    grid_size_m: int,
    shap_filename: str = "shap_par_parcelle_v6.csv",
    generated_from: str = "python pipeline.py",
) -> str:
    feature_lines = "\n".join(
        f"  {feature:35s} [{next((group for group, items in feature_groups.items() if feature in items), '?')}]"
        for feature in feat_cols
    )

    return f"""# Terra-IA - Dataset ML V6 - README pour le coequipier ML

## Version : v6.0
## Fichier : {output_csv_ml_v6.name}
Genere par `{generated_from}` - Avril 2026

## Features disponibles ({len(feat_cols)})
```
{feature_lines}
```

## Colonne cible : proxy_label
  1  = constructible probable  (H1 LiDAR + H3 DVF si disponible)
  0  = contraignant             (H2 LiDAR + H4 DVF si disponible)
  -1 = ABSTAIN (exclu du dataset - non present dans ce fichier)

## Colonne CPI_v3
  Score deterministe V3 [0-100] - baseline de comparaison

## Colonne block_id
  Identifiant bloc spatial (grille {grid_size_m}m) - OBLIGATOIRE pour GroupKFold

## Modele recommande - XGBoost LambdaMART
```python
import xgboost as xgb
import numpy as np

# Trier par bloc pour LambdaMART
df_sorted = df.sort_values('block_id')
X = df_sorted[FEAT_COLS].values
y = df_sorted['proxy_label'].values

_, qid_inv = np.unique(df_sorted['block_id'], return_inverse=True)
_, counts  = np.unique(qid_inv, return_counts=True)

model = xgb.XGBRanker(
    objective='rank:ndcg',
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X, y, group=counts)
```

## Validation croisee - GroupKFold spatial OBLIGATOIRE
```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
for tr_idx, val_idx in gkf.split(X, y, groups=df['block_id']):
    blocks_tr = df['block_id'].values[tr_idx]
    _, qid_tr = np.unique(blocks_tr, return_inverse=True)
    sort_idx  = np.argsort(qid_tr)
    _, cnts   = np.unique(qid_tr[sort_idx], return_counts=True)
    model.fit(X[tr_idx][sort_idx], y[tr_idx][sort_idx], group=cnts)
    scores = model.predict(X[val_idx])
```

## Metriques cibles (par ordre d'importance)
  1. NDCG@20       - qualite du ranking (principale)
  2. Precision@20  - top-k pertinence metier
  3. AUC-ROC       - discrimination generale
  4. Spearman vs CPI_v3 - coherence avec score deterministe

## SHAP - explicabilite
```python
import shap
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
```

## Points de vigilance (issus analyse V2/V3)
  - NE PAS utiliser surface_m2 comme feature ML (redondance avec labels)
  - Verifier distribution classes par bloc avant CV (blocs monoclasses)
  - XGBRanker necessite le tri par block_id avant fit()
  - NDCG = 1.000 pour RF suggere une redondance feature/label - normal si DVF absent

## Fichier SHAP par parcelle : {shap_filename}
Colonnes : id_parcelle, shap_slope_p50, shap_slope_p90, ...,
           shap_groupe_SLOPE, shap_groupe_HYDROLOGY,
           shap_groupe_MORPHOLOGY, shap_groupe_SUNLIGHT,
           CPI_ML_v6, CPI_v6, CPI_technique

## References
  LambdaMART   : Chen & Guestrin (2016) XGBoost - KDD 2016
  SHAP         : Lundberg & Lee (2017) NeurIPS 30
  Spatial CV   : Valavi et al. (2019) Methods Ecol Evol 10(2):225-232
  Snorkel      : Ratner et al. (2017) VLDB Journal 26:793-817
  Burges       : Burges et al. (2006) Learning to Rank - ICML
"""
