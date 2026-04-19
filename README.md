# Terra-IA

Terra-IA is a V6 capstone project focused on parcel-level constructibility scoring for Chambery. It combines open geospatial data, feature engineering on LiDAR rasters, weak supervision, ranking models, and a Streamlit demo designed for a live defense.

This README describes the current V6 handover state. A few historical names are intentionally preserved for compatibility with committed outputs and previous notebooks: `CPI_v3` is the deterministic baseline score, `CPI_ML_v3` is the ML score column used by the V6 exports, and `data/lidar_chamberey` remains the default local data path.

The repository is now organized around stable entrypoints so it is easier to install, run, explain, and hand over on a third-party machine: 

- `pipeline.py`: clean CLI entrypoint for the end-to-end data and modeling pipeline.
- `demo.py`: clean CLI entrypoint for the Streamlit demo.
- `preflight.py`: quick readiness check before a checkpoint or live defense.
- `terra_ia_pipeline_v6.py`: main scientific pipeline implementation.
- `terra_ia_demo_v6.py`: main Streamlit application.
- `src/terra_ia/`: reusable package for runtime configuration, CLI logic, and preflight checks.
- `environment.yml`: reproducible environment for the geospatial and ML stack.

## Repository layout

The repo now follows a lightweight but clearer split between entrypoints, reusable infrastructure, and domain scripts:

- `pipeline.py`, `demo.py`, `preflight.py`: thin wrappers used by teammates and the jury.
- `src/terra_ia/project.py`: shared project root and path helpers.
- `src/terra_ia/catalog.py`: shared domain constants, feature groups, and IGN tile URL catalog.
- `src/terra_ia/downloads.py`: reusable download helpers for open cadastral and DVF sources.
- `src/terra_ia/consensus.py`: reusable consensus-scoring logic across multiple parcel ranking approaches.
- `src/terra_ia/exports.py`: export helpers for generated ML documentation.
- `src/terra_ia/hazards.py`: reusable BRGM/geohazard loading and parcel overlay helpers.
- `src/terra_ia/labeling.py`: reusable weak-labeling, DVF prep, clustering validation, and spatial block helpers.
- `src/terra_ia/ml.py`: reusable model comparison, LambdaMART training, SHAP export, and bootstrap uncertainty logic.
- `src/terra_ia/raster_features.py`: reusable raster derivation and zonal-stat helpers for terrain feature engineering.
- `src/terra_ia/reporting.py`: reusable dataset export and JSON/report generation helpers.
- `src/terra_ia/scoring.py`: reusable parcel filtering, parcel geometry metrics, CES, and CPI scoring helpers.
- `src/terra_ia/urban_data.py`: reusable urban-planning data access for PLU and BD TOPO building layers.
- `src/terra_ia/spatial_data.py`: reusable geospatial IO for tile merging, raster validation, parcel loading, and PLU joins.
- `src/terra_ia/pipeline_runtime.py`: pipeline paths, output naming, and flag resolution.
- `src/terra_ia/pipeline_resilience.py`: checkpoint management, live log teeing, and OSM cache helpers.
- `src/terra_ia/demo_runtime.py`: demo asset resolution and setup guidance.
- `src/terra_ia/pipeline_cli.py`, `demo_cli.py`, `preflight_cli.py`: CLI implementations.
- `terra_ia_pipeline_v6.py`, `terra_ia_demo_v6.py`: scientific and UI implementations kept intact but now wired to the shared runtime package.

## Why this project fits the capstone rubric

The capstone document asks for a full AI lifecycle, not only a model. Terra-IA covers:

- Problem definition: identify parcels with strong densification potential under terrain constraints.
- Data collection: IGN LiDAR HD, cadastral parcels, DVF, PLU, BRGM, BD TOPO, plus optional auxiliary sources.
- Cleaning and preprocessing: downloads, joins, reprojections, raster derivatives, parcel-level zonal statistics, filtering, and weak labels.
- Modeling and evaluation: deterministic CPI baseline, Snorkel-style proxy labels, spatial cross-validation, model comparison, SHAP, clustering validation, and bootstrap uncertainty.
- Deployment/demo: Streamlit application for live exploration of parcel scores and explanations.

## Installation

The recommended setup is Conda because the project uses several geospatial libraries.

```bash
conda env create -f environment.yml
conda activate terra-ia-capstone
python preflight.py
```

## How to run

Run the full pipeline:

```bash
python pipeline.py
```

Reuse existing downloads or intermediate outputs:

```bash
python pipeline.py --skip-download
python pipeline.py --skip-download --skip-features
python pipeline.py --skip-download --skip-zonal
```

Resume an interrupted run and keep a live log on disk:

```bash
python pipeline.py --resume
python pipeline.py --resume --skip-download --skip-features --skip-bootstrap
python pipeline.py --log-file logs/pipeline_defense.log
```

Refresh the cached OSM roads layer if needed:

```bash
python pipeline.py --refresh-osm
```

Write outputs to a dedicated folder:

```bash
python pipeline.py --output-dir outputs
```

Launch the Streamlit demo:

```bash
python demo.py
```

Run the demo on another port or without opening a browser:

```bash
python demo.py --port 8502 --no-browser
```

Point the demo to a custom output folder:

```bash
python demo.py --output-dir outputs
```

## Data policy

Raw and intermediate geospatial assets are not committed because they are heavy and environment-specific. This includes downloaded LiDAR tiles, merged rasters, runtime checkpoints, logs, local caches, and virtual environments.

By default, the pipeline reads and writes local raw/intermediate assets under `data/lidar_chamberey`. The spelling is legacy compatibility, not a new project scope; use `python pipeline.py --data-dir <path>` to point the pipeline at another local data folder.

The final V6 deliverables are committed so the defense demo can open immediately after cloning the repository:

- `features_parcelles_v6.csv`
- `ml_dataset_v6.csv`
- `shap_par_parcelle_v6.csv`
- `rapport_stats_v6.json`
- `cluster_scores_v6.csv`
- `README_ML_dataset_v6.md`

The demo now fails gracefully: if these assets are missing, it opens a setup page instead of crashing during a live presentation.

Internal runtime assets generated by the pipeline:

- `checkpoints/pipeline_state.json`
- `checkpoints/stage3_features.parquet`
- `checkpoints/stage6_labels.parquet`
- `checkpoints/stage9_scores.parquet`
- `logs/pipeline_<timestamp>.log`

These runtime assets are intentionally ignored by Git and can be regenerated with `python pipeline.py --resume`.

## Demo notes

The Streamlit app can optionally use the OpenAI API for parcel-level discussion support. To enable it, create `.streamlit/secrets.toml` from the provided example file and add your key there.

## Suggested defense flow

1. Run `python preflight.py`.
2. Show that the repo has a stable `README`, environment file, and clean entrypoints.
3. Run `python pipeline.py --skip-download` if the data already exists locally.
4. Launch `python demo.py`.
5. Use the app to justify both business value and technical choices.

## Current limitations

- The project is optimized for Chambery and not yet packaged as a multi-city product.
- Some optional features depend on libraries that may be absent from a lightweight environment.
- In the final local run, `richdem` and `rvt-py` were unavailable, so TWI/thalweg/SVF features are reported as unavailable in `rapport_stats_v6.json`.
- The local PLU layer downloaded during the final run was empty, so `zone_plu` is `inconnu`; BRGM local layers were also absent.
- Historical score column names such as `CPI_v3` and `CPI_ML_v3` are kept so existing V6 CSVs, SHAP exports, and demo code remain compatible.

## License and caution

Terra-IA is an academic decision-support prototype for pre-screening. It does not replace geotechnical, legal, or urban planning expertise, and it should not be used alone for investment decisions.
