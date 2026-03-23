# Terra-IA

Terra-IA is an AI/data project for **morphological buildability pre-screening** using **IGN LiDAR HD**, cadastral parcels, and transaction-based weak supervision.

This repository contains the **V3 reference version** of the project used for the Capstone:
- deterministic baseline scoring (**CPI**),
- machine learning ranking/classification workflow,
- spatial cross-validation,
- SHAP explainability,
- demo-ready outputs.

## Project status
This repository is being cleaned and structured from the original V3 pipeline into a reproducible Capstone-compatible project.

## Scope of this repository
This repo keeps only the materials relevant to the **V3 version**:
- source code,
- documentation,
- lightweight outputs,
- report material,
- demo app.

Old exploratory files and pre-V3 materials are intentionally excluded or archived separately.

## Planned structure
- `src/terra_ia/` → project source code
- `app/` → Streamlit demo
- `scripts/` → execution scripts
- `data/` → raw / interim / processed data
- `outputs/` → figures, metrics, SHAP, samples
- `reports/` → Capstone report and internal report
- `docs/` → project documentation and Capstone checklist

## Current entry point
At this stage, the original V3 pipeline is temporarily kept as a single file before modular refactoring.

## Setup
A first `requirements.txt` is provided. It will be refined after the codebase is split into modules and the environment is stabilized.

## Capstone objectives
The repository is being aligned with the Capstone requirements:
- clean and commented source code,
- GitHub repository,
- reproducible workflow,
- baseline + ML comparison,
- explainability,
- error analysis,
- demo,
- final report.
