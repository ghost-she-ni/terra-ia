# Capstone Checklist - Terra-IA V6

Use this checklist for the current V6 defense state. Historical names such as `CPI_v3`, `CPI_ML_v3`, and `data/lidar_chamberey` are retained only for compatibility with existing V6 deliverables.

## 1. Problem definition
- [ ] Problem is clearly stated
- [ ] Business / practical value is explicit
- [ ] Scope is realistic and well bounded
- [ ] Success criteria are defined

## 2. Data
- [ ] Data sources are documented
- [ ] Data collection process is explained
- [ ] Data cleaning steps are explained
- [ ] Legacy local path `data/lidar_chamberey` is explained, with `--data-dir` documented for custom runs
- [ ] Limitations and data quality issues are documented
- [ ] Licensing / legal considerations are addressed

## 3. Baseline
- [ ] Deterministic baseline is implemented
- [ ] `CPI_v3` is explained as the historical deterministic baseline column retained in V6 outputs
- [ ] Baseline logic is explained
- [ ] Baseline metrics are reported

## 4. Machine learning
- [ ] ML dataset is documented
- [ ] `CPI_ML_v3` is explained as the historical ML score column retained for V6 compatibility
- [ ] Train/validation strategy is justified
- [ ] Spatial cross-validation is documented
- [ ] Model selection is explained
- [ ] Hyperparameters are traceable
- [ ] Final metrics are reported

## 5. Explainability and analysis
- [ ] SHAP or equivalent explainability is included
- [ ] Error analysis is included
- [ ] Failure cases are discussed
- [ ] Bias / weak supervision limitations are discussed

## 6. Engineering quality
- [ ] GitHub repository exists
- [ ] Code is clean and commented
- [ ] Project structure is modular
- [ ] Runtime paths and flags have one source of truth in `src/terra_ia/*_runtime.py`
- [ ] `requirements.txt` or `environment.yml` is provided
- [ ] README explains setup and execution
- [ ] Outputs are organized

## 7. Demo / deployment
- [ ] Demo exists
- [ ] Demo scenario is defined
- [ ] Inputs and outputs are understandable
- [ ] The app supports the project story clearly

## 8. Reporting and presentation
- [ ] Final report is structured
- [ ] Figures are readable and relevant
- [ ] Slides are ready
- [ ] Oral defense message is clear
