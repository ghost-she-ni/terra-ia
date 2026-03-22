# Terra-IA

**AI-assisted morphological buildability scoring from IGN LiDAR HD data**

Terra-IA is a Capstone project that scores the **morphological buildability potential** of urban parcels using high-resolution LiDAR data from IGN. The goal is to support **land pre-screening** for urban densification by combining a deterministic baseline (**CPI**) with an explainable machine learning model (**XGBoost + SHAP**).

> Terra-IA does **not** predict legal buildability. It estimates **physical and morphological constraints** that may affect real-world feasibility.

---

## Official project scope

The official version of the project presented for the Capstone is:

- **Pilot area:** Chambéry (INSEE 73065)
- **Current operational footprint:** partial LiDAR coverage focused on the city center
- **Pipeline:** data ingestion → parcel features → deterministic CPI baseline → ML ranking model → SHAP explainability
- **Positioning:** decision-support tool for **pre-filtering**, not final legal validation

### Official V3 figures
- 15,480 cadastral parcels
- 2,536 valid parcels
- 2,315 labeled parcels
- 10 LiDAR-based ML features

These figures are the reference values for the current Capstone version.

---

## Problem addressed

In the context of urban densification and the French ZAN framework, identifying promising parcels is still often a manual, slow, expensive and poorly reproducible process. Existing tools focus mainly on cadastral and regulatory dimensions, while **physical terrain constraints** remain underused in early-stage site screening.

Terra-IA addresses this gap by extracting parcel-level morphological signals from LiDAR data:
- slope and terrain variability
- hydrological proxies
- enclosure / sky openness
- solar exposure proxies
- surrounding height signals

---

## Business value

Terra-IA is designed for:
- real estate developers
- local authorities
- land investment teams
- planning / engineering consultancies

Main value delivered:
- faster first-pass parcel screening
- reproducible scoring logic
- complementary physical reading of the terrain
- explainable prioritization through SHAP

---

## Repository structure

```text
terra-ia/
├── README.md
├── PROJECT_SCOPE.md
├── src/
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
├── reports/
│   └── final_sources/
├── docs/
│   └── archive/
└── notebooks/
```

---

## Data sources

- IGN LiDAR HD (MNT / MNH, 50 cm)
- French cadastral parcel data
- DVF 2023 transaction data

---

## Current status

The project already includes:
- a full V3 pipeline execution trace
- parcel-level feature exports
- a deterministic CPI score logic
- a trained XGBoost LambdaMART ranking model
- SHAP importance outputs
- a V3 validation report

Day 1 objective is to **freeze the official scope** and align all future deliverables with the same version of the project.

---

## Out of scope for the official presentation

The following items are **not claimed as delivered results** unless they are regenerated and validated before final submission:
- full-city LiDAR coverage with 24 tiles
- 70–80% valid parcel coverage
- fully operational road-access feature through OSMnx
- PLU/legal buildability integration
- national-scale deployment
- direct point-cloud processing

---

## Next steps

1. Clean and modularize the codebase
2. Fix dataset export inconsistencies
3. Align hyperparameters and final training logic
4. Prepare the technical report
5. Build a reliable demo

---

## Capstone deliverables reminder

The Capstone expects:
- clean and commented source code on GitHub
- a technical report or blog article
- an oral presentation with a live demo

This repository is being structured to match those requirements.
