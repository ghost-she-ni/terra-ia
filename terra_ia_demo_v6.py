import os
import random
import sys
from pathlib import Path
from typing import Dict, List

SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from terra_ia import catalog as shared_catalog
from terra_ia.demo_runtime import (
    build_demo_runtime_config,
    missing_demo_assets as shared_missing_demo_assets,
    recommended_setup_commands,
)

import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

st.set_page_config(
    page_title="Terra-IA — Scoring Constructibilité Chambéry",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

THEME_TOKENS: Dict[str, Dict[str, str]] = {
    "dark": {
        "bg": "#080F1E",
        "bg_grad_1": "rgba(13,148,136,0.09)",
        "bg_grad_2": "rgba(26,86,219,0.12)",
        "sidebar": "#0D1A2E",
        "card": "#0F1C30",
        "card_soft": "rgba(15,28,48,0.72)",
        "card_alt": "#13243C",
        "border": "#24344C",
        "text": "#F1F5F9",
        "text_soft": "#94A3B8",
        "accent": "#0D9488",
        "accent_2": "#1A56DB",
        "shadow": "0 22px 36px rgba(8, 15, 30, 0.45)",
    },
    "white": {
        "bg": "#F4F7FC",
        "bg_grad_1": "rgba(13,148,136,0.08)",
        "bg_grad_2": "rgba(26,86,219,0.08)",
        "sidebar": "#FFFFFF",
        "card": "#FFFFFF",
        "card_soft": "rgba(255,255,255,0.82)",
        "card_alt": "#EEF4FF",
        "border": "#DCE5F2",
        "text": "#0F172A",
        "text_soft": "#64748B",
        "accent": "#0D9488",
        "accent_2": "#1A56DB",
        "shadow": "0 14px 34px rgba(15, 23, 42, 0.10)",
    },
}


def _theme_mode() -> str:
    return st.session_state.get("theme_mode", "dark")


def theme_tokens() -> Dict[str, str]:
    return THEME_TOKENS["white"] if _theme_mode() == "white" else THEME_TOKENS["dark"]


def plotly_layout_defaults() -> Dict[str, object]:
    colors = theme_tokens()
    return {
        "template": "plotly_white" if _theme_mode() == "white" else "plotly_dark",
        "paper_bgcolor": colors["card"],
        "plot_bgcolor": colors["card"],
        "font": {"color": colors["text"]},
        "colorway": [colors["accent_2"], colors["accent"], "#16A34A", "#D97706", "#DC2626"],
        "legend": {"bgcolor": "rgba(0,0,0,0)", "borderwidth": 0, "font": {"color": colors["text_soft"]}},
    }


def apply_theme_css(mode: str):
    colors = THEME_TOKENS["white"] if mode == "white" else THEME_TOKENS["dark"]
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
    :root {{
        --terra-bg: {colors["bg"]};
        --terra-bg-grad-1: {colors["bg_grad_1"]};
        --terra-bg-grad-2: {colors["bg_grad_2"]};
        --terra-sidebar: {colors["sidebar"]};
        --terra-card: {colors["card"]};
        --terra-card-soft: {colors["card_soft"]};
        --terra-card-alt: {colors["card_alt"]};
        --terra-border: {colors["border"]};
        --terra-text: {colors["text"]};
        --terra-text-soft: {colors["text_soft"]};
        --terra-accent: {colors["accent"]};
        --terra-accent-2: {colors["accent_2"]};
        --terra-shadow: {colors["shadow"]};
    }}
    * {{ font-family: 'DM Sans', sans-serif; }}
    html {{ scroll-behavior: smooth; }}
    body, .stApp {{
        color: var(--terra-text);
        background: radial-gradient(circle at 12% 0%, var(--terra-bg-grad-1), transparent 30%),
                    radial-gradient(circle at 88% 0%, var(--terra-bg-grad-2), transparent 28%),
                    var(--terra-bg);
    }}
    h1, h2, h3 {{ letter-spacing: -0.02em; }}
    .terra-page-header {{
        border: 1px solid var(--terra-border);
        background: linear-gradient(150deg, var(--terra-card-soft), color-mix(in srgb, var(--terra-card-alt) 70%, transparent 30%));
        border-radius: 18px;
        padding: 14px 16px 12px 16px;
        margin-bottom: 12px;
        box-shadow: var(--terra-shadow);
        animation: terraFadeUp 340ms ease-out;
    }}
    .terra-page-header h1 {{
        margin: 0;
        font-size: clamp(30px, 2.6vw, 42px);
        line-height: 1.15;
        font-weight: 800;
        color: var(--terra-text);
    }}
    .terra-page-header p {{
        margin: 6px 0 0 0;
        font-size: 14px;
        color: var(--terra-text-soft);
    }}
    @keyframes terraFadeUp {{
      from {{ opacity: 0; transform: translateY(8px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    .block-container {{ padding-top: 1rem; }}
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, var(--terra-sidebar), color-mix(in srgb, var(--terra-sidebar) 86%, var(--terra-bg) 14%));
        border-right: 1px solid var(--terra-border);
    }}
    section[data-testid="stSidebar"] * {{ color: var(--terra-text) !important; }}
    .terra-logo {{
        font-size: 22px;
        font-weight: 800;
        letter-spacing: 0.3px;
        color: var(--terra-accent);
    }}
    .terra-subtitle {{
        color: var(--terra-text-soft);
        font-size: 14px;
        margin-top: 2px;
        margin-bottom: 4px;
    }}
    .terra-badge {{
        display: inline-block;
        border-radius: 999px;
        padding: 6px 10px;
        font-size: 12px;
        font-weight: 700;
        color: var(--terra-text);
        background: linear-gradient(120deg, color-mix(in srgb, var(--terra-accent) 34%, transparent 66%), color-mix(in srgb, var(--terra-accent-2) 28%, transparent 72%));
        border: 1px solid color-mix(in srgb, var(--terra-accent) 34%, var(--terra-border) 66%);
    }}
    .stMetric {{
        background: var(--terra-card-soft);
        border: 1px solid var(--terra-border);
        border-radius: 16px;
        padding: 12px;
        box-shadow: var(--terra-shadow);
        backdrop-filter: blur(6px);
        transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
    }}
    .stMetric:hover {{
        transform: translateY(-1px);
        border-color: color-mix(in srgb, var(--terra-accent) 32%, var(--terra-border) 68%);
    }}
    div[data-testid="stMetricValue"] {{
        font-weight: 800;
        letter-spacing: -0.01em;
    }}
    div[data-testid="stMetricLabel"] {{
        color: var(--terra-text-soft);
    }}
    .stTabs [data-baseweb="tab-list"] {{
        border-bottom: 1px solid color-mix(in srgb, var(--terra-accent) 50%, var(--terra-border) 50%);
    }}
    .stTabs [data-baseweb="tab"] {{
        color: var(--terra-text-soft);
        font-weight: 500;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        color: var(--terra-accent);
        border-bottom: 2px solid var(--terra-accent);
    }}
    [data-baseweb="radio"] label {{
        border-radius: 10px;
        padding: 6px 10px;
        border: 1px solid transparent;
    }}
    [data-baseweb="radio"] label:hover {{
        border-color: color-mix(in srgb, var(--terra-accent) 40%, transparent 60%);
        background: color-mix(in srgb, var(--terra-accent) 14%, transparent 86%);
    }}
    [data-baseweb="radio"] label:has(input:checked) {{
        border-color: color-mix(in srgb, var(--terra-accent) 45%, var(--terra-border) 55%);
        background: linear-gradient(145deg, color-mix(in srgb, var(--terra-accent) 26%, transparent 74%), color-mix(in srgb, var(--terra-accent-2) 16%, transparent 84%));
        box-shadow: 0 8px 16px rgba(13,148,136,0.18);
    }}
    div[data-baseweb="select"] > div,
    div[data-baseweb="base-input"] > div {{
        background: var(--terra-card-soft);
        border-color: var(--terra-border);
        border-radius: 12px;
    }}
    .stMultiSelect [data-baseweb="tag"] {{
        border-radius: 999px;
        border: 1px solid color-mix(in srgb, var(--terra-accent) 35%, var(--terra-border) 65%);
        background: color-mix(in srgb, var(--terra-accent) 18%, transparent 82%);
    }}
    div[data-baseweb="slider"] [role="slider"] {{
        border: 2px solid color-mix(in srgb, var(--terra-accent) 82%, white 18%);
        box-shadow: 0 0 0 6px color-mix(in srgb, var(--terra-accent) 22%, transparent 78%);
    }}
    .stButton > button, .stDownloadButton > button {{
        border-radius: 12px;
        border: 1px solid color-mix(in srgb, var(--terra-accent) 46%, var(--terra-border) 54%);
        background: linear-gradient(150deg, color-mix(in srgb, var(--terra-accent) 30%, var(--terra-card) 70%), color-mix(in srgb, var(--terra-accent-2) 16%, var(--terra-card) 84%));
        color: var(--terra-text);
        font-weight: 700;
        transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease;
        box-shadow: 0 12px 20px rgba(15, 23, 42, 0.14);
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        transform: translateY(-1px);
        filter: saturate(1.05);
    }}
    .stButton > button:focus, .stDownloadButton > button:focus {{
        outline: none;
        box-shadow: 0 0 0 3px color-mix(in srgb, var(--terra-accent) 25%, transparent 75%);
    }}
    .stDataFrame, div[data-testid="stTable"] {{
        border-radius: 14px;
        border: 1px solid var(--terra-border);
        overflow: hidden;
        box-shadow: var(--terra-shadow);
    }}
    div[data-testid="stInfo"], div[data-testid="stSuccess"], div[data-testid="stWarning"], div[data-testid="stError"] {{
        border-radius: 12px;
        border-width: 1px;
    }}
    div[data-testid="stChatInput"] {{
        border: 1px solid var(--terra-border);
        border-radius: 14px;
        background: var(--terra-card-soft);
        box-shadow: var(--terra-shadow);
        padding: 2px;
    }}
    .terra-ai-chip {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border-radius: 999px;
        border: 1px solid color-mix(in srgb, var(--terra-accent) 40%, var(--terra-border) 60%);
        padding: 5px 10px;
        margin-bottom: 8px;
        font-size: 12px;
        font-weight: 700;
        color: var(--terra-text);
        background: color-mix(in srgb, var(--terra-accent) 20%, transparent 80%);
    }}
    .terra-map-legend {{
        position: absolute;
        left: 12px;
        bottom: 12px;
        z-index: 9999;
        background: var(--terra-card-soft);
        border: 1px solid var(--terra-border);
        border-radius: 12px;
        padding: 10px 12px;
        font-size: 12px;
        color: var(--terra-text);
        box-shadow: var(--terra-shadow);
        backdrop-filter: blur(6px);
    }}
    .terra-map-legend .dot {{
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 4px;
        vertical-align: middle;
    }}
    .stChatMessage {{
        border: 1px solid var(--terra-border);
        border-radius: 12px;
        background: var(--terra-card-soft);
    }}
    footer, #MainMenu {{ visibility: hidden; }}
    ::-webkit-scrollbar {{ width: 7px; height: 7px; }}
    ::-webkit-scrollbar-track {{ background: var(--terra-sidebar); }}
    ::-webkit-scrollbar-thumb {{ background: var(--terra-accent-2); border-radius: 8px; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

FEATURE_GROUPS = shared_catalog.FEATURE_GROUPS
FEAT_COLS = shared_catalog.ALL_FEATURES
GUIDED_STEPS = [
    ("🗺️ Carte & Scoring", "Étape 1/4 — Filtrer", "Définissez votre stratégie et construisez une shortlist."),
    ("🔍 Analyse parcelle", "Étape 2/4 — Analyser", "Lisez le verdict métier d'une parcelle candidate."),
    ("⚖️ Comparer 2 parcelles", "Étape 3/4 — Comparer", "Arbitrez entre 2 options avec les mêmes critères."),
    ("📊 Science & Méthode", "Étape 4/4 — Justifier", "Conservez les éléments de preuve méthodologique."),
]
BUSINESS_PROFILES = {
    "Équilibré": {"filter_min_cpi": 30, "filter_min_conf": 0.5, "filter_max_slope": 30, "filter_min_surface": 100},
    "Densification rapide": {"filter_min_cpi": 45, "filter_min_conf": 0.45, "filter_max_slope": 25, "filter_min_surface": 120},
    "Prudent (risque faible)": {"filter_min_cpi": 55, "filter_min_conf": 0.7, "filter_max_slope": 18, "filter_min_surface": 180},
    "Premium (fort potentiel)": {"filter_min_cpi": 70, "filter_min_conf": 0.75, "filter_max_slope": 15, "filter_min_surface": 250},
}
GLOSSARY = {
    "CPI_technique": "Score global [0-100] de constructibilité morphologique. Plus il est élevé, plus la parcelle est favorable.",
    "consensus_confidence": "Accord entre plusieurs approches (déterministe, clustering, marché). >0.80 = alignement fort.",
    "slope_p90": "Pente maximale sur les zones critiques de la parcelle. Au-delà de 25°, le terrain devient contraint.",
    "cpi_ml_std": "Variabilité du score ML (bootstrap). Bas = score stable, haut = incertitude plus forte.",
    "has_thalweg_mean": "Présence d'un axe de drainage naturel (eau). Plus c'est élevé, plus il faut vérifier l'hydrologie.",
    "surface_m2": "Surface cadastrale exploitable. La capacité de projet dépend aussi du PLU et des retraits.",
est ce     "svf_mean": "Sky View Factor : ouverture au ciel. Valeur élevée = meilleur ensoleillement potentiel.",
}

DEMO_RUNTIME = build_demo_runtime_config()
PROJECT_ROOT = DEMO_RUNTIME.project_root
OUTPUT_DIR = DEMO_RUNTIME.output_dir
DATA_FEATURES_PATH = DEMO_RUNTIME.features_path
DATA_SHAP_PATH = DEMO_RUNTIME.shap_path
DATA_GEO_PATH = DEMO_RUNTIME.geo_path


def missing_demo_assets() -> List[tuple[str, Path, str]]:
    return shared_missing_demo_assets(DEMO_RUNTIME)


def render_setup_page(missing_assets: List[tuple[str, Path, str]]):
    render_page_header(
        "Live demo setup required",
        "The interface is ready, but the generated capstone datasets are still missing.",
    )
    st.warning(
        "The app now explains what is missing instead of crashing during a checkpoint or live defense."
    )
    rows = [
        {"Asset": label, "Expected path": str(path), "Why it matters": reason}
        for label, path, reason in missing_assets
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown("**Recommended commands**")
    st.code(
        recommended_setup_commands(),
        language="bash",
    )
    st.caption(
        "Optional file: `shap_par_parcelle_v6.csv`. The demo can start without it, but parcel-level explainability will stay hidden."
    )

def first_existing(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    df[candidates[0]] = np.nan
    return candidates[0]

def ensure_column(df: pd.DataFrame, col: str):
    if col not in df.columns:
        df[col] = np.nan

def score_to_color(score: float) -> str:
    if pd.isna(score):
        return "#475569"
    if score < 20:
        return "#DC2626"
    if score < 40:
        return "#F97316"
    if score < 60:
        return "#EAB308"
    if score < 75:
        return "#86EFAC"
    return "#16A34A"

def score_to_label(score: float) -> str:
    if pd.isna(score):
        return "Non calculé"
    if score < 20:
        return "Éliminatoire"
    if score < 38:
        return "Contraint"
    if score < 62:
        return "Faisable"
    if score < 80:
        return "Favorable"
    return "Optimal"

def confidence_to_emoji(conf: float) -> str:
    if pd.isna(conf):
        return "⬜"
    if conf >= 0.85:
        return "🟢"
    if conf >= 0.65:
        return "🟡"
    return "🔴"

def feature_label_fr(name: str) -> str:
    labels = {
        "slope_p50": "Pente médiane",
        "slope_p90": "Pente maximale (zone critique)",
        "slope_std": "Régularité du terrain",
        "twi_mean": "Accumulation d'eau (TWI)",
        "has_thalweg_mean": "Axe de drainage (thalweg)",
        "tri_mean": "Rugosité du terrain (TRI)",
        "profile_curvature_mean": "Courbure du profil",
        "hillshade_winter_mean": "Ensoleillement hivernal",
        "aspect_south_ratio_mean": "Orientation plein sud",
        "svf_mean": "Ciel visible (SVF)",
    }
    return labels.get(name, name)

def feature_unit(name: str) -> str:
    units = {
        "slope_p50": "°",
        "slope_p90": "°",
        "slope_std": "°",
        "twi_mean": "indice",
        "has_thalweg_mean": "ratio [0-1]",
        "tri_mean": "m",
        "profile_curvature_mean": "1/m²",
        "hillshade_winter_mean": "indice [0-1]",
        "aspect_south_ratio_mean": "ratio [0-1]",
        "svf_mean": "indice [0-1]",
    }
    return units.get(name, "")

def percentile(val: float, series: pd.Series) -> float:
    if pd.isna(val) or series.empty:
        return np.nan
    return (series < val).mean()

def warn_missing(df: pd.DataFrame, col: str):
    if col not in df.columns:
        st.caption(f"⚠️ Colonne manquante : {col} (NaN)")
        df[col] = np.nan


def ensure_parcel_id_column(df: pd.DataFrame, gdf=None, col_name: str = "_parcel_id") -> pd.DataFrame:
    if col_name in df.columns and df[col_name].notna().any():
        return df
    if "id_parcelle" in df.columns:
        df[col_name] = df["id_parcelle"].astype(str)
        return df
    if gdf is not None and "id" in gdf.columns:
        aligned = gdf["id"].reindex(df.index)
        if aligned.notna().any():
            df[col_name] = aligned.astype(str)
            return df
    df[col_name] = df.index.astype(str)
    return df


def normalize_parcel_id(parcel_id: str) -> str:
    return str(parcel_id or "").strip().upper().replace(" ", "").replace("-", "")


def trigger_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def render_glossary(terms: List[str]):
    terms = [t for t in terms if t in GLOSSARY]
    if not terms:
        return
    with st.expander("🧾 Glossaire rapide"):
        for term in terms:
            st.markdown(f"- **{term}** — {GLOSSARY[term]}")


def build_business_verdict(row: pd.Series) -> Dict[str, object]:
    cpi = row.get("CPI_technique", np.nan)
    conf = row.get("consensus_confidence", np.nan)
    slope = row.get("slope_p90", np.nan)
    thalweg = row.get("has_thalweg_mean", np.nan)
    surface = row.get("surface_m2", np.nan)
    ci_std = row.get("cpi_ml_std", np.nan)

    if pd.notna(cpi) and cpi >= 65:
        verdict = "FAVORABLE ✅"
        summary = "Ce terrain est globalement favorable à une opération standard, sous réserve des vérifications réglementaires."
    elif pd.notna(cpi) and cpi >= 40:
        verdict = "À ÉTUDIER ⚠️"
        summary = "Le potentiel existe, mais le montage doit intégrer des précautions techniques et une validation de faisabilité."
    else:
        verdict = "RISQUÉ 🔴"
        summary = "Les contraintes morphologiques sont marquées. Le projet doit être revu ou très encadré techniquement."

    strengths: List[str] = []
    risks: List[str] = []
    actions: List[str] = []

    if pd.notna(slope) and slope < 10:
        strengths.append(f"Pente max modérée ({slope:.1f}°), favorable au coût d'aménagement.")
    elif pd.notna(slope) and slope > 25:
        risks.append(f"Pente max élevée ({slope:.1f}°), risque de surcoût terrassement.")
        actions.append("Lancer un pré-chiffrage terrassement + étude géotechnique ciblée.")

    if pd.notna(thalweg) and thalweg < 0.10:
        strengths.append("Signal de drainage central faible (thalweg limité).")
    elif pd.notna(thalweg) and thalweg >= 0.30:
        risks.append("Drainage naturel marqué (thalweg), vigilance hydrologique.")
        actions.append("Vérifier gestion des eaux pluviales et servitudes hydrauliques.")

    if pd.notna(surface) and surface >= 250:
        strengths.append(f"Surface confortable ({surface:.0f} m²) pour flexibilité du programme.")
    elif pd.notna(surface) and surface < 120:
        risks.append(f"Surface réduite ({surface:.0f} m²), capacité de projet limitée.")

    if pd.notna(conf) and conf >= 0.8:
        strengths.append(f"Confiance de consensus élevée ({conf:.0%}), signaux convergents.")
    elif pd.notna(conf) and conf < 0.5:
        risks.append(f"Consensus faible ({conf:.0%}), signaux contradictoires.")
        actions.append("Demander une vérification terrain avant engagement ferme.")

    if pd.notna(ci_std) and ci_std > 6:
        risks.append(f"Incertitude ML notable (écart-type {ci_std:.1f} pts).")
    elif pd.notna(ci_std) and ci_std <= 3:
        strengths.append(f"Score stable en bootstrap (±{ci_std:.1f} pts).")

    if pd.notna(row.get("brgm_mvt_terrain_flag")) and bool(row.get("brgm_mvt_terrain_flag")):
        risks.append("Signal BRGM mouvement de terrain à proximité.")
        actions.append("Exiger un avis géotechnique préliminaire (G1/G2).")
    if pd.notna(row.get("brgm_argiles_flag")) and bool(row.get("brgm_argiles_flag")):
        risks.append("Contexte argileux BRGM : aléa retrait-gonflement possible.")

    if pd.notna(row.get("gate_reason")):
        risks.append(f"Gate technique déclenchée : {row.get('gate_reason')}.")

    if not actions:
        actions = [
            "Vérifier PLU (emprise, hauteur, retraits) avec la mairie.",
            "Confirmer accès/réseaux et servitudes avant offre finale.",
            "Lancer une visite terrain rapide avec géomètre.",
        ]
    strengths = strengths[:3] if strengths else ["Aucun signal fort positif détecté avec les seuils actuels."]
    risks = risks[:3] if risks else ["Pas de risque majeur détecté avec les indicateurs disponibles."]
    actions = actions[:3]
    confidence_txt = f"{conf:.0%}" if pd.notna(conf) else "N/A"
    return {
        "verdict": verdict,
        "summary": summary,
        "strengths": strengths,
        "risks": risks,
        "actions": actions,
        "confidence_txt": confidence_txt,
    }


def build_decision_sheet(parcel_id: str, row: pd.Series, cpi_ml_col: str, verdict: Dict[str, object]) -> str:
    lines = [
        f"FICHE DECISION — Parcelle {parcel_id}",
        "Terra-IA V6 · Chambéry 73065",
        "",
        f"Verdict: {verdict['verdict']}",
        f"CPI_technique: {_fmt_num(row.get('CPI_technique'), 1)} / 100",
        f"{cpi_ml_col}: {_fmt_num(row.get(cpi_ml_col), 1)} / 100",
        f"Consensus: {_fmt_num(row.get('consensus_score'), 1)} · Confiance: {verdict['confidence_txt']}",
        f"Pente max (slope_p90): {_fmt_num(row.get('slope_p90'), 1, '°')}",
        f"Surface: {_fmt_num(row.get('surface_m2'), 0, ' m²')}",
        "",
        "Atouts:",
    ]
    lines.extend([f"- {x}" for x in verdict["strengths"]])
    lines.append("")
    lines.append("Points de vigilance:")
    lines.extend([f"- {x}" for x in verdict["risks"]])
    lines.append("")
    lines.append("Actions recommandées:")
    lines.extend([f"- {x}" for x in verdict["actions"]])
    lines.append("")
    lines.append("Disclaimer: Outil de pré-filtrage morphologique. Ne remplace pas une expertise géotechnique/réglementaire.")
    return "\n".join(lines)


def apply_business_profile(profile_name: str):
    profile = BUSINESS_PROFILES.get(profile_name, BUSINESS_PROFILES["Équilibré"])
    for k, v in profile.items():
        st.session_state[k] = v


def render_guided_stepper(current_page: str) -> str:
    if not st.session_state.get("guided_mode", False):
        return current_page
    pages = [x[0] for x in GUIDED_STEPS]
    page_to_step = {p: i for i, p in enumerate(pages)}
    if current_page in page_to_step:
        st.session_state["guided_step"] = page_to_step[current_page]
    step = int(st.session_state.get("guided_step", 0))
    step = max(0, min(step, len(GUIDED_STEPS) - 1))
    page, title, desc = GUIDED_STEPS[step]
    st.session_state["guided_step"] = step
    st.session_state["current_page"] = page
    st.markdown(
        f"<div class='card'><b>{title}</b><br><span style='color:{theme_tokens()['text_soft']}'>{desc}</span></div>",
        unsafe_allow_html=True,
    )
    st.progress((step + 1) / len(GUIDED_STEPS))
    pcol, ccol, ncol = st.columns([1, 4, 1])
    if pcol.button("← Précédent", key="guided_prev", disabled=step == 0):
        st.session_state["guided_step"] = max(0, step - 1)
        st.session_state["current_page"] = GUIDED_STEPS[st.session_state["guided_step"]][0]
        trigger_rerun()
    ccol.caption(f"Mode guidé actif · Étape {step + 1}/{len(GUIDED_STEPS)}")
    if ncol.button("Suivant →", key="guided_next", disabled=step == len(GUIDED_STEPS) - 1):
        st.session_state["guided_step"] = min(len(GUIDED_STEPS) - 1, step + 1)
        st.session_state["current_page"] = GUIDED_STEPS[st.session_state["guided_step"]][0]
        trigger_rerun()
    return page


def render_page_header(title: str, subtitle: str = ""):
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"<div class='terra-page-header'><h1>{title}</h1>{subtitle_html}</div>",
        unsafe_allow_html=True,
    )


def _fmt_num(value, decimals: int = 2, unit: str = "") -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.{decimals}f}{unit}"


def _parcel_context(selected_id: str, row: pd.Series, cpi_ml_col: str) -> str:
    argiles_flag = row.get("brgm_argiles_flag")
    mvt_flag = row.get("brgm_mvt_terrain_flag")
    return (
        f"Parcelle {selected_id} (Chambéry 73065)\n"
        f"CPI_technique: {_fmt_num(row.get('CPI_technique'), 1)} / 100 ({score_to_label(row.get('CPI_technique'))})\n"
        f"{cpi_ml_col}: {_fmt_num(row.get(cpi_ml_col), 1)} / 100\n"
        f"cluster_score: {_fmt_num(row.get('cluster_score'), 1)}\n"
        f"consensus_score: {_fmt_num(row.get('consensus_score'), 1)}\n"
        f"consensus_confidence: {_fmt_num(row.get('consensus_confidence') * 100 if pd.notna(row.get('consensus_confidence')) else np.nan, 0, '%')}\n"
        f"slope_p50: {_fmt_num(row.get('slope_p50'), 2, '°')}\n"
        f"slope_p90: {_fmt_num(row.get('slope_p90'), 2, '°')}\n"
        f"slope_std: {_fmt_num(row.get('slope_std'), 2, '°')}\n"
        f"surface_m2: {_fmt_num(row.get('surface_m2'), 0, ' m²')}\n"
        f"twi_mean: {_fmt_num(row.get('twi_mean'), 2)}\n"
        f"has_thalweg_mean: {_fmt_num(row.get('has_thalweg_mean'), 3)}\n"
        f"tri_mean: {_fmt_num(row.get('tri_mean'), 3)}\n"
        f"profile_curvature_mean: {_fmt_num(row.get('profile_curvature_mean'), 4)}\n"
        f"hillshade_winter_mean: {_fmt_num(row.get('hillshade_winter_mean'), 3)}\n"
        f"aspect_south_ratio_mean: {_fmt_num(row.get('aspect_south_ratio_mean'), 3)}\n"
        f"svf_mean: {_fmt_num(row.get('svf_mean'), 3)}\n"
        f"Argiles BRGM: {'Oui' if pd.notna(argiles_flag) and bool(argiles_flag) else 'Non'}\n"
        f"Mouvement terrain BRGM: {'Oui' if pd.notna(mvt_flag) and bool(mvt_flag) else 'Non'}"
    )


def render_ai_chatbot(selected_id: str, row: pd.Series, cpi_ml_col: str):
    st.subheader("💬 Assistant IA")
    st.caption("Assistant conversationnel basé sur les données de la parcelle sélectionnée.")
    st.markdown("<div class='terra-ai-chip'>✨ Terra-IA Copilot · contexte parcelle actif</div>", unsafe_allow_html=True)
    if not OPENAI_AVAILABLE:
        st.info("Installez `openai` pour activer l'assistant : `pip install openai`.")
        return

    api_key = ""
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        api_key = ""
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        st.info("Ajoutez `OPENAI_API_KEY` dans `.streamlit/secrets.toml` ou dans vos variables d'environnement.")
        return

    try:
        client = OpenAI(api_key=api_key)
    except Exception as exc:
        st.error(f"Impossible d'initialiser le client OpenAI : {exc}")
        return

    context = _parcel_context(selected_id, row, cpi_ml_col)
    system_prompt = (
        "Tu es Terra-IA, assistant expert en faisabilité foncière. "
        "Réponds en français, avec chiffres concrets, puis termine par une recommandation courte. "
        "Structure: Atouts / Points de vigilance / Recommandation."
    )
    summary_key = f"ai_summary_{selected_id}"
    if summary_key not in st.session_state:
        with st.spinner("Analyse IA de la parcelle..."):
            try:
                resp = client.chat.completions.create(
                    model="gpt-5.4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Analyse la parcelle suivante:\n{context}"},
                    ],
                    max_completion_tokens=500,
                    temperature=0.25,
                )
                st.session_state[summary_key] = resp.choices[0].message.content
            except Exception as exc:
                st.session_state[summary_key] = f"Erreur API : {exc}"

    st.info(st.session_state[summary_key])

    history_key = f"chat_history_{selected_id}"
    history = st.session_state.setdefault(history_key, [])
    for message in history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input(f"Posez une question sur la parcelle {selected_id}...")
    if not prompt:
        return

    history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    chat_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Contexte parcelle:\n{context}"},
    ] + history

    with st.chat_message("assistant"):
        with st.spinner("Réponse IA..."):
            try:
                resp = client.chat.completions.create(
                    model="gpt-5.4",
                    messages=chat_messages,
                    max_completion_tokens=700,
                    temperature=0.35,
                )
                answer = resp.choices[0].message.content
                st.write(answer)
                history.append({"role": "assistant", "content": answer})
            except Exception as exc:
                st.error(f"Erreur API : {exc}")

@st.cache_data
def load_features() -> pd.DataFrame:
    return pd.read_csv(DATA_FEATURES_PATH, index_col=0, low_memory=False)

@st.cache_data
def load_shap() -> pd.DataFrame:
    try:
        df = pd.read_csv(DATA_SHAP_PATH, index_col=0, low_memory=False)
        if "id_parcelle" in df.columns and df.index.name != "id_parcelle":
            df = df.set_index("id_parcelle")
        return df
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_geodata():
    import geopandas as gpd
    gdf = gpd.read_file(DATA_GEO_PATH)
    gdf = gdf.to_crs("EPSG:4326")
    return gdf

def sidebar() -> tuple[str, str, str, bool]:
    st.sidebar.markdown("<div class='terra-logo'>🏗️ TERRA-IA</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='terra-subtitle'>Scoring morphologique LiDAR HD</div>", unsafe_allow_html=True)
    theme_index = 1 if st.session_state.get("theme_mode") == "white" else 0
    theme_choice = st.sidebar.radio("Apparence", ["Dark", "White"], index=theme_index)
    theme_mode = "white" if theme_choice == "White" else "dark"
    audience_index = 0 if st.session_state.get("audience_mode", "metier") == "metier" else 1
    audience_choice = st.sidebar.radio("Mode lecture", ["Métier (recommandé)", "Technique"], index=audience_index)
    audience_mode = "metier" if audience_choice.startswith("Métier") else "technique"
    guided_mode = st.sidebar.toggle("Parcours guidé (pas-à-pas)", value=st.session_state.get("guided_mode", False))
    st.sidebar.markdown("---")
    nav_options = ["🗺️ Carte & Scoring", "🔍 Analyse parcelle", "⚖️ Comparer 2 parcelles", "📊 Science & Méthode", "ℹ️ À propos"]
    current_page = st.session_state.get("current_page", nav_options[0])
    nav_index = nav_options.index(current_page) if current_page in nav_options else 0
    page = st.sidebar.radio(
        "",
        nav_options,
        index=nav_index,
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("⚠️ Outil de pré-filtrage morphologique.\nNe remplace pas une expertise géotechnique.")
    st.sidebar.markdown(
        "<div style='margin-top:8px;'>"
        "<span class='terra-badge'>V6 · Avril 2026</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Sources : LiDAR IGN HD · DVF 2023 · BRGM 2026")
    return page, theme_mode, audience_mode, guided_mode

# -----------------------------------------------------------------------------
# PAGE 1 — CARTE & SCORING
# -----------------------------------------------------------------------------
def page_map_scoring(df: pd.DataFrame, shap_df: pd.DataFrame, gdf):
    render_page_header(
        "🗺️ Carte & Scoring",
        "Chambéry 73065 · 15 480 parcelles · LiDAR HD 50cm · Score CPI_technique [0-100]",
    )

    audience_mode = st.session_state.get("audience_mode", "metier")
    guided_mode = st.session_state.get("guided_mode", False)
    cpi_ml_col = first_existing(df, ["CPI_ML_v6", "CPI_ML_v3", "CPI_ML_v5"]) if len(df) else "CPI_ML_v6"
    for col in ["CPI_technique", "consensus_confidence", "cluster_score", "is_valid", "slope_p90", "surface_m2"]:
        warn_missing(df, col)

    if audience_mode == "metier":
        st.info("Mode métier actif : focus sur la décision. Les détails ML/SHAP restent disponibles à la demande.")
        profile = st.selectbox(
            "Profil de recherche",
            list(BUSINESS_PROFILES.keys()),
            index=list(BUSINESS_PROFILES.keys()).index(st.session_state.get("business_profile", "Équilibré"))
            if st.session_state.get("business_profile", "Équilibré") in BUSINESS_PROFILES
            else 0,
            key="business_profile",
            help="Préréglages pour gagner du temps selon votre stratégie foncière.",
        )
        if st.button("Appliquer ce profil", key="apply_business_profile"):
            apply_business_profile(profile)
            trigger_rerun()
    if guided_mode and st.session_state.get("guided_step", 0) == 0:
        st.caption("👉 Commencez par régler les filtres puis ajoutez 2 à 5 parcelles en shortlist.")

    with st.form("filters_form_v6", clear_on_submit=False):
        col1, col2, col3, col4 = st.columns(4)
        min_cpi = col1.slider("Score CPI minimum", 0, 100, key="filter_min_cpi")
        col1.caption(f"Parcelles avec CPI ≥ {min_cpi}")
        min_conf = col2.slider("Confiance consensus minimum", 0.0, 1.0, step=0.05, key="filter_min_conf")
        col2.caption("🟢 Haute confiance = 3 méthodes s'accordent")
        max_slope = col3.slider("Pente max (slope_p90)", 0, 40, step=1, key="filter_max_slope")
        col3.caption("En degrés — gate à 25°")
        min_surface = col4.slider("Surface min (m²)", 0, 1000, step=50, key="filter_min_surface")

        selected_cats = st.multiselect(
            "Catégorie CPI",
            ["Éliminatoire", "Contraint", "Faisable", "Favorable", "Optimal"],
            key="filter_categories",
        )
        only_shap = st.checkbox("Seulement les parcelles avec SHAP disponible", key="filter_only_shap")
        apply_filters = st.form_submit_button("Appliquer les filtres", use_container_width=True)
    if apply_filters:
        st.session_state["map_cache_key"] = None
    if audience_mode == "metier":
        render_glossary(["CPI_technique", "consensus_confidence", "slope_p90", "surface_m2"])

    df_f = df.copy()
    if "_parcel_id" not in df_f.columns:
        df_f = ensure_parcel_id_column(df_f, gdf=gdf)
    df_f = df_f[df_f["CPI_technique"] >= min_cpi]
    df_f = df_f[df_f["consensus_confidence"] >= min_conf]
    df_f = df_f[df_f["slope_p90"] <= max_slope]
    df_f = df_f[df_f["surface_m2"] >= min_surface]
    df_f = df_f[df_f["is_valid"] == True]
    if only_shap and not shap_df.empty:
        ids_shap = set(shap_df.index.astype(str).tolist())
        df_f = df_f[df_f["_parcel_id"].astype(str).isin(ids_shap)]
    df_f["cat"] = df_f["CPI_technique"].apply(score_to_label)
    df_f = df_f[df_f["cat"].isin(selected_cats)]

    n_filtered = len(df_f)
    cpi_mean = df_f["CPI_technique"].mean() if n_filtered else np.nan
    n_high_conf = (df_f["consensus_confidence"] >= 0.8).sum()
    n_gates = df_f.get("gate_reason", pd.Series(index=df_f.index)).notna().sum() if "gate_reason" in df_f else 0
    n_shap = len(df_f[df_f["_parcel_id"].astype(str).isin(shap_df.index.astype(str))]) if not shap_df.empty else 0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Parcelles affichées", f"{n_filtered:,}")
    m2.metric("CPI moyen", f"{cpi_mean:.1f}/100" if n_filtered else "N/A")
    m3.metric("Haute confiance", f"{n_high_conf:,}" + (f" ({n_high_conf/max(n_filtered,1)*100:.0f}%)" if n_filtered else ""))
    m4.metric("Parcelles gates", f"{n_gates:,}")
    m5.metric("SHAP disponible", f"{n_shap:,}")

    if n_filtered == 0:
        st.info("Aucune parcelle ne correspond aux filtres.")
        return

    if n_filtered > 3000:
        df_f = df_f.nlargest(3000, "CPI_technique")
        st.info(f"Affichage des {len(df_f)} meilleures parcelles (performance). Ajustez les filtres pour voir d'autres zones.")

    theme = _theme_mode()
    map_key = (theme, tuple(df_f.index.tolist()))
    cached_key = st.session_state.get("map_cache_key")
    if cached_key == map_key and st.session_state.get("map_cache_obj") is not None:
        m = st.session_state["map_cache_obj"]
    else:
        with st.spinner("Rendu cartographique..."):
            try:
                gdf_sub = gdf.loc[df_f.index]
            except Exception:
                gdf_sub = gdf.copy()
            overlap = gdf_sub.columns.intersection(df_f.columns)
            overlap = overlap.drop("geometry") if "geometry" in overlap else overlap
            if len(overlap):
                gdf_sub = gdf_sub.drop(columns=list(overlap))
            gdf_sub = gdf_sub.join(df_f, how="inner")

            map_tiles = "CartoDB positron" if theme == "white" else "CartoDB dark_matter"
            stroke_color = "#334155" if theme == "white" else "#1E293B"
            fill_opacity = 0.62 if theme == "white" else 0.72
            m = folium.Map(location=[45.5646, 5.9170], zoom_start=13, tiles=map_tiles)
            for idx, row in gdf_sub.iterrows():
                cpi = row.get("CPI_technique", np.nan)
                parcel_id = str(row.get("_parcel_id", idx))
                label = score_to_label(cpi)
                conf = row.get("consensus_confidence", np.nan)
                slope = row.get("slope_p90", np.nan)
                surface = row.get("surface_m2", np.nan)
                color = score_to_color(cpi)
                conf_str = f"{conf:.0%}" if pd.notna(conf) else "N/A"
                tooltip = folium.Tooltip(
                    f"<div style='font-family:monospace;font-size:12px;'>"
                    f"<b>ID:</b> {parcel_id}<br>"
                    f"<b>CPI:</b> {cpi:.0f}/100 — {label}<br>"
                    f"<b>Confiance:</b> {confidence_to_emoji(conf)} {conf_str}<br>"
                    f"<b>Pente max:</b> {slope:.1f}°<br>"
                    f"<b>Surface:</b> {surface:.0f} m²"
                    f"</div>",
                    sticky=True,
                )
                popup = folium.Popup(
                    f"<div style='width:200px;'>"
                    f"<h4>Parcelle {parcel_id}</h4>"
                    f"<b>CPI_technique:</b> {cpi:.0f}/100<br>"
                    f"<b>Catégorie:</b> {label}<br>"
                    f"<b>Confiance:</b> {conf_str}<br>"
                    f"</div>",
                    max_width=250,
                )
                folium.GeoJson(
                    row.geometry,
                    name=parcel_id,
                    style_function=lambda x, col=color: {
                        "fillColor": col,
                        "color": stroke_color,
                        "weight": 0.5,
                        "fillOpacity": fill_opacity,
                    },
                    tooltip=tooltip,
                    popup=popup,
                ).add_to(m)

            legend_bg = "rgba(255,255,255,0.90)" if theme == "white" else "rgba(15,28,48,0.82)"
            legend_fg = "#0F172A" if theme == "white" else "#F1F5F9"
            legend_border = "#CBD5E1" if theme == "white" else "#334155"
            legend_html = f"""
            <div style="position:absolute;left:12px;bottom:12px;z-index:9999;background:{legend_bg};
                        border:1px solid {legend_border};border-radius:10px;padding:10px 12px;font-size:12px;color:{legend_fg};">
              <div style="font-weight:700;margin-bottom:6px;">Légende CPI</div>
              <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;background:#DC2626;"></span>Éliminatoire (&lt;20)</div>
              <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;background:#F97316;"></span>Contraint (20-38)</div>
              <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;background:#EAB308;"></span>Faisable (38-62)</div>
              <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;background:#86EFAC;"></span>Favorable (62-80)</div>
              <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;background:#16A34A;"></span>Optimal (&gt;80)</div>
            </div>
            """
            m.get_root().html.add_child(folium.Element(legend_html))
            st.session_state["map_cache_key"] = map_key
            st.session_state["map_cache_obj"] = m

    st_data = st_folium(m, width=None, height=520, returned_objects=["last_object_clicked"])
    if st_data and st_data.get("last_object_clicked"):
        clicked = st_data["last_object_clicked"]
        sel_id = str(clicked.get("id") or clicked.get("name") or "")
        if sel_id:
            st.session_state["selected_parcel_id"] = sel_id
            st.success(f"✅ Parcelle {sel_id} sélectionnée — allez sur 'Analyse parcelle' pour le détail")

    st.subheader("🏆 Top 10 parcelles selon les filtres")
    df_top10 = df_f.nlargest(10, "CPI_technique").copy()
    df_top10.insert(0, "Rang", range(1, len(df_top10) + 1))
    df_top10.insert(1, "ID Parcelle", df_top10["_parcel_id"].astype(str))
    df_top10 = df_top10[["Rang", "ID Parcelle", "CPI_technique", "consensus_confidence", "slope_p90", "surface_m2"]]
    st.dataframe(df_top10.style.format({"CPI_technique": "{:.1f}", "consensus_confidence": "{:.2f}", "slope_p90": "{:.1f}", "surface_m2": "{:.0f}"}), use_container_width=True)

    st.subheader("🧺 Shortlist décision")
    shortlist_ids = st.session_state.setdefault("shortlist_ids", [])
    valid_id_set = set(df["_parcel_id"].astype(str).tolist())
    normalized_to_canonical = {normalize_parcel_id(x): x for x in valid_id_set}
    with st.container(border=True):
        st.caption("Ajoutez par copier-coller d'ID cadastral ou depuis les meilleures parcelles filtrées.")
        with st.form("shortlist_add_form", clear_on_submit=False):
            add_col1, add_col2 = st.columns([2, 2])
            manual_id = add_col1.text_input(
                "Ajouter par ID (copier-coller)",
                value="",
                placeholder="Ex: 73065000AB0101",
                help="Format cadastral complet recommandé.",
            )
            add_candidate = add_col2.selectbox(
                "Ou sélectionner rapidement",
                options=[""] + df_f["_parcel_id"].astype(str).head(150).tolist(),
                key="shortlist_add_candidate",
            )
            add_submit = st.form_submit_button("➕ Ajouter à la shortlist", use_container_width=True)

        if add_submit:
            candidate_norm = normalize_parcel_id(manual_id) if manual_id.strip() else normalize_parcel_id(add_candidate)
            if not candidate_norm:
                st.warning("Entrez un ID ou choisissez une parcelle.")
            elif candidate_norm not in normalized_to_canonical:
                st.error(f"ID introuvable : {candidate_norm}. Vérifiez le format (ex: 73065000AB0101).")
            else:
                canonical_id = normalized_to_canonical[candidate_norm]
                if canonical_id in shortlist_ids:
                    st.info(f"{canonical_id} est déjà dans la shortlist.")
                else:
                    shortlist_ids.append(canonical_id)
                    st.success(f"Parcelle {canonical_id} ajoutée.")

        act1, act2, act3 = st.columns([2, 1, 1])
        remove_id = act1.selectbox(
            "Retirer une parcelle",
            options=[""] + shortlist_ids,
            key="shortlist_remove_id",
            placeholder="Choisir un ID à retirer",
        )
        if act2.button("➖ Retirer", use_container_width=True, disabled=not remove_id):
            st.session_state["shortlist_ids"] = [x for x in shortlist_ids if x != remove_id]
            st.success(f"Parcelle {remove_id} retirée.")
            trigger_rerun()
        if act3.button("🧹 Vider", use_container_width=True, disabled=not shortlist_ids):
            st.session_state["shortlist_ids"] = []
            st.success("Shortlist vidée.")
            trigger_rerun()

        open_col, export_col = st.columns(2)
        if open_col.button("🔍 Ouvrir la première en analyse", disabled=not st.session_state["shortlist_ids"], use_container_width=True):
            st.session_state["selected_parcel_id"] = st.session_state["shortlist_ids"][0]
            st.session_state["current_page"] = "🔍 Analyse parcelle"
            if st.session_state.get("guided_mode", False):
                st.session_state["guided_step"] = 1
            trigger_rerun()
        export_text = "\n".join(st.session_state["shortlist_ids"]) if st.session_state["shortlist_ids"] else ""
        export_col.download_button(
            "⬇️ Export IDs shortlist",
            data=export_text.encode("utf-8"),
            file_name="shortlist_ids.txt",
            mime="text/plain",
            disabled=not st.session_state["shortlist_ids"],
            use_container_width=True,
        )

        if st.session_state["shortlist_ids"]:
            short_df = df[df["_parcel_id"].astype(str).isin(st.session_state["shortlist_ids"])].copy()
            short_df = short_df[["_parcel_id", "CPI_technique", "consensus_confidence", "slope_p90", "surface_m2"]].rename(columns={"_parcel_id": "ID Parcelle"})
            short_df = short_df.sort_values("CPI_technique", ascending=False)
            st.dataframe(
                short_df.style.format(
                    {"CPI_technique": "{:.1f}", "consensus_confidence": "{:.2f}", "slope_p90": "{:.1f}", "surface_m2": "{:.0f}"}
                ),
                use_container_width=True,
            )
        else:
            st.caption("Aucune parcelle en shortlist pour le moment.")

    if guided_mode and st.session_state.get("guided_step", 0) == 0 and len(st.session_state["shortlist_ids"]) >= 2:
        st.success("✅ Shortlist prête. Passez à l'étape suivante pour analyser une parcelle.")
        if st.button("➡️ Continuer vers Analyse parcelle", key="guided_to_analysis"):
            st.session_state["guided_step"] = 1
            st.session_state["current_page"] = "🔍 Analyse parcelle"
            if st.session_state["shortlist_ids"] and not st.session_state.get("selected_parcel_id"):
                st.session_state["selected_parcel_id"] = st.session_state["shortlist_ids"][0]
            trigger_rerun()

# -----------------------------------------------------------------------------
# PAGE 2 — ANALYSE PARCELLE
# -----------------------------------------------------------------------------
def _shap_top_positive_negative(shap_row: pd.Series, top_pos: int = 3, top_neg: int = 2):
    shap_cols = [c for c in shap_row.index if c.startswith("shap_") and "groupe" not in c]
    vals = shap_row[shap_cols]
    pos = vals[vals > 0].sort_values(ascending=False).head(top_pos)
    neg = vals[vals < 0].sort_values(ascending=True).head(top_neg)
    return pos, neg


def page_parcel_analysis(df: pd.DataFrame, shap_df: pd.DataFrame):
    render_page_header("🔍 Analyse détaillée", "Lecture technique, alertes et explications IA par parcelle")
    audience_mode = st.session_state.get("audience_mode", "metier")
    guided_mode = st.session_state.get("guided_mode", False)
    ensure_column(df, "is_valid")
    if "_parcel_id" not in df.columns:
        df = ensure_parcel_id_column(df)
    cpi_ml_col = first_existing(df, ["CPI_ML_v6", "CPI_ML_v3", "CPI_ML_v5"]) if len(df) else "CPI_ML_v6"
    valid_ids = df[df["is_valid"] == True]["_parcel_id"].astype(str).tolist()

    col_input, col_btn = st.columns([4, 1])
    default_idx = 0
    if st.session_state.get("selected_parcel_id") and st.session_state["selected_parcel_id"] in valid_ids:
        default_idx = valid_ids.index(st.session_state["selected_parcel_id"]) + 1
    selected_id = col_input.selectbox(
        "Sélectionner une parcelle",
        options=[""] + valid_ids,
        index=default_idx,
        placeholder="Tapez un ID parcelle ou sélectionnez...",
    )
    pasted_id = col_input.text_input("Ou collez un ID cadastral", value="", placeholder="73065000AB0101")
    if pasted_id:
        normalized = normalize_parcel_id(pasted_id)
        id_lookup = {normalize_parcel_id(x): x for x in valid_ids}
        if normalized in id_lookup:
            selected_id = id_lookup[normalized]
            st.session_state["selected_parcel_id"] = selected_id
    if col_btn.button("🎲 Aléatoire") and valid_ids:
        selected_id = random.choice(valid_ids)
        st.session_state["selected_parcel_id"] = selected_id
        trigger_rerun()

    if not selected_id:
        st.info("👆 Sélectionnez une parcelle ci-dessus ou cliquez sur une parcelle sur la carte.")
        return
    if selected_id not in df["_parcel_id"].astype(str).values:
        st.warning("Parcelle introuvable dans les données.")
        return

    row = df.loc[df["_parcel_id"].astype(str) == selected_id].iloc[0]
    shap_row = (
        shap_df.loc[shap_df.index.astype(str) == selected_id].iloc[0]
        if not shap_df.empty and selected_id in shap_df.index.astype(str)
        else None
    )

    cpi = row.get("CPI_technique", np.nan)
    conf = row.get("consensus_confidence", np.nan)
    ci_low = row.get("cpi_ml_ci_low", np.nan)
    ci_high = row.get("cpi_ml_ci_high", np.nan)
    ci_range = ci_high - ci_low if pd.notna(ci_high) and pd.notna(ci_low) else np.nan
    verdict = build_business_verdict(row)

    st.subheader("🧭 Verdict métier (20 sec)")
    v1, v2, v3 = st.columns(3)
    v1.metric("Verdict", verdict["verdict"])
    v2.metric("Score CPI_technique", f"{cpi:.1f}/100" if pd.notna(cpi) else "N/A")
    v3.metric("Confiance consensus", verdict["confidence_txt"])
    st.markdown(f"**Synthèse:** {verdict['summary']}")

    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown("**✅ Atouts clés**")
        for txt in verdict["strengths"]:
            st.caption(f"• {txt}")
    with b2:
        st.markdown("**⚠️ Points de vigilance**")
        for txt in verdict["risks"]:
            st.caption(f"• {txt}")
    with b3:
        st.markdown("**🎯 Actions recommandées**")
        for txt in verdict["actions"]:
            st.caption(f"• {txt}")

    decision_sheet = build_decision_sheet(selected_id, row, cpi_ml_col, verdict)
    st.download_button(
        "⬇️ Export fiche décision (TXT)",
        data=decision_sheet.encode("utf-8"),
        file_name=f"fiche_decision_{selected_id}.txt",
        mime="text/plain",
    )
    render_glossary(["CPI_technique", "consensus_confidence", "slope_p90", "cpi_ml_std", "has_thalweg_mean"])

    if guided_mode and st.session_state.get("guided_step", 0) == 1:
        st.caption("👉 Vérifiez la synthèse, puis avancez vers la comparaison de 2 parcelles.")
        if st.button("➡️ Continuer vers Comparer 2 parcelles", key="guided_to_compare"):
            st.session_state["guided_step"] = 2
            st.session_state["current_page"] = "⚖️ Comparer 2 parcelles"
            shortlist_ids = st.session_state.setdefault("shortlist_ids", [])
            if selected_id not in shortlist_ids:
                shortlist_ids.append(selected_id)
            trigger_rerun()

    st.subheader("📊 Scores & Confiance")
    c1, c2, c3, c4 = st.columns(4)
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=cpi if pd.notna(cpi) else 0,
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": score_to_color(cpi)},
                "steps": [
                    {"range": [0, 20], "color": "#DC2626"},
                    {"range": [20, 38], "color": "#F97316"},
                    {"range": [38, 62], "color": "#EAB308"},
                    {"range": [62, 80], "color": "#86EFAC"},
                    {"range": [80, 100], "color": "#16A34A"},
                ],
            },
            number={"suffix": "/100"},
            title={"text": "CPI_technique"},
        )
    )
    gauge.update_layout(height=240, paper_bgcolor=theme_tokens()["card"], margin=dict(t=20, b=0), font={"color": theme_tokens()["text"]})
    c1.plotly_chart(gauge, use_container_width=True)

    c2.metric("Confiance consensus", f"{conf:.0%}" if pd.notna(conf) else "N/A", delta="3 méthodes s'accordent" if conf and conf > 0.8 else "Méthodes divergent")
    c3.metric("Précision ML", f"±{ci_range:.1f} pts" if pd.notna(ci_range) else "N/A", delta="Stable" if ci_range and ci_range < 5 else "Variable")
    c4.metric(
        "Scores complémentaires",
        f"Cluster {row.get('cluster_score', np.nan):.1f}" if pd.notna(row.get("cluster_score", np.nan)) else "Cluster N/A",
        delta=f"{cpi_ml_col}: {_fmt_num(row.get(cpi_ml_col), 1)}",
    )
    if "gate_reason" in row and pd.notna(row["gate_reason"]):
        st.warning(f"⚠️ Gate déclenchée : {row['gate_reason']}")

    st.subheader("🚨 Alertes")
    alerts = []
    if row.get("brgm_argiles_flag"):
        alerts.append("⚠️ Zone argileuse BRGM 2026 — consultez PPR auprès de la DDT.")
    if row.get("brgm_mvt_terrain_flag"):
        alerts.append("🔴 Point de mouvement terrain BRGM à proximité.")
    if pd.notna(row.get("shape_warning")):
        alerts.append(f"📐 {row.get('shape_warning')}")
    if pd.notna(row.get("ces_warning")):
        alerts.append(f"🏗️ {row.get('ces_warning')}")
    if alerts:
        for al in alerts:
            st.info(al)
    else:
        st.caption("Aucune alerte signalée.")

    default_tech_view = audience_mode == "technique"
    show_technical = st.toggle(
        "Afficher les détails techniques (features + SHAP)",
        value=default_tech_view,
        key=f"show_technical_{selected_id}",
    )

    if not show_technical:
        st.info("Mode simplifié actif : vous avez le verdict métier. Activez le détail technique pour voir SHAP et indicateurs complets.")
        if shap_row is not None:
            pos, neg = _shap_top_positive_negative(shap_row, top_pos=2, top_neg=2)
            pcol, ncol = st.columns(2)
            with pcol:
                st.markdown("**Facteurs favorables dominants**")
                for feat, _ in pos.items():
                    st.caption(f"• {feature_label_fr(feat.replace('shap_', ''))}")
            with ncol:
                st.markdown("**Freins principaux**")
                for feat, _ in neg.items():
                    st.caption(f"• {feature_label_fr(feat.replace('shap_', ''))}")
    else:
        st.subheader("🔬 Détail des indicateurs terrain")
        st.caption("10 features calculées depuis LiDAR HD 50cm IGN")
        tabs = st.tabs(["⛰️ Pente", "💧 Hydrologie", "🪨 Morphologie", "☀️ Ensoleillement"])
        for tab, grp in zip(tabs, FEATURE_GROUPS.keys()):
            with tab:
                group_feats = FEATURE_GROUPS[grp]
                cols = st.columns(3)
                for i, feat in enumerate(group_feats):
                    val = row.get(feat, np.nan)
                    pct = percentile(val, df[feat]) * 100 if feat in df else np.nan
                    pct_disp = f"{pct:.0f}e pct" if pd.notna(pct) else "N/A"
                    cols[i % 3].metric(
                        f"{feature_label_fr(feat)}",
                        f"{val:.2f} {feature_unit(feat)}" if pd.notna(val) else "N/A",
                        delta=pct_disp,
                        delta_color="normal",
                    )
                st.caption("Valeurs colorées selon la distribution Chambéry. Percentile calculé sur toutes les parcelles.")

        st.subheader("🧠 Pourquoi ce score ?")
        if shap_row is None:
            st.info("💡 Explication SHAP non disponible pour cette parcelle (hors dataset ML — parcelle sans label Snorkel). Le score CPI_technique reste valide.")
        else:
            level = st.radio(
                "Niveau d'explication",
                ["🟢 Simple — Je suis promoteur", "🟡 Détaillé — Je comprends les données", "🔵 Expert — Je suis géotechnicien"],
                horizontal=True,
            )

            if level.startswith("🟢"):
                pos, neg = _shap_top_positive_negative(shap_row)
                st.markdown("##### ✅ Facteurs favorables")
                for feat, v in pos.items():
                    st.success(f"{feature_label_fr(feat.replace('shap_', ''))} favorise la constructibilité (SHAP {v:+.3f})")
                st.markdown("##### ⚠️ Facteurs pénalisants")
                for feat, v in neg.items():
                    st.warning(f"{feature_label_fr(feat.replace('shap_', ''))} pénalise le score (SHAP {v:+.3f})")
                verdict_text = "✅ Ce terrain est techniquement favorable à la construction." if cpi >= 65 else "⚠️ Faisable avec précautions." if cpi >= 40 else "🚫 Contraintes significatives."
                st.markdown(f"### {verdict_text}")
                st.caption("📋 Explication automatique — contactez un géotechnicien pour une étude G2.")

            elif level.startswith("🟡"):
                shap_groups = {g: shap_row.get(f"shap_groupe_{g}", np.nan) for g in ["SLOPE", "HYDROLOGY", "MORPHOLOGY", "SUNLIGHT"]}
                vals = list(shap_groups.values())
                labels = ["⛰️ Pente", "💧 Hydrologie", "🪨 Morphologie", "☀️ Ensoleillement"]
                fig = go.Figure(
                    go.Bar(
                        x=vals,
                        y=labels,
                        orientation="h",
                        marker_color=["#16A34A" if v >= 0 else "#DC2626" for v in vals],
                        text=[f"{v:+.3f}" if pd.notna(v) else "NA" for v in vals],
                        textposition="outside",
                    )
                )
                fig.update_layout(height=320, xaxis_title="Contribution SHAP au score", showlegend=False, **plotly_layout_defaults())
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Une valeur positive augmente le score, une valeur négative le diminue.")

            else:
                shap_cols = [c for c in shap_row.index if c.startswith("shap_") and "groupe" not in c]
                shap_values = [shap_row[c] for c in shap_cols]
                feature_names_fr = [feature_label_fr(c.replace("shap_", "")) for c in shap_cols]
                fig = go.Figure(
                    go.Waterfall(
                        name="SHAP",
                        orientation="h",
                        measure=["relative"] * len(shap_values) + ["total"],
                        y=feature_names_fr + ["Score final"],
                        x=shap_values + [sum(shap_values)],
                        connector={"line": {"color": "#334155", "width": 1}},
                        increasing={"marker": {"color": "#16A34A"}},
                        decreasing={"marker": {"color": "#DC2626"}},
                        totals={"marker": {"color": "#1A56DB"}},
                        text=[f"{v:+.3f}" for v in shap_values] + [""],
                        textposition="outside",
                    )
                )
                fig.update_layout(title=f"Décomposition SHAP — Parcelle {selected_id}", height=450, xaxis_title="Contribution au score ML", **plotly_layout_defaults())
                st.plotly_chart(fig, use_container_width=True)

    st.divider()
    render_ai_chatbot(selected_id, row, cpi_ml_col)

# -----------------------------------------------------------------------------
# PAGE 3 — COMPARER 2 PARCELLES
# -----------------------------------------------------------------------------
def page_compare(df: pd.DataFrame, shap_df: pd.DataFrame):
    render_page_header("⚖️ Comparer deux parcelles", "Vue côte à côte pour décision d'investissement")
    guided_mode = st.session_state.get("guided_mode", False)
    ensure_column(df, "is_valid")
    if "_parcel_id" not in df.columns:
        df = ensure_parcel_id_column(df)
    ids = df[df["is_valid"] == True]["_parcel_id"].astype(str).tolist()
    if len(ids) < 2:
        st.info("Pas assez de parcelles valides pour comparer.")
        return
    shortlist_ids = [x for x in st.session_state.get("shortlist_ids", []) if x in ids]
    default_a = shortlist_ids[0] if len(shortlist_ids) >= 1 else (ids[0] if ids else "")
    default_b = shortlist_ids[1] if len(shortlist_ids) >= 2 else (ids[1] if len(ids) > 1 else default_a)
    col_a, col_sep, col_b = st.columns([5, 1, 5])
    col_a.subheader("📍 Parcelle A")
    id_a = col_a.selectbox("Parcelle A", ids, key="parcel_a", index=ids.index(default_a) if default_a in ids else 0)
    col_sep.markdown("<div style='text-align:center;font-size:32px;margin-top:40px;'>⚖️</div>", unsafe_allow_html=True)
    col_b.subheader("📍 Parcelle B")
    id_b = col_b.selectbox("Parcelle B", ids, key="parcel_b", index=ids.index(default_b) if default_b in ids else 0)

    if not id_a or not id_b:
        return
    if id_a == id_b:
        st.warning("Sélectionnez deux parcelles différentes.")
        return

    row_a = df.loc[df["_parcel_id"].astype(str) == id_a].iloc[0]
    row_b = df.loc[df["_parcel_id"].astype(str) == id_b].iloc[0]
    cpi_ml_col = first_existing(df, ["CPI_ML_v6", "CPI_ML_v3", "CPI_ML_v5"])
    scores = ["CPI_technique", cpi_ml_col, "cluster_score", "consensus_score", "consensus_confidence"]
    labels = ["CPI Technique", "CPI ML", "Cluster Score", "Consensus", "Confiance"]
    st.subheader("📊 Comparaison des scores")
    metrics_cols = st.columns(len(scores))
    for col, score, label in zip(metrics_cols, scores, labels):
        ensure_column(df, score)
        val_a = row_a.get(score, np.nan)
        val_b = row_b.get(score, np.nan)
        delta = val_a - val_b if pd.notna(val_a) and pd.notna(val_b) else np.nan
        col.metric(label, f"{val_a:.1f}" if pd.notna(val_a) else "N/A", delta=f"{delta:+.1f} vs B" if pd.notna(delta) else "N/A", delta_color="normal")

    st.subheader("🕵️‍♂️ Radar des 10 features")
    norms = {}
    for f in FEAT_COLS:
        vals = df[f] if f in df else pd.Series(dtype=float)
        minv, maxv = vals.min(skipna=True), vals.max(skipna=True)
        if pd.isna(minv) or pd.isna(maxv) or maxv == minv:
            norm_a = norm_b = 0.5
        else:
            norm_a = (row_a.get(f, 0) - minv) / (maxv - minv)
            norm_b = (row_b.get(f, 0) - minv) / (maxv - minv)
        norms[f] = (norm_a, norm_b)
    fig_radar = go.Figure()
    fig_radar.add_trace(
        go.Scatterpolar(
            r=[norms[f][0] for f in FEAT_COLS],
            theta=[feature_label_fr(f) for f in FEAT_COLS],
            fill='toself',
            name=f"Parcelle {id_a}",
            line_color="#1A56DB",
            fillcolor="rgba(26,86,219,0.15)",
        )
    )
    fig_radar.add_trace(
        go.Scatterpolar(
            r=[norms[f][1] for f in FEAT_COLS],
            theta=[feature_label_fr(f) for f in FEAT_COLS],
            fill='toself',
            name=f"Parcelle {id_b}",
            line_color="#0D9488",
            fillcolor="rgba(13,148,136,0.15)",
            line_dash="dot",
        )
    )
    colors = theme_tokens()
    fig_radar.update_layout(
        polar=dict(
            bgcolor=colors["card"],
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=colors["border"], linecolor=colors["border"]),
            angularaxis=dict(gridcolor=colors["border"], linecolor=colors["border"]),
        ),
        height=420,
        **plotly_layout_defaults(),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.subheader("🎯 Synthèse comparative")
    winner = id_a if row_a["CPI_technique"] > row_b["CPI_technique"] else id_b
    margin = abs(row_a["CPI_technique"] - row_b["CPI_technique"])
    st.markdown(
        f"La parcelle **{winner}** présente un meilleur CPI_technique "
        f"({max(row_a['CPI_technique'], row_b['CPI_technique']):.0f} vs {min(row_a['CPI_technique'], row_b['CPI_technique']):.0f}, "
        f"soit +{margin:.0f} pts)."
    )
    decision_text = (
        f"Comparaison Parcelles {id_a} vs {id_b}\n"
        f"Winner: {winner}\n"
        f"CPI {id_a}: {_fmt_num(row_a.get('CPI_technique'), 1)}\n"
        f"CPI {id_b}: {_fmt_num(row_b.get('CPI_technique'), 1)}\n"
        f"Ecart: {margin:.1f} points\n"
        f"Confiance {id_a}: {_fmt_num(row_a.get('consensus_confidence') * 100 if pd.notna(row_a.get('consensus_confidence')) else np.nan, 0, '%')}\n"
        f"Confiance {id_b}: {_fmt_num(row_b.get('consensus_confidence') * 100 if pd.notna(row_b.get('consensus_confidence')) else np.nan, 0, '%')}\n"
    )
    st.download_button(
        "⬇️ Export note de comparaison (TXT)",
        data=decision_text.encode("utf-8"),
        file_name=f"comparaison_{id_a}_vs_{id_b}.txt",
        mime="text/plain",
    )
    if guided_mode and st.session_state.get("guided_step", 0) == 2:
        st.caption("👉 Vous pouvez maintenant ouvrir l'étape 4 pour documenter la méthode et les limites.")
        if st.button("➡️ Continuer vers Science & Méthode", key="guided_to_science"):
            st.session_state["guided_step"] = 3
            st.session_state["current_page"] = "📊 Science & Méthode"
            trigger_rerun()

    if not shap_df.empty and id_a in shap_df.index.astype(str) and id_b in shap_df.index.astype(str):
        shap_a = shap_df.loc[shap_df.index.astype(str) == id_a].iloc[0]
        shap_b = shap_df.loc[shap_df.index.astype(str) == id_b].iloc[0]
        shap_groups = [f"shap_groupe_{g}" for g in ["SLOPE", "HYDROLOGY", "MORPHOLOGY", "SUNLIGHT"]]
        cols_exist = [c for c in shap_groups if c in shap_df.columns]
        if cols_exist:
            data = {
                "Groupe": ["⛰️ Pente", "💧 Hydrologie", "🪨 Morphologie", "☀️ Ensoleillement"],
                "A": [shap_a.get(c, np.nan) for c in shap_groups],
                "B": [shap_b.get(c, np.nan) for c in shap_groups],
            }
            fig = go.Figure()
            fig.add_trace(go.Bar(y=data["Groupe"], x=data["A"], orientation="h", name=f"{id_a}", marker_color="#1A56DB"))
            fig.add_trace(go.Bar(y=data["Groupe"], x=data["B"], orientation="h", name=f"{id_b}", marker_color="#0D9488"))
            fig.update_layout(
                barmode="group",
                height=360,
                title="SHAP par groupe",
                **plotly_layout_defaults(),
            )
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# PAGE 4 — SCIENCE & MÉTHODE
# -----------------------------------------------------------------------------
def page_science(df: pd.DataFrame, shap_df: pd.DataFrame):
    render_page_header("📊 Science & Méthode", "Transparence sur la validation scientifique du pipeline V6")
    if st.session_state.get("guided_mode", False) and st.session_state.get("guided_step", 0) == 3:
        st.success("✅ Parcours guidé terminé. Vous avez les éléments métier + méthode pour une décision argumentée.")
    tabs = st.tabs(["🔬 4 Approches", "📈 SHAP Global", "🔗 Corrélations", "⚡ Features stables", "📉 Distributions", "⚠️ Biais & Limites"])

    with tabs[0]:
        st.subheader("Comparaison des 4 méthodes de scoring")
        data_approches = {
            "Approche": ["1 — Snorkel V6", "2 — Clustering", "3 — DVF Régression", "4 — Revealed Preference"],
            "Supervision": ["Règles H1-H5", "KMeans (aucune)", "Prix DVF continu", "Comparaisons prix"],
            "Dépendance formule": ["Réduite V6", "Zéro", "Zéro", "Zéro"],
            "SHAP #1": ["slope_p90", "slope_p90", "slope_std", "aspect_south"],
            "Corrél. CPI": ["0.849", "0.701", "−0.202", "0.438"],
            "Message clé": [
                "ML confirme la formule",
                "Structure naturelle confirme",
                "Marché ≠ constructibilité",
                "Orientation décide intra-bloc",
            ],
        }
        st.dataframe(pd.DataFrame(data_approches), use_container_width=True)
        st.success("✅ slope_p90 apparaît SHAP #1 dans 3 méthodes sur 4 sans coordination → signal robuste et objectif")
        st.info("ℹ️ La corrélation négative de l'App3 prouve que le marché mesure autre chose que la constructibilité.")

    with tabs[1]:
        st.subheader("Importance SHAP globale — 9 453 parcelles V6")
        if shap_df.empty:
            st.info("SHAP indisponible.")
        else:
            shap_cols = [c for c in shap_df.columns if c.startswith("shap_") and "groupe" not in c]
            mean_shap = shap_df[shap_cols].abs().mean().sort_values()
            fig = go.Figure(
                go.Bar(
                    x=mean_shap.values,
                    y=[feature_label_fr(c.replace("shap_", "")) for c in mean_shap.index],
                    orientation="h",
                    marker_color=["#1A56DB" if "slope" in c else "#16A34A" for c in mean_shap.index],
                    text=[f"{v:.4f}" for v in mean_shap.values],
                    textposition="outside",
                )
            )
            fig.update_layout(height=420, xaxis_title="|SHAP| moyen", **plotly_layout_defaults())
            st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.subheader("Corrélations des scores")
        cols = [c for c in ["CPI_v3", "CPI_technique", "consensus_score", "cluster_score", "CPI_ML_v6", "CPI_ML_v3"] if c in df.columns]
        if len(cols) < 2:
            st.info("Colonnes insuffisantes pour corrélation.")
        else:
            corr = df[cols].corr()
            fig = px.imshow(
                corr,
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                text_auto=".3f",
                title="Corrélations entre les différents scores Terra-IA V6",
            )
            fig.update_layout(height=420, **plotly_layout_defaults())
            st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        st.subheader("3 features robustes — validées par 4 méthodes")
        st.markdown(
            """
            - slope_p90 : Top 3 dans 3/4 approches  
            - slope_std : Top 3 dans 4/4 approches  
            - has_thalweg_mean : Top 3 dans 3/4 approches  
            """
        )
        st.caption("Ces features émergent sans coordination entre méthodes. Leur convergence prouve la robustesse du signal.")

    with tabs[4]:
        st.subheader("Distributions")
        col1, col2 = st.columns(2)
        if "CPI_technique" in df.columns:
            col1.plotly_chart(
                px.histogram(df, x="CPI_technique", nbins=20, title="Distribution CPI_technique").update_layout(**plotly_layout_defaults()),
                use_container_width=True,
            )
        if "cpi_ml_std" in df.columns:
            col2.plotly_chart(
                px.histogram(df, x="cpi_ml_std", nbins=20, title="Stabilité du score ML (±IC 95% Bootstrap)").update_layout(**plotly_layout_defaults()),
                use_container_width=True,
            )
        if {"CPI_technique", "cluster_score"}.issubset(df.columns):
            scatter = px.scatter(
                df,
                x="CPI_technique",
                y="cluster_score",
                color="consensus_confidence" if "consensus_confidence" in df else None,
                size="surface_m2" if "surface_m2" in df else None,
                title="CPI_technique vs cluster_score",
                color_continuous_scale="Viridis",
            )
            scatter.update_layout(**plotly_layout_defaults())
            st.plotly_chart(scatter, use_container_width=True)

    with tabs[5]:
        st.subheader("⚠️ Limites documentées — nous les assumons")
        st.write(
            """
            • Label leakage résiduel : slope_p90 utilisé dans H1 et comme feature ML.  
            • DVF proxy bruité : App3 montre que le prix mesure autre chose que la constructibilité.  
            • AUC extrêmes = 1.000 : zones très pentues/plates faciles à classer.  
            • Couverture biaisée : centre-ville sur-représenté, périphérie moins couverte.  
            """
        )

# -----------------------------------------------------------------------------
# PAGE 5 — À PROPOS
# -----------------------------------------------------------------------------
def page_about():
    render_page_header("ℹ️ À propos de Terra-IA", "Scoring morphologique automatisé pour le pré-filtrage foncier")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "Terra-IA est un outil de pré-filtrage morphologique développé dans le cadre du Capstone SKEMA 2026. "
            "Il analyse automatiquement 15 480 parcelles cadastrales de Chambéry (73065) depuis des données LiDAR HD open source de l'IGN. "
            "L'objectif : répondre à la politique ZAN (Zéro Artificialisation Nette) en identifiant le potentiel de densification."
        )
        st.markdown(
            "Références académiques : Horn (1981) · Beven & Kirkby (1979) · Riley (1999) · Zakšek et al. (2011) · "
            "Zevenbergen & Thorne (1987) · O'Callaghan & Mark (1984) · Valavi et al. (2019) · "
            "Ratner et al. (2017) · Burges et al. (2006) · Rosen (1974) · Nicoletti M. (2025)"
        )
    with col2:
        st.markdown("**Sources de données**")
        sources = {
            "IGN LiDAR HD": "open.ign.fr · Etalab 2.0",
            "DVF": "data.gouv.fr · Etalab 2.0",
            "Cadastre Etalab": "cadastre.data.gouv.fr · Etalab 2.0",
            "BRGM Géorisques": "georisques.gouv.fr · Etalab 2.0",
            "BD TOPO": "geoservices.ign.fr · Etalab 2.0",
        }
        st.table(pd.DataFrame.from_dict(sources, orient="index", columns=["Source"]))
    st.error(
        "⚠️ AVERTISSEMENT LÉGAL\n"
        "Terra-IA est un outil de recherche académique destiné au pré-filtrage morphologique. "
        "Il ne constitue pas une expertise géotechnique, urbanistique ou réglementaire. "
        "Les données BRGM sont reproduites à titre informatif. Toute décision d'investissement doit s'appuyer "
        "sur des études professionnelles (G1, G2, PLU, PPR). Les seuils sont des heuristiques internes — pas des valeurs normatives."
    )


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    if "selected_parcel_id" not in st.session_state:
        st.session_state["selected_parcel_id"] = None
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "🗺️ Carte & Scoring"
    if "theme_mode" not in st.session_state:
        st.session_state["theme_mode"] = "dark"
    if "audience_mode" not in st.session_state:
        st.session_state["audience_mode"] = "metier"
    if "guided_mode" not in st.session_state:
        st.session_state["guided_mode"] = False
    if "guided_step" not in st.session_state:
        st.session_state["guided_step"] = 0
    if "shortlist_ids" not in st.session_state:
        st.session_state["shortlist_ids"] = []
    st.session_state.setdefault("filter_min_cpi", 30)
    st.session_state.setdefault("filter_min_conf", 0.5)
    st.session_state.setdefault("filter_max_slope", 30)
    st.session_state.setdefault("filter_min_surface", 100)
    st.session_state.setdefault("filter_categories", ["Faisable", "Favorable", "Optimal"])
    st.session_state.setdefault("filter_only_shap", False)
    st.session_state.setdefault("business_profile", "Équilibré")
    apply_theme_css(st.session_state["theme_mode"])

    missing_assets = missing_demo_assets()
    if missing_assets:
        render_setup_page(missing_assets)
        return

    df = load_features()
    shap_df = load_shap()
    gdf = load_geodata()
    df = ensure_parcel_id_column(df, gdf=gdf)

    page, theme_mode, audience_mode, guided_mode = sidebar()
    if theme_mode != st.session_state["theme_mode"]:
        st.session_state["theme_mode"] = theme_mode
        apply_theme_css(theme_mode)
    st.session_state["audience_mode"] = audience_mode
    st.session_state["guided_mode"] = guided_mode
    st.session_state["current_page"] = page
    page = render_guided_stepper(page)

    if page == "🗺️ Carte & Scoring":
        page_map_scoring(df, shap_df, gdf)
    elif page == "🔍 Analyse parcelle":
        page_parcel_analysis(df, shap_df)
    elif page == "⚖️ Comparer 2 parcelles":
        page_compare(df, shap_df)
    elif page == "📊 Science & Méthode":
        page_science(df, shap_df)
    else:
        page_about()


if __name__ == "__main__":
    main()
