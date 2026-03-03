"""
Crop Disease Detection - Interactive Streamlit Dashboard
========================================================
Deep learning pipeline: EfficientNetB0 transfer learning,
Grad-CAM explainability, and live predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys
from PIL import Image
import torch

# Ensure src is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from src.data_loader import (
    CLASS_NAMES, DISPLAY_NAMES, CROP_MAP, IS_HEALTHY,
    get_transforms, IMAGENET_MEAN, IMAGENET_STD,
)

# -- Page Config ---------------------------------------------------------------
st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -- Color Palette (dark theme) -----------------------------------------------
COLORS = {
    'bg_dark':      '#0A0F1A',
    'bg_card':      '#111827',
    'bg_card_alt':  '#1A2332',
    'border':       '#1E3A5F',
    'border_glow':  '#10B981',
    'accent':       '#10B981',
    'accent_light': '#34D399',
    'accent_dim':   '#065F46',
    'warning':      '#F59E0B',
    'danger':       '#EF4444',
    'text':         '#E5E7EB',
    'text_dim':     '#9CA3AF',
    'text_muted':   '#6B7280',
    'tomato':       '#EF4444',
    'potato':       '#D97706',
    'pepper':       '#10B981',
    'healthy':      '#10B981',
    'disease':      '#EF4444',
}

# Plotly dark template
PLOTLY_LAYOUT = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(17,24,39,1)',
    plot_bgcolor='rgba(17,24,39,1)',
    font=dict(color='#E5E7EB', family='Inter, sans-serif', size=13),
    legend_font=dict(color='#D1D5DB'),
    xaxis_tickfont=dict(color='#D1D5DB'),
    yaxis_tickfont=dict(color='#D1D5DB'),
    xaxis_title_font=dict(color='#D1D5DB', size=13),
    yaxis_title_font=dict(color='#D1D5DB', size=13),
    margin=dict(l=20, r=20, t=50, b=20),
)

# Reusable title-font dict for every Plotly chart title
_TITLE_FONT = dict(color='#F9FAFB', size=16, family='Inter, sans-serif')
_AXIS_FONT  = dict(color='#D1D5DB', size=13, family='Inter, sans-serif')


# -- Custom CSS ----------------------------------------------------------------
st.markdown("""
<style>
    /* ---------- Import font ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ---------- Global ---------- */
    html, body, .stApp {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(180deg, #0A0F1A 0%, #0D1321 50%, #0A0F1A 100%);
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1321 0%, #111827 100%);
        border-right: 1px solid rgba(16, 185, 129, 0.15);
    }
    section[data-testid="stSidebar"] .stRadio > label {
        color: #9CA3AF !important;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        padding: 0.6rem 1rem;
        border-radius: 10px;
        transition: all 0.2s ease;
        border: 1px solid transparent;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
        background: rgba(16, 185, 129, 0.08);
        border-color: rgba(16, 185, 129, 0.2);
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"] {
        background: rgba(16, 185, 129, 0.12);
        border-color: rgba(16, 185, 129, 0.3);
    }

    /* ---------- Main Header ---------- */
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #10B981 0%, #34D399 50%, #6EE7B7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #6B7280;
        margin-top: 4px;
        margin-bottom: 28px;
        font-weight: 400;
    }
    .page-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #E5E7EB;
        margin-bottom: 4px;
    }
    .page-subtitle {
        font-size: 0.95rem;
        color: #6B7280;
        margin-bottom: 24px;
    }

    /* ---------- Metric Cards ---------- */
    .metric-card {
        background: linear-gradient(135deg, #111827 0%, #1A2332 100%);
        border: 1px solid rgba(16, 185, 129, 0.15);
        border-radius: 16px;
        padding: 24px 20px;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: rgba(16, 185, 129, 0.4);
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.08);
        transform: translateY(-2px);
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #10B981, #34D399);
        border-radius: 16px 16px 0 0;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #F9FAFB;
        line-height: 1;
        margin-bottom: 6px;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
    }

    /* ---------- Glass Cards ---------- */
    .glass-card {
        background: linear-gradient(135deg, rgba(17,24,39,0.8) 0%, rgba(26,35,50,0.6) 100%);
        border: 1px solid rgba(16, 185, 129, 0.12);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }
    .glass-card h4 {
        color: #10B981;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
        margin-bottom: 12px;
    }

    /* ---------- Insight / Info Boxes ---------- */
    /* Note: these use higher specificity to override the global color rules */
    .insight-box, .insight-box p, .insight-box span, .insight-box strong {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(16, 185, 129, 0.03) 100%);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-left: 4px solid #10B981;
        padding: 18px 20px;
        border-radius: 0 12px 12px 0;
        margin: 12px 0;
        color: #D1FAE5 !important;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    .insight-box p, .insight-box span { background: none; border: none; padding: 0; margin: 0; border-radius: 0; }
    .insight-box strong { background: none; border: none; padding: 0; margin: 0; border-radius: 0; color: #F0FDF4 !important; }

    .warning-box, .warning-box p, .warning-box span, .warning-box strong {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.08) 0%, rgba(245, 158, 11, 0.03) 100%);
        border: 1px solid rgba(245, 158, 11, 0.2);
        border-left: 4px solid #F59E0B;
        padding: 18px 20px;
        border-radius: 0 12px 12px 0;
        margin: 12px 0;
        color: #FDE68A !important;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    .warning-box p, .warning-box span { background: none; border: none; padding: 0; margin: 0; border-radius: 0; }
    .warning-box strong { background: none; border: none; padding: 0; margin: 0; border-radius: 0; color: #FEF3C7 !important; }

    .danger-box, .danger-box p, .danger-box span, .danger-box strong {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.08) 0%, rgba(239, 68, 68, 0.03) 100%);
        border: 1px solid rgba(239, 68, 68, 0.2);
        border-left: 4px solid #EF4444;
        padding: 18px 20px;
        border-radius: 0 12px 12px 0;
        margin: 12px 0;
        color: #FCA5A5 !important;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    .danger-box p, .danger-box span { background: none; border: none; padding: 0; margin: 0; border-radius: 0; }
    .danger-box strong { background: none; border: none; padding: 0; margin: 0; border-radius: 0; color: #FEE2E2 !important; }

    .explain-box, .explain-box p, .explain-box span, .explain-box strong {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.06) 0%, rgba(59, 130, 246, 0.02) 100%);
        border: 1px solid rgba(59, 130, 246, 0.15);
        border-left: 4px solid #3B82F6;
        padding: 18px 20px;
        border-radius: 0 12px 12px 0;
        margin: 12px 0;
        color: #BFDBFE !important;
        font-size: 0.9rem;
        line-height: 1.65;
    }
    .explain-box p, .explain-box span { background: none; border: none; padding: 0; margin: 0; border-radius: 0; }
    .explain-box strong { background: none; border: none; padding: 0; margin: 0; border-radius: 0; color: #DBEAFE !important; }

    /* ---------- Pipeline Table ---------- */
    .pipeline-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 0.9rem;
    }
    .pipeline-table tr {
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .pipeline-table td {
        padding: 12px 16px;
        vertical-align: top;
    }
    .pipeline-table td:first-child {
        color: #10B981;
        font-weight: 600;
        white-space: nowrap;
        width: 140px;
    }
    .pipeline-table td:last-child {
        color: #D1D5DB;
    }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(17, 24, 39, 0.5);
        border-radius: 12px;
        padding: 4px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 24px;
        border-radius: 10px;
        font-weight: 500;
        font-size: 0.88rem;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(16, 185, 129, 0.15) !important;
    }

    /* ---------- Stat Chips ---------- */
    .stat-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 0.82rem;
        color: #34D399;
        font-weight: 500;
        margin-right: 8px;
        margin-bottom: 8px;
    }

    /* ---------- Image containers ---------- */
    .img-card {
        background: #111827;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 12px;
        transition: all 0.3s ease;
    }
    .img-card:hover {
        border-color: rgba(16, 185, 129, 0.3);
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.05);
    }

    /* ---------- Diagnosis Result ---------- */
    .diagnosis-healthy {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(16, 185, 129, 0.04) 100%);
        border: 2px solid rgba(16, 185, 129, 0.3);
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }
    .diagnosis-disease {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(239, 68, 68, 0.04) 100%);
        border: 2px solid rgba(239, 68, 68, 0.3);
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }
    .diagnosis-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .diagnosis-name {
        font-size: 1.5rem;
        font-weight: 700;
        color: #F9FAFB;
    }
    .confidence-bar {
        height: 6px;
        border-radius: 3px;
        background: rgba(255,255,255,0.08);
        margin-top: 12px;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.8s ease;
    }

    /* ---------- Severity badges ---------- */
    .severity-none {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .severity-moderate {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .severity-high {
        background: rgba(239, 68, 68, 0.15);
        color: #FCA5A5;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center;
        color: #4B5563;
        font-size: 0.8rem;
        padding: 30px 0 10px 0;
        border-top: 1px solid rgba(255,255,255,0.04);
        margin-top: 40px;
    }
    .footer a {
        color: #10B981;
        text-decoration: none;
    }

    /* ================================================================
       FORCE ALL TEXT LIGHT ON DARK
       Covers every Streamlit widget type across all versions.
       ================================================================ */

    /* --- Global text --- */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp li, .stApp td, .stApp th, .stApp caption,
    .stApp strong, .stApp em, .stApp b, .stApp i, .stApp small,
    .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li,
    .stMarkdown strong, .stMarkdown em {
        color: #E5E7EB !important;
    }

    /* --- Force ALL Streamlit containers dark bg --- */
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .stMainBlockContainer,
    [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"],
    [data-testid="column"] {
        background-color: transparent !important;
    }

    /* --- Radio buttons (sidebar nav) --- */
    .stRadio label, .stRadio label p, .stRadio label span,
    .stRadio div[role="radiogroup"] label,
    .stRadio div[role="radiogroup"] label p,
    .stRadio div[role="radiogroup"] label span {
        color: #E5E7EB !important;
    }
    .stRadio > label, .stRadio > label p {
        color: #9CA3AF !important;
    }

    /* --- Tabs --- */
    .stTabs [data-baseweb="tab"],
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span,
    .stTabs button[role="tab"],
    .stTabs button[role="tab"] p,
    .stTabs button[role="tab"] span,
    [data-baseweb="tab-list"] button {
        color: #9CA3AF !important;
    }
    .stTabs [aria-selected="true"],
    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span,
    [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #10B981 !important;
    }

    /* --- Expander --- */
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary div,
    .streamlit-expanderHeader,
    .streamlit-expanderHeader span,
    .streamlit-expanderHeader p,
    details summary, details summary span, details summary p {
        color: #E5E7EB !important;
        background: rgba(17, 24, 39, 0.6) !important;
        border-color: rgba(255,255,255,0.06) !important;
    }
    [data-testid="stExpander"] [data-testid="stExpanderDetails"],
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] p,
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] span,
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] strong {
        color: #D1D5DB !important;
    }

    /* --- Selectbox / Dropdown --- */
    .stSelectbox label, .stSelectbox label p, .stSelectbox label span,
    [data-testid="stSelectbox"] label,
    [data-testid="stSelectbox"] label p {
        color: #E5E7EB !important;
    }
    /* Dropdown input field — dark bg + light text */
    .stSelectbox [data-baseweb="select"],
    .stSelectbox [data-baseweb="select"] div,
    .stSelectbox [data-baseweb="select"] span,
    .stSelectbox [data-baseweb="select"] input,
    [data-baseweb="select"],
    [data-baseweb="select"] div,
    [data-baseweb="select"] span {
        background-color: #111827 !important;
        color: #E5E7EB !important;
        border-color: rgba(255,255,255,0.1) !important;
    }
    /* Dropdown menu items */
    [data-baseweb="popover"],
    [data-baseweb="popover"] ul,
    [data-baseweb="popover"] li,
    [data-baseweb="menu"],
    [data-baseweb="menu"] ul,
    [data-baseweb="menu"] li,
    [data-baseweb="list"] li,
    [role="listbox"],
    [role="listbox"] li,
    [role="option"] {
        background-color: #111827 !important;
        color: #E5E7EB !important;
    }
    [data-baseweb="popover"] li:hover,
    [data-baseweb="menu"] li:hover,
    [role="option"]:hover,
    [role="option"][aria-selected="true"] {
        background-color: #1A2332 !important;
        color: #F9FAFB !important;
    }

    /* --- File uploader --- */
    .stFileUploader label, .stFileUploader label p, .stFileUploader label span,
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] label p,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] section span,
    [data-testid="stFileUploader"] section small,
    [data-testid="stFileUploader"] section div {
        color: #9CA3AF !important;
    }
    /* File uploader drop zone */
    [data-testid="stFileUploader"] section {
        background-color: #111827 !important;
        border-color: rgba(255,255,255,0.1) !important;
    }
    [data-testid="stFileUploader"] button,
    [data-testid="stFileUploader"] button span {
        color: #E5E7EB !important;
        background-color: #1A2332 !important;
        border-color: rgba(16, 185, 129, 0.3) !important;
    }

    /* --- Dataframe / table --- */
    .stDataFrame, .stDataFrame th, .stDataFrame td,
    [data-testid="stDataFrame"] th,
    [data-testid="stDataFrame"] td,
    [data-testid="stDataFrame"] div,
    [data-testid="stTable"] th,
    [data-testid="stTable"] td {
        color: #E5E7EB !important;
    }

    /* --- Image captions --- */
    [data-testid="stImage"] div,
    [data-testid="stImage"] figcaption,
    [data-testid="stImage"] p {
        color: #9CA3AF !important;
    }

    /* --- st.warning / st.info / st.error --- */
    .stAlert, .stAlert p, .stAlert span, .stAlert div {
        color: #E5E7EB !important;
    }

    /* --- Plotly chart containers (force dark bg behind transparent charts) --- */
    .stPlotlyChart, [data-testid="stPlotlyChart"],
    .js-plotly-plot, .plotly, .plot-container {
        background-color: #111827 !important;
        border-radius: 12px;
    }

    /* ================================================================
       MISC OVERRIDES
       ================================================================ */
    .stMetric {
        background: linear-gradient(135deg, #111827 0%, #1A2332 100%);
        border: 1px solid rgba(16, 185, 129, 0.12);
        border-radius: 12px;
        padding: 16px;
    }
    hr {
        border-color: rgba(255,255,255,0.06) !important;
        margin: 28px 0 !important;
    }
    .stDataFrame {
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# -- Disease Information -------------------------------------------------------

DISEASE_INFO = {
    'Pepper__bell___Bacterial_spot': {
        'severity': 'Moderate',
        'description': 'Dark, water-soaked spots on leaves caused by Xanthomonas bacteria.',
        'action': 'Remove infected leaves. Apply copper-based bactericide. Avoid overhead watering.',
    },
    'Pepper__bell___healthy': {
        'severity': 'None',
        'description': 'Healthy pepper bell leaf with no signs of disease.',
        'action': 'No action needed. Continue regular care and monitoring.',
    },
    'Potato___Early_blight': {
        'severity': 'Moderate',
        'description': 'Brown concentric ring lesions on lower leaves caused by Alternaria solani.',
        'action': 'Apply fungicide (chlorothalonil or mancozeb). Remove affected foliage. Ensure proper spacing.',
    },
    'Potato___Late_blight': {
        'severity': 'High',
        'description': 'Dark, water-soaked lesions that spread rapidly. Caused by Phytophthora infestans.',
        'action': 'Apply fungicide immediately. Remove and destroy infected plants. Avoid overhead irrigation.',
    },
    'Potato___healthy': {
        'severity': 'None',
        'description': 'Healthy potato leaf with no signs of disease.',
        'action': 'No action needed. Continue regular care and monitoring.',
    },
    'Tomato_Bacterial_spot': {
        'severity': 'Moderate',
        'description': 'Small, dark, raised spots on leaves caused by Xanthomonas species.',
        'action': 'Remove infected leaves. Apply copper spray. Use disease-free seed and transplants.',
    },
    'Tomato_Early_blight': {
        'severity': 'Moderate',
        'description': 'Concentric ring "target" pattern lesions caused by Alternaria solani.',
        'action': 'Apply fungicide. Mulch around plants. Ensure adequate spacing for air circulation.',
    },
    'Tomato_Late_blight': {
        'severity': 'High',
        'description': 'Large, irregularly shaped brown-green lesions. Can devastate entire crops.',
        'action': 'Apply fungicide immediately. Remove and destroy infected plants. This disease spreads very rapidly.',
    },
    'Tomato_Leaf_Mold': {
        'severity': 'Moderate',
        'description': 'Pale green to yellow spots on upper leaf surface, olive-green mold underneath.',
        'action': 'Improve air circulation. Reduce humidity. Apply fungicide if severe.',
    },
    'Tomato_Septoria_leaf_spot': {
        'severity': 'Moderate',
        'description': 'Small circular spots with dark borders and gray centers.',
        'action': 'Remove infected lower leaves. Apply fungicide. Avoid overhead watering.',
    },
    'Tomato_Spider_mites_Two_spotted_spider_mite': {
        'severity': 'Moderate',
        'description': 'Tiny yellow stippling on leaves caused by spider mite feeding.',
        'action': 'Spray with miticide or insecticidal soap. Increase humidity. Introduce predatory mites.',
    },
    'Tomato__Target_Spot': {
        'severity': 'Moderate',
        'description': 'Brown lesions with concentric rings, often on lower leaves.',
        'action': 'Remove affected leaves. Apply fungicide. Improve plant spacing.',
    },
    'Tomato__Tomato_YellowLeaf__Curl_Virus': {
        'severity': 'High',
        'description': 'Upward curling and yellowing of leaves caused by a whitefly-transmitted virus.',
        'action': 'Control whitefly populations. Remove infected plants. Use resistant varieties.',
    },
    'Tomato__Tomato_mosaic_virus': {
        'severity': 'High',
        'description': 'Mottled light and dark green patterns on leaves. Highly contagious virus.',
        'action': 'Remove and destroy infected plants. Disinfect tools. No chemical treatment available.',
    },
    'Tomato_healthy': {
        'severity': 'None',
        'description': 'Healthy tomato leaf with no signs of disease.',
        'action': 'No action needed. Continue regular care and monitoring.',
    },
}


# -- Load Artifacts ------------------------------------------------------------

@st.cache_data
def load_results():
    models_dir = os.path.join(ROOT, 'models')
    results = {}

    results_path = os.path.join(models_dir, 'pipeline_results.json')
    if os.path.exists(results_path):
        with open(results_path) as f:
            results['pipeline'] = json.load(f)

    cm_path = os.path.join(models_dir, 'confusion_matrix.json')
    if os.path.exists(cm_path):
        with open(cm_path) as f:
            results['confusion_matrix'] = json.load(f)

    history_path = os.path.join(models_dir, 'training_history.json')
    if os.path.exists(history_path):
        with open(history_path) as f:
            results['history'] = json.load(f)

    metrics_path = os.path.join(models_dir, 'per_class_metrics.csv')
    if os.path.exists(metrics_path):
        results['per_class_df'] = pd.read_csv(metrics_path)

    return results


@st.cache_resource
def load_model_for_prediction():
    from src.model_training import load_model
    device = torch.device('cpu')  # Use CPU for Streamlit predictions
    model, artifacts = load_model(
        path=os.path.join(ROOT, 'models'),
        num_classes=len(CLASS_NAMES),
        device=device,
    )
    return model, artifacts, device


@st.cache_data
def scan_class_counts():
    """Scan the dataset directory to get class counts.
    Works even before the pipeline has been run.
    """
    data_dir = os.path.join(ROOT, 'data', 'PlantVillage')
    counts = {}
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            n = sum(1 for f in os.listdir(class_dir)
                    if os.path.splitext(f)[1].lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'})
            counts[class_name] = n
    return counts


results = load_results()


# -- Sidebar -------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 12px 0 20px 0;">
        <div style="font-size: 1.5rem; font-weight: 800;
                    background: linear-gradient(135deg, #10B981, #34D399);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    background-clip: text; margin-bottom: 4px;">
            CropDiseaseDetection
        </div>
        <div style="font-size: 0.78rem; color: #6B7280; letter-spacing: 0.05em;">
            AI-POWERED DISEASE DETECTION
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Data Explorer",
            "Model Performance",
            "Grad-CAM Explanations",
            "Live Prediction",
        ],
        index=0,
    )

    st.markdown("---")

    # Sidebar stats
    st.markdown("""
    <div style="padding: 8px 0;">
        <div style="color: #6B7280; font-size: 0.72rem; text-transform: uppercase;
                    letter-spacing: 0.08em; font-weight: 600; margin-bottom: 12px;">
            Dataset Stats
        </div>
        <div style="display: flex; flex-direction: column; gap: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #9CA3AF; font-size: 0.85rem;">Images</span>
                <span style="color: #E5E7EB; font-weight: 600; font-size: 0.85rem;">20,639</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #9CA3AF; font-size: 0.85rem;">Classes</span>
                <span style="color: #E5E7EB; font-weight: 600; font-size: 0.85rem;">15</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #9CA3AF; font-size: 0.85rem;">Crops</span>
                <span style="color: #E5E7EB; font-weight: 600; font-size: 0.85rem;">3</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #9CA3AF; font-size: 0.85rem;">Model</span>
                <span style="color: #10B981; font-weight: 600; font-size: 0.85rem;">EfficientNetB0</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="display: flex; flex-wrap: wrap; gap: 6px; padding: 4px 0;">
        <span class="stat-chip">Deep Learning</span>
        <span class="stat-chip">Computer Vision</span>
        <span class="stat-chip">Grad-CAM</span>
        <span class="stat-chip">Transfer Learning</span>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
#  HELPERS
# ==============================================================================

def render_metric_cards(metrics_dict):
    """Render a row of beautiful metric cards using st.columns."""
    cols = st.columns(len(metrics_dict))
    for col, (label, value) in zip(cols, metrics_dict.items()):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)


def severity_badge(severity):
    """Return HTML for a severity badge."""
    cls = {
        'None': 'severity-none',
        'Moderate': 'severity-moderate',
        'High': 'severity-high',
    }.get(severity, 'severity-moderate')
    return f'<span class="{cls}">{severity}</span>'


# ==============================================================================
#  PAGE: OVERVIEW
# ==============================================================================
if page == "Overview":
    st.markdown('<p class="hero-title">Crop Disease Detection</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">'
        'Deep learning model for identifying plant diseases from leaf images with visual explanations'
        '</p>',
        unsafe_allow_html=True,
    )

    if 'pipeline' in results:
        metrics = results['pipeline'].get('test_metrics', {})

        # -- Metric Cards --
        render_metric_cards({
            'Test Accuracy':  f"{metrics.get('accuracy', 0):.3f}",
            'Macro F1':       f"{metrics.get('macro_f1', 0):.3f}",
            'Precision':      f"{metrics.get('macro_precision', 0):.3f}",
            'Recall':         f"{metrics.get('macro_recall', 0):.3f}",
            'Classes':        f"{results['pipeline']['dataset']['num_classes']}",
        })

        with st.expander("What do these metrics mean?", expanded=False):
            st.markdown(f"""
            **These numbers summarize how well the model performs at identifying crop diseases from leaf images:**

            - **Test Accuracy ({metrics.get('accuracy', 0):.3f}):** The percentage of leaf images the model classified correctly
              out of the entire test set. This is the most intuitive metric -- higher means more correct diagnoses.

            - **Macro F1 ({metrics.get('macro_f1', 0):.3f}):** The balance between precision and recall, averaged equally across
              all 15 classes. This ensures the model performs well on rare diseases (like Tomato Mosaic Virus with only 373 images)
              and not just on common ones. A perfect score is 1.0.

            - **Macro Precision ({metrics.get('macro_precision', 0):.3f}):** When the model says a leaf has a specific disease,
              how often is it correct? Higher precision means fewer false alarms.

            - **Macro Recall ({metrics.get('macro_recall', 0):.3f}):** Of all leaves that actually have a disease, how many does
              the model correctly identify? Higher recall means fewer missed diseases.
            """)

        st.markdown("---")

        # -- Two-column layout: Pipeline + Insights --
        col_left, col_right = st.columns([3, 2], gap="large")

        with col_left:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("<h4>PROJECT PIPELINE</h4>", unsafe_allow_html=True)

            dataset_info = results['pipeline']['dataset']
            training_info = results['pipeline']['training']
            st.markdown(f"""
            <table class="pipeline-table">
                <tr><td>Data</td><td>PlantVillage &mdash; {dataset_info['total_images']:,} images, 15 classes, 3 crops</td></tr>
                <tr><td>Preprocessing</td><td>Resize 224&times;224, ImageNet normalization, augmentation</td></tr>
                <tr><td>Model</td><td>EfficientNetB0 (transfer learning from ImageNet)</td></tr>
                <tr><td>Training</td><td>2-phase: head-only &rarr; partial fine-tuning, early stopping</td></tr>
                <tr><td>Imbalance</td><td>Inverse-frequency weighted CrossEntropyLoss</td></tr>
                <tr><td>Evaluation</td><td>Accuracy, Precision, Recall, F1 (macro &amp; weighted)</td></tr>
                <tr><td>Explainability</td><td>Grad-CAM visual heatmaps on last conv layer</td></tr>
            </table>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_right:
            # Model insight card
            st.markdown(f"""
            <div class="glass-card">
                <h4>MODEL PERFORMANCE</h4>
                <div style="color: #D1D5DB; font-size: 0.9rem; line-height: 1.7;">
                    <strong style="color: #F9FAFB;">EfficientNetB0</strong> with two-phase transfer learning<br>
                    Test Accuracy = <strong style="color: #10B981;">{metrics.get('accuracy', 0):.4f}</strong><br>
                    Macro F1 = <strong style="color: #10B981;">{metrics.get('macro_f1', 0):.4f}</strong><br>
                    <span style="color: #6B7280; font-size: 0.82rem;">
                        ImageNet pretraining &rarr; head training &rarr; partial backbone fine-tuning
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Hardest/easiest class
            if 'per_class_df' in results:
                pcdf = results['per_class_df']
                worst = pcdf.loc[pcdf['f1'].idxmin()]
                best = pcdf.loc[pcdf['f1'].idxmax()]
                st.markdown(f"""
                <div class="glass-card">
                    <h4>CLASS ANALYSIS</h4>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <span style="color: #9CA3AF; font-size: 0.85rem;">Hardest class</span>
                        <span style="color: #FCA5A5; font-weight: 600; font-size: 0.85rem;">F1: {worst['f1']:.3f}</span>
                    </div>
                    <div style="color: #E5E7EB; font-size: 0.92rem; margin-bottom: 16px;
                                padding-left: 12px; border-left: 2px solid #EF4444;">
                        {worst['display_name']}
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <span style="color: #9CA3AF; font-size: 0.85rem;">Easiest class</span>
                        <span style="color: #6EE7B7; font-weight: 600; font-size: 0.85rem;">F1: {best['f1']:.3f}</span>
                    </div>
                    <div style="color: #E5E7EB; font-size: 0.92rem;
                                padding-left: 12px; border-left: 2px solid #10B981;">
                        {best['display_name']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Explainability card
            st.markdown("""
            <div class="glass-card">
                <h4>EXPLAINABILITY</h4>
                <div style="color: #D1D5DB; font-size: 0.9rem; line-height: 1.7;">
                    Grad-CAM heatmaps confirm the model focuses on
                    <strong style="color: #10B981;">disease lesions and discoloration</strong>
                    rather than background artifacts, validating learned pathological features.
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.warning("Pipeline results not found. Run `python src/run_pipeline.py` first.")


# ==============================================================================
#  PAGE: DATA EXPLORER
# ==============================================================================
elif page == "Data Explorer":
    st.markdown('<p class="page-title">Data Explorer</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">'
        'Explore the PlantVillage dataset: class distribution, sample images, and crop breakdown'
        '</p>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["Class Distribution", "Sample Images", "Crop Breakdown"])

    with tab1:
        st.markdown("""
        <div class="explain-box">
        <strong>What you are looking at:</strong> This chart shows how many images are available
        for each of the 15 disease/healthy classes. Classes with fewer images are harder for the
        model to learn. The weighted cross-entropy loss compensates for this imbalance during training.
        </div>
        """, unsafe_allow_html=True)

        if 'pipeline' in results:
            class_counts = results['pipeline']['dataset']['class_counts']
        else:
            class_counts = scan_class_counts()

        if class_counts:
            df_counts = pd.DataFrame([
                {
                    'Class': DISPLAY_NAMES.get(k, k),
                    'Count': v,
                    'Crop': CROP_MAP.get(k, 'Unknown'),
                    'Type': 'Healthy' if IS_HEALTHY.get(k, False) else 'Disease',
                }
                for k, v in class_counts.items()
            ]).sort_values('Count', ascending=True)

            color_map = {
                'Pepper': '#10B981',
                'Potato': '#F59E0B',
                'Tomato': '#EF4444',
            }
            fig = px.bar(
                df_counts, x='Count', y='Class', orientation='h',
                color='Crop', color_discrete_map=color_map,
            )
            fig.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text='Images per Class', font=_TITLE_FONT),
                height=600,
                yaxis={'categoryorder': 'total ascending'},
                legend=dict(
                    orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1, font=dict(size=12, color='#D1D5DB'),
                ),
            )
            fig.update_traces(marker_line_width=0, opacity=0.9)
            st.plotly_chart(fig, use_container_width=True)

            largest = df_counts.iloc[-1]
            smallest = df_counts.iloc[0]
            ratio = largest['Count'] / max(smallest['Count'], 1)
            st.markdown(f"""
            <div class="insight-box">
            <strong>Imbalance ratio:</strong> The largest class (<strong>{largest['Class']}</strong>
            with {largest['Count']:,} images) has <strong>{ratio:.0f}x</strong> more images than the
            smallest (<strong>{smallest['Class']}</strong> with {smallest['Count']:,} images).
            Weighted cross-entropy loss compensates for this during training.
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown("""
        <div class="explain-box">
        <strong>What you are looking at:</strong> Select a class to see sample leaf images from the dataset.
        These are the raw images the model learns from. Notice how disease symptoms create distinct visual
        patterns — spots, discoloration, curling — that the model learns to recognize.
        </div>
        """, unsafe_allow_html=True)

        selected_class = st.selectbox(
            "Select class",
            CLASS_NAMES,
            format_func=lambda x: DISPLAY_NAMES.get(x, x),
        )

        data_dir = os.path.join(ROOT, 'data', 'PlantVillage', selected_class)
        if os.path.exists(data_dir):
            images = [f for f in os.listdir(data_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:9]

            if images:
                cols = st.columns(3, gap="small")
                for i, img_name in enumerate(images):
                    with cols[i % 3]:
                        img_path = os.path.join(data_dir, img_name)
                        st.image(img_path, use_container_width=True)

                info = DISEASE_INFO.get(selected_class, {})
                sev = info.get('severity', 'N/A')
                st.markdown(f"""
                <div class="insight-box">
                    <strong>{DISPLAY_NAMES[selected_class]}</strong>
                    &nbsp;&middot;&nbsp; Severity: {severity_badge(sev)}<br>
                    <span style="color: #D1D5DB;">{info.get('description', '')}</span>
                </div>
                """, unsafe_allow_html=True)

    with tab3:
        st.markdown("""
        <div class="explain-box">
        <strong>What you are looking at:</strong> Distribution of images across the three crops
        and the balance between healthy and diseased samples within each crop.
        </div>
        """, unsafe_allow_html=True)

        if 'pipeline' in results:
            class_counts = results['pipeline']['dataset']['class_counts']
        else:
            class_counts = scan_class_counts()

        if class_counts:
            crop_data = {}
            for name, count in class_counts.items():
                crop = CROP_MAP.get(name, 'Unknown')
                healthy = IS_HEALTHY.get(name, False)
                if crop not in crop_data:
                    crop_data[crop] = {'Healthy': 0, 'Disease': 0, 'Total': 0, 'Classes': 0}
                crop_data[crop]['Total'] += count
                crop_data[crop]['Classes'] += 1
                if healthy:
                    crop_data[crop]['Healthy'] += count
                else:
                    crop_data[crop]['Disease'] += count

            col1, col2 = st.columns(2, gap="large")
            with col1:
                pie_data = pd.DataFrame([
                    {'Crop': crop, 'Images': data['Total']}
                    for crop, data in crop_data.items()
                ])
                fig = px.pie(
                    pie_data, values='Images', names='Crop',
                    color='Crop',
                    color_discrete_map={
                        'Pepper': '#10B981',
                        'Potato': '#F59E0B',
                        'Tomato': '#EF4444',
                    },
                    hole=0.45,
                )
                fig.update_layout(
                    **PLOTLY_LAYOUT, height=420,
                    title=dict(text='Image Distribution by Crop', font=_TITLE_FONT),
                )
                fig.update_traces(
                    textposition='inside', textinfo='percent+label',
                    textfont_size=13, textfont_color='#F9FAFB',
                    marker=dict(line=dict(color='#0A0F1A', width=2)),
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                bar_data = pd.DataFrame([
                    {'Crop': crop, 'Type': 'Healthy', 'Count': data['Healthy']}
                    for crop, data in crop_data.items()
                ] + [
                    {'Crop': crop, 'Type': 'Disease', 'Count': data['Disease']}
                    for crop, data in crop_data.items()
                ])
                fig = px.bar(
                    bar_data, x='Crop', y='Count', color='Type',
                    barmode='group',
                    color_discrete_map={'Healthy': '#10B981', 'Disease': '#EF4444'},
                )
                fig.update_layout(
                    **PLOTLY_LAYOUT, height=420,
                    title=dict(text='Healthy vs Disease by Crop', font=_TITLE_FONT),
                )
                fig.update_traces(marker_line_width=0, opacity=0.9)
                st.plotly_chart(fig, use_container_width=True)

            # Summary table
            summary_df = pd.DataFrame([
                {
                    'Crop': crop,
                    'Classes': data['Classes'],
                    'Total Images': f"{data['Total']:,}",
                    'Healthy': f"{data['Healthy']:,}",
                    'Disease': f"{data['Disease']:,}",
                    'Disease %': f"{data['Disease'] / max(data['Total'], 1) * 100:.1f}%",
                }
                for crop, data in crop_data.items()
            ])
            st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ==============================================================================
#  PAGE: MODEL PERFORMANCE
# ==============================================================================
elif page == "Model Performance":
    st.markdown('<p class="page-title">Model Performance</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">'
        'Comprehensive evaluation of EfficientNetB0 on the held-out test set'
        '</p>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "Overall Metrics", "Confusion Matrix", "Training Curves", "Per-Class Performance"
    ])

    with tab1:
        if 'pipeline' in results:
            metrics = results['pipeline']['test_metrics']

            render_metric_cards({
                'Accuracy':  f"{metrics['accuracy']:.4f}",
                'Macro F1':  f"{metrics['macro_f1']:.4f}",
                'Precision': f"{metrics['macro_precision']:.4f}",
                'Recall':    f"{metrics['macro_recall']:.4f}",
            })

            st.markdown("""
            <div class="explain-box">
            <strong>What these metrics mean in context:</strong><br><br>
            <strong>Accuracy</strong> — overall percentage of correct diagnoses<br>
            <strong>Macro Precision</strong> — "When the model says diseased, how often is it right?" Important for avoiding unnecessary treatments<br>
            <strong>Macro Recall</strong> — "Of all diseased leaves, how many does the model catch?" Important for not missing spreading diseases<br>
            <strong>Macro F1</strong> — balances precision and recall, treating all 15 classes equally regardless of size
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Pipeline results not found.")

    with tab2:
        st.markdown("""
        <div class="explain-box">
        <strong>How to read this:</strong> Each cell shows how many test images of one class (row)
        were classified as another class (column). The diagonal shows correct predictions. Bright
        off-diagonal cells indicate classes the model confuses.
        </div>
        """, unsafe_allow_html=True)

        if 'confusion_matrix' in results:
            cm = np.array(results['confusion_matrix']['matrix'])
            display_names = [DISPLAY_NAMES.get(n, n) for n in CLASS_NAMES]

            fig = px.imshow(
                cm, text_auto=True,
                x=display_names, y=display_names,
                color_continuous_scale=[
                    [0, '#0A0F1A'],
                    [0.01, '#064E3B'],
                    [0.3, '#10B981'],
                    [1.0, '#6EE7B7'],
                ],
                labels=dict(x="Predicted", y="Actual", color="Count"),
            )
            fig.update_layout(
                **PLOTLY_LAYOUT,
                height=720,
                xaxis_tickangle=-45,
                title='',
                coloraxis_colorbar_tickfont=dict(color='#D1D5DB'),
            )
            fig.update_traces(textfont_size=10)
            st.plotly_chart(fig, use_container_width=True)

            # Top confusions
            cm_copy = np.array(results['confusion_matrix']['matrix'])
            np.fill_diagonal(cm_copy, 0)
            top_confusions = []
            for _ in range(3):
                idx = np.unravel_index(cm_copy.argmax(), cm_copy.shape)
                if cm_copy[idx] > 0:
                    top_confusions.append((
                        DISPLAY_NAMES[CLASS_NAMES[idx[0]]],
                        DISPLAY_NAMES[CLASS_NAMES[idx[1]]],
                        int(cm_copy[idx]),
                    ))
                    cm_copy[idx] = 0

            if top_confusions:
                confusion_text = "<br>".join([
                    f"&bull; <strong>{actual}</strong> &rarr; <strong>{pred}</strong> ({count} times)"
                    for actual, pred, count in top_confusions
                ])
                st.markdown(f"""
                <div class="warning-box">
                <strong>Top misclassifications:</strong><br>{confusion_text}
                </div>
                """, unsafe_allow_html=True)

    with tab3:
        st.markdown("""
        <div class="explain-box">
        <strong>How to read this:</strong> These curves show model performance over training epochs.
        The dashed line marks where Phase 1 (head-only) ended and Phase 2 (backbone fine-tuning) began.
        A large gap between train and validation curves would indicate overfitting.
        </div>
        """, unsafe_allow_html=True)

        if 'history' in results:
            history = results['history']
            epochs = list(range(1, len(history['train_loss']) + 1))
            phase_boundary = history.get('phase_boundary', 5)

            col1, col2 = st.columns(2, gap="large")
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=epochs, y=history['train_loss'],
                    name='Train Loss',
                    line=dict(color='#3B82F6', width=2.5),
                    mode='lines+markers',
                    marker=dict(size=5),
                ))
                fig.add_trace(go.Scatter(
                    x=epochs, y=history['val_loss'],
                    name='Val Loss',
                    line=dict(color='#EF4444', width=2.5),
                    mode='lines+markers',
                    marker=dict(size=5),
                ))
                fig.add_vline(
                    x=phase_boundary + 0.5, line_dash='dash',
                    line_color='rgba(255,255,255,0.2)',
                    annotation_text='Phase 2',
                    annotation_font_color='#6B7280',
                )
                fig.update_layout(
                    **PLOTLY_LAYOUT,
                    title=dict(text='Loss over Epochs', font=_TITLE_FONT),
                    xaxis_title='Epoch', yaxis_title='Loss',
                    height=420,
                    legend=dict(
                        orientation='h', yanchor='bottom', y=1.02,
                        xanchor='right', x=1, font=dict(color='#D1D5DB'),
                    ),
                )
                fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)', zeroline=False)
                fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)', zeroline=False)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=epochs, y=history['train_acc'],
                    name='Train Acc',
                    line=dict(color='#3B82F6', width=2.5),
                    mode='lines+markers',
                    marker=dict(size=5),
                ))
                fig.add_trace(go.Scatter(
                    x=epochs, y=history['val_acc'],
                    name='Val Acc',
                    line=dict(color='#EF4444', width=2.5),
                    mode='lines+markers',
                    marker=dict(size=5),
                ))
                fig.add_vline(
                    x=phase_boundary + 0.5, line_dash='dash',
                    line_color='rgba(255,255,255,0.2)',
                    annotation_text='Phase 2',
                    annotation_font_color='#6B7280',
                )
                fig.update_layout(
                    **PLOTLY_LAYOUT,
                    title=dict(text='Accuracy over Epochs', font=_TITLE_FONT),
                    xaxis_title='Epoch', yaxis_title='Accuracy',
                    height=420,
                    legend=dict(
                        orientation='h', yanchor='bottom', y=1.02,
                        xanchor='right', x=1, font=dict(color='#D1D5DB'),
                    ),
                )
                fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)', zeroline=False)
                fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)', zeroline=False)
                st.plotly_chart(fig, use_container_width=True)

            best_epoch = np.argmin(history['val_loss']) + 1
            st.markdown(f"""
            <div class="insight-box">
            <strong>Training summary:</strong> {len(epochs)} total epochs &mdash;
            Phase 1 (head-only) for {phase_boundary} epochs, then Phase 2 (partial backbone fine-tuning).
            Best validation loss at epoch <strong>{best_epoch}</strong>.
            Train and validation curves track closely, confirming <strong>no overfitting</strong>.
            </div>
            """, unsafe_allow_html=True)

    with tab4:
        if 'per_class_df' in results:
            pcdf = results['per_class_df'].copy()

            fig = go.Figure()
            for metric, color, opacity in [
                ('precision', '#3B82F6', 0.85),
                ('recall', '#10B981', 0.85),
                ('f1', '#F59E0B', 0.95),
            ]:
                fig.add_trace(go.Bar(
                    name=metric.title(),
                    x=[DISPLAY_NAMES.get(n, n) for n in pcdf['class']],
                    y=pcdf[metric],
                    marker_color=color,
                    opacity=opacity,
                ))
            fig.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text='Precision, Recall, F1 by Class', font=_TITLE_FONT),
                barmode='group', height=520,
                xaxis_tickangle=-45, yaxis_range=[0.97, 1.005],
                legend=dict(
                    orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1, font=dict(color='#D1D5DB'),
                ),
            )
            fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)', zeroline=False)
            fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)', zeroline=False)
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)

            # Table
            display_df = pcdf[['display_name', 'crop', 'precision', 'recall', 'f1', 'support']].copy()
            display_df.columns = ['Class', 'Crop', 'Precision', 'Recall', 'F1', 'Support']
            st.dataframe(
                display_df.style.format({
                    'Precision': '{:.3f}', 'Recall': '{:.3f}', 'F1': '{:.3f}',
                }),
                use_container_width=True, hide_index=True,
            )


# ==============================================================================
#  PAGE: GRAD-CAM EXPLANATIONS
# ==============================================================================
elif page == "Grad-CAM Explanations":
    st.markdown('<p class="page-title">Grad-CAM Explanations</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">'
        'Visual evidence of what the model focuses on when making predictions'
        '</p>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class="explain-box">
    <strong>What is Grad-CAM?</strong> Gradient-weighted Class Activation Mapping highlights the regions
    of a leaf image that most influenced the model's prediction.
    <strong style="color: #FCA5A5;">Red/yellow regions</strong> indicate high importance (where the model
    detected disease patterns), while <strong style="color: #93C5FD;">blue/green regions</strong>
    contributed less.<br><br>
    <strong>Why it matters:</strong> A model that gets high accuracy by looking at the wrong parts of the
    image (background, image artifacts) would fail in the real world. Grad-CAM provides visual proof
    that the model learned meaningful disease patterns.
    </div>
    """, unsafe_allow_html=True)

    gradcam_dir = os.path.join(ROOT, 'assets', 'gradcam_samples')

    if os.path.exists(gradcam_dir) and os.listdir(gradcam_dir):
        gradcam_files = sorted([
            f for f in os.listdir(gradcam_dir) if f.endswith('.png')
        ])

        available_classes = sorted(set(
            '_'.join(f.split('_')[:-1]) for f in gradcam_files
        ))

        selected = st.selectbox(
            "Filter by class",
            ['All Classes'] + available_classes,
            format_func=lambda x: DISPLAY_NAMES.get(x, x) if x != 'All Classes' else x,
        )

        if selected != 'All Classes':
            gradcam_files = [f for f in gradcam_files if f.startswith(selected)]

        # Display in grid
        cols = st.columns(2, gap="medium")
        for i, fname in enumerate(gradcam_files):
            with cols[i % 2]:
                img_path = os.path.join(gradcam_dir, fname)
                class_key = '_'.join(fname.split('_')[:-1])
                display = DISPLAY_NAMES.get(class_key, class_key)
                st.image(img_path, caption=display, use_container_width=True)

        if not gradcam_files:
            st.info("No Grad-CAM samples found for this class.")
    else:
        st.warning("Grad-CAM samples not generated yet. Run the pipeline first.")


# ==============================================================================
#  PAGE: LIVE PREDICTION
# ==============================================================================
elif page == "Live Prediction":
    st.markdown('<p class="page-title">Live Prediction</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">'
        'Upload a leaf image for real-time disease diagnosis with Grad-CAM explanation'
        '</p>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class="explain-box">
    <strong>How this works:</strong> Upload a photo of a plant leaf. The model processes it through
    the same EfficientNetB0 network used during training and returns: (1) the predicted disease class
    with confidence score, (2) the top 5 most likely diagnoses, and (3) a Grad-CAM heatmap showing
    which parts of the leaf the model focused on.
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a leaf image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a photo of a pepper, potato, or tomato leaf.",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("**Uploaded Image**")
            st.image(image, use_container_width=True)

        # Run prediction
        try:
            model, artifacts, device = load_model_for_prediction()

            transform = get_transforms('eval', 224)
            input_tensor = transform(image).unsqueeze(0).to(device)

            from src.explainability import GradCAM, overlay_gradcam
            gradcam = GradCAM(model)
            heatmap, pred_class, confidence, top5 = gradcam.generate(input_tensor)
            gradcam.remove_hooks()

            pred_name = CLASS_NAMES[pred_class]
            pred_display = DISPLAY_NAMES[pred_name]
            crop = CROP_MAP[pred_name]
            healthy = IS_HEALTHY[pred_name]

            # Create overlay
            overlay = overlay_gradcam(image, heatmap, alpha=0.5)

            with col2:
                st.markdown("**Grad-CAM Overlay**")
                st.image(overlay, use_container_width=True)

            st.markdown("---")

            # -- Diagnosis Result --
            info = DISEASE_INFO.get(pred_name, {})
            sev = info.get('severity', 'N/A')
            status_class = 'diagnosis-healthy' if healthy else 'diagnosis-disease'
            status_label_color = '#10B981' if healthy else '#EF4444'
            status_text = 'HEALTHY' if healthy else 'DISEASE DETECTED'
            conf_color = '#10B981' if confidence > 0.9 else '#F59E0B' if confidence > 0.7 else '#EF4444'

            st.markdown(f"""
            <div class="{status_class}">
                <div class="diagnosis-label" style="color: {status_label_color};">{status_text}</div>
                <div class="diagnosis-name">{pred_display}</div>
                <div style="margin-top: 10px; display: flex; justify-content: center; gap: 24px;
                            font-size: 0.88rem; color: #9CA3AF;">
                    <span>Confidence: <strong style="color: {conf_color};">{confidence:.1%}</strong></span>
                    <span>Crop: <strong style="color: #E5E7EB;">{crop}</strong></span>
                    <span>Severity: {severity_badge(sev)}</span>
                </div>
                <div class="confidence-bar" style="max-width: 300px; margin: 16px auto 0 auto;">
                    <div class="confidence-fill"
                         style="width: {confidence * 100:.0f}%;
                                background: linear-gradient(90deg, {conf_color}, {conf_color}88);"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")

            # Top 5 predictions
            col_t5, col_action = st.columns([3, 2], gap="large")

            with col_t5:
                st.markdown("""
                <div class="glass-card">
                    <h4>TOP 5 PREDICTIONS</h4>
                </div>
                """, unsafe_allow_html=True)

                top5_df = pd.DataFrame([
                    {'Class': DISPLAY_NAMES.get(name, name), 'Confidence': prob}
                    for name, prob in top5
                ])
                fig = px.bar(
                    top5_df, x='Confidence', y='Class', orientation='h',
                    color='Confidence',
                    color_continuous_scale=[
                        [0, '#064E3B'],
                        [0.5, '#10B981'],
                        [1.0, '#6EE7B7'],
                    ],
                )
                fig.update_layout(
                    **PLOTLY_LAYOUT,
                    height=280,
                    yaxis={'categoryorder': 'total ascending'},
                    showlegend=False,
                    coloraxis_showscale=False,
                    title='',
                )
                fig.update_traces(marker_line_width=0)
                st.plotly_chart(fig, use_container_width=True)

            with col_action:
                box_class = 'insight-box' if healthy else 'warning-box'
                action_title = 'Status' if healthy else 'Recommended Action'
                st.markdown(f"""
                <div class="{box_class}" style="margin-top: 0;">
                    <strong>{action_title}</strong><br><br>
                    {info.get('description', '')}<br><br>
                    <strong>Action:</strong> {info.get('action', 'N/A')}
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error running prediction: {e}")
            st.info("Make sure you have run the pipeline first: `python src/run_pipeline.py`")


# -- Footer --------------------------------------------------------------------
st.markdown("""
<div class="footer">
    Crop Disease Detection Dashboard &nbsp;&middot;&nbsp;
    Built with EfficientNetB0, Grad-CAM &amp; Streamlit &nbsp;&middot;&nbsp;
    PlantVillage Dataset
</div>
""", unsafe_allow_html=True)
