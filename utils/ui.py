import base64
from pathlib import Path

import streamlit as st
from utils.i18n import t


SIGNIFY_ICON_PATH = Path(__file__).resolve().parents[1] / "assets" / "signify-icon.png" # Absolute icon path built from this module location


def icon_data_uri():
    """Return the Signify icon as an embeddable image URL for custom HTML."""

    encoded = base64.b64encode(SIGNIFY_ICON_PATH.read_bytes()).decode("ascii") # Base64 turns binary image bytes into text safe for HTML

    return f"data:image/png;base64,{encoded}"


def load_custom_css():
    st.markdown("""
    <style>
    :root {
        --signify-primary: #6d5dfc;
        --signify-primary-hover: #5948e8;
        --signify-surface: rgba(255, 255, 255, 0.72);
        --signify-border: rgba(109, 93, 252, 0.16);
        --signify-text-muted: #667085;
    }

    .stApp {
        background:
            radial-gradient(circle at 12% 8%, rgba(109, 93, 252, 0.10), transparent 28%),
            radial-gradient(circle at 88% 16%, rgba(14, 165, 233, 0.08), transparent 24%),
            #f8f9fc;
        color: #262153;
    }

    [data-testid="stAppViewContainer"] > .main {
        background: transparent;
    }

    /* Transparent header keeps the application background gradient continuous. */
    [data-testid="stHeader"] {
        background: transparent;
        border-bottom: none;
    }

    /* Sidebar controls ------> open button stays in the header and close button stays inside the sidebar. */
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapseButton"] button {
        background: #6d5dfc !important;
        border: none !important;
        border-radius: 12px !important;
        width: 38px !important;
        height: 38px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 6px 18px rgba(109, 93, 252, 0.35) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }

    [data-testid="stExpandSidebarButton"]:hover,
    [data-testid="stSidebarCollapseButton"] button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 22px rgba(109, 93, 252, 0.55) !important;
    }

    [data-testid="stExpandSidebarButton"] *,
    [data-testid="stSidebarCollapseButton"] svg,
    [data-testid="stSidebarCollapseButton"] span {
        color: #ffffff !important;
        fill: #ffffff !important;
    }

    .block-container {
        max-width: 1280px;
        padding-top: 1rem;
        padding-right: 3rem;
        padding-bottom: 4rem;
        padding-left: 3rem;
    }

    [data-testid="stSidebar"] {
        min-width: 260px;
        background:
            radial-gradient(circle at 35% 78%, rgba(109, 93, 252, 0.20), transparent 26%),
            linear-gradient(180deg, #15152f 0%, #111229 58%, #17163a 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 14px 0 34px rgba(17, 18, 41, 0.12);
        /* Transition creates a smooth sidebar opening and closing movement. */
        transition:
            min-width 0.28s ease,
            max-width 0.28s ease,
            width 0.28s ease,
            transform 0.28s ease;
    }

    /* Collapsed sidebar width becomes zero ------> hidden 260px space cannot push the main content away from center. */
    [data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 0 !important;
        max-width: 0 !important;
        width: 0 !important;
    }

    /* Smaller sidebar header keeps the Signify title close to the top while leaving room for the close button. */
    [data-testid="stSidebarHeader"] {
        height: 2.2rem !important;
        min-height: 0 !important;
        padding-top: 0.45rem !important;
        padding-bottom: 0 !important;
    }

    [data-testid="stSidebarUserContent"] {
        margin-top: -0.7rem !important;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 4px 0 2px;
        color: #ffffff;
        font-size: 29px;
        font-weight: 800;
        letter-spacing: -0.04em;
    }

    .sidebar-brand span {
        line-height: 1;
    }

    .brand-mark {
        display: grid;
        place-items: center;
        width: 40px;
        height: 40px;
        border-radius: 13px;
        background: transparent;
        color: #ffffff;
        font-size: 21px;
        font-weight: 900;
        box-shadow: none;
        overflow: hidden;
    }

    .brand-mark img {
        width: 100%;
        height: 100%;
        object-fit: contain;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #a9abc5 !important;
        line-height: 1.55;
    }

    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #c9cbe0 !important;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: rgba(255, 255, 255, 0.07);
        border-color: rgba(255, 255, 255, 0.12);
        color: #ffffff;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-baseweb="select"] svg {
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 7px;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label {
        padding: 12px 14px;
        border: 1px solid transparent;
        border-radius: 12px;
        transition: background 150ms ease, border-color 150ms ease;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label:hover {
        background: rgba(255, 255, 255, 0.07);
        border-color: rgba(255, 255, 255, 0.08);
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(135deg, rgba(109, 93, 252, 0.82), rgba(80, 63, 194, 0.78));
        border-color: rgba(154, 141, 255, 0.30);
        box-shadow: 0 10px 22px rgba(44, 35, 126, 0.25);
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p {
        color: #ffffff !important;
        font-weight: 700;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label p {
        color: #d7d8e8 !important;
    }

    .sidebar-spacer {
        height: 24px;
    }

    .st-key-person_selector {
        margin: 8px 0 10px;
    }

    .st-key-person_selector div.stButton > button {
        min-height: 42px;
        justify-content: center;
        padding: 8px 14px;
        border-radius: 11px;
        text-align: center;
        box-shadow: 0 5px 16px rgba(38, 33, 83, 0.05);
    }

    .st-key-person_selector div.stButton > button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.88);
        border-color: rgba(109, 93, 252, 0.14);
        color: #344054;
        box-shadow: 0 5px 16px rgba(38, 33, 83, 0.04);
    }

    .st-key-person_selector div.stButton > button[kind="secondary"] p,
    .st-key-person_selector div.stButton > button[kind="secondary"] span {
        color: #344054 !important;
    }

    .st-key-person_selector div.stButton > button[kind="secondary"]:hover {
        background: rgba(109, 93, 252, 0.06);
        border-color: rgba(109, 93, 252, 0.26);
        transform: translateY(-1px);
    }

    .st-key-person_selector div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6d5dfc, #5948e8);
        border-color: #5948e8;
    }

    .main-title {
        display: flex;
        align-items: center;
        gap: 14px;
        font-size: clamp(42px, 7vw, 58px);
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: -0.045em;
        color: #262153;
        margin-bottom: 8px;
    }

    .main-title img {
        width: 52px;
        height: 52px;
        border-radius: 14px;
        object-fit: cover;
        box-shadow: 0 8px 20px rgba(109, 93, 252, 0.22);
    }

    .verify-hero {
        position: relative;
        display: flex;
        align-items: center;
        gap: 24px;
        min-height: 150px;
        margin: -8px -12px 28px;
        padding: 28px 32px;
        overflow: hidden;
        border: 1px solid rgba(109, 93, 252, 0.12);
        border-radius: 22px;
        background:
            radial-gradient(circle at 86% 28%, rgba(109, 93, 252, 0.11), transparent 30%),
            rgba(255, 255, 255, 0.82);
        box-shadow: 0 15px 38px rgba(38, 33, 83, 0.06);
    }

    .verify-hero-icon {
        position: relative;
        display: grid;
        place-items: center;
        width: 72px;
        height: 82px;
        flex: 0 0 72px;
        background: linear-gradient(145deg, #8b7cff, #5745e8);
        clip-path: polygon(50% 0, 94% 16%, 88% 70%, 50% 100%, 12% 70%, 6% 16%);
        filter: drop-shadow(0 10px 12px rgba(87, 69, 232, 0.28));
    }

    .verify-hero-icon span {
        width: 24px;
        height: 13px;
        border-left: 4px solid #ffffff;
        border-bottom: 4px solid #ffffff;
        transform: translateY(-3px) rotate(-45deg);
    }

    .verify-hero-copy {
        position: relative;
        z-index: 2;
        max-width: 690px;
    }

    .verify-hero-copy h1 {
        margin: 0 0 8px;
        color: #1f2348;
        font-size: clamp(34px, 5vw, 48px);
        font-weight: 850;
        letter-spacing: -0.045em;
    }

    .verify-hero-copy p {
        margin: 0;
        color: #667085 !important;
        font-size: 16px;
    }

    .signature-illustration {
        position: absolute;
        right: 38px;
        top: 26px;
        width: 166px;
        height: 102px;
        border: 1px solid rgba(109, 93, 252, 0.11);
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.88);
        box-shadow: 0 18px 36px rgba(73, 61, 169, 0.13);
        transform: rotate(-3deg);
    }

    .signature-lines::before,
    .signature-lines::after {
        content: "";
        position: absolute;
        left: 30px;
        height: 4px;
        border-radius: 99px;
        background: #c8c4e8;
    }

    .signature-lines::before { top: 26px; width: 78px; }
    .signature-lines::after { top: 39px; width: 52px; }

    .signature-stroke {
        position: absolute;
        right: 35px;
        bottom: 10px;
        color: #6254d8;
        font-family: cursive;
        font-size: 46px;
        font-style: italic;
        transform: rotate(-12deg);
    }

    .st-key-verification_mode_cards [role="radiogroup"] {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 18px;
        width: 100%;
    }

    .st-key-verification_mode_cards [role="radiogroup"] > label {
        min-height: 104px;
        padding: 22px 24px;
        border: 1px solid rgba(109, 93, 252, 0.15);
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.78);
        box-shadow: 0 8px 22px rgba(38, 33, 83, 0.05);
        transition: transform 150ms ease, border-color 150ms ease, background 150ms ease;
    }

    .st-key-verification_mode_cards [role="radiogroup"] > label:hover {
        transform: translateY(-2px);
        border-color: rgba(109, 93, 252, 0.35);
    }

    .st-key-verification_mode_cards [role="radiogroup"] > label:has(input:checked) {
        border: 2px solid #6d5dfc;
        background: linear-gradient(135deg, rgba(109, 93, 252, 0.10), rgba(255, 255, 255, 0.92));
        box-shadow: 0 12px 28px rgba(109, 93, 252, 0.12);
    }

    .st-key-verification_mode_cards [role="radiogroup"] > label p {
        color: #22264b !important;
        font-size: 16px;
        font-weight: 750;
        line-height: 1.7;
        white-space: pre-line;
    }

    .upload-guidance {
        display: flex;
        align-items: center;
        gap: 16px;
        margin: 12px 0 26px;
        padding: 19px 22px;
        border: 1px solid rgba(99, 102, 241, 0.19);
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(238, 242, 255, 0.92), rgba(255, 255, 255, 0.84));
    }

    .guidance-icon {
        display: grid;
        place-items: center;
        width: 42px;
        height: 42px;
        flex: 0 0 42px;
        border-radius: 999px;
        background: linear-gradient(145deg, #7768ff, #5543e7);
        color: #ffffff;
        font-family: Georgia, serif;
        font-size: 24px;
        font-weight: 700;
        box-shadow: 0 8px 18px rgba(87, 69, 232, 0.26);
    }

    .guidance-title {
        margin-bottom: 3px;
        color: #4338ca;
        font-size: 17px;
        font-weight: 800;
    }

    .guidance-copy {
        color: #59627a;
        font-size: 14px;
        line-height: 1.55;
    }

    .st-key-reference_upload_card,
    .st-key-questioned_upload_card {
        min-height: 350px;
        padding: 5px;
    }

    .st-key-reference_upload_card [data-testid="stVerticalBlockBorderWrapper"],
    .st-key-questioned_upload_card [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: rgba(109, 93, 252, 0.20) !important;
        background: rgba(255, 255, 255, 0.82);
        box-shadow: 0 12px 30px rgba(38, 33, 83, 0.05);
    }

    .st-key-reference_upload_card h3,
    .st-key-questioned_upload_card h3 {
        color: #272b57;
        font-size: 20px;
    }

    .st-key-reference_upload_card [data-testid="stFileUploaderDropzone"],
    .st-key-questioned_upload_card [data-testid="stFileUploaderDropzone"] {
        min-height: 165px;
        padding: 28px 20px;
        border: 2px dashed rgba(109, 93, 252, 0.32);
        border-radius: 16px;
        background: linear-gradient(180deg, rgba(248, 247, 255, 0.92), rgba(255, 255, 255, 0.92));
    }

    .st-key-reference_upload_card [data-testid="stFileUploaderDropzone"]:hover,
    .st-key-questioned_upload_card [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #6d5dfc;
        background: rgba(109, 93, 252, 0.055);
    }

    .st-key-reference_upload_card [data-testid="stFileUploaderDropzone"] svg,
    .st-key-questioned_upload_card [data-testid="stFileUploaderDropzone"] svg {
        width: 38px;
        height: 38px;
        color: #6d5dfc;
    }

    .upload-waiting-bar {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-top: 22px;
        padding: 17px 20px;
        border: 1px solid rgba(109, 93, 252, 0.12);
        border-radius: 13px;
        background: linear-gradient(90deg, rgba(109, 93, 252, 0.18), rgba(109, 93, 252, 0.09));
        color: #5f6190;
        font-weight: 700;
    }

    .st-key-verification_action {
        margin-top: 22px;
        padding: 18px;
        border: 1px solid rgba(109, 93, 252, 0.18);
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(109, 93, 252, 0.09), rgba(255, 255, 255, 0.88));
        box-shadow: 0 12px 28px rgba(38, 33, 83, 0.06);
    }

    .verification-ready {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 14px;
        color: #4338ca;
        font-size: 14px;
        font-weight: 750;
    }

    .ready-check {
        display: grid;
        place-items: center;
        width: 24px;
        height: 24px;
        border-radius: 999px;
        background: #dcfce7;
        color: #15803d;
        font-size: 15px;
        font-weight: 900;
    }

    .st-key-verification_action div.stButton > button {
        min-height: 54px;
        background: linear-gradient(135deg, #7465ff, #5140db);
        border: 0;
        border-radius: 13px;
        font-size: 16px;
        box-shadow: 0 12px 24px rgba(81, 64, 219, 0.24);
    }

    .waiting-lock {
        position: relative;
        width: 15px;
        height: 13px;
        border: 2px solid #8b83d7;
        border-radius: 3px;
    }

    .waiting-lock::before {
        content: "";
        position: absolute;
        left: 2px;
        top: -10px;
        width: 7px;
        height: 8px;
        border: 2px solid #8b83d7;
        border-bottom: 0;
        border-radius: 7px 7px 0 0;
    }

    .waiting-arrow {
        margin-left: auto;
        color: #7c70de;
        font-size: 22px;
    }

    .subtitle {
        font-size: 18px;
        line-height: 1.65;
        color: var(--signify-text-muted);
        margin-bottom: 28px;
        max-width: 720px;
    }

    h1, h2, h3 {
        letter-spacing: -0.025em;
        color: #262153;
    }

    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] li,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] {
        color: #344054;
    }

    [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] p,
    [data-testid="stAppViewContainer"] [role="radiogroup"] label p,
    [data-testid="stAppViewContainer"] [data-baseweb="radio"] p {
        color: #344054 !important;
        font-weight: 600;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--signify-surface);
        border: 1px solid var(--signify-border) !important;
        border-radius: 18px;
        box-shadow: 0 12px 34px rgba(38, 33, 83, 0.06);
    }

    div.stButton > button,
    div.stDownloadButton > button {
        min-height: 46px;
        border-radius: 12px;
        border: 1px solid var(--signify-primary);
        background: var(--signify-primary);
        color: #ffffff;
        font-weight: 700;
        box-shadow: 0 8px 18px rgba(109, 93, 252, 0.18);
        transition: transform 150ms ease, box-shadow 150ms ease, background 150ms ease;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        border-color: var(--signify-primary-hover);
        background: var(--signify-primary-hover);
        color: #ffffff;
        transform: translateY(-1px);
        box-shadow: 0 10px 24px rgba(109, 93, 252, 0.24);
    }

    div.stButton > button p,
    div.stDownloadButton > button p,
    div.stButton > button span,
    div.stDownloadButton > button span {
        color: #ffffff !important;
    }

    div.stButton > button:disabled {
        background: #d7d9e5;
        border-color: #d7d9e5;
        color: #85899b;
        box-shadow: none;
        transform: none;
    }

    [data-baseweb="input"] > div,
    [data-baseweb="select"] > div,
    [data-testid="stFileUploaderDropzone"] {
        border-radius: 12px;
        border-color: var(--signify-border);
    }

    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255, 255, 255, 0.88);
        border: 1px dashed rgba(109, 93, 252, 0.34);
    }

    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] div,
    [data-testid="stFileUploaderDropzone"] p {
        color: #475467 !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
        background: #ffffff;
        border: 1px solid rgba(109, 93, 252, 0.28);
        color: #4338a8;
        border-radius: 10px;
        font-weight: 700;
    }

    [data-testid="stFileUploaderDropzone"] button:hover {
        background: #f2f0ff;
        border-color: var(--signify-primary);
        color: #322a78;
    }

    [data-testid="stFileUploaderDropzone"] svg {
        fill: #6d5dfc;
        color: #6d5dfc;
    }

    /* Camera widgets use a dark native shell; force readable controls in both themes. */
    [data-testid="stCameraInput"] {
        padding: 12px;
        border: 1px solid rgba(109, 93, 252, 0.22);
        border-radius: 14px;
        background: #ffffff;
    }

    [data-testid="stCameraInput"] button {
        background: #6d5dfc !important;
        border-color: #6d5dfc !important;
    }

    [data-testid="stCameraInput"] button,
    [data-testid="stCameraInput"] button p,
    [data-testid="stCameraInput"] button span,
    [data-testid="stCameraInput"] button svg {
        color: #ffffff !important;
        fill: #ffffff !important;
    }

    [data-testid="stCameraInput"] p,
    [data-testid="stCameraInput"] span {
        color: #344054 !important;
    }

    [data-testid="stMetric"] {
        padding: 16px 18px;
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid var(--signify-border);
        border-radius: 14px;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p {
        color: #667085 !important;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] > div {
        color: #262153 !important;
    }

    [data-testid="stAlert"] {
        border-radius: 14px;
    }

    [data-testid="stAlert"] p {
        color: inherit !important;
    }

    [data-testid="stExpander"] {
        overflow: hidden;
        border-radius: 14px;
        border-color: var(--signify-border);
    }

    hr {
        border-color: rgba(102, 112, 133, 0.16) !important;
    }

    .result-card {
        padding: 30px;
        border-radius: 24px;
        margin-top: 20px;
        margin-bottom: 20px;
        box-shadow: 0 14px 34px rgba(16, 24, 40, 0.08);
    }

    .genuine {
        background: linear-gradient(135deg, rgba(22, 163, 74, 0.16), rgba(74, 222, 128, 0.08));
        border: 1px solid rgba(22, 163, 74, 0.32);
    }

    .forged {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.15), rgba(248, 113, 113, 0.07));
        border: 1px solid rgba(220, 38, 38, 0.30);
    }

    .uncertain {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.17), rgba(251, 191, 36, 0.07));
        border: 1px solid rgba(217, 119, 6, 0.30);
    }

    .result-title {
        font-size: 30px;
        font-weight: 800;
        letter-spacing: -0.025em;
        color: #262153 !important;
    }

    .small-muted {
        color: #475467 !important;
        font-size: 14px;
        margin-top: 5px;
    }

    .result-eyebrow {
        margin-bottom: 20px;
        color: #667085;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 0.09em;
        text-transform: uppercase;
    }

    .result-hero {
        display: flex;
        align-items: center;
        gap: 24px;
    }

    .result-icon {
        display: grid;
        place-items: center;
        width: 94px;
        height: 94px;
        flex: 0 0 94px;
        border-radius: 999px;
        font-size: 56px;
        font-weight: 800;
        line-height: 1;
    }

    .genuine .result-icon {
        background: rgba(22, 163, 74, 0.13);
        color: #16a34a;
    }

    .forged .result-icon {
        background: rgba(220, 38, 38, 0.12);
        color: #dc2626;
    }

    .uncertain .result-icon {
        background: rgba(245, 158, 11, 0.15);
        color: #d97706;
    }

    .genuine .result-title,
    .genuine .confidence-value {
        color: #15803d !important;
    }

    .forged .result-title,
    .forged .confidence-value {
        color: #b91c1c !important;
    }

    .uncertain .result-title,
    .uncertain .confidence-value {
        color: #b45309 !important;
    }

    .result-separator {
        height: 1px;
        margin: 26px 0 22px;
        background: rgba(102, 112, 133, 0.18);
    }

    .confidence-label {
        color: #475467;
        font-size: 14px;
        font-weight: 700;
    }

    .confidence-value {
        margin: 5px 0 12px;
        font-size: 38px;
        font-weight: 800;
        letter-spacing: -0.04em;
    }

    .confidence-track {
        height: 10px;
        overflow: hidden;
        border-radius: 999px;
        background: rgba(148, 163, 184, 0.24);
    }

    .confidence-fill {
        height: 100%;
        border-radius: inherit;
    }

    .genuine .confidence-fill { background: #16a34a; }
    .forged .confidence-fill { background: #dc2626; }
    .uncertain .confidence-fill { background: #d97706; }

    .confidence-caption {
        margin-top: 10px;
        color: #667085;
        font-size: 13px;
        font-weight: 600;
    }

    /* Keep sidebar text readable after the light-content theme rules above. */
    [data-testid="stSidebar"] [role="radiogroup"] > label p {
        color: #d7d8e8 !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p {
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #c9cbe0 !important;
    }

    @media (max-width: 640px) {
        .block-container {
            padding-top: 1.25rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .subtitle {
            font-size: 16px;
        }

        .result-card {
            padding: 22px;
        }

        .result-hero {
            align-items: flex-start;
            gap: 16px;
        }

        .result-icon {
            width: 66px;
            height: 66px;
            flex-basis: 66px;
            font-size: 40px;
        }

        .result-title {
            font-size: 24px;
        }

        .verify-hero {
            min-height: auto;
            padding: 22px;
        }

        .verify-hero-icon {
            width: 56px;
            height: 64px;
            flex-basis: 56px;
        }

        .signature-illustration {
            display: none;
        }

        .st-key-verification_mode_cards [role="radiogroup"] {
            grid-template-columns: 1fr;
        }

        .st-key-reference_upload_card,
        .st-key-questioned_upload_card {
            min-height: auto;
        }
    }
    </style>
    """, unsafe_allow_html = True) # Allows custom CSS because Streamlit Markdown escapes HTML by default


def show_header():
    st.markdown(
        f'<div class="main-title"><img src="{icon_data_uri()}" alt="Signify">Signify</div>',
        unsafe_allow_html = True
    )

    st.markdown(
        f'<div class="subtitle">{t("subtitle")}</div>',
        unsafe_allow_html = True
    )

    st.divider()


def show_summary_card(
    person_name,
    result,
    confidence,
    distance,
    threshold,
    best_reference = None,
    compared_count = None
):
    result_style = { # Result name maps to CSS class localized title and localized message
        "Genuine": ("genuine", t("genuine_title"), t("same_person")),
        "Uncertain": ("uncertain", t("uncertain_title"), t("uncertain_message")),
        "Forgery": ("forged", t("forged_title"), t("different_person")),
    }
    card_class, result_title, result_message = result_style[result]
    result_icon = {"Genuine": "&#10003;", "Uncertain": "?", "Forgery": "&times;"}[result] # HTML entity supplies the check or cross symbol
    confidence_width = max(0, min(100, float(confidence))) # Clamp keeps the progress-bar width between 0% and 100%

    confidence_caption = t(
        "confidence_high" if confidence >= 80 else
        "confidence_medium" if confidence >= 60 else
        "confidence_low"
    )

    st.markdown(f"""
    <div class="result-card {card_class}">
        <div class="result-eyebrow">{t("verification_result")}</div>
        <div class="result-hero">
            <div class="result-icon">{result_icon}</div>
            <div>
                <div class="result-title">{result_title}</div>
                <div class="small-muted">{result_message}</div>
            </div>
        </div>
        <div class="result-separator"></div>
        <div class="confidence-label">{t("decision_confidence")}</div>
        <div class="confidence-value">{confidence:.1f}%</div>
        <div class="confidence-track">
            <div class="confidence-fill" style="width: {confidence_width:.1f}%"></div>
        </div>
        <div class="confidence-caption">{confidence_caption}</div>
    </div>
    """, unsafe_allow_html = True)

    with st.container(border=True):
        st.markdown(f"### {t('summary')}")
        col1, col2 = st.columns(2)

        with col1:
            st.metric(t("distance"), f"{distance:.4f}")

        with col2:
            st.metric(t("threshold"), f"{threshold:.2f}")

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.caption(t("person"))
            st.write(person_name)

        with col2:
            st.caption(t("result"))
            st.write(t({
                "Genuine": "result_genuine",
                "Uncertain": "result_uncertain",
                "Forgery": "result_forgery",
            }[result]))

        if best_reference is not None or compared_count is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**{t('best_reference')}**")
                st.write(best_reference)

            with col2:
                st.write(f"**{t('compared_references')}**")
                st.write(compared_count)
