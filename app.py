#!/usr/bin/env python3
"""
app.py — Streamlit web UI for the phishing URL analyser.

Run with:
  streamlit run app.py
"""

import csv
import datetime
import io
import math
import os
import pathlib
import pickle
import re

import numpy as np
import shap
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import url as urlmod
import llm_url
from streamlit_quill import st_quill

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fish the Phishing",
    layout="wide",
)

# Build the whitelist SLD lookup set once per session
urlmod._build_safe_sld_set()

# ---------------------------------------------------------------------------
# Sidebar — LLM backend configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### LLM Settings")
    _backend_choice = st.selectbox(
        "Backend",
        ["ollama (local, free)", "OpenAI API"],
        index=0,
        help="Ollama runs on your machine — no API key needed. Install: pip install ollama",
    )
    if _backend_choice == "ollama (local, free)":
        _ollama_model = st.text_input(
            "Model name", value="llama3.2",
            help="Pull first: ollama pull llama3.2",
        )
        os.environ["LLM_BACKEND"]  = "ollama"
        os.environ["OLLAMA_MODEL"] = _ollama_model
    else:
        _oai_key = st.text_input(
            "OpenAI API key", type="password",
            value=os.environ.get("OPENAI_API_KEY", ""),
        )
        if _oai_key:
            os.environ["OPENAI_API_KEY"] = _oai_key
        os.environ["LLM_BACKEND"]  = "openai"
        os.environ["OPENAI_MODEL"] = st.text_input("Model", value="gpt-4o-mini")
    st.markdown("---")
    st.caption(
        "Enable LLM analysis with the checkbox in the Analyse tab. "
        "Each URL is sent to the LLM for semantic reasoning on top of the rule-based signals."
    )

# ---------------------------------------------------------------------------
# Load XGBoost model (trained by classify.py)
# ---------------------------------------------------------------------------
_MODEL_PATH = pathlib.Path(__file__).parent / "model.pkl"

@st.cache_resource
def _load_model():
    """Load model.pkl — returns (xgb_model, feature_names) or (None, None)."""
    if not _MODEL_PATH.exists():
        return None, None
    with open(_MODEL_PATH, "rb") as f:
        pkg = pickle.load(f)
    return pkg["model"], pkg["features"]


def _extract_model_features(per_urls: list, result: dict) -> dict:
    """Extract the same 15 features classify.py uses, from a live analysis result."""
    pus = per_urls
    url_count    = len(pus)
    has_ip       = int(result["has_ip"])
    has_short    = int(bool(result["shorteners"]))
    typosquat    = int(result["suspect_typosquat"])
    brand_sub    = int(any(p["brand_in_subdomain"] for p in pus))
    susp_path    = int(any(p["suspicious_path"]    for p in pus))
    high_ent     = int(any(p["high_entropy"]       for p in pus))
    max_score    = result["max_risk_score"]
    min_lev      = result["min_lev"] if result["min_lev"] is not None else 0

    max_url_len  = max((len(p["url"])              for p in pus), default=0)
    max_spec     = max((sum(c in "@-_=%;/" for c in p["url"]) for p in pus), default=0)
    max_digits   = max((sum(c.isdigit() for c in p["url"]) for p in pus), default=0)
    max_sub_dep  = max((
        (p["domain"].lower().split(".").index(p["sld"])
         if p["sld"] and p["sld"] in p["domain"].lower().split(".")
         else 0)
        for p in pus), default=0)
    from urllib.parse import urlparse as _up
    max_path_dep = max((
        len([s for s in _up(p["url"]).path.split("/") if s])
        for p in pus), default=0)
    max_entropy  = max((p["entropy"] for p in pus), default=0.0)
    double_ext   = int(any(p.get("double_ext", 0) for p in pus))
    susp_port    = int(any(p.get("susp_port",  0) for p in pus))
    hex_domain   = int(any(p.get("hex_domain", 0) for p in pus))

    return {
        "has_ip_url":             has_ip,
        "has_shortener":          has_short,
        "suspect_typosquat":      typosquat,
        "any_brand_in_subdomain": brand_sub,
        "any_suspicious_path":    susp_path,
        "any_high_entropy":       high_ent,
        "max_risk_score":         max_score,
        "url_count":              url_count,
        "min_levenshtein":        min_lev,
        "max_url_length":         max_url_len,
        "max_special_char_count": max_spec,
        "max_digit_count":        max_digits,
        "max_subdomain_depth":    max_sub_dep,
        "max_path_depth":         max_path_dep,
        "max_entropy_score":      round(max_entropy, 3),
        "any_double_ext":         double_ext,
        "any_susp_port":          susp_port,
        "any_hex_domain":         hex_domain,
    }

# ---------------------------------------------------------------------------
# Custom theme overrides — keep red for high-risk only
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Analyse button */
    button[kind="primary"] {
        background-color: #1a6fb5 !important;
        border-color:     #1a6fb5 !important;
        color: white !important;
    }
    button[kind="primary"]:hover {
        background-color: #155a96 !important;
        border-color:     #155a96 !important;
    }

    /* Active tab underline + text */
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #1a6fb5 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #1a6fb5 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HISTORY_FILE = pathlib.Path(__file__).parent / "history.csv"
HISTORY_COLS = ["timestamp", "input_snippet", "urls_found", "max_score", "verdict", "signals_fired"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def verdict(score: int) -> tuple:
    """Return (prefix, label, bg_colour, text_colour) for a risk score."""
    if score == 0:
        return "[SAFE]", "LIKELY SAFE",                "#d4edda", "#155724"
    if score <= 20:
        return "[LOW]",  "LOW RISK",                   "#fff3cd", "#856404"
    if score <= 50:
        return "[MED]",  "MEDIUM RISK",                "#ffe5b4", "#7d4000"
    return "[HIGH]",     "HIGH RISK — LIKELY PHISHING", "#f8d7da", "#721c24"


def _ensure_history_file():
    """Create history.csv with headers if it doesn't exist yet."""
    if not HISTORY_FILE.exists():
        with open(HISTORY_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(HISTORY_COLS)


def _append_history(input_text: str, urls_found: int, max_score: int,
                    verdict_label: str, per_urls: list):
    """Append one analysis record to history.csv."""
    _ensure_history_file()
    signals = []
    for p in per_urls:
        if p.get("is_ip"):             signals.append("ip_host")
        if p.get("is_shortener"):      signals.append("shortener")
        if p.get("punycode"):          signals.append("punycode")
        if p.get("suspect_typosquat"): signals.append("typosquat")
        if p.get("brand_in_subdomain"):signals.append("brand_subdomain")
        if p.get("suspicious_path"):   signals.append("susp_path")
        if p.get("high_entropy"):      signals.append("high_entropy")
        if p.get("susp_tld"):          signals.append("susp_tld")
        if p.get("excess_subdomains"): signals.append("excess_subdomains")
        if p.get("long_url"):          signals.append("long_url")
        if p.get("at_in_url"):         signals.append("at_in_url")
        if p.get("http_only"):         signals.append("http_only")
        if p.get("redirect_param"):    signals.append("redirect_param")
        if p.get("numeric_domain"):    signals.append("numeric_domain")
        if p.get("double_ext"):         signals.append("double_ext")
        if p.get("susp_port"):          signals.append("susp_port")
        if p.get("hex_domain"):         signals.append("hex_domain")
    unique_signals = list(dict.fromkeys(signals))

    with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            input_text[:80].replace("\n", " "),
            urls_found,
            max_score,
            verdict_label,
            ", ".join(unique_signals) or "none",
        ])


def _signal_card(label: str, colour: str, description: str):
    """Render a coloured signal card."""
    bg   = {"red": "#f8d7da", "orange": "#fff3cd", "green": "#d4edda"}.get(colour, "#f0f0f0")
    bdr  = {"red": "#dc3545", "orange": "#ffc107", "green": "#28a745"}.get(colour, "#999")
    txt  = {"red": "#721c24", "orange": "#856404", "green": "#155724"}.get(colour, "#212529")
    desc = {"red": "#5a1a1a", "orange": "#5a4000", "green": "#1a3a20"}.get(colour, "#333")
    st.markdown(
        f"<div style='background:{bg};padding:10px 14px;border-radius:8px;"
        f"margin:4px 0;border-left:4px solid {bdr};color:{txt}'>"
        f"<b>{label}</b><br>"
        f"<span style='font-size:0.88em;color:{desc}'>{description}</span></div>",
        unsafe_allow_html=True,
    )


def SCORE_ATTR(name: str) -> int:
    """Look up a SCORE_* constant from urlmod by name."""
    return getattr(urlmod, name, 0)


def _strip_html(html: str) -> str:
    """Strip HTML tags from Quill output, returning readable plain text.

    Converts block-level tags to newlines so that URLs and content
    on separate lines are preserved, then decodes common HTML entities.
    """
    if not html:
        return ""
    text = re.sub(r'<br\s*/?>', '\n', html, flags=re.I)
    text = re.sub(r'</(?:p|div|li|h[1-6])>', '\n', text, flags=re.I)
    text = re.sub(r'<[^>]+>', '', text)
    text = (
        text
        .replace('&amp;', '&')
        .replace('&lt;', '<')
        .replace('&gt;', '>')
        .replace('&nbsp;', ' ')
        .replace('&#39;', "'")
        .replace('&quot;', '"')
    )
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _confidence_text(score: int) -> str:
    if score == 0:   return "Very likely safe — no suspicious signals found."
    if score <= 20:  return "Probably safe — minor indicators only."
    if score <= 50:  return "Suspicious — treat with caution before clicking."
    if score <= 75:  return "Likely phishing — do not interact with this link."
    return "Almost certainly phishing — block immediately."


def _pdf_safe(text: str) -> str:
    """Sanitise text for fpdf2 core fonts (latin-1)."""
    return (
        str(text)
        .replace("\u2014", "--").replace("\u2013", "-")
        .replace("\u2018", "'").replace("\u2019", "'")
        .replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u2026", "...").replace("\u202f", " ")
        .replace("\u00b7", ".")
        .encode("latin-1", errors="replace").decode("latin-1")
    )


def _build_pdf_report(display_text, per_urls, result, max_score, label,
                      gauge_fig, shap_fig, ml_prob, ml_pred,
                      llm_verdicts, now_str):
    """Generate a PDF report. Returns (pdf_bytes, None) or (None, err_str)."""
    try:
        from fpdf import FPDF, XPos, YPos
    except ImportError:
        return None, "fpdf2 not installed -- run: pip install fpdf2"

    # Export Plotly charts to PNG via kaleido (skipped gracefully if unavailable)
    _gauge_png = _shap_png = None
    try:
        if gauge_fig is not None:
            _gauge_png = gauge_fig.to_image(format="png", width=540, height=200, scale=2)
        if shap_fig is not None:
            _shap_png = shap_fig.to_image(format="png", width=580, height=420, scale=2)
    except Exception:
        pass

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Header ──────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 12, "Fish the Phishing -- Analysis Report",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, _pdf_safe(f"Generated: {now_str}"),
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    # ── Input snippet ────────────────────────────────────────────────────────
    snippet = display_text[:120].replace("\n", " ")
    if len(display_text) > 120:
        snippet += "..."
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 60)
    pdf.multi_cell(0, 6, _pdf_safe(f"Input: {snippet}"),
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    # ── Overall verdict banner ───────────────────────────────────────────────
    _FILL = {
        "SAFE":            (212, 237, 218,  21,  87,  36),
        "LOW RISK":        (212, 237, 218,  21,  87,  36),
        "MEDIUM RISK":     (255, 243, 205, 133, 100,   4),
        "LIKELY PHISHING": (255, 229, 180, 150,  80,   0),
        "PHISHING":        (248, 215, 218, 114,  28,  36),
    }
    _f = (230, 230, 230, 60, 60, 60)
    for _k, _v in _FILL.items():
        if _k in label.upper():
            _f = _v
            break
    pdf.set_fill_color(_f[0], _f[1], _f[2])
    pdf.set_text_color(_f[3], _f[4], _f[5])
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, _pdf_safe(f"  {label}  --  Risk Score: {max_score} / 100"),
             fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # ── Stats row ────────────────────────────────────────────────────────────
    _sig_count = sum(
        1 for p in per_urls
        for k in ["is_ip", "is_shortener", "punycode", "suspect_typosquat",
                  "brand_in_subdomain", "suspicious_path", "high_entropy", "susp_tld",
                  "excess_subdomains", "long_url", "at_in_url", "http_only",
                  "redirect_param", "numeric_domain", "double_ext", "susp_port", "hex_domain"]
        if p.get(k)
    )
    _safe_count = sum(1 for p in per_urls if p.get("exact_safe_match"))
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(
        0, 6,
        f"URLs Found: {len(per_urls)}    Signals Fired: {_sig_count}    Safe Domains: {_safe_count}",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )
    pdf.ln(4)

    # ── Gauge chart image ────────────────────────────────────────────────────
    if _gauge_png:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(0, 7, "Risk Gauge", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.image(io.BytesIO(_gauge_png), x=30, w=150)
        pdf.ln(4)

    # ── Per-URL breakdown ────────────────────────────────────────────────────
    _n = len(per_urls)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, f"Per-URL Breakdown ({_n} URL{'s' if _n != 1 else ''})",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    _SIG_LABELS = {
        "is_ip":              "IP address as host",
        "is_shortener":       "URL shortener",
        "punycode":           "Punycode / IDN domain",
        "suspect_typosquat":  "Typosquat / homoglyph spoof",
        "brand_in_subdomain": "Brand name in subdomain",
        "suspicious_path":    "Suspicious path keywords",
        "high_entropy":       "High domain entropy (DGA)",
        "susp_tld":           "Suspicious free TLD",
        "excess_subdomains":  "Excessive subdomain depth",
        "long_url":           "Very long URL",
        "at_in_url":          "@ symbol in URL",
        "http_only":          "Plain HTTP (no HTTPS)",
        "redirect_param":     "Redirect parameter",
        "numeric_domain":     "Numeric-heavy domain",
        "double_ext":         "Double file extension",
        "susp_port":          "Non-standard port",
        "hex_domain":         "Hex-encoded domain",
        "exact_safe_match":   "Whitelisted trusted domain (safe)",
    }
    for _i, _p in enumerate(per_urls, 1):
        _, _ul, _, _ = verdict(_p["risk_score"])
        _url_d = _p["url"] if len(_p["url"]) <= 90 else _p["url"][:87] + "..."
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(30, 30, 30)
        pdf.multi_cell(0, 6, _pdf_safe(f"URL {_i}: {_url_d}"),
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(80, 80, 80)
        pdf.multi_cell(
            0, 5,
            _pdf_safe(f"  Score: {_p['risk_score']}/100  |  {_ul}  |"
                      f"  Domain: {_p['domain']}  |  Entropy: {_p['entropy']}"),
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )
        if _p.get("closest_safe") and _p.get("lev") is not None:
            pdf.cell(
                0, 5,
                _pdf_safe(f"  Closest safe domain: {_p['closest_safe']} (distance {_p['lev']})"),
                new_x=XPos.LMARGIN, new_y=YPos.NEXT,
            )
        _active = [_SIG_LABELS[k] for k in _SIG_LABELS if _p.get(k)]
        if _active:
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(140, 30, 30)
            pdf.multi_cell(0, 5, _pdf_safe(f"  Signals: {', '.join(_active)}"),
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        if llm_verdicts and _i - 1 < len(llm_verdicts):
            _lv = llm_verdicts[_i - 1].strip()
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(50, 50, 150)
            pdf.multi_cell(0, 5, _pdf_safe(f"  {_lv}"),
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(4)

    # ── ML prediction ─────────────────────────────────────────────────────────
    if ml_prob is not None:
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 8, "ML Model Prediction (XGBoost)",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)
        if ml_pred == 1:
            pdf.set_fill_color(248, 215, 218)
            pdf.set_text_color(114, 28, 36)
        else:
            pdf.set_fill_color(212, 237, 218)
            pdf.set_text_color(21, 87, 36)
        pdf.set_font("Helvetica", "B", 11)
        _ml_str = "PHISHING" if ml_pred == 1 else "LEGITIMATE"
        pdf.cell(
            0, 9,
            f"  ML Verdict: {_ml_str}  |  Phishing Probability: {ml_prob * 100:.1f}%",
            fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )
        pdf.ln(4)
        if _shap_png:
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(50, 50, 50)
            pdf.cell(0, 7, "Feature Contributions (SHAP)",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.image(io.BytesIO(_shap_png), x=10, w=190)
            pdf.ln(4)

    # ── LLM Analysis section ──────────────────────────────────────────────────
    if llm_verdicts:
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 8, "LLM Analysis (Semantic Reasoning)",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)
        _VERDICT_COLORS = {
            "PHISHING":   (248, 215, 218, 114, 28, 36),
            "SUSPICIOUS": (255, 243, 205, 133, 100, 4),
            "LEGIT":      (212, 237, 218, 21, 87, 36),
            "ERROR":      (230, 230, 230, 80, 80, 80),
        }
        for _i, _lv_raw in enumerate(llm_verdicts, 1):
            _lv_text = _lv_raw.strip()
            # Determine fill colour from verdict keyword in the text
            _vc = (240, 240, 240, 60, 60, 60)
            for _kw, _col in _VERDICT_COLORS.items():
                if _kw in _lv_text.upper():
                    _vc = _col
                    break
            pdf.set_fill_color(_vc[0], _vc[1], _vc[2])
            pdf.set_text_color(_vc[3], _vc[4], _vc[5])
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, f"URL {_i}", fill=True,
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(50, 50, 50)
            pdf.multi_cell(0, 5, _pdf_safe(_lv_text),
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(3)

    # ── Footer ────────────────────────────────────────────────────────────────
    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(160, 160, 160)
    pdf.cell(0, 5, "Fish the Phishing -- Phishing URL Analyser",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    return bytes(pdf.output()), None


def _run_analysis(input_text: str, run_llm: bool = False):
    """Core analysis logic — shared by the Analyse tab.

    *input_text* may be raw HTML (from st_quill) or plain text.
    URL extraction uses the raw value so that <a href="..."> links are
    captured; display/history/report use the stripped plain-text version.
    """
    # Strip HTML for human-readable display; keep raw for URL extraction
    _display_text = _strip_html(input_text)
    urls = urlmod.extract_urls(input_text.strip())
    if not urls:
        st.error("No URLs found. Make sure links start with http:// or https://")
        return

    result    = urlmod.analyze_urls(urls)
    per_urls  = result["per_url"]
    max_score = result["max_risk_score"]
    icon, label, bg, text_col = verdict(max_score)

    # Overall verdict banner
    st.markdown(
        f"<div style='background:{bg};color:{text_col};padding:16px 20px;border-radius:10px;"
        f"font-size:1.25em;font-weight:bold;margin-bottom:12px'>"
        f"{label} — Risk score: {max_score} / 100</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"*{_confidence_text(max_score)}*")
    st.caption(
        "Score guide: 0\u202f=\u202fsafe \u00b7 1\u201320\u202f=\u202flow risk "
        "\u00b7 21\u201350\u202f=\u202fmedium risk "
        "\u00b7 51\u201375\u202f=\u202flikely phishing "
        "\u00b7 76\u2013100\u202f=\u202falmost certainly phishing"
    )
    gauge_col, meta_col = st.columns([3, 1])
    with gauge_col:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=max_score,
            number={"suffix": " / 100", "font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": text_col, "thickness": 0.3},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0,  20],  "color": "#d4edda"},
                    {"range": [20, 50],  "color": "#fff3cd"},
                    {"range": [50, 75],  "color": "#ffe5b4"},
                    {"range": [75, 100], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line": {"color": text_col, "width": 4},
                    "thickness": 0.75,
                    "value": max_score,
                },
            },
        ))
        fig.update_layout(
            height=220,
            margin=dict(t=20, b=30, l=50, r=50),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    with meta_col:
        sig_count  = sum(
            1 for p in per_urls
            for k in ["is_ip","is_shortener","punycode","suspect_typosquat",
                      "brand_in_subdomain","suspicious_path","high_entropy","susp_tld",
                      "excess_subdomains","long_url","at_in_url","http_only",
                      "redirect_param","numeric_domain","double_ext","susp_port","hex_domain"]
            if p.get(k)
        )
        safe_count = sum(1 for p in per_urls if p.get("exact_safe_match"))
        score_color = (
            "#dc3545" if max_score > 50
            else "#e67e22" if max_score > 20
            else "#28a745"
        )
        for val, lbl, col in [
            (len(per_urls),       "URLs Found",     "#1a6fb5"),
            (f"{max_score}/100",  "Risk Score",     score_color),
            (sig_count,           "Signals Fired",  "#e67e22"),
            (safe_count,          "Safe Domains",   "#28a745"),
        ]:
            st.markdown(
                f"<div style='text-align:center;padding:8px 4px'>"
                f"<div style='font-size:1.8rem;font-weight:700;color:{col}'>{val}</div>"
                f"<div style='font-size:0.75rem;color:#666;text-transform:uppercase;"
                f"letter-spacing:0.5px'>{lbl}</div></div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # Fired signals summary
    st.markdown("#### Signals detected")
    SIGNAL_META = [
        ("is_ip",             "IP address as host",        "red",    "SCORE_IP"),
        ("is_shortener",      "URL shortener",              "red",    "SCORE_SHORTENER"),
        ("punycode",          "Punycode / IDN domain",      "orange", "SCORE_PUNYCODE"),
        ("suspect_typosquat", "Typosquat / homoglyph spoof","red",    "SCORE_TYPOSQUAT"),
        ("brand_in_subdomain","Brand name in subdomain",    "red",    "SCORE_BRAND_SUBDOMAIN"),
        ("suspicious_path",   "Suspicious path keywords",   "orange", "SCORE_SUSP_PATH"),
        ("high_entropy",      "High domain entropy (DGA)",  "orange", "SCORE_HIGH_ENTROPY"),
        ("susp_tld",          "Suspicious free TLD",        "orange", "SCORE_SUSP_TLD"),
        ("excess_subdomains", "Excessive subdomain depth",  "orange", "SCORE_EXCESS_SUBDOMAIN"),
        ("long_url",          "Very long URL",              "orange", "SCORE_LONG_URL"),
        ("at_in_url",         "@ symbol in URL",            "red",    "SCORE_AT_IN_URL"),
        ("http_only",         "Plain HTTP (no HTTPS)",      "orange", "SCORE_HTTP_ONLY"),
        ("redirect_param",    "Redirect parameter",         "orange", "SCORE_REDIRECT"),
        ("numeric_domain",    "Numeric-heavy domain",       "orange", "SCORE_NUMERIC_DOMAIN"),
        ("double_ext",        "Double file extension",       "red",    "SCORE_DOUBLE_EXT"),
        ("susp_port",         "Non-standard port",           "orange", "SCORE_SUSP_PORT"),
        ("hex_domain",        "Hex-encoded domain",          "red",    "SCORE_HEX_DOMAIN"),
        ("exact_safe_match",  "Whitelisted trusted domain",  "green",  "SCORE_SAFE_BONUS"),
    ]
    red_count    = sum(1 for k, _, c, _ in SIGNAL_META if c == "red"    and any(p.get(k) for p in per_urls))
    orange_count = sum(1 for k, _, c, _ in SIGNAL_META if c == "orange" and any(p.get(k) for p in per_urls))

    any_fired = red_count > 0 or orange_count > 0
    if any_fired:
        _pill_css = {
            "red":    ("#f8d7da", "#721c24", "#f5c6cb"),
            "orange": ("#fff3cd", "#856404", "#ffeeba"),
        }
        pills_html = "<div style='display:flex;flex-wrap:wrap;gap:8px;margin:6px 0'>"
        for k, lbl, colour, _ in SIGNAL_META:
            if colour in _pill_css and any(p.get(k) for p in per_urls):
                bg, txt, bdr = _pill_css[colour]
                pills_html += (
                    f"<span style='background:{bg};color:{txt};border:1px solid {bdr};"
                    f"padding:4px 12px;border-radius:20px;font-size:0.82rem;"
                    f"font-weight:600'>{lbl}</span>"
                )
        pills_html += "</div>"
        st.markdown(pills_html, unsafe_allow_html=True)
    else:
        st.info("No suspicious signals detected.")

    st.divider()

    # Per-URL expanders
    st.markdown(f"#### Per-URL breakdown ({len(per_urls)} URL{'s' if len(per_urls) != 1 else ''})")
    for i, p in enumerate(per_urls, 1):
        url_score = p["risk_score"]
        _, url_label, _, _ = verdict(url_score)
        with st.expander(
            f"URL {i}: `{p['url']}`  —  score **{url_score}/100** · {url_label}",
            expanded=(i == 1),
        ):
            c1, c2 = st.columns(2)
            c1.markdown(f"**Domain:** `{p['domain']}`")
            c1.markdown(f"**Registered SLD:** `{p['sld'] or '—'}`")
            c2.markdown(f"**Entropy:** `{p['entropy']}`")
            c2.markdown(
                f"**Closest safe domain:** `{p['closest_safe'] or '—'}` "
                f"(dist {p['lev']})"
            )
            st.markdown("---")

            sigs = []
            if p["is_ip"]:
                sigs.append(("IP address as host", "red",
                    "Uses a raw IPv4 address — legitimate services almost never do this."))
            if p["is_shortener"]:
                sigs.append(("URL shortener", "red",
                    f"`{p['domain']}` is a known URL shortener — real destination hidden."))
            if p["punycode"]:
                sigs.append(("Punycode / IDN domain", "orange",
                    "Contains `xn--` encoding used in homograph attacks."))
            if p["exact_safe_match"]:
                sigs.append(("Trusted domain (whitelisted)", "green",
                    f"`{p['domain']}` matched a known-safe domain. Suspicion reduced."))
            if p["suspect_typosquat"]:
                dist    = p.get("lev", "?")
                closest = p.get("closest_safe", "?")
                sld_raw  = p.get("sld", "")
                sld_norm = urlmod.normalize_homoglyphs(sld_raw) if sld_raw else ""
                if dist == 0 and sld_norm != sld_raw:
                    sigs.append(("Homoglyph spoof", "red",
                        f"Looks identical to `{closest}` using look-alike characters "
                        f"(e.g. `1`→`l`, `0`→`o`)."))
                else:
                    sigs.append(("Typosquat", "red",
                        f"Only **{dist}** edit(s) away from `{closest}`."))
            if p["brand_in_subdomain"]:
                sigs.append(("Brand name in subdomain", "red",
                    f"Brand **`{p['impersonated_brand']}`** used as a subdomain of an "
                    "attacker-controlled domain."))
            if p["suspicious_path"]:
                kws = ", ".join(f"`{k}`" for k in p.get("path_keywords", []))
                sigs.append(("Suspicious path keywords", "orange",
                    f"Credential-harvesting keywords in path: {kws}."))
            if p["high_entropy"]:
                sigs.append(("High domain entropy", "orange",
                    f"Entropy = **{p['entropy']}** (threshold 3.5) — likely DGA-generated."))
            if p["susp_tld"]:
                tld = p["domain"].rsplit(".", 1)[-1]
                sigs.append(("Suspicious free TLD", "orange",
                    f"`.{tld}` is a free/abused TLD registry overwhelmingly used in phishing."))
            if p["excess_subdomains"]:
                sigs.append(("Excessive subdomain depth", "orange",
                    f"More than {urlmod.MAX_SUBDOMAIN_DEPTH} subdomain labels — "
                    "unusual for legitimate sites."))
            if p["long_url"]:
                sigs.append(("Very long URL", "orange",
                    f"URL is {len(p['url'])} chars (threshold {urlmod.SUSP_URL_LENGTH}) — "
                    "padding to hide the real domain."))
            if p["at_in_url"]:
                sigs.append(("@ symbol in URL", "red",
                    "Browsers ignore everything before '@' — "
                    "e.g. paypal.com@evil.com navigates to evil.com."))
            if p["http_only"]:
                sigs.append(("Plain HTTP (no HTTPS)", "orange",
                    "Serving over unencrypted HTTP — typical of credential-harvesting pages."))
            if p["redirect_param"]:
                sigs.append(("Redirect parameter", "orange",
                    "Query string contains ?url=, ?redirect=, or similar — "
                    "victims bounced through a trusted domain."))
            if p["numeric_domain"]:
                sigs.append(("Numeric-heavy domain", "orange",
                    f"SLD `{p['sld']}` contains a long digit sequence — "
                    "mass-generated phishing domain pattern."))
            if p.get("double_ext"):
                sigs.append(("Double file extension", "red",
                    "Path ends with a benign extension followed by an executable — "
                    "e.g. `invoice.pdf.exe` — disguising a malicious file."))
            if p.get("susp_port"):
                sigs.append(("Non-standard port", "orange",
                    "URL specifies an unusual port number — legitimate web services "
                    "almost always serve on port 80 (HTTP) or 443 (HTTPS)."))
            if p.get("hex_domain"):
                sigs.append(("Hex-encoded domain", "red",
                    "Domain contains percent-encoded characters (%XX) — "
                    "used to obfuscate the real hostname and bypass string-matching filters."))

            if sigs:
                st.markdown("**Signals:**")
                for sig_label, colour, desc in sigs:
                    _signal_card(sig_label, colour, desc)
            else:
                st.success("No suspicious signals for this URL.")

    # Log to history
    _append_history(_display_text, len(per_urls), max_score, label, per_urls)
    st.caption("Saved to history.")

    # -----------------------------------------------------------------------
    # ML model prediction (XGBoost + SHAP)
    # -----------------------------------------------------------------------
    _shap_fig = None
    _ml_prob  = None
    _ml_pred  = None
    xgb_model, feature_names = _load_model()
    if xgb_model is None or feature_names is None:
        st.info("ML prediction unavailable — run `python classify.py` to generate `model.pkl`.")
    else:
        st.divider()
        st.markdown("#### ML model prediction (XGBoost)")

        feats = _extract_model_features(per_urls, result)
        X_row = np.array([[feats[f] for f in feature_names]], dtype=float)
        prob  = float(xgb_model.predict_proba(X_row)[0, 1])
        pred  = int(xgb_model.predict(X_row)[0])
        _ml_prob, _ml_pred = prob, pred

        if pred == 1:
            st.markdown(
                f"<div style='background:#f8d7da;color:#721c24;padding:12px 18px;"
                f"border-radius:10px;font-size:1.1em;font-weight:bold'>"
                f"ML verdict: PHISHING &nbsp;|&nbsp; Probability: {prob*100:.1f}%</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='background:#d4edda;color:#155724;padding:12px 18px;"
                f"border-radius:10px;font-size:1.1em;font-weight:bold'>"
                f"ML verdict: LEGITIMATE &nbsp;|&nbsp; Phishing probability: {prob*100:.1f}%</div>",
                unsafe_allow_html=True,
            )

        st.markdown("**Feature contributions (SHAP)**")
        st.caption("Red bars push toward phishing • green bars push away from phishing")
        explainer  = shap.TreeExplainer(xgb_model)
        shap_vals  = explainer.shap_values(X_row)[0]
        shap_pairs = sorted(
            zip(feature_names, shap_vals, X_row[0]),
            key=lambda t: abs(t[1]), reverse=True,
        )

        names  = [p[0] for p in shap_pairs]
        values = [p[1] for p in shap_pairs]
        colors = ["#dc3545" if v > 0 else "#28a745" for v in values]

        shap_fig = go.Figure(go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in values],
            textposition="outside",
        ))
        shap_fig.update_layout(
            height=460,
            margin=dict(t=10, b=20, l=10, r=80),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="SHAP value",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(shap_fig, use_container_width=True)
        _shap_fig = shap_fig

        with st.expander("Feature values used for prediction", expanded=False):
            feat_df = pd.DataFrame([
                {"Feature": f, "Value": feats[f], "SHAP": round(s, 4)}
                for f, s, _ in shap_pairs
            ])
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------    # LLM analysis
    # -----------------------------------------------------------------------
    _llm_verdicts = []   # collected for inclusion in the download report
    if run_llm:
        st.divider()
        st.markdown("#### LLM analysis")
        st.caption(
            f"Semantic reasoning via **{os.environ.get('LLM_BACKEND', 'ollama').upper()}** — "
            "each URL is analysed individually."
        )
        _VERDICT_STYLE = {
            "phishing":   ("#f8d7da", "#721c24"),
            "suspicious": ("#fff3cd", "#856404"),
            "legit":      ("#d4edda", "#155724"),
        }
        # Cache LLM results so a download-button rerun does not re-call the LLM
        if "llm_cache" not in st.session_state:
            st.session_state["llm_cache"] = {}
        for _i, _p in enumerate(per_urls, 1):
            _cache_key = _p["url"]
            if _cache_key in st.session_state["llm_cache"]:
                _lr = st.session_state["llm_cache"][_cache_key]
            else:
                with st.spinner(f"LLM analysing URL {_i}/{len(per_urls)}…"):
                    _lr = llm_url.llm_classify_url(_p["url"], _p)
                st.session_state["llm_cache"][_cache_key] = _lr

            if "error" in _lr:
                st.error(f"URL {_i}: {_lr['error']}")
                _llm_verdicts.append(f"  LLM   : ERROR — {_lr['error']}")
                continue

            _v    = _lr["verdict"]
            _conf = _lr["confidence"]
            _bg, _txt = _VERDICT_STYLE.get(_v, ("#f0f0f0", "#333"))
            st.markdown(
                f"<div style='background:{_bg};color:{_txt};padding:12px 16px;"
                f"border-radius:8px;margin:6px 0'>"
                f"<b>URL {_i}</b> — LLM verdict: <b>{_v.upper()}</b>"
                f"&nbsp;|&nbsp; Confidence: {_conf*100:.0f}%<br>"
                f"<span style='font-size:0.9em'><i>{_lr['reasoning']}</i></span></div>",
                unsafe_allow_html=True,
            )
            _llm_verdicts.append(
                f"  LLM   : {_v.upper()} ({_conf*100:.0f}% confidence) — {_lr['reasoning']}"
            )

    # -----------------------------------------------------------------------    # Downloadable report
    # -----------------------------------------------------------------------
    st.divider()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines = [
        "Fish the Phishing \u2014 Analysis Report",
        f"Generated : {now_str}",
        "=" * 50,
        f"Input     : {_display_text[:80].replace(chr(10), ' ')}",
        f"URLs found: {len(per_urls)}",
        f"Risk score: {max_score} / 100",
        f"Verdict   : {label}",
        "",
    ]
    for i, p in enumerate(per_urls, 1):
        _, ul, _, _ = verdict(p["risk_score"])
        report_lines.append(f"URL {i}: {p['url']}")
        report_lines.append(f"  Score  : {p['risk_score']}/100 | {ul}")
        report_lines.append(f"  Domain : {p['domain']} | Entropy: {p['entropy']}")
        active = [k for k in [
            "is_ip", "is_shortener", "punycode", "suspect_typosquat",
            "brand_in_subdomain", "suspicious_path", "high_entropy", "susp_tld",
            "excess_subdomains", "long_url", "at_in_url", "http_only",
            "redirect_param", "numeric_domain", "double_ext", "susp_port", "hex_domain",
        ] if p.get(k)]
        report_lines.append(f"  Signals: {', '.join(active) if active else 'none'}")
        if _llm_verdicts and i - 1 < len(_llm_verdicts):
            report_lines.append(_llm_verdicts[i - 1])
        report_lines.append("")
    _xgb, _feats = _load_model()
    if _xgb is not None and _feats is not None:
        _fv = _extract_model_features(per_urls, result)
        _X  = np.array([[_fv[f] for f in _feats]], dtype=float)
        _prob = float(_xgb.predict_proba(_X)[0, 1])
        _pred = int(_xgb.predict(_X)[0])
        report_lines.append(
            f"ML prediction: {'PHISHING' if _pred == 1 else 'LEGITIMATE'} "
            f"({_prob * 100:.1f}% phishing probability)"
        )
    report_text = "\n".join(report_lines)

    _pdf_bytes, _pdf_err = _build_pdf_report(
        _display_text, per_urls, result, max_score, label,
        fig, _shap_fig, _ml_prob, _ml_pred, _llm_verdicts, now_str,
    )
    _pdf_fname = f"phishing_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    if _pdf_bytes:
        st.download_button(
            "\u2b07 Download PDF report",
            data=_pdf_bytes,
            file_name=_pdf_fname,
            mime="application/pdf",
        )
    else:
        st.warning(f"PDF unavailable ({_pdf_err}) — downloading plain text instead.")
        st.download_button(
            "\u2b07 Download text report",
            data=report_text,
            file_name="phishing_analysis_report.txt",
            mime="text/plain",
        )


# ===========================================================================
# Layout — three tabs
# ===========================================================================
st.title("Fish the Phishing")
st.caption("Phishing URL analyser — static analysis only, no DNS lookups or live HTTP requests made.")

tab_analyse, tab_history = st.tabs(["Analyse", "History"])

# ---------------------------------------------------------------------------
# TAB 1: Analyse
# ---------------------------------------------------------------------------
with tab_analyse:
    st.markdown("### Paste a URL or email body to analyse")

    st.session_state.setdefault("input_counter", 0)

    # Rich text editor — preserves hyperlinks pasted from formatted emails.
    # st_quill returns HTML; <a href="..."> tags are extracted by extract_urls.
    st.caption(
        "Paste a plain URL, raw text, or a formatted email — "
        "hyperlinks are automatically preserved from your clipboard."
    )
    email_content_html = st_quill(
        placeholder="Paste the email content (Rich Text)...",
        html=True,
        key=f"main_input_{st.session_state['input_counter']}",
    )

    # Quill emits None or '<p><br></p>' when the editor is empty
    _QUILL_EMPTY = {None, "", "<p><br></p>", "<p></p>"}
    input_text = email_content_html if email_content_html not in _QUILL_EMPTY else ""

    if input_text:
        _detected = urlmod.extract_urls(input_text)
        _n = len(_detected)
        _plain_len = len(_strip_html(input_text))
        if _n == 0:
            st.caption("No URLs detected yet.")
        elif _n > 1 or _plain_len > (len(_detected[0]) + 20):
            st.caption(f"Detected: email / text body — {_n} URL{'s' if _n != 1 else ''} found.")
        elif _n == 1:
            st.caption("Detected: single URL.")
        else:
            st.caption(f"Detected: {_n} URLs.")

    _, btn_area, _ = st.columns([1, 4, 1])
    with btn_area:
        llm_enabled = st.checkbox(
            "Enable LLM analysis",
            value=False,
            help=(
                "Send each URL to the configured LLM for semantic phishing reasoning. "
                "Requires Ollama running locally or a valid OpenAI API key in the sidebar."
            ),
        )
        btn_col, reset_col = st.columns([5, 1])
        with btn_col:
            analyse_clicked = btn_col.button("Analyse", type="primary", use_container_width=True)
        with reset_col:
            if reset_col.button("Reset", use_container_width=True):
                st.session_state["input_counter"] += 1
                st.session_state.pop("analysis_input", None)
                st.session_state.pop("analysis_run_llm", None)
                st.session_state.pop("llm_cache", None)
                st.rerun()

    if analyse_clicked:
        if not input_text:
            st.warning("Please enter a URL or some text first.")
        else:
            # Persist input so results survive reruns (e.g. download button click)
            st.session_state["analysis_input"] = input_text
            st.session_state["analysis_run_llm"] = llm_enabled
            st.session_state.pop("llm_cache", None)  # force fresh LLM run
            with st.spinner("Analysing URLs..."):
                _run_analysis(input_text, run_llm=llm_enabled)
    elif st.session_state.get("analysis_input"):
        # Rerun caused by download button or other widget — re-render without re-calling LLM
        _run_analysis(
            st.session_state["analysis_input"],
            run_llm=st.session_state.get("analysis_run_llm", False),
        )

    # Batch file upload
    st.markdown("---")
    st.markdown("**Or upload a file** (\\.txt or \\.csv)")
    uploaded_file = st.file_uploader(
        "Upload a plain-text or CSV file containing URLs or email text",
        type=["txt", "csv"],
        key=f"uploader_{st.session_state['input_counter']}",
    )
    if uploaded_file is not None:
        file_content = uploaded_file.read().decode("utf-8", errors="ignore")
        if st.button("Analyse uploaded file", type="primary", key="analyse_upload"):
            _run_analysis(file_content, run_llm=llm_enabled)

# ---------------------------------------------------------------------------
# TAB 2: History
# ---------------------------------------------------------------------------
with tab_history:
    st.markdown("### URL/Link Analysis history")
    _ensure_history_file()

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            hist_rows = list(csv.DictReader(f))
    except Exception as e:
        hist_rows = []
        st.error(f"Could not read history file: {e}")

    if not hist_rows:
        st.info("No analyses recorded yet. Run an analysis in the **Analyse** tab first.")
    else:
        df = pd.DataFrame(reversed(hist_rows))  # newest first

        def _style_verdict(val) -> str:
            if "PHISHING" in val or "HIGH" in val:
                return "background-color:#f8d7da;color:#721c24;font-weight:bold"
            if "MEDIUM" in val:
                return "background-color:#ffe5b4;color:#856404"
            if "LOW" in val:
                return "background-color:#fff3cd;color:#856404"
            return "background-color:#d4edda;color:#155724"

        styled = df.style.map(_style_verdict, subset=["verdict"])
        st.dataframe(styled, use_container_width=True, height=420)

        st.markdown("#### Visualisations")
        vc1, vc2, vc3 = st.columns(3)

        with vc1:
            st.markdown("**Verdict distribution**")
            st.bar_chart(df["verdict"].value_counts(), color="#1a6fb5")

        with vc2:
            st.markdown("**Risk score over time**")
            score_df = df[["timestamp", "max_score"]].copy()
            score_df["max_score"] = pd.to_numeric(score_df["max_score"], errors="coerce")
            st.line_chart(score_df.set_index("timestamp")["max_score"])

        with vc3:
            st.markdown("**Most common signals**")
            all_sigs = []
            for row in hist_rows:
                val = row.get("signals_fired", "none")
                if val and val != "none":
                    all_sigs.extend(s.strip() for s in val.split(","))
            if all_sigs:
                st.bar_chart(pd.Series(all_sigs).value_counts().head(8), color="#1a6fb5")
            else:
                st.info("No signals recorded yet.")

        st.divider()
        col_dl, col_clr = st.columns(2)
        with col_dl:
            st.download_button(
                "Download history.csv",
                data=HISTORY_FILE.read_bytes(),
                file_name="history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_clr:
            if st.button("Clear history", use_container_width=True):
                with open(HISTORY_FILE, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(HISTORY_COLS)
                st.rerun()




