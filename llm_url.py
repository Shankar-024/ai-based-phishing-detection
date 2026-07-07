#!/usr/bin/env python3
"""LLM-based URL phishing classifier — Ollama (local/free) or OpenAI API.

Backend is selected via the LLM_BACKEND env var ("ollama" default, "openai").
Model names are controlled via OLLAMA_MODEL and OPENAI_MODEL env vars.

Typical usage from app.py:
    from llm_url import llm_classify_url
    result = llm_classify_url(url_string, per_url_signal_dict)
    # result: {"verdict": "phishing"|"suspicious"|"legit",
    #          "confidence": 0.0-1.0, "reasoning": "..."}
"""

import json
import os

# ---------------------------------------------------------------------------
# Backend / model configuration (overridable via env vars or app.py sidebar)
# ---------------------------------------------------------------------------
LLM_BACKEND    = os.environ.get("LLM_BACKEND",    "ollama")
OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL",   "llama3.2")
OPENAI_MODEL   = os.environ.get("OPENAI_MODEL",   "gpt-4o-mini")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are a cybersecurity expert specialising in phishing URL detection. "
    "Respond ONLY with valid JSON — no markdown fences, no extra text."
)

_USER_TEMPLATE = """\
Analyse this URL for phishing indicators:

URL: {url}

Rule-based signals already detected by our analysis engine:
{signals}

Rule-based risk score: {score}/100

Based on the URL structure, domain name, path, the signals above, and your
cybersecurity knowledge, determine:
  1. Is this URL phishing, suspicious, or legitimate?
  2. Your confidence (0.0 = uncertain, 1.0 = certain).
  3. A clear 1-2 sentence explanation of your reasoning.

Respond ONLY with this exact JSON (no extra text):
{{"verdict": "phishing"|"suspicious"|"legit", "confidence": 0.0-1.0, "reasoning": "your explanation"}}\
"""


# ---------------------------------------------------------------------------
# Signal summary builder
# ---------------------------------------------------------------------------

def build_signal_summary(per: dict) -> str:
    """Convert a per-URL signal dict into a human-readable bullet list.

    This is fed into the LLM prompt so the model can reason on top of the
    already-computed rule-based evidence rather than starting from scratch.
    """
    lines = []
    if per.get("is_ip"):
        lines.append("- Raw IPv4 address as host (no domain name)")
    if per.get("is_shortener"):
        lines.append(f"- Known URL shortener ({per.get('domain')}) — real destination hidden")
    if per.get("punycode"):
        lines.append("- Punycode/IDN encoding (xn--) — possible homograph/visual-spoof attack")
    if per.get("suspect_typosquat"):
        lines.append(
            f"- Typosquat/homoglyph: '{per.get('sld')}' resembles "
            f"'{per.get('closest_safe')}' (edit distance {per.get('lev')})"
        )
    if per.get("brand_in_subdomain"):
        lines.append(
            f"- Brand '{per.get('impersonated_brand')}' used as a subdomain of "
            "an attacker-controlled domain"
        )
    if per.get("suspicious_path"):
        kws = ", ".join(per.get("path_keywords", []))
        lines.append(f"- Credential-harvesting path keywords detected: {kws}")
    if per.get("high_entropy"):
        lines.append(
            f"- High domain entropy ({per.get('entropy')}) — "
            "looks algorithmically generated (DGA)"
        )
    if per.get("susp_tld"):
        tld = per.get("domain", "").rsplit(".", 1)[-1]
        lines.append(f"- Suspicious free TLD (.{tld}) — heavily abused by phishers")
    if per.get("excess_subdomains"):
        lines.append("- Excessive subdomain depth — unusual for legitimate sites")
    if per.get("at_in_url"):
        lines.append("- @ symbol in URL netloc — browser ignores everything before @")
    if per.get("http_only"):
        lines.append("- Plain HTTP, no HTTPS — unencrypted, typical of credential pages")
    if per.get("redirect_param"):
        lines.append("- Open redirect parameter (?url=, ?redirect=, etc.)")
    if per.get("numeric_domain"):
        lines.append(
            f"- Numeric-heavy SLD '{per.get('sld')}' — "
            "mass-registration phishing domain pattern"
        )
    if per.get("double_ext"):
        lines.append("- Double file extension in path (e.g. .pdf.exe) — malware disguise")
    if per.get("susp_port"):
        lines.append("- Non-standard port number — unusual for legitimate web services")
    if per.get("hex_domain"):
        lines.append("- Percent-encoded characters in hostname — obfuscation technique")
    if per.get("exact_safe_match"):
        lines.append(f"- Exact match to trusted whitelist: {per.get('domain')}")
    if per.get("long_url"):
        lines.append(
            f"- Very long URL ({per.get('url_length')} chars) — "
            "padding to hide the real domain"
        )
    return "\n".join(lines) if lines else "- No rule-based signals detected"


# ---------------------------------------------------------------------------
# LLM backend callers
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str) -> dict:
    """Send prompt to a local Ollama model. Returns a parsed dict."""
    try:
        import ollama  # pip install ollama
    except ImportError:
        return {
            "error": (
                "ollama package not installed. "
                "Run: pip install ollama  "
                "Then pull a model: ollama pull llama3.2"
            )
        }
    try:
        # Re-read env at call time so sidebar changes take effect immediately
        model = os.environ.get("OLLAMA_MODEL", OLLAMA_MODEL)
        resp  = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            options={"temperature": 0},
        )
        text = resp["message"]["content"].strip()
        # Strip markdown code fences some models add despite instructions
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except json.JSONDecodeError as exc:
        return {"error": f"LLM returned non-JSON output: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Ollama error: {exc}"}


def _call_openai(prompt: str) -> dict:
    """Send prompt to OpenAI API. Returns a parsed dict."""
    try:
        from openai import OpenAI  # pip install openai
    except ImportError:
        return {"error": "openai package not installed. Run: pip install openai"}

    api_key = os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY)
    if not api_key:
        return {"error": "OPENAI_API_KEY environment variable is not set."}

    try:
        model  = os.environ.get("OPENAI_MODEL", OPENAI_MODEL)
        client = OpenAI(api_key=api_key)
        resp   = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError as exc:
        return {"error": f"LLM returned non-JSON output: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"OpenAI error: {exc}"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def llm_classify_url(url: str, per: dict) -> dict:
    """Classify a single URL using an LLM.

    Args:
        url: The URL string to analyse.
        per: The per-URL signal dict produced by url.py's analyze_urls().

    Returns:
        On success:
            {"verdict": "phishing"|"suspicious"|"legit",
             "confidence": float (0.0-1.0),
             "reasoning": str}
        On failure:
            {"error": str}
    """
    prompt = _USER_TEMPLATE.format(
        url=url,
        signals=build_signal_summary(per),
        score=per.get("risk_score", 0),
    )

    backend = os.environ.get("LLM_BACKEND", LLM_BACKEND).lower()
    result  = _call_openai(prompt) if backend == "openai" else _call_ollama(prompt)

    if "error" in result:
        return result

    # Normalise and validate fields
    v = str(result.get("verdict", "suspicious")).lower()
    if v not in {"phishing", "suspicious", "legit"}:
        v = "suspicious"

    try:
        conf = round(min(1.0, max(0.0, float(result.get("confidence", 0.5)))), 2)
    except (TypeError, ValueError):
        conf = 0.5

    return {
        "verdict":    v,
        "confidence": conf,
        "reasoning":  str(result.get("reasoning", "No reasoning provided.")),
    }
