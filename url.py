#!/usr/bin/env python3
"""Extract URLs from CSV email datasets and flag IP links, shorteners, and typosquatting.

Usage:
  python url.py --input <file_or_dir> --output results.csv

The script looks for a `urls` column first. If that column holds only 0/1
or is absent, it falls back to extracting URLs from `body` / `text` fields.

Analysis features (all static — no live HTTP requests):
  - Raw IP address as URL host              → high phishing signal
  - Known URL shortener domains             → destination hidden
  - Punycode / IDN (xn--) domains          → visual spoofing
  - Typosquatting via Levenshtein on SLD   → e.g. paypa1.com ≈ paypal.com
  - Homoglyph normalisation before compare → 0→o, 1→l, rn→m, etc.
  - Brand name in subdomain                → e.g. paypal.attacker.com
  - Suspicious path keywords               → /login /verify /confirm etc.
  - Domain entropy (DGA detection)         → randomly generated domain names
  - Per-URL additive risk score 0-100      → higher = more suspicious
  - Email-level max_risk_score             → worst link wins
"""

import argparse
import csv
import math
import re
from pathlib import Path
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# CSV field-size guard — some email bodies exceed Python's default limit
# ---------------------------------------------------------------------------
try:
    csv.field_size_limit(10_000_000)
except OverflowError:
    csv.field_size_limit(2**31 - 1)

# ---------------------------------------------------------------------------
# Known URL-shortener domains
# A match (exact or subdomain) is treated as HIGH risk because the true
# destination is hidden. Extend this list freely.
# ---------------------------------------------------------------------------
SHORTENERS = {
    "bit.ly", "t.co", "tinyurl.com", "goo.gl", "ow.ly", "is.gd", "tiny.cc",
    "buff.ly", "rb.gy", "v.gd", "tr.im", "cli.gs", "shorte.st", "adf.ly",
    "cut.ly", "cutt.ly", "bitly.com", "lnkd.in", "rebrand.ly", "t.ly",
    "soo.gd", "2.gp", "bl.ink", "short.io", "s.id", "clck.ru", "qr.ae",
    "x.co", "mcaf.ee", "po.st", "snipurl.com", "snurl.com", "short.link",
    "tny.im", "wp.me", "buzurl.com", "u.to",
}

# ---------------------------------------------------------------------------
# Trusted / well-known domains used as the comparison baseline for
# typosquatting detection. Only the registered SLD label is compared
# (see extract_sld()), so "paypal" from "paypal.co.uk" is in the set.
# ---------------------------------------------------------------------------
TOP_SAFE = [
    # --- Search & portals ---
    "google.com", "google.co.uk", "google.com.au", "google.ca", "google.de",
    "google.fr", "google.co.in", "google.co.jp", "google.com.br",
    "bing.com", "yahoo.com", "yahoo.co.jp", "duckduckgo.com", "baidu.com",
    "yandex.ru", "yandex.com", "ask.com", "aol.com",

    # --- Social media ---
    "facebook.com", "instagram.com", "twitter.com", "x.com", "linkedin.com",
    "tiktok.com", "pinterest.com", "tumblr.com", "reddit.com", "snapchat.com",
    "whatsapp.com", "telegram.org", "discord.com", "twitch.tv",
    "ok.ru", "vk.com", "weibo.com", "qq.com", "wechat.com",

    # --- Video & streaming ---
    "youtube.com", "netflix.com", "hulu.com", "disneyplus.com", "hbomax.com",
    "primevideo.com", "peacocktv.com", "paramountplus.com", "crunchyroll.com",
    "vimeo.com", "dailymotion.com", "twitch.tv",

    # --- Tech / software ---
    "microsoft.com", "apple.com", "amazon.com", "amazon.co.uk", "amazon.com.au",
    "amazon.ca", "amazon.de", "amazon.fr", "amazon.co.jp",
    "github.com", "gitlab.com", "bitbucket.org", "stackoverflow.com",
    "adobe.com", "salesforce.com", "oracle.com", "ibm.com", "intel.com",
    "cisco.com", "dell.com", "hp.com", "lenovo.com", "samsung.com",
    "sony.com", "lg.com", "nvidia.com", "amd.com", "qualcomm.com",
    "vmware.com", "atlassian.com", "docker.com", "kubernetes.io",
    "cloudflare.com", "fastly.com", "akamai.com", "cdnjs.com",

    # --- Microsoft services ---
    "live.com", "outlook.com", "hotmail.com", "office.com", "office365.com",
    "onedrive.com", "sharepoint.com", "teams.microsoft.com", "azure.com",
    "msn.com", "bing.com", "skype.com", "xbox.com",

    # --- Google services ---
    "gmail.com", "drive.google.com", "docs.google.com", "maps.google.com",
    "icloud.com", "me.com",

    # --- Cloud & hosting ---
    "aws.amazon.com", "cloud.google.com", "azure.microsoft.com",
    "digitalocean.com", "heroku.com", "netlify.com", "vercel.com",
    "godaddy.com", "namecheap.com", "bluehost.com", "hostgator.com",

    # --- E-commerce & retail ---
    "ebay.com", "ebay.co.uk", "walmart.com", "target.com", "bestbuy.com",
    "costco.com", "homedepot.com", "lowes.com", "wayfair.com", "overstock.com",
    "newegg.com", "bhphotovideo.com", "shopify.com", "etsy.com",
    "alibaba.com", "aliexpress.com", "taobao.com", "jd.com",

    # --- Financial & banking ---
    "paypal.com", "paypal.co.uk", "paypal.com.au",
    "bankofamerica.com", "chase.com", "wellsfargo.com", "citibank.com",
    "capitalone.com", "usbank.com", "tdbank.com", "pnc.com", "regions.com",
    "ally.com", "discover.com", "americanexpress.com", "amex.com",
    "visa.com", "mastercard.com", "stripe.com", "square.com",
    "fidelity.com", "schwab.com", "vanguard.com", "etrade.com",
    "robinhood.com", "coinbase.com", "kraken.com",
    "barclays.co.uk", "lloydsbank.com", "hsbc.com", "hsbc.co.uk",
    "natwest.com", "rbs.co.uk", "santander.co.uk", "halifax.co.uk",
    "commbank.com.au", "westpac.com.au", "nab.com.au", "anz.com.au",

    # --- Productivity & collaboration ---
    "zoom.us", "slack.com", "dropbox.com", "box.com", "notion.so",
    "trello.com", "asana.com", "monday.com", "basecamp.com",
    "confluence.atlassian.com", "jira.atlassian.com",
    "wordpress.com", "squarespace.com", "wix.com", "medium.com",
    "substack.com", "mailchimp.com", "hubspot.com", "zendesk.com",

    # --- News & media ---
    "bbc.co.uk", "bbc.com", "cnn.com", "nytimes.com", "theguardian.com",
    "reuters.com", "apnews.com", "washingtonpost.com", "wsj.com",
    "forbes.com", "bloomberg.com", "businessinsider.com", "techcrunch.com",
    "wired.com", "theverge.com", "engadget.com", "arstechnica.com",
    "nbcnews.com", "abcnews.go.com", "cbsnews.com", "foxnews.com",
    "abc.net.au", "smh.com.au", "theage.com.au",

    # --- Shipping & logistics ---
    "ups.com", "fedex.com", "usps.com", "dhl.com", "auspost.com.au",
    "royalmail.com", "canadapost.ca", "parcelforce.com",

    # --- Government (US) ---
    "irs.gov", "fbi.gov", "cdc.gov", "ssa.gov", "medicare.gov",
    "usa.gov", "whitehouse.gov", "senate.gov", "house.gov",
    "state.gov", "defense.gov", "hhs.gov", "dhs.gov", "fda.gov",
    "ftc.gov", "sec.gov", "treasury.gov", "dol.gov", "ed.gov",

    # --- Government (AU/UK/other) ---
    "gov.uk", "nhs.uk", "hmrc.gov.uk", "police.uk",
    "australia.gov.au", "ato.gov.au", "mygov.au",
    "canada.ca", "gc.ca",

    # --- Education ---
    "mit.edu", "harvard.edu", "stanford.edu", "berkeley.edu", "columbia.edu",
    "yale.edu", "princeton.edu", "cornell.edu", "ox.ac.uk", "cam.ac.uk",
    "coursera.org", "edx.org", "khanacademy.org", "udemy.com", "udacity.com",

    # --- Telecommunications ---
    "att.com", "verizon.com", "tmobile.com", "sprint.com",
    "comcast.com", "charter.com", "cox.com", "frontier.com",
    "bt.com", "sky.com", "virginmedia.com", "o2.co.uk", "vodafone.com",
    "telstra.com.au", "optus.com.au",

    # --- Travel & hospitality ---
    "booking.com", "airbnb.com", "expedia.com", "tripadvisor.com",
    "hotels.com", "kayak.com", "priceline.com", "orbitz.com",
    "united.com", "delta.com", "aa.com", "southwest.com", "britishairways.com",
    "qantas.com", "virginaustralia.com",

    # --- Other popular services ---
    "wikipedia.org", "imgur.com", "spotify.com", "apple.com",
    "soundcloud.com", "bandcamp.com", "deviantart.com",
    "quora.com", "yelp.com", "glassdoor.com", "indeed.com",
    "craigslist.org", "meetup.com", "eventbrite.com",
    "surveymonkey.com", "typeform.com", "docusign.com",
    "lastpass.com", "1password.com", "norton.com", "mcafee.com",
    "malwarebytes.com", "avast.com", "avg.com",
]

# Pre-computed set of SLD labels from TOP_SAFE (populated in main() via
# _build_safe_sld_set). Used for O(1) exact-whitelist lookup.
SAFE_SLDS: set = set()

# ---------------------------------------------------------------------------
# Multi-part TLD table — required to correctly extract the SLD for domains
# like "paypal.co.uk" (SLD = "paypal", not "co").
# ---------------------------------------------------------------------------
MULTI_PART_TLDS = {
    "co.uk", "co.nz", "co.in", "co.jp", "co.za", "co.ke", "co.au",
    "com.au", "com.br", "com.cn", "com.mx", "com.ar", "com.sg",
    "org.uk", "net.uk", "ac.uk", "gov.uk", "me.uk",
    "gov.au", "edu.au", "net.au",
}

# ---------------------------------------------------------------------------
# Homoglyph / confusable-character map.
# Domains are normalised through this table before Levenshtein comparison so
# that "paypa1.com" → "paypal.com" and is caught as a typosquat.
# ---------------------------------------------------------------------------
HOMOGLYPHS = str.maketrans({
    "0": "o",  # zero  → o
    "1": "l",  # one   → l
    "3": "e",  # 3     → e
    "4": "a",  # 4     → a
    "5": "s",  # 5     → s
    "6": "g",  # 6     → g
    "7": "t",  # 7     → t
    "8": "b",  # 8     → b
    "@": "a",  # @     → a
    "!": "i",  # !     → i
    "$": "s",  # $     → s
})

# ---------------------------------------------------------------------------
# Compiled regular expressions
# ---------------------------------------------------------------------------

# Match a bare IPv4 address (used to distinguish IP hosts from domain names)
IP_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")

# Match domain-like tokens in free text (used for bare-domain extraction)
DOMAIN_RE = re.compile(r"\b(?:[a-z0-9][a-z0-9\-]{0,63}\.)+[a-z]{2,63}\b", re.I)

# ---------------------------------------------------------------------------
# Suspicious URL path keywords — their presence in a URL path/query string
# is a strong phishing indicator (credential harvesting pages).
# ---------------------------------------------------------------------------
SUSPICIOUS_PATH_KEYWORDS = {
    "login", "logon", "signin", "sign-in", "verify", "verification",
    "confirm", "account", "update", "secure", "security", "password",
    "banking", "credential", "billing", "payment", "suspend", "suspended",
    "limited", "unlock", "recover", "recovery", "authenticate", "auth",
    "webscr", "cmd=", "dispatch", "resolution", "validate",
}

# Entropy threshold above which a domain SLD is considered suspiciously random.
# Readable words typically score 2.5–3.2; DGA domains often exceed 3.5.
ENTROPY_THRESHOLD = 3.5

# ---------------------------------------------------------------------------
# Suspicious / abused TLDs — free or dirt-cheap registries overwhelmingly
# used in phishing campaigns (Freenom TLDs, cheap generics, etc.).
# ---------------------------------------------------------------------------
SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq",     # Freenom free TLDs — heavily abused
    "xyz", "top", "club", "work",      # cheap mass-registration TLDs
    "click", "link", "live",           # action-oriented TLDs common in phishing
    "online", "site", "website",       # generic low-cost TLDs
    "info", "biz",                     # historically abused generic TLDs
}

# URLs longer than this character count are suspicious (padded to hide domain).
SUSP_URL_LENGTH = 75

# More subdomain labels than this before the SLD is unusual.
# "a.b.c.attacker.com" has 3 labels before the SLD → flagged.
MAX_SUBDOMAIN_DEPTH = 3

# Redirect/forwarding parameter names used to bounce victims through a
# trusted domain before landing on the attacker-controlled page.
REDIRECT_PARAM_RE = re.compile(
    r"[?&](url|redirect|redir|next|return|returnurl|goto|link|target|dest|destination)=",
    re.I,
)

# Long numeric sequences in the SLD signal mass-registration phishing domains.
# e.g. "secure-12345-update" or "amazon-order-98765".
NUMERIC_SLD_RE = re.compile(r"\d{4,}")

# Double-extension pattern: a benign document extension immediately followed by
# an executable/script extension — used to disguise malicious files.
# e.g. "invoice.pdf.exe", "document.docx.js"
DOUBLE_EXT_RE = re.compile(
    r"\.(pdf|doc|docx|xls|xlsx|ppt|pptx|txt|png|jpg|jpeg|gif|zip|rar|7z)"
    r"\.(exe|php|asp|aspx|js|html?|bat|cmd|sh|vbs|ps1)\b",
    re.I,
)

# Standard web ports — any URL specifying a different port is suspicious.
SAFE_PORTS = {80, 443}

# ---------------------------------------------------------------------------
# Risk-score weights (additive, capped to [0, 100])
# These reflect the relative phishing danger of each indicator.
# ---------------------------------------------------------------------------
SCORE_IP               = 35   # raw IPv4 host           — legitimate services rarely use IPs
SCORE_SHORTENER        = 40   # shortener               — destination completely hidden
SCORE_PUNYCODE         = 25   # xn-- / IDN              — classic visual-spoofing technique
SCORE_TYPOSQUAT        = 30   # near-miss SLD           — e.g. "paypa1" vs "paypal"
SCORE_BRAND_SUBDOMAIN  = 35   # brand in subdomain      — e.g. paypal.attacker.com
SCORE_SUSP_PATH        = 15   # suspicious path         — /login /verify /confirm etc.
SCORE_HIGH_ENTROPY     = 20   # high domain entropy     — looks DGA-generated
SCORE_SUSP_TLD         = 20   # suspicious/free TLD     — .tk .ml .xyz etc.
SCORE_EXCESS_SUBDOMAIN = 15   # excessive subdomains    — >3 labels before SLD
SCORE_LONG_URL         = 10   # very long URL           — >75 chars, hides real domain
SCORE_AT_IN_URL        = 30   # @ in URL netloc         — browser ignores pre-@ part
SCORE_HTTP_ONLY        = 10   # plain HTTP not HTTPS    — unencrypted credential page
SCORE_REDIRECT         = 20   # redirect parameter      — ?url=, ?redirect= etc.
SCORE_NUMERIC_DOMAIN   = 15   # numeric-heavy SLD       — e.g. secure-98765-update
SCORE_SAFE_BONUS       = -40  # exact whitelist match   — reduces suspicion score
SCORE_DOUBLE_EXT       = 30   # double file extension   — benign ext + executable ext
SCORE_SUSP_PORT        = 15   # non-standard port       — unusual port for a web service
SCORE_HEX_DOMAIN       = 20   # percent-encoded domain  — %XX chars in hostname


# ===========================================================================
# Helper functions
# ===========================================================================

def _build_safe_sld_set() -> None:
    """Populate SAFE_SLDS with the SLD label of every TOP_SAFE entry.

    Called once at startup so whitelist lookups are O(1) rather than O(n).
    """
    for domain in TOP_SAFE:
        sld = extract_sld(domain)
        if sld:
            SAFE_SLDS.add(sld)


def extract_sld(domain: str) -> str:
    """Return the registered second-level domain (SLD) label of *domain*.

    Handles common multi-part TLDs from MULTI_PART_TLDS so that, e.g.,
    "paypal.co.uk" correctly returns "paypal" rather than "co".

    Examples:
        "paypal.com"       → "paypal"
        "paypal.co.uk"     → "paypal"
        "mail.google.com"  → "google"
        "192.168.1.1"      → ""        (IP addresses return empty string)
    """
    domain = domain.lower().strip(".")
    if IP_RE.match(domain):
        return ""  # IP addresses have no SLD
    if domain.startswith("www."):
        domain = domain[4:]
    parts = domain.split(".")
    if len(parts) < 2:
        return parts[0] if parts else ""
    # Check for a known two-part TLD suffix (e.g. "co.uk")
    if len(parts) >= 3 and ".".join(parts[-2:]) in MULTI_PART_TLDS:
        return parts[-3]
    # Default: the label immediately before the TLD
    return parts[-2]


def normalize_homoglyphs(s: str) -> str:
    """Replace look-alike characters with their ASCII equivalents.

    Applies single-char substitutions from HOMOGLYPHS, then handles
    multi-char confusables ("rn"→"m", "vv"→"w", "cl"→"d").
    This lets us catch substitution-based typosquats before Levenshtein.
    """
    s = s.lower().translate(HOMOGLYPHS)
    # Multi-character visual substitutions
    s = s.replace("rn", "m").replace("vv", "w").replace("cl", "d")
    return s


def levenshtein(a: str, b: str) -> int:
    """Return the Levenshtein edit distance between strings *a* and *b*.

    Uses a single-row DP approach for O(min(m,n)) space.
    """
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def domain_entropy(domain: str) -> float:
    """Compute the Shannon entropy of *domain*'s characters.

    High entropy suggests a randomly generated (DGA) domain name rather than
    a human-readable word.  Typical readable SLDs score 2.5–3.2; values above
    ENTROPY_THRESHOLD (3.5) are treated as suspicious.

    Example:
        "paypal"         → ~2.25  (low, readable)
        "xr7q2kf9"       → ~3.0   (medium)
        "a1b2c3d4e5f6"   → ~3.58  (high, DGA-like)
    """
    if not domain:
        return 0.0
    counts: dict = {}
    for ch in domain:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(domain)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def check_brand_in_subdomain(domain: str) -> tuple:
    """Return (flag, matched_brand) if a trusted brand name appears as a
    subdomain label of *domain*.

    Phishing technique: use the real brand as a subdomain of an attacker-owned
    domain to fool users scanning only the beginning of the URL.

    Examples:
        "paypal.attacker.com"     → (1, "paypal")
        "secure.apple.phish.net" → (1, "apple")
        "paypal.com"              → (0, "")   ← legitimate, not a subdomain spoof

    Strategy:
      1. Find the position of the registered SLD in the label list.
      2. Check every label *before* the SLD (i.e. actual subdomains).
      3. Flag if any subdomain label exactly matches a known trusted SLD OR
         is within Levenshtein distance 1 (catches misspellings like "paypa1").
    """
    parts = domain.lower().split(".")
    sld = extract_sld(domain)
    if not sld or sld not in parts:
        return 0, ""

    sld_index = parts.index(sld)
    subdomain_labels = parts[:sld_index]  # labels to the left of the SLD

    for label in subdomain_labels:
        # Exact match against trusted SLDs
        if label in SAFE_SLDS:
            return 1, label
        # Near-miss (lev ≤ 1) — catches homoglyph subdomains like "paypa1"
        norm_label = normalize_homoglyphs(label)
        for safe_sld in SAFE_SLDS:
            if levenshtein(norm_label, safe_sld) <= 1:
                return 1, safe_sld
    return 0, ""


def check_suspicious_path(url: str) -> tuple:
    """Return (flag, matched_keywords) if the URL path or query string
    contains keywords commonly found on phishing credential-harvesting pages.

    Only the path and query portions are inspected — the domain is handled by
    the other detectors.  Matching is case-insensitive substring search.

    Examples:
        "https://evil.com/login.php"           → (1, ["login"])
        "https://evil.com/webscr?cmd=_login"   → (1, ["webscr", "cmd=", "login"])
        "https://google.com/search?q=weather"  → (0, [])
    """
    try:
        p = urlparse(url)
        # Combine path and query for a single scan
        target = (p.path + "?" + p.query).lower()
        found = [kw for kw in SUSPICIOUS_PATH_KEYWORDS if kw in target]
        return (1 if found else 0), found
    except Exception:
        return 0, []


def check_suspicious_tld(domain: str) -> bool:
    """Return True if the domain's TLD is in the SUSPICIOUS_TLDS set."""
    parts = domain.lower().rstrip(".").split(".")
    return parts[-1] in SUSPICIOUS_TLDS if parts else False


def check_excess_subdomains(domain: str) -> bool:
    """Return True if the domain has more subdomain labels than MAX_SUBDOMAIN_DEPTH.

    e.g. "a.b.c.attacker.com" has 3 subdomain labels → flagged.
    """
    sld = extract_sld(domain)
    if not sld:
        return False
    parts = domain.lower().split(".")
    try:
        sld_idx = parts.index(sld)
    except ValueError:
        return False
    return sld_idx > MAX_SUBDOMAIN_DEPTH


def check_at_in_url(url: str) -> bool:
    """Return True if the URL netloc contains '@'.

    Browsers silently ignore everything before '@' in a URL, so
    http://paypal.com@evil.com actually navigates to evil.com.
    """
    try:
        return "@" in urlparse(url).netloc
    except Exception:
        return False


def check_redirect_param(url: str) -> bool:
    """Return True if the URL query string contains a redirect/forwarding parameter.

    e.g. http://legit.com/click?url=http://evil.com bounces victims through
    a trusted domain before landing on the attacker's page.
    """
    try:
        query = urlparse(url).query or ""
        return bool(REDIRECT_PARAM_RE.search("?" + query))
    except Exception:
        return False


def check_numeric_domain(sld: str) -> bool:
    """Return True if the SLD contains a 4+-digit numeric sequence.

    Mass-registration phishing patterns pad random numbers into domain names,
    e.g. "secure-12345-update" or "amazon-order-98765".
    """
    return bool(NUMERIC_SLD_RE.search(sld))


def check_double_extension(url: str) -> bool:
    """Return True if the URL path ends with a double extension (benign + executable).

    e.g. "invoice.pdf.exe" or "document.docx.js" — used to disguise malicious files.
    """
    try:
        return bool(DOUBLE_EXT_RE.search(urlparse(url).path))
    except Exception:
        return False


def check_suspicious_port(url: str) -> bool:
    """Return True if the URL uses a non-standard port (not 80 or 443).

    Legitimate web services almost always serve on port 80 (HTTP) or 443 (HTTPS).
    Any other port is unusual and worth flagging.
    """
    try:
        port = urlparse(url).port
        return port is not None and port not in SAFE_PORTS
    except Exception:
        return False


def check_hex_encoded_domain(url: str) -> bool:
    """Return True if the URL netloc contains percent-encoded (%XX) characters.

    Percent-encoding in the hostname is used to obfuscate the real domain and
    bypass naive string-matching filters.
    """
    try:
        return "%" in urlparse(url).netloc
    except Exception:
        return False


def compute_risk_score(per: dict) -> int:
    """Compute an additive risk score (0–100) for a single URL."""
    score = 0
    if per.get("is_ip"):               score += SCORE_IP
    if per.get("is_shortener"):        score += SCORE_SHORTENER
    if per.get("punycode"):            score += SCORE_PUNYCODE
    if per.get("suspect_typosquat"):   score += SCORE_TYPOSQUAT
    if per.get("brand_in_subdomain"):  score += SCORE_BRAND_SUBDOMAIN
    if per.get("suspicious_path"):     score += SCORE_SUSP_PATH
    if per.get("high_entropy"):        score += SCORE_HIGH_ENTROPY
    if per.get("susp_tld"):            score += SCORE_SUSP_TLD
    if per.get("excess_subdomains"):   score += SCORE_EXCESS_SUBDOMAIN
    if per.get("long_url"):            score += SCORE_LONG_URL
    if per.get("at_in_url"):           score += SCORE_AT_IN_URL
    if per.get("http_only"):           score += SCORE_HTTP_ONLY
    if per.get("redirect_param"):      score += SCORE_REDIRECT
    if per.get("numeric_domain"):      score += SCORE_NUMERIC_DOMAIN
    if per.get("double_ext"):          score += SCORE_DOUBLE_EXT
    if per.get("susp_port"):           score += SCORE_SUSP_PORT
    if per.get("hex_domain"):          score += SCORE_HEX_DOMAIN
    if per.get("exact_safe_match"):    score += SCORE_SAFE_BONUS   # negative
    return max(0, min(100, score))


# ===========================================================================
# URL extraction
# ===========================================================================

def extract_urls(text: str) -> list:
    """Extract all URLs and bare domains from *text*.

    Extraction order (highest confidence first):
      1. Scheme-ful URLs — http:// or https://
      2. href="..." attribute values from HTML email bodies
      3. Bare domain tokens — filtered to skip email addresses

    Bare domains are prefixed with "http://" so that urlparse() works
    uniformly for all returned strings.

    Returns a deduplicated list preserving first-seen order.
    """
    if not text:
        return []

    urls = []

    # 1. Explicit http / https URLs
    urls.extend(re.findall(r"https?://[^\s'\"<>]+", text, flags=re.I))

    # 2. href attribute values (common in HTML-formatted phishing emails)
    for m in re.findall(r'href=[\'"]([^\'"]+)[\'"]', text, flags=re.I):
        urls.append(m)

    # 3. Bare domain tokens not already captured above
    for m in DOMAIN_RE.findall(text):
        # Skip email addresses — the domain after "@" is not a URL
        start = text.lower().find(m.lower())
        if start > 0 and text[start - 1] == "@":
            continue
        # Skip if this domain string is already part of a captured URL
        if any(m.lower() in u.lower() for u in urls):
            continue
        # Prefix with scheme so urlparse() can parse it normally
        urls.append("http://" + m)

    # Deduplicate while preserving first-seen order
    seen: set = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


# ===========================================================================
# Domain utilities
# ===========================================================================

def domain_from_url(url: str) -> str:
    """Parse *url* and return its hostname in lower-case.

    Strips leading "www.", port numbers, and user-info (user@host) fragments.
    Returns an empty string if parsing fails.
    """
    try:
        p = urlparse(url)
        host = p.netloc.split("@")[-1].split(":")[0]
        if host.startswith("www."):
            host = host[4:]
        return host.lower()
    except Exception:
        return ""


def is_ip_host(host: str) -> bool:
    """Return True if *host* is a raw IPv4 address (e.g. "192.168.1.1")."""
    return bool(IP_RE.match(host))


# ===========================================================================
# CSV row processing
# ===========================================================================

def process_row(row: dict, headers: list) -> list:
    """Extract URLs from a single CSV row.

    Strategy:
      - Look for a dedicated URL column ('urls', 'url', 'links').
        If the value is a binary indicator (0 or 1) skip it.
      - Fall back to email body fields ('body', 'text', 'message').

    Returns a deduplicated list of URL strings for this row.
    """
    # Locate a dedicated URL column if one exists
    urls_field = None
    for candidate in ("urls", "url", "links"):
        if candidate in headers:
            urls_field = candidate
            break

    found = []

    # Try the URL column first (skip binary 0/1 indicators)
    if urls_field and row.get(urls_field):
        v = row[urls_field].strip()
        if v not in ("0", "1"):
            found.extend(extract_urls(v))

    # Fall back to raw email body / text fields
    if not found:
        for candidate in ("body", "text", "message"):
            if candidate in headers and row.get(candidate):
                found.extend(extract_urls(row[candidate]))
                if found:
                    break  # stop at the first field that yields URLs

    return list(dict.fromkeys(found))  # deduplicate, preserve order


# ===========================================================================
# Core analysis
# ===========================================================================

def analyze_urls(urls: list) -> dict:
    """Analyse a list of URLs and return aggregated + per-URL phishing signals.

    Aggregated result keys:
      has_ip            – 1 if any URL uses a raw IPv4 host
      shorteners        – comma-separated shortener domains found
      closest_safe      – trusted domain closest to any URL by Levenshtein
      min_lev           – lowest edit distance to a trusted domain SLD
      suspect_typosquat – 1 if any URL is a probable typosquat
      max_risk_score    – highest per-URL risk score (0-100); use as email feature
      per_url           – list of per-URL detail dicts (see below)

    Per-URL detail keys:
      url, domain, sld, is_ip, is_shortener, punycode, exact_safe_match,
      brand_in_subdomain, impersonated_brand, suspicious_path, path_keywords,
      entropy, high_entropy, closest_safe, lev, suspect_typosquat, risk_score
    """
    results = {
        "has_ip": 0,
        "shorteners": set(),
        "closest_safe": "",
        "min_lev": None,
        "suspect_typosquat": 0,
        "max_risk_score": 0,
        "per_url": [],
    }

    for u in urls:
        domain = domain_from_url(u)
        if not domain:
            continue  # skip unparseable URLs

        # Initialise per-URL record with neutral defaults
        per = {
            "url": u,
            "domain": domain,
            "sld": extract_sld(domain),
            "is_ip": 0,
            "is_shortener": 0,
            "punycode": 0,
            "exact_safe_match": 0,
            "brand_in_subdomain": 0,
            "impersonated_brand": "",
            "suspicious_path": 0,
            "path_keywords": [],
            "entropy": 0.0,
            "high_entropy": 0,
            "susp_tld": 0,
            "excess_subdomains": 0,
            "long_url": 0,
            "at_in_url": 0,
            "http_only": 0,
            "redirect_param": 0,
            "numeric_domain": 0,
            "closest_safe": "",
            "lev": None,
            "suspect_typosquat": 0,
            "risk_score": 0,
            # Raw numeric features
            "url_length": 0,
            "special_char_count": 0,
            "digit_count": 0,
            "subdomain_depth": 0,
            "path_depth": 0,
            # New signals
            "double_ext": 0,
            "susp_port": 0,
            "hex_domain": 0,
        }

        # --- IP-address host detection ---
        if is_ip_host(domain):
            results["has_ip"] = 1
            per["is_ip"] = 1

        # --- Punycode / Internationalised Domain Name detection ---
        # "xn--" labels are the ASCII-compatible encoding of Unicode domains
        # and are commonly used in homograph attacks (e.g. pаypal.com in Cyrillic).
        if "xn--" in domain:
            per["punycode"] = 1

        # --- URL shortener detection (exact domain OR subdomain match) ---
        if domain in SHORTENERS or any(
            domain == s or domain.endswith("." + s) for s in SHORTENERS
        ):
            results["shorteners"].add(domain)
            per["is_shortener"] = 1

        # --- Whitelist exact-match at SLD level ---
        # "google-login.com" has SLD "google" which IS in SAFE_SLDS, so it is
        # NOT marked safe here — the typosquat check below will catch it.
        sld = per["sld"]
        if sld and sld in SAFE_SLDS:
            # Only mark safe if the FULL domain also matches a TOP_SAFE entry
            full_match = any(
                domain == d or domain.endswith("." + d) for d in TOP_SAFE
            )
            if full_match:
                per["exact_safe_match"] = 1

        # --- Subdomain brand-impersonation detection ---
        # Check whether a trusted brand name appears as a subdomain label.
        # e.g. "paypal.attacker.com" has subdomain "paypal" → flagged.
        # Legitimate domains like "paypal.com" are excluded (SLD IS the brand).
        bsub, bsub_brand = check_brand_in_subdomain(domain)
        if bsub:
            per["brand_in_subdomain"] = 1
            per["impersonated_brand"] = bsub_brand
            results["suspect_typosquat"] = 1  # treat as overall phishing signal

        # --- Suspicious path / query keyword detection ---
        # Checks for credential-harvesting keywords in the URL path/query.
        spath, skeywords = check_suspicious_path(u)
        if spath:
            per["suspicious_path"] = 1
            per["path_keywords"] = skeywords

        # --- Suspicious TLD ---
        # Free/abused TLDs (.tk, .ml, .xyz, etc.) cost nothing to register and
        # are overwhelmingly used in phishing campaigns.
        if check_suspicious_tld(domain):
            per["susp_tld"] = 1

        # --- Excessive subdomain depth ---
        # More than MAX_SUBDOMAIN_DEPTH labels before the SLD is unusual and
        # is often used to pad out URLs to appear more legitimate.
        if check_excess_subdomains(domain):
            per["excess_subdomains"] = 1

        # --- Long URL ---
        # Very long URLs push the malicious domain out of the visible address bar.
        if len(u) > SUSP_URL_LENGTH:
            per["long_url"] = 1

        # --- @ symbol in URL netloc ---
        # Browsers ignore everything before '@' — so paypal.com@evil.com → evil.com.
        if check_at_in_url(u):
            per["at_in_url"] = 1

        # --- Plain HTTP (not HTTPS) ---
        # A non-shortener, non-IP URL served over HTTP on a path that also has
        # suspicious keywords is a red flag (credential page without encryption).
        if (u.lower().startswith("http://")
                and not per["is_ip"]
                and not per["is_shortener"]):
            per["http_only"] = 1

        # --- Redirect / forwarding parameter ---
        # ?url=, ?redirect=, ?next= etc. bounce victims through a trusted domain.
        if check_redirect_param(u):
            per["redirect_param"] = 1

        # --- Numeric-heavy SLD ---
        # Long digit sequences padded into the SLD signal mass-generated phishing
        # domains (e.g. "amazon-order-98765.com").
        if per["sld"] and check_numeric_domain(per["sld"]):
            per["numeric_domain"] = 1

        # --- Double file extension in path ---
        # e.g. invoice.pdf.exe — benign extension disguising an executable.
        if check_double_extension(u):
            per["double_ext"] = 1

        # --- Non-standard port ---
        # Legitimate web services almost always use port 80 or 443.
        if check_suspicious_port(u):
            per["susp_port"] = 1

        # --- Percent-encoded characters in hostname ---
        # %XX encoding in the netloc obfuscates the real domain.
        if check_hex_encoded_domain(u):
            per["hex_domain"] = 1

        # --- Domain entropy (DGA / randomly generated domain detection) ---
        # Compute Shannon entropy of the SLD; high entropy → looks machine-generated.
        sld_for_entropy = per["sld"] or domain.split(".")[0]
        ent = domain_entropy(sld_for_entropy)
        per["entropy"] = round(ent, 3)
        if ent > ENTROPY_THRESHOLD:
            per["high_entropy"] = 1

        # --- Typosquatting detection via Levenshtein on normalised SLD ---
        # Steps:
        #   1. Normalise homoglyphs in the SLD (paypa1 → paypal)
        #   2. Compare against every trusted SLD
        #   3. Flag as typosquat if distance ≤ adaptive threshold AND dist > 0
        #      (dist == 0 means it IS the safe domain, already handled above)
        sld_norm = normalize_homoglyphs(sld) if sld else ""
        best_domain, best_dist = None, 999

        for safe in TOP_SAFE:
            safe_sld = extract_sld(safe)
            if not safe_sld:
                continue
            dist = levenshtein(sld_norm, safe_sld)
            if dist < best_dist:
                best_dist = dist
                best_domain = safe

        per["closest_safe"] = best_domain or ""
        per["lev"] = best_dist if best_domain else None

        # Adaptive threshold: 20% of safe SLD length, minimum 1.
        # e.g. "paypal" (6 chars) → thresh=1; "microsoft" (9 chars) → thresh=2
        if best_domain and isinstance(best_dist, int):
            safe_sld_len = len(extract_sld(best_domain))
            thresh = max(1, round(0.2 * max(3, safe_sld_len)))

            # Two ways a domain is a typosquat:
            #   1. Homoglyph substitution: normalization made dist==0 but the
            #      ORIGINAL SLD differs (e.g. "paypa1" normalises to "paypal")
            #   2. Edit-distance match: original SLD is within the threshold
            #      but is not an exact match (dist > 0, dist ≤ thresh)
            is_homoglyph_spoof = (best_dist == 0 and sld_norm != sld)
            is_edit_dist_spoof = (0 < best_dist <= thresh)

            if is_homoglyph_spoof or is_edit_dist_spoof:
                per["suspect_typosquat"] = 1
                results["suspect_typosquat"] = 1

        # --- Update aggregated closest-match tracking ---
        if results["min_lev"] is None or (
            isinstance(per["lev"], int) and per["lev"] < results["min_lev"]
        ):
            results["min_lev"] = per["lev"]
            results["closest_safe"] = per["closest_safe"]

        # --- Raw numeric features ---
        per["url_length"] = len(u)
        per["special_char_count"] = sum(c in "@-_=%;/" for c in u)
        per["digit_count"] = sum(c.isdigit() for c in u)
        _parts = domain.lower().split(".")
        _sld = per["sld"]
        per["subdomain_depth"] = _parts.index(_sld) if (_sld and _sld in _parts) else 0
        try:
            per["path_depth"] = len([s for s in urlparse(u).path.split("/") if s])
        except Exception:
            per["path_depth"] = 0

        # --- Compute and track per-URL risk score ---
        per["risk_score"] = compute_risk_score(per)
        if per["risk_score"] > results["max_risk_score"]:
            results["max_risk_score"] = per["risk_score"]

        results["per_url"].append(per)

    # Serialise the shortener set to a sorted comma-separated string
    results["shorteners"] = ",".join(sorted(results["shorteners"]))
    return results


# ===========================================================================
# File discovery
# ===========================================================================

def find_csvs(path: Path) -> list:
    """Return a list of CSV paths under *path*.

    If *path* is a file, returns it directly.
    If *path* is a directory, recurses to find all *.csv files.
    """
    if path.is_file():
        return [path]
    return list(path.rglob("*.csv"))


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyse URLs in CSV email datasets for phishing indicators."
    )
    parser.add_argument("--input", "-i", required=True, help="CSV file or folder")
    parser.add_argument(
        "--output", "-o", default="url_analysis_results.csv",
        help="Output CSV path (default: url_analysis_results.csv)",
    )
    args = parser.parse_args()

    # Build the whitelist SLD lookup set once before processing any files
    _build_safe_sld_set()

    src = Path(args.input)
    files = find_csvs(src)
    if not files:
        print("No CSV files found at", args.input)
        return

    # Output columns — binary flags + raw numeric features for ML pipelines.
    output_cols = [
        "source_file", "row_index", "found_urls",
        "has_ip_url", "shortener_domains",
        "closest_safe_domain", "min_levenshtein",
        "suspect_typosquat",
        "any_brand_in_subdomain",
        "any_suspicious_path",
        "any_high_entropy",
        "max_risk_score",
        # Raw numeric features
        "max_url_length",
        "max_special_char_count",
        "max_digit_count",
        "max_subdomain_depth",
        "max_path_depth",
        "max_entropy_score",
        "any_double_ext",
        "any_susp_port",
        "any_hex_domain",
        "per_url_flags",
    ]

    with open(args.output, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(output_cols)

        for fp in files:
            with open(fp, "r", encoding="utf-8", errors="replace") as in_f:
                reader = csv.DictReader(in_f)
                headers = [h.strip() for h in reader.fieldnames] if reader.fieldnames else []

                for row_idx, row in enumerate(reader):
                    urls = process_row(row, headers)
                    analysis = analyze_urls(urls)

                    # Aggregate new email-level flags across all URLs
                    any_brand_sub  = int(any(p["brand_in_subdomain"] for p in analysis["per_url"]))
                    any_susp_path  = int(any(p["suspicious_path"]     for p in analysis["per_url"]))
                    any_high_ent   = int(any(p["high_entropy"]        for p in analysis["per_url"]))
                    any_double_ext = int(any(p["double_ext"]          for p in analysis["per_url"]))
                    any_susp_port  = int(any(p["susp_port"]           for p in analysis["per_url"]))
                    any_hex_domain = int(any(p["hex_domain"]          for p in analysis["per_url"]))

                    # Aggregate raw numeric features (max across all URLs)
                    max_url_len    = max((p["url_length"]         for p in analysis["per_url"]), default=0)
                    max_spec_chars = max((p["special_char_count"] for p in analysis["per_url"]), default=0)
                    max_digits     = max((p["digit_count"]         for p in analysis["per_url"]), default=0)
                    max_sub_depth  = max((p["subdomain_depth"]     for p in analysis["per_url"]), default=0)
                    max_path_dep   = max((p["path_depth"]          for p in analysis["per_url"]), default=0)
                    max_entropy    = max((p["entropy"]             for p in analysis["per_url"]), default=0.0)

                    # Build compact per-URL flag string:
                    # "https://bit.ly/xyz|shortener,score=40;192.0.0.1/path|ip,score=35"
                    per_flags = []
                    for p in analysis["per_url"]:
                        flags = []
                        if p["is_ip"]:            flags.append("ip")
                        if p["is_shortener"]:     flags.append("shortener")
                        if p["punycode"]:         flags.append("punycode")
                        if p["exact_safe_match"]: flags.append("safe")
                        if p["brand_in_subdomain"]:
                            flags.append(f"brand_sub({p['impersonated_brand']})")
                        if p["suspicious_path"]:
                            flags.append(f"susp_path({'+'.join(p['path_keywords'])})")
                        if p["high_entropy"]:     flags.append(f"entropy({p['entropy']})")
                        if p["suspect_typosquat"]:
                            flags.append(f"typo({p['closest_safe']}:{p['lev']})")
                        if p["susp_tld"]:         flags.append("susp_tld")
                        if p["excess_subdomains"]:flags.append("excess_subdomain")
                        if p["long_url"]:         flags.append("long_url")
                        if p["at_in_url"]:        flags.append("at_in_url")
                        if p["http_only"]:        flags.append("http_only")
                        if p["redirect_param"]:   flags.append("redirect_param")
                        if p["numeric_domain"]:   flags.append("numeric_domain")
                        if p["double_ext"]:        flags.append("double_ext")
                        if p["susp_port"]:         flags.append("susp_port")
                        if p["hex_domain"]:        flags.append("hex_domain")
                        flags.append(f"score={p['risk_score']}")
                        per_flags.append(f"{p['url']}|{','.join(flags)}")

                    writer.writerow([
                        str(fp),
                        row_idx,
                        ";".join(urls),
                        analysis["has_ip"],
                        analysis["shorteners"],
                        analysis["closest_safe"],
                        analysis["min_lev"],
                        analysis["suspect_typosquat"],
                        any_brand_sub,
                        any_susp_path,
                        any_high_ent,
                        analysis["max_risk_score"],
                        max_url_len,
                        max_spec_chars,
                        max_digits,
                        max_sub_depth,
                        max_path_dep,
                        round(max_entropy, 3),
                        any_double_ext,
                        any_susp_port,
                        any_hex_domain,
                        ";".join(per_flags),
                    ])

    print(f"Done. Results written to {args.output}")


if __name__ == "__main__":
    main()

