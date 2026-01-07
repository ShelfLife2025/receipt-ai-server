"""
main.py — ShelfLife receipt + image service (Render-ready)

- Deploy-safe
- /parse-receipt (Google Vision OCR)
- /image (packshot proxy → fallback map → tiny png)
- /health
- /instacart/create-list

Core goals:
1) High-precision line extraction: aggressively filter non-items (totals, headers, addresses, etc.).
2) Parse quantities and clean price/unit artifacts.
3) Expand abbreviations + preserve common multi-word phrases.
4) BEST-EFFORT "full name enrichment" to get closer to branded product names:
   - Learned mapping (store-specific translations) with feedback endpoint + persistence
   - Open Food Facts search (no key required)
   - Confidence gating + caching (never confidently guess wrong, unless you enable FORCE mode)
5) Return only grocery items (Food) by default (existing behavior).

Notes:
- OFF (Open Food Facts) will NEVER be 100% on its own. The learned map loop is what closes the gap.
- This file includes:
  - /admin/pending (see which lines OFF couldn't enrich)
  - /admin/learned-map (inspect learned map size)
  - /admin/feedback (POST corrections to the learned map, secured by ADMIN_KEY)
"""

from __future__ import annotations

import os
import re
import io
import json
import time
import base64
import urllib.parse
import difflib
from typing import Any, Optional, Tuple

import httpx
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, Query, Request, HTTPException
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageOps, ImageFilter

from google.cloud import vision

# ============================================================
# App + Lifespan clients (re-use connections; much faster)
# ============================================================

OFF_CLIENT: Optional[httpx.AsyncClient] = None
IMG_CLIENT: Optional[httpx.AsyncClient] = None

app = FastAPI()


@app.on_event("startup")
async def _startup():
    global OFF_CLIENT, IMG_CLIENT
    OFF_CLIENT = httpx.AsyncClient(
        timeout=float(os.getenv("ENRICH_TIMEOUT_SECONDS", "4.0")),
        follow_redirects=True,
    )
    IMG_CLIENT = httpx.AsyncClient(
        timeout=12.0,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    _init_google_credentials_file()
    _load_learned_map()
    _load_pending_map()
    print("Startup complete.")


@app.on_event("shutdown")
async def _shutdown():
    global OFF_CLIENT, IMG_CLIENT
    if OFF_CLIENT:
        await OFF_CLIENT.aclose()
    if IMG_CLIENT:
        await IMG_CLIENT.aclose()
    OFF_CLIENT = None
    IMG_CLIENT = None


# ============================================================
# Google Vision setup
# ============================================================

VISION_TMP_PATH = "/tmp/gcloud_key.json"


def _init_google_credentials_file() -> None:
    creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not creds_json:
        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            print("Google creds: using GOOGLE_APPLICATION_CREDENTIALS already set.")
        else:
            print("Google creds: GOOGLE_APPLICATION_CREDENTIALS_JSON not set (OCR will fail).")
        return

    try:
        parsed = json.loads(creds_json)
        with open(VISION_TMP_PATH, "w", encoding="utf-8") as f:
            json.dump(parsed, f)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = VISION_TMP_PATH
        print("Google creds: wrote GOOGLE_APPLICATION_CREDENTIALS_JSON to /tmp and set GOOGLE_APPLICATION_CREDENTIALS.")
    except Exception as e:
        print("Google creds: failed to parse/write credentials JSON:", e)


# ============================================================
# Parsing helpers (noise filters, cleaning, quantity)
# ============================================================

NOISE_PATTERNS = [
    # totals/tenders
    r"\btotal\b", r"\bsub\s*total\b", r"\bsubtotal\b", r"\btax\b", r"\bbalance\b",
    r"\bchange\b", r"\bamount\b", r"\bamnt\b", r"\btender\b", r"\bcash\b",
    r"\bpayment\b", r"\bdebit\b", r"\bcredit\b",
    r"\bvisa\b", r"\bmastercard\b", r"\bamex\b", r"\bdiscover\b",
    r"\bauth\b", r"\bapproval\b", r"\bapproved\b",
    r"\bref(?:erence)?\b", r"\btrace\b", r"\btrx\b", r"\btransaction\b", r"\bterminal\b",
    r"\bcard\b", r"\bchip\b", r"\bpin\b", r"\bsignature\b",

    # store/meta
    r"\bregister\b", r"\bcashier\b", r"\bmanager\b", r"\bstore\b",
    r"\breceipt\b", r"\bserved\b", r"\bguest\b", r"\bvisit\b",
    r"\bmember\b", r"\brewards?\b", r"\bpoints?\b",

    # coupons/discounts
    r"\bcoupon\b", r"\bdiscount\b", r"\bpromo\b", r"\bsave\b", r"\bsavings\b", r"\byou saved\b",
    r"\bclub\b", r"\bdeal\b", r"\boff\b",

    # thank-you/footer
    r"\bthank you\b", r"\bthanks\b", r"\bcome again\b", r"\breturn\b", r"\brefund\b",

    # contact/web
    r"\bphone\b", r"\btel\b", r"\bwww\.", r"\.com\b",

    # common “items count” footer/header
    r"\bitems?\b\s*\d+\b",
    r"^\s*#\s*\d+\s*$",

    # specific junk token seen in scans
    r"\bvov\b",
]
NOISE_RE = re.compile("|".join(f"(?:{p})" for p in NOISE_PATTERNS), re.IGNORECASE)

ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
PHONEISH_RE = re.compile(r"\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b")

ADDR_SUFFIX_RE = re.compile(
    r"\b("
    r"st|street|ave|avenue|rd|road|blvd|boulevard|dr|drive|ln|lane|ct|court|"
    r"pkwy|parkway|hwy|highway|trl|trail|pl|place|cir|circle|way"
    r")\b",
    re.IGNORECASE,
)
DIRECTION_RE = re.compile(r"\b(north|south|east|west|ne|nw|se|sw)\b", re.IGNORECASE)
STATE_RE = re.compile(r"\b(AL|AK|AZ|AR|CA|CO|CT|DC|DE|FL|GA|HI|IA|ID|IL|IN|KS|KY|LA|MA|MD|ME|MI|MN|MO|MS|MT|NC|ND|NE|NH|NJ|NM|NV|NY|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VA|VT|WA|WI|WV|WY)\b")

# Allow dot or comma decimals for OCR
UNIT_PRICE_RE = re.compile(r"\b\d+\s*@\s*\$?\s*\d+(?:[.,]\s*\d{1,2})?\b", re.IGNORECASE)

# --- IMPORTANT FIX: OCR-tolerant money patterns ---
# Matches: 3.99, 3,99, 3. 99, $ 3.99, and "3 99" (OCR drops the decimal).
MONEY_TOKEN_RE = re.compile(
    r"(?:\$?\s*)\b\d{1,6}(?:[.,]\s*\d{2})\b|\b\d{1,6}\s+\d{2}\b"
)
ONLY_MONEYISH_RE = re.compile(
    r"^\s*(?:\$?\s*)\d+(?:[.,]\s*\d{2})?\s*$|^\s*\d+\s+\d{2}\s*$"
)
PRICE_FLAG_RE = re.compile(
    r"^\s*(?:\$?\s*)\d+(?:[.,]\s*\d{2})\s*[A-Za-z]{1,2}\s*$|^\s*\d+\s+\d{2}\s*[A-Za-z]{1,2}\s*$"
)

WEIGHT_ONLY_RE = re.compile(r"^\s*\$?\d+(?:[.,]\s*\d+)?\s*(lb|lbs|oz|g|kg|ct)\s*$", re.IGNORECASE)
PER_UNIT_PRICE_RE = re.compile(r"\b\d+(?:[.,]\s*\d+)?\s*/\s*(lb|lbs|oz|g|kg|ea|each|ct)\b", re.IGNORECASE)

STOP_ITEM_WORDS = {
    "lb", "lbs", "oz", "g", "kg", "ct", "ea", "each",
    "w", "wt", "weight",
    "at", "x",
    "vov",
}

HOUSEHOLD_WORDS = {
    "paper", "towel", "towels", "toilet", "tissue", "napkin", "napkins",
    "detergent", "bleach", "cleaner", "wipes", "wipe", "soap", "dish", "dawn",
    "shampoo", "conditioner", "deodorant", "toothpaste", "floss", "razor",
    "trash", "garbage", "bag", "bags", "foil", "wrap", "parchment",
    "rubbing", "alcohol", "isopropyl", "cotton", "swab", "swabs",
    "battery", "batteries", "lightbulb", "lighter", "matches",
    "pet", "litter",
}


def dedupe_key(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def title_case(s: str) -> str:
    small = {"and", "or", "of", "the", "a", "an", "to", "in", "on", "for", "with"}
    words = [w for w in re.split(r"\s+", (s or "").strip()) if w]

    def cap_word(w: str, is_first: bool) -> str:
        raw = w.strip()
        if not raw:
            return raw
        if raw.isalpha() and raw.upper() == raw and 2 <= len(raw) <= 5:
            return raw

        lw = raw.lower()
        if not is_first and lw in small:
            return lw

        if "-" in raw:
            parts = [p for p in raw.split("-") if p != ""]
            outp: list[str] = []
            for i, p in enumerate(parts):
                pl = p.lower()
                if i != 0 and pl in small:
                    outp.append(pl)
                else:
                    outp.append(pl[:1].upper() + pl[1:])
            return "-".join(outp)

        if "'" in raw:
            parts = raw.split("'")
            base = parts[0].lower()
            base = base[:1].upper() + base[1:] if base else base
            rest = "'".join(parts[1:])
            return base + ("'" + rest.lower() if rest else "")

        return lw[:1].upper() + lw[1:]

    out: list[str] = []
    for i, w in enumerate(words):
        out.append(cap_word(w, is_first=(i == 0)))
    return " ".join(out).strip()


def _is_header_or_address(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return True
    if PHONEISH_RE.search(s):
        return True
    if ZIP_RE.search(s):
        return True
    if DATE_RE.search(s) or TIME_RE.search(s):
        return True
    if re.search(r"^\s*\d{2,6}\b", s) and (ADDR_SUFFIX_RE.search(s) or DIRECTION_RE.search(s)):
        return True
    if STATE_RE.search(s):
        words = re.findall(r"[A-Za-z]+", s)
        has_money = bool(MONEY_TOKEN_RE.search(s))
        if 2 <= len(words) <= 12 and not has_money:
            return True
    return False


def _is_junk_line_gate(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return True
    if len(s) < 3:
        return True
    if not re.search(r"[A-Za-z]", s):
        return True
    if NOISE_RE.search(s):
        return True
    if _is_header_or_address(s):
        return True
    if ONLY_MONEYISH_RE.match(s):
        return True
    if PRICE_FLAG_RE.match(s):
        return True
    if WEIGHT_ONLY_RE.match(s):
        return True
    letters = len(re.findall(r"[A-Za-z]", s))
    digits = len(re.findall(r"\d", s))
    if digits >= 4 and letters <= 2:
        return True
    return False


def _has_valid_item_words(line: str) -> bool:
    words = [w.lower() for w in re.findall(r"[A-Za-z]{2,}", line or "")]
    words = [w for w in words if w not in STOP_ITEM_WORDS]
    return len(words) > 0


def _looks_like_item(line: str) -> bool:
    if not line:
        return False
    s = (line or "").strip()
    if ONLY_MONEYISH_RE.match(s) or PRICE_FLAG_RE.match(s) or WEIGHT_ONLY_RE.match(s):
        return False
    if PER_UNIT_PRICE_RE.search(s) and not _has_valid_item_words(s):
        return False
    if NOISE_RE.search(s) or _is_header_or_address(s):
        return False
    letters = len(re.findall(r"[A-Za-z]", s))
    digits = len(re.findall(r"\d", s))
    if letters == 0:
        return False
    if digits >= 4 and letters <= 2:
        return False
    if not _has_valid_item_words(s):
        return False
    if len(s) > 64:
        return False
    return True


def _clean_line(line: str) -> str:
    s = (line or "").strip()

    # --- IMPORTANT FIX: strip OCR-style trailing prices ---
    # Handles: " 3.99", " 3,99", " 3. 99", "$ 3.99", and " 3 99"
    s = re.sub(r"(?:\s+\$?\s*\d{1,6}(?:[.,]\s*\d{2})\s*)\s*$", "", s)
    s = re.sub(r"(?:\s+\d{1,6}\s+\d{2}\s*)\s*$", "", s)

    s = UNIT_PRICE_RE.sub("", s).strip()
    s = re.sub(r"(?:\s+\$?\s*\d{1,6}(?:[.,]\s*\d{2}))+\s*$", "", s).strip()
    s = re.sub(r"\b\d+(?:[.,]\s*\d+)?\s*/\s*(lb|lbs|oz|g|kg|ea|each|ct)\b", "", s, flags=re.IGNORECASE).strip()
    s = s.replace("—", "-")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_quantity(line: str) -> Tuple[int, str]:
    s = (line or "").strip()
    m = re.search(r"(.*?)\b[xX]\s*(\d+)\s*$", s)
    if m:
        return max(int(m.group(2)), 1), m.group(1).strip()
    m = re.match(r"^\s*(\d+)\s*[xX]\s+(.*)$", s)
    if m:
        return max(int(m.group(1)), 1), m.group(2).strip()
    m = re.match(r"^\s*(\d+)\s+(.*)$", s)
    if m:
        qty = int(m.group(1))
        if 1 <= qty <= 50:
            return qty, m.group(2).strip()
    return 1, s


def _classify(name: str) -> str:
    key = dedupe_key(name)
    tokens = set(key.split())
    if tokens & HOUSEHOLD_WORDS:
        return "Household"
    return "Food"


def _dedupe_and_merge(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for it in items:
        k = dedupe_key(it["name"])
        if not k:
            continue
        if k not in merged:
            merged[k] = it
        else:
            merged[k]["quantity"] = int(merged[k]["quantity"]) + int(it["quantity"])
            if merged[k].get("category") == "Household" or it.get("category") == "Household":
                merged[k]["category"] = "Household"
    return sorted(merged.values(), key=lambda x: x["name"].lower())


def _public_base_url(request: Request) -> str:
    xf_proto = request.headers.get("x-forwarded-proto")
    xf_host = request.headers.get("x-forwarded-host")
    if xf_host:
        scheme = xf_proto or request.url.scheme or "https"
        return f"{scheme}://{xf_host}".rstrip("/")
    return str(request.base_url).rstrip("/")


def _image_url_for_item(base_url: str, name: str) -> str:
    qname = urllib.parse.quote((name or "").strip())
    return f"{base_url}/image?name={qname}"


# ============================================================
# Price-aware keep gate (reduces random receipt junk)
# ============================================================

def _raw_line_has_price_or_qty_hint(s: str) -> bool:
    """
    Item lines almost always contain a price token or unit-price token.
    We require one of these hints BEFORE we even try _looks_like_item.
    This removes most random headers, promos, rewards lines, etc.

    IMPORTANT: This is OCR-tolerant via MONEY_TOKEN_RE + UNIT_PRICE_RE.
    """
    if not s:
        return False

    if MONEY_TOKEN_RE.search(s):
        return True

    if UNIT_PRICE_RE.search(s):
        return True

    if re.search(r"\b[xX]\s*\d+\b", s) or re.search(r"\b\d+\s*[xX]\b", s):
        return True

    return False


# ============================================================
# Store hint
# ============================================================

STORE_HINTS: list[tuple[str, re.Pattern]] = [
    ("publix", re.compile(r"\bpublix\b", re.IGNORECASE)),
    ("walmart", re.compile(r"\bwalmart\b|\bwal[-\s]*mart\b", re.IGNORECASE)),
    ("target", re.compile(r"\btarget\b", re.IGNORECASE)),
    ("costco", re.compile(r"\bcostco\b", re.IGNORECASE)),
    ("kroger", re.compile(r"\bkroger\b", re.IGNORECASE)),
    ("aldi", re.compile(r"\baldi\b", re.IGNORECASE)),
    ("whole foods", re.compile(r"\bwhole\s+foods\b", re.IGNORECASE)),
    ("trader joe's", re.compile(r"\btrader\s+joe'?s\b", re.IGNORECASE)),
]


def detect_store_hint(raw_lines: list[str]) -> str:
    blob = " \n ".join(raw_lines[:80]).lower()
    for name, pat in STORE_HINTS:
        if pat.search(blob):
            return name
    return ""


# ============================================================
# Abbrev expansion + phrase preservation
# ============================================================

ABBREV_TOKEN_MAP: dict[str, str] = {
    # store-brand tokens
    "pblx": "publix",
    "publx": "publix",
    "pub": "publix",
    "pbl": "publix",

    # dairy
    "crm": "cream",
    "chs": "cheese",
    "whp": "whipping",
    "whp.": "whipping",
    "hvy": "heavy",
    "hvy.": "heavy",
    "cr": "cream",
    "chs.": "cheese",
    "bttr": "butter",
    "marg": "margarine",
    "yog": "yogurt",

    # staples
    "veg": "vegetable",
    "org": "organic",
    "wb": "whole",
    "grd": "ground",
    "bf": "beef",
    "chk": "chicken",
    "ckn": "chicken",
    "chkn": "chicken",
    "brst": "breast",
    "bnls": "boneless",
    "sknls": "skinless",
    "flr": "flour",
    "pdr": "powdered",
    "sug": "sugar",
    "brn": "brown",
    "van": "vanilla",
    "ext": "extract",
    "vngr": "vinegar",

    # household-ish
    "alc": "alcohol",
    "iso": "isopropyl",
    "isoprop": "isopropyl",
    "isopropyl": "isopropyl",
}

PHRASE_MAP: dict[str, str] = {
    "half and half": "half-and-half",
    "h and h": "half-and-half",
    "hnh": "half-and-half",
    "hf and hf": "half-and-half",

    "heavy whipping cream": "heavy whipping cream",
    "cream cheese": "cream cheese",
    "sour cream": "sour cream",

    "ground beef": "ground beef",
    "chicken breast": "chicken breast",
    "boneless chicken breast": "boneless chicken breast",
    "boneless skinless chicken breast": "boneless skinless chicken breast",

    "olive oil": "olive oil",
    "vegetable oil": "vegetable oil",

    "all purpose flour": "all-purpose flour",
    "powdered sugar": "powdered sugar",
    "brown sugar": "brown sugar",
    "vanilla extract": "vanilla extract",
    "apple cider vinegar": "apple cider vinegar",
    "balsamic vinegar": "balsamic vinegar",
}


def _normalize_for_phrase_match(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def expand_abbreviations(name: str) -> str:
    if not name:
        return ""
    raw = (name or "").strip()
    raw = raw.replace("&", " and ")
    raw = raw.replace("-", " ")
    raw = re.sub(r"[^\w\s]", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()

    toks = [t for t in raw.split(" ") if t]
    expanded: list[str] = []
    for t in toks:
        tl = t.lower()
        expanded.append(ABBREV_TOKEN_MAP.get(tl, tl))

    joined = " ".join(expanded).strip()
    norm = _normalize_for_phrase_match(joined)

    for k in sorted(PHRASE_MAP.keys(), key=lambda x: len(x.split()), reverse=True):
        if k in norm:
            norm = re.sub(rf"\b{re.escape(k)}\b", PHRASE_MAP[k], norm)

    return re.sub(r"\s+", " ", norm).strip()


# ============================================================
# Name enrichment (OFF + Learned Map + Pending collector)
# ============================================================

ENABLE_NAME_ENRICH = (os.getenv("ENABLE_NAME_ENRICH", "1").strip() == "1")

ENRICH_MIN_CONF = float(os.getenv("ENRICH_MIN_CONF", "0.58"))
ENRICH_TIMEOUT_SECONDS = float(os.getenv("ENRICH_TIMEOUT_SECONDS", "4.0"))

ENRICH_FORCE_BEST_EFFORT = (os.getenv("ENRICH_FORCE_BEST_EFFORT", "1").strip() == "1")
ENRICH_FORCE_SCORE_FLOOR = float(os.getenv("ENRICH_FORCE_SCORE_FLOOR", "0.50"))

NAME_MAP_JSON = (os.getenv("NAME_MAP_JSON") or "").strip()
NAME_MAP_PATH = (os.getenv("NAME_MAP_PATH") or "/tmp/name_map.json").strip()

ADMIN_KEY = (os.getenv("ADMIN_KEY") or "").strip()  # set this on Render

_ENRICH_CACHE: dict[str, Tuple[float, str, str, float]] = {}
_ENRICH_CACHE_TTL = int(os.getenv("ENRICH_CACHE_TTL_SECONDS", "86400"))

OFF_SEARCH_URLS: list[str] = [
    (os.getenv("OFF_SEARCH_URL_US") or "https://us.openfoodfacts.org/cgi/search.pl").strip(),
    (os.getenv("OFF_SEARCH_URL_WORLD") or "https://world.openfoodfacts.org/cgi/search.pl").strip(),
]
OFF_PAGE_SIZE = int(os.getenv("OFF_PAGE_SIZE", "30"))
OFF_FIELDS = (os.getenv("OFF_FIELDS") or "product_name,product_name_en,brands,quantity").strip()

_LEARNED_MAP: dict[str, str] = {}

PENDING_PATH = (os.getenv("PENDING_PATH") or "/tmp/pending_map.json").strip()
PENDING_ENABLED = (os.getenv("PENDING_ENABLED", "1").strip() == "1")
_PENDING: dict[str, dict[str, Any]] = {}


def _atomic_write_json(path: str, obj: Any) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _load_learned_map() -> None:
    global _LEARNED_MAP
    _LEARNED_MAP = {}

    if NAME_MAP_JSON:
        try:
            obj = json.loads(NAME_MAP_JSON)
            if isinstance(obj, dict):
                _LEARNED_MAP = {str(k): str(v) for k, v in obj.items()}
                print(f"Learned map: loaded {len(_LEARNED_MAP)} entries from NAME_MAP_JSON")
                return
        except Exception as e:
            print("Learned map: failed to parse NAME_MAP_JSON:", e)

    try:
        if os.path.exists(NAME_MAP_PATH):
            with open(NAME_MAP_PATH, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                _LEARNED_MAP = {str(k): str(v) for k, v in obj.items()}
                print(f"Learned map: loaded {len(_LEARNED_MAP)} entries from {NAME_MAP_PATH}")
    except Exception as e:
        print("Learned map: failed to load file:", e)


def _load_pending_map() -> None:
    global _PENDING
    _PENDING = {}
    try:
        if os.path.exists(PENDING_PATH):
            with open(PENDING_PATH, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                _PENDING = obj
                print(f"Pending map: loaded {len(_PENDING)} entries from {PENDING_PATH}")
    except Exception as e:
        print("Pending map: failed to load:", e)


def _pending_add(store_hint: str, raw_line: str, cleaned: str, expanded: str) -> None:
    if not PENDING_ENABLED:
        return
    key = dedupe_key(f"{store_hint}:{expanded}" if store_hint else expanded)
    if not key:
        return
    rec = _PENDING.get(key) or {
        "store_hint": store_hint,
        "raw_examples": [],
        "cleaned": cleaned,
        "expanded": expanded,
        "count": 0,
        "last_seen": 0,
    }
    rec["count"] = int(rec.get("count", 0)) + 1
    rec["last_seen"] = int(time.time())
    ex = rec.get("raw_examples") or []
    if raw_line and raw_line not in ex:
        ex = (ex + [raw_line])[:8]
    rec["raw_examples"] = ex
    _PENDING[key] = rec
    try:
        _atomic_write_json(PENDING_PATH, _PENDING)
    except Exception as e:
        print("Pending map: write failed:", e)


def _enrich_cache_get(key: str) -> Optional[Tuple[str, str, float]]:
    rec = _ENRICH_CACHE.get(key)
    if not rec:
        return None
    expires_at, name, source, score = rec
    if time.time() > expires_at:
        _ENRICH_CACHE.pop(key, None)
        return None
    return name, source, score


def _enrich_cache_set(key: str, name: str, source: str, score: float) -> None:
    _ENRICH_CACHE[key] = (time.time() + _ENRICH_CACHE_TTL, name, source, score)


_ENRICH_SCORE_STOPWORDS = {
    "the", "and", "or", "of", "a", "an", "with", "for",
    "pack", "ct", "count", "oz", "lb", "lbs", "g", "kg", "ml", "l",
}


def _tokenize_for_score(s: str) -> list[str]:
    k = dedupe_key(s)
    return [t for t in k.split() if t and t not in _ENRICH_SCORE_STOPWORDS]


def _has_any_digits(s: str) -> bool:
    return bool(re.search(r"\d", s or ""))


def _score_candidate(query: str, candidate: str, store_hint: str = "") -> float:
    q = dedupe_key(query)
    c = dedupe_key(candidate)
    if not q or not c:
        return 0.0

    q_toks = _tokenize_for_score(q)
    c_toks = _tokenize_for_score(c)
    if not q_toks or not c_toks:
        return 0.0

    q_set = set(q_toks)
    c_set = set(c_toks)

    overlap = len(q_set & c_set) / max(len(q_set), 1)
    fuzzy = difflib.SequenceMatcher(None, q, c).ratio()

    if overlap < 0.28 and fuzzy < 0.55:
        return 0.0

    score = 0.70 * overlap + 0.30 * fuzzy

    if store_hint:
        sh = dedupe_key(store_hint)
        if sh and sh in c:
            score += 0.06
        if sh and sh in q and sh in c:
            score += 0.04

    if _has_any_digits(query) and _has_any_digits(candidate):
        score += 0.03

    return max(0.0, min(1.0, score))


def _learned_map_lookup(raw_line: str, expanded_name: str, store_hint: str = "") -> Optional[str]:
    raw = (raw_line or "").strip()
    if not raw:
        return None

    base_keys = [
        raw,
        dedupe_key(raw),
        expanded_name,
        dedupe_key(expanded_name),
    ]

    keys: list[str] = []
    for k in base_keys:
        if not k:
            continue
        keys.append(k)
        if store_hint:
            keys.append(f"{store_hint}:{k}")
            keys.append(f"{dedupe_key(store_hint)}:{k}")

    for k in keys:
        v = _LEARNED_MAP.get(k)
        if v and str(v).strip():
            return str(v).strip()
    return None


def _build_off_candidate(p: dict[str, Any]) -> str:
    product_name = (p.get("product_name_en") or p.get("product_name") or "").strip()
    brands = (p.get("brands") or "").strip()
    qty = (p.get("quantity") or "").strip()
    if "," in brands:
        brands = brands.split(",")[0].strip()
    return " ".join(x for x in [brands, product_name, qty] if x).strip()


async def _off_get(url: str, params: dict[str, Any]) -> Optional[dict[str, Any]]:
    global OFF_CLIENT
    if not OFF_CLIENT:
        return None
    try:
        r = await OFF_CLIENT.get(url, params=params, headers={"User-Agent": "ShelfLife/1.0"})
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


async def _openfoodfacts_best_match(name: str, store_hint: str = "") -> Optional[Tuple[str, float, str, str]]:
    key = dedupe_key(name)
    if len(key) < 5:
        return None

    variants: list[str] = [name]

    if store_hint and dedupe_key(store_hint) not in key:
        variants.append(f"{store_hint} {name}")

    if "publix" in key:
        variants.append(re.sub(r"\bpublix\b", "", key).strip())

    toks = [t for t in key.split() if t not in {"publix"}]
    if len(toks) >= 2:
        variants.append(" ".join(toks[:4]))

    seen: set[str] = set()
    query_variants: list[str] = []
    for v in variants:
        v = re.sub(r"\s+", " ", (v or "").strip())
        if not v:
            continue
        if v.lower() in seen:
            continue
        seen.add(v.lower())
        query_variants.append(v)

    params_base = {
        "search_simple": 1,
        "action": "process",
        "json": 1,
        "page_size": OFF_PAGE_SIZE,
        "sort_by": "unique_scans_n",
        "fields": OFF_FIELDS,
    }

    best_name: Optional[str] = None
    best_score = 0.0
    best_url = ""
    best_query = ""

    for q in query_variants:
        params = dict(params_base)
        params["search_terms"] = q

        for url in OFF_SEARCH_URLS:
            if not url:
                continue
            data = await _off_get(url, params=params)
            if not data:
                continue
            products = data.get("products") or []
            if not products:
                continue

            for p in products[:OFF_PAGE_SIZE]:
                cand = _build_off_candidate(p)
                if not cand:
                    continue
                s = _score_candidate(q, cand, store_hint=store_hint)
                if s > best_score:
                    best_score = s
                    best_name = cand
                    best_url = url
                    best_query = q

    if best_name:
        return best_name, best_score, best_url, best_query
    return None


async def enrich_full_name(raw_line: str, cleaned_name: str, expanded_name: str, store_hint: str = "") -> Tuple[str, str, float, str]:
    if not ENABLE_NAME_ENRICH:
        return expanded_name, "none", 0.0, ""

    cache_key = dedupe_key(f"{store_hint}:{expanded_name}" if store_hint else expanded_name)
    if cache_key:
        cached = _enrich_cache_get(cache_key)
        if cached:
            return cached[0], cached[1], cached[2], ""

    learned = _learned_map_lookup(raw_line, expanded_name, store_hint=store_hint)
    if learned:
        if cache_key:
            _enrich_cache_set(cache_key, learned, "learned_map", 1.0)
        return learned, "learned_map", 1.0, ""

    off = await _openfoodfacts_best_match(expanded_name, store_hint=store_hint)
    if off:
        candidate, score, _url, query_used = off

        if score >= ENRICH_MIN_CONF:
            if cache_key:
                _enrich_cache_set(cache_key, candidate, "openfoodfacts", score)
            return candidate, "openfoodfacts", score, query_used

        if ENRICH_FORCE_BEST_EFFORT and score >= ENRICH_FORCE_SCORE_FLOOR:
            if cache_key:
                _enrich_cache_set(cache_key, candidate, "openfoodfacts_forced", score)
            return candidate, "openfoodfacts_forced", score, query_used

    _pending_add(store_hint=store_hint, raw_line=raw_line, cleaned=cleaned_name, expanded=expanded_name)

    if cache_key:
        _enrich_cache_set(cache_key, expanded_name, "none", 0.0)
    return expanded_name, "none", 0.0, ""


# ============================================================
# OCR
# ============================================================

def _preprocess_image_bytes(data: bytes) -> bytes:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)
    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


def _vision_client() -> vision.ImageAnnotatorClient:
    return vision.ImageAnnotatorClient()


def ocr_text_google_vision(image_bytes: bytes) -> str:
    client = _vision_client()
    image = vision.Image(content=image_bytes)
    resp = client.document_text_detection(image=image)

    if resp.error and resp.error.message:
        raise RuntimeError(resp.error.message)

    if resp.full_text_annotation and resp.full_text_annotation.text:
        return resp.full_text_annotation.text

    if resp.text_annotations:
        return resp.text_annotations[0].description or ""

    return ""


# ============================================================
# Routes
# ============================================================

@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True}


@app.post("/parse-receipt")
async def parse_receipt(
    request: Request,
    file: UploadFile = File(...),
    debug: bool = Query(False, description="include debug fields"),
):
    """
    Returns ONLY grocery items (Food) and includes image_url for each item.
    """
    raw = await file.read()
    if not raw:
        return JSONResponse(status_code=400, content={"error": "Empty file"})

    try:
        pre = _preprocess_image_bytes(raw)
        text = ocr_text_google_vision(pre)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": f"OCR failed: {str(e)}",
                "hint": "Check GOOGLE_APPLICATION_CREDENTIALS_JSON / GOOGLE_APPLICATION_CREDENTIALS",
            },
        )

    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    raw_lines = lines[:]

    store_hint = detect_store_hint(raw_lines)

    # Debug: capture why lines are dropped (helps you confirm the scanner is gating correctly)
    dropped_lines: list[dict[str, Any]] = []

    # early junk-line gate (tracked)
    filtered: list[str] = []
    for ln in lines:
        if _is_junk_line_gate(ln):
            if debug:
                dropped_lines.append({"line": ln, "stage": "junk_gate"})
            continue
        filtered.append(ln)
    lines = filtered

    # Keep BOTH raw and cleaned lines, and require price/qty hints on the raw line
    kept: list[tuple[str, str]] = []  # (raw_line, cleaned_line)
    for ln in lines:
        raw_ln = ln

        if not _raw_line_has_price_or_qty_hint(raw_ln):
            if debug:
                dropped_lines.append({"line": raw_ln, "stage": "price_hint"})
            continue

        if not _looks_like_item(raw_ln):
            if debug:
                dropped_lines.append({"line": raw_ln, "stage": "looks_like_item_raw"})
            continue

        cleaned = _clean_line(raw_ln)
        if not cleaned:
            if debug:
                dropped_lines.append({"line": raw_ln, "stage": "clean_line_empty"})
            continue

        if not _looks_like_item(cleaned):
            if debug:
                dropped_lines.append({"line": raw_ln, "stage": "looks_like_item_clean", "cleaned": cleaned})
            continue

        kept.append((raw_ln, cleaned))

    parsed: list[dict[str, Any]] = []
    enrich_debug: list[dict[str, Any]] = []

    for raw_ln, cleaned_ln in kept:
        qty, name = _parse_quantity(cleaned_ln)
        name_cleaned = _clean_line(name)

        expanded = expand_abbreviations(name_cleaned)

        enriched, source, score, query_used = await enrich_full_name(
            raw_line=raw_ln,
            cleaned_name=name_cleaned,
            expanded_name=expanded,
            store_hint=store_hint,
        )

        final_name = (enriched or "").strip()

        if debug:
            enrich_debug.append(
                {
                    "raw_line": raw_ln,
                    "cleaned_line": cleaned_ln,
                    "qty": qty,
                    "name_cleaned": name_cleaned,
                    "name_expanded": expanded,
                    "name_enriched": enriched,
                    "enrich_source": source,
                    "enrich_score": score,
                    "enrich_query_used": query_used,
                }
            )

        if not final_name:
            continue
        if ONLY_MONEYISH_RE.match(final_name) or PRICE_FLAG_RE.match(final_name) or WEIGHT_ONLY_RE.match(final_name):
            continue
        if NOISE_RE.search(final_name) or _is_header_or_address(final_name):
            continue
        if not _has_valid_item_words(final_name):
            continue

        category = _classify(final_name)
        if category != "Food":
            continue

        parsed.append({"name": title_case(final_name), "quantity": int(qty), "category": category})

    parsed = _dedupe_and_merge(parsed)

    base_url = _public_base_url(request)
    for it in parsed:
        it["image_url"] = _image_url_for_item(base_url, it["name"])

    if debug:
        return {
            "items": parsed,
            "raw_line_count": len(raw_lines),
            "kept_line_count": len(kept),
            "kept_lines": [c for (_r, c) in kept][:200],
            "base_url": base_url,
            "store_hint": store_hint,
            "enrichment_debug": enrich_debug[:200],
            "enrich_enabled": ENABLE_NAME_ENRICH,
            "enrich_min_conf": ENRICH_MIN_CONF,
            "enrich_force_best_effort": ENRICH_FORCE_BEST_EFFORT,
            "enrich_force_score_floor": ENRICH_FORCE_SCORE_FLOOR,
            "off_page_size": OFF_PAGE_SIZE,
            "off_search_urls": OFF_SEARCH_URLS,
            "learned_map_entries": len(_LEARNED_MAP),
            "pending_entries": len(_PENDING),
            "debug": {"dropped_lines": dropped_lines[:200]},
        }

    return parsed


# ============================================================
# Instacart list link
# ============================================================

class InstacartLineItem(BaseModel):
    name: str
    quantity: float = 1.0
    unit: str = "each"


class InstacartCreateListRequest(BaseModel):
    title: str = "ShelfLife Shopping List"
    items: list[InstacartLineItem]


@app.post("/instacart/create-list")
@app.post("/instacart/create-list/")
@app.post("/instacart/create_list")
@app.post("/instacart/create_list/")
async def instacart_create_list(req: InstacartCreateListRequest):
    api_key = (os.getenv("INSTACART_API_KEY") or "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing INSTACART_API_KEY env var on server")

    payload = {
        "title": req.title,
        "link_type": "shopping_list",
        "line_items": [{"name": i.name, "quantity": i.quantity, "unit": i.unit} for i in req.items],
    }

    url = "https://connect.instacart.com/idp/v1/products/products_link"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers=headers, json=payload)

    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    data = r.json()
    return {"products_link_url": data.get("products_link_url"), "raw": data}


# ============================================================
# Image delivery
# ============================================================

_IMAGE_CACHE: dict[str, bytes] = {}
_IMAGE_CONTENT_TYPE_CACHE: dict[str, str] = {}
_MAX_CACHE_ITEMS = 2000

PACKSHOT_SERVICE_URL = (os.getenv("PACKSHOT_SERVICE_URL") or "").strip().rstrip("/")
PACKSHOT_SERVICE_KEY = (os.getenv("PACKSHOT_SERVICE_KEY") or "").strip()

PRODUCT_IMAGE_MAP: dict[str, str] = {
    "sargento artisan blends parmesan cheese": "https://images.unsplash.com/photo-1601004890684-d8cbf643f5f2?w=512&q=80",
    "philadelphia cream cheese": "https://images.unsplash.com/photo-1589301763197-9713a1e1e5c0?w=512&q=80",
    "dr pepper": "https://images.unsplash.com/photo-1621451532593-49f463c06d65?w=512&q=80",
    "starbucks nespresso pike place roast": "https://images.unsplash.com/photo-1541167760496-1628856ab772?w=512&q=80",
    "rao's marinara sauce": "https://images.unsplash.com/photo-1611075389455-2f43fa462446?w=512&q=80",
    "bread": "https://images.unsplash.com/photo-1608198093002-de0e3580bb67?w=512&q=80",
    "garlic": "https://images.unsplash.com/photo-1506806732259-39c2d0268443?w=512&q=80",
    "tomatoes": "https://images.unsplash.com/photo-1567306226416-28f0efdc88ce?w=512&q=80",
    "grapes": "https://images.unsplash.com/photo-1601004890684-d8cbf643f5f2?w=512&q=80",
}
FALLBACK_PRODUCT_IMAGE = "https://images.unsplash.com/photo-1604908177071-6c2b7b66010c?w=512&q=80"


def _cache_key(name: str, upc: str | None, product_id: str | None) -> str:
    if product_id and product_id.strip():
        return f"pid:{product_id.strip()}"
    if upc and upc.strip():
        return f"upc:{re.sub(r'[^0-9]', '', upc)}"
    return f"name:{dedupe_key(name)}"


def _trim_caches_if_needed() -> None:
    if len(_IMAGE_CACHE) <= _MAX_CACHE_ITEMS:
        return
    _IMAGE_CACHE.clear()
    _IMAGE_CONTENT_TYPE_CACHE.clear()


async def fetch_image(url: str, headers: dict[str, str] | None = None) -> Optional[Tuple[bytes, str]]:
    global IMG_CLIENT
    if not IMG_CLIENT:
        return None
    try:
        r = await IMG_CLIENT.get(url, headers=headers or {})
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if not ctype.startswith("image/"):
            return None
        return r.content, ctype
    except Exception as e:
        print("fetch_image error:", e)
        return None


@app.get("/image")
async def get_product_image(
    name: str = Query(..., description="product name to fetch image for"),
    upc: str | None = Query(None, description="UPC/GTIN if available"),
    product_id: str | None = Query(None, description="catalog product id if available"),
):
    ck = _cache_key(name, upc, product_id)

    if ck in _IMAGE_CACHE and ck in _IMAGE_CONTENT_TYPE_CACHE:
        return Response(content=_IMAGE_CACHE[ck], media_type=_IMAGE_CONTENT_TYPE_CACHE[ck])

    if PACKSHOT_SERVICE_URL:
        qp: list[str] = []
        if product_id:
            qp.append(f"product_id={urllib.parse.quote(product_id.strip())}")
        if upc:
            qp.append(f"upc={urllib.parse.quote(upc.strip())}")
        if name:
            qp.append(f"name={urllib.parse.quote(name.strip())}")

        url = f"{PACKSHOT_SERVICE_URL}/image"
        if qp:
            url += "?" + "&".join(qp)

        headers = {"User-Agent": "ShelfLife/1.0"}
        if PACKSHOT_SERVICE_KEY:
            headers["Authorization"] = f"Bearer {PACKSHOT_SERVICE_KEY}"

        result = await fetch_image(url, headers=headers)
        if result:
            img_bytes, ctype = result
            _IMAGE_CACHE[ck] = img_bytes
            _IMAGE_CONTENT_TYPE_CACHE[ck] = ctype
            _trim_caches_if_needed()
            return Response(content=img_bytes, media_type=ctype)

    key = dedupe_key(name)
    img_url = PRODUCT_IMAGE_MAP.get(key, FALLBACK_PRODUCT_IMAGE)

    result = await fetch_image(img_url)
    if result:
        img_bytes, ctype = result
        _IMAGE_CACHE[ck] = img_bytes
        _IMAGE_CONTENT_TYPE_CACHE[ck] = ctype
        _trim_caches_if_needed()
        return Response(content=img_bytes, media_type=ctype)

    tiny_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
    )
    _IMAGE_CACHE[ck] = tiny_bytes
    _IMAGE_CONTENT_TYPE_CACHE[ck] = "image/png"
    _trim_caches_if_needed()
    return Response(content=tiny_bytes, media_type="image/png")


# ============================================================
# Admin endpoints (close the “last mile” to 100%)
# ============================================================

def _require_admin(key: str | None) -> None:
    if not ADMIN_KEY:
        raise HTTPException(status_code=500, detail="ADMIN_KEY not set on server")
    if not key or key.strip() != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/admin/learned-map")
async def admin_learned_map(key: str | None = Query(None)):
    _require_admin(key)
    return {"entries": len(_LEARNED_MAP), "path": NAME_MAP_PATH}


@app.get("/admin/pending")
async def admin_pending(key: str | None = Query(None), limit: int = Query(200)):
    """
    Shows unresolved expanded names. These are the ones you convert into learned map entries.
    """
    _require_admin(key)
    items = list(_PENDING.items())
    items.sort(key=lambda kv: int((kv[1] or {}).get("count", 0)), reverse=True)
    out = [{"key": k, **v} for k, v in items[: max(1, min(limit, 2000))]]
    return {"pending": out, "total": len(_PENDING), "path": PENDING_PATH}


class FeedbackBody(BaseModel):
    store_hint: str = ""
    expanded: str
    full_name: str
    raw_example: str = ""


@app.post("/admin/feedback")
async def admin_feedback(body: FeedbackBody, key: str | None = Query(None)):
    """
    Promote a pending entry into the learned map.
    Use keys that match your lookup logic:
      - store_hint:expanded
      - expanded
      - dedupe variants are also looked up automatically
    """
    _require_admin(key)

    expanded = (body.expanded or "").strip()
    full_name = (body.full_name or "").strip()
    store_hint = (body.store_hint or "").strip()

    if not expanded or not full_name:
        raise HTTPException(status_code=400, detail="expanded and full_name required")

    updates: dict[str, str] = {}
    updates[expanded] = full_name
    updates[dedupe_key(expanded)] = full_name
    if store_hint:
        updates[f"{store_hint}:{expanded}"] = full_name
        updates[f"{dedupe_key(store_hint)}:{expanded}"] = full_name
        updates[f"{store_hint}:{dedupe_key(expanded)}"] = full_name
        updates[f"{dedupe_key(store_hint)}:{dedupe_key(expanded)}"] = full_name

    current: dict[str, str] = {}
    try:
        if os.path.exists(NAME_MAP_PATH):
            with open(NAME_MAP_PATH, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                current = {str(k): str(v) for k, v in obj.items()}
    except Exception:
        current = {}

    current.update(updates)
    try:
        _atomic_write_json(NAME_MAP_PATH, current)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write learned map: {e}")

    _load_learned_map()

    pend_key = dedupe_key(f"{store_hint}:{expanded}" if store_hint else expanded)
    if pend_key in _PENDING:
        _PENDING.pop(pend_key, None)
        try:
            _atomic_write_json(PENDING_PATH, _PENDING)
        except Exception:
            pass

    return {"ok": True, "learned_map_entries": len(_LEARNED_MAP)}


# ============================================================
# Logging middleware
# ============================================================

@app.middleware("http")
async def _log_requests(request: Request, call_next):
    try:
        resp = await call_next(request)
        return resp
    finally:
        print(f"{request.method} {request.url.path}")
