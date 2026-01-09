from __future__ import annotations

import os
import re
import io
import json
import time
import base64
import difflib
import urllib.parse
from typing import Any, Optional, Tuple, List, Dict

import anyio
import httpx
from fastapi import FastAPI, File, UploadFile, Query, Request, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from PIL import Image, ImageOps, ImageFilter

from google.cloud import vision

# ============================================================
# Performance-first configuration
# ============================================================

# Overall per-request budget (server-side). Keep below iOS client timeout.
# If you set iOS timeout to e.g. 15s, keep this ~10-12s.
REQUEST_BUDGET_SECONDS = float(os.getenv("REQUEST_BUDGET_SECONDS", "10.0"))

# OCR settings
OCR_MODE = (os.getenv("OCR_MODE", "document").strip().lower())  # "document" or "text"
MAX_IMAGE_DIM = int(os.getenv("MAX_IMAGE_DIM", "1700"))  # downscale big photos
OCR_THREADPOOL = True  # keep True: prevents blocking event loop

# Enrichment settings (OpenFoodFacts)
ENABLE_NAME_ENRICH = (os.getenv("ENABLE_NAME_ENRICH", "1").strip() == "1")
ENRICH_MAX_ITEMS = int(os.getenv("ENRICH_MAX_ITEMS", "10"))  # only enrich top N lines
ENRICH_TOTAL_BUDGET_SECONDS = float(os.getenv("ENRICH_TOTAL_BUDGET_SECONDS", "2.2"))
ENRICH_PER_QUERY_TIMEOUT = float(os.getenv("ENRICH_PER_QUERY_TIMEOUT", "1.2"))
ENRICH_MIN_CONF = float(os.getenv("ENRICH_MIN_CONF", "0.60"))
ENRICH_FORCE_BEST_EFFORT = (os.getenv("ENRICH_FORCE_BEST_EFFORT", "1").strip() == "1")
ENRICH_FORCE_SCORE_FLOOR = float(os.getenv("ENRICH_FORCE_SCORE_FLOOR", "0.52"))
ENRICH_CACHE_TTL_SECONDS = int(os.getenv("ENRICH_CACHE_TTL_SECONDS", "86400"))

# OpenFoodFacts endpoints
OFF_SEARCH_URL_US = (os.getenv("OFF_SEARCH_URL_US") or "https://us.openfoodfacts.org/cgi/search.pl").strip()
OFF_SEARCH_URL_WORLD = (os.getenv("OFF_SEARCH_URL_WORLD") or "https://world.openfoodfacts.org/cgi/search.pl").strip()
OFF_PAGE_SIZE = int(os.getenv("OFF_PAGE_SIZE", "18"))
OFF_FIELDS = (os.getenv("OFF_FIELDS") or "product_name,product_name_en,brands,quantity").strip()

# Learned map & pending map
NAME_MAP_JSON = (os.getenv("NAME_MAP_JSON") or "").strip()
NAME_MAP_PATH = (os.getenv("NAME_MAP_PATH") or "/tmp/name_map.json").strip()
PENDING_PATH = (os.getenv("PENDING_PATH") or "/tmp/pending_map.json").strip()
PENDING_ENABLED = (os.getenv("PENDING_ENABLED", "1").strip() == "1")
ADMIN_KEY = (os.getenv("ADMIN_KEY") or "").strip()

# Packshot service
PACKSHOT_SERVICE_URL = (os.getenv("PACKSHOT_SERVICE_URL") or "").strip().rstrip("/")
PACKSHOT_SERVICE_KEY = (os.getenv("PACKSHOT_SERVICE_KEY") or "").strip()

VISION_TMP_PATH = "/tmp/gcloud_key.json"

# ============================================================
# Globals (reused clients)
# ============================================================

OFF_CLIENT: Optional[httpx.AsyncClient] = None
IMG_CLIENT: Optional[httpx.AsyncClient] = None
VISION_CLIENT: Optional[vision.ImageAnnotatorClient] = None

_LEARNED_MAP: Dict[str, str] = {}
_PENDING: Dict[str, Dict[str, Any]] = {}

# Cache: key -> (expires_at, name, source, score)
_ENRICH_CACHE: Dict[str, Tuple[float, str, str, float]] = {}

# Image caches
_IMAGE_CACHE: Dict[str, bytes] = {}
_IMAGE_CONTENT_TYPE_CACHE: Dict[str, str] = {}
_MAX_CACHE_ITEMS = 2000

app = FastAPI()


@app.on_event("startup")
async def startup():
    global OFF_CLIENT, IMG_CLIENT, VISION_CLIENT
    _init_google_credentials_file()

    # Reuse clients (keep-alive) – critical on Render
    OFF_CLIENT = httpx.AsyncClient(
        timeout=httpx.Timeout(ENRICH_PER_QUERY_TIMEOUT, connect=1.0),
        follow_redirects=True,
        limits=httpx.Limits(max_connections=30, max_keepalive_connections=20),
        headers={"User-Agent": "ShelfLife/1.0"},
    )
    IMG_CLIENT = httpx.AsyncClient(
        timeout=httpx.Timeout(12.0, connect=3.0),
        follow_redirects=True,
        limits=httpx.Limits(max_connections=30, max_keepalive_connections=20),
        headers={"User-Agent": "Mozilla/5.0"},
    )

    # Create Vision client ONCE
    try:
        VISION_CLIENT = vision.ImageAnnotatorClient()
    except Exception as e:
        print("Google Vision client init failed:", e)
        VISION_CLIENT = None

    _load_learned_map()
    _load_pending_map()
    print("Startup complete.")


@app.on_event("shutdown")
async def shutdown():
    global OFF_CLIENT, IMG_CLIENT, VISION_CLIENT
    if OFF_CLIENT:
        await OFF_CLIENT.aclose()
    if IMG_CLIENT:
        await IMG_CLIENT.aclose()
    OFF_CLIENT = None
    IMG_CLIENT = None
    VISION_CLIENT = None


# ============================================================
# Google Vision credentials setup
# ============================================================

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
    r"\bregister\b", r"\bcashier\b", r"\bmanager\b",
    r"\breceipt\b", r"\bserved\b", r"\bguest\b", r"\bvisit\b",

    # coupons/discounts (hard terms only)
    r"\bcoupon\b", r"\bdiscount\b", r"\bpromo\b", r"\byou saved\b",

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

UNIT_PRICE_RE = re.compile(r"\b\d+\s*@\s*\$?\s*\d+(?:[.,]\s*\d{1,2})?\b", re.IGNORECASE)

MONEY_TOKEN_RE = re.compile(r"(?:\$?\s*)\b\d{1,6}(?:[.,]\s*\d{2})\b|\b\d{1,6}\s+\d{2}\b")
ONLY_MONEYISH_RE = re.compile(r"^\s*(?:\$?\s*)\d+(?:[.,]\s*\d{2})?\s*$|^\s*\d+\s+\d{2}\s*$")
PRICE_FLAG_RE = re.compile(r"^\s*(?:\$?\s*)\d+(?:[.,]\s*\d{2})\s*[A-Za-z]{1,2}\s*$|^\s*\d+\s+\d{2}\s*[A-Za-z]{1,2}\s*$")

WEIGHT_ONLY_RE = re.compile(r"^\s*\$?\d+(?:[.,]\s*\d+)?\s*(lb|lbs|oz|g|kg|ct)\s*$", re.IGNORECASE)
PER_UNIT_PRICE_RE = re.compile(r"\b\d+(?:[.,]\s*\d+)?\s*/\s*(lb|lbs|oz|g|kg|ea|each|ct)\b", re.IGNORECASE)

STOP_ITEM_WORDS = {"lb", "lbs", "oz", "g", "kg", "ct", "ea", "each", "w", "wt", "weight", "at", "x", "vov"}

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
    if len(s) > 72:
        return False
    return True


def _clean_line(line: str) -> str:
    s = (line or "").strip()
    # strip trailing prices (many OCR variants)
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
# Price-aware keep gate (lookahead hints)
# ============================================================

def _raw_line_has_price_or_qty_hint(s: str) -> bool:
    if not s:
        return False
    if MONEY_TOKEN_RE.search(s):
        return True
    if UNIT_PRICE_RE.search(s):
        return True
    if re.search(r"\b[xX]\s*\d+\b", s) or re.search(r"\b\d+\s*[xX]\b", s):
        return True
    return False


def _is_price_like_line(s: str) -> bool:
    if not s:
        return False
    ss = (s or "").strip()
    if ONLY_MONEYISH_RE.match(ss):
        return True
    if PRICE_FLAG_RE.match(ss):
        return True
    if re.match(r"^\s*-\s*\d+(?:[.,]\s*\d{2})\s*[A-Za-z]{0,2}\s*$", ss):
        return True
    if re.match(r"^\s*\d+\s+\d{2}\s*[A-Za-z]{0,2}\s*$", ss):
        return True
    return False


def _is_weight_or_unit_price_line(s: str) -> bool:
    if not s:
        return False
    ss = (s or "").strip()
    if UNIT_PRICE_RE.search(ss):
        return True
    if PER_UNIT_PRICE_RE.search(ss):
        return True
    if re.search(r"\blb\b", ss, flags=re.IGNORECASE) and ("@" in ss or "/" in ss):
        return True
    return False


# ============================================================
# Store hints
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
    "pblx": "publix", "publx": "publix", "pub": "publix", "pbl": "publix",
    "crm": "cream", "chs": "cheese", "whp": "whipping", "whp.": "whipping",
    "hvy": "heavy", "hvy.": "heavy", "cr": "cream", "chs.": "cheese",
    "bttr": "butter", "marg": "margarine", "yog": "yogurt",
    "veg": "vegetable", "org": "organic", "wb": "whole",
    "grd": "ground", "bf": "beef", "chk": "chicken", "ckn": "chicken",
    "chkn": "chicken", "brst": "breast", "bnls": "boneless",
    "sknls": "skinless", "flr": "flour", "pdr": "powdered",
    "sug": "sugar", "brn": "brown", "van": "vanilla",
    "ext": "extract", "vngr": "vinegar",
    "alc": "alcohol", "iso": "isopropyl", "isoprop": "isopropyl", "isopropyl": "isopropyl",
}

PHRASE_MAP: dict[str, str] = {
    "half and half": "half-and-half",
    "h and h": "half-and-half",
    "hnh": "half-and-half",
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
# Learned map + pending collector
# ============================================================

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


# ============================================================
# Enrichment (OpenFoodFacts) - time budgeted and cached
# ============================================================

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
    _ENRICH_CACHE[key] = (time.time() + ENRICH_CACHE_TTL_SECONDS, name, source, score)


def _build_off_candidate(p: dict[str, Any]) -> str:
    product_name = (p.get("product_name_en") or p.get("product_name") or "").strip()
    brands = (p.get("brands") or "").strip()
    qty = (p.get("quantity") or "").strip()
    if "," in brands:
        brands = brands.split(",")[0].strip()
    return " ".join(x for x in [brands, product_name, qty] if x).strip()


async def _off_search(search_terms: str) -> list[str]:
    global OFF_CLIENT
    if not OFF_CLIENT:
        return []
    params = {
        "search_simple": 1,
        "action": "process",
        "json": 1,
        "page_size": OFF_PAGE_SIZE,
        "sort_by": "unique_scans_n",
        "fields": OFF_FIELDS,
        "search_terms": search_terms,
    }

    out: list[str] = []

    # Try US first, then WORLD if time permits (but keep it minimal)
    for url in [OFF_SEARCH_URL_US, OFF_SEARCH_URL_WORLD]:
        if not url:
            continue
        try:
            r = await OFF_CLIENT.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            products = data.get("products") or []
            for p in products[:OFF_PAGE_SIZE]:
                cand = _build_off_candidate(p)
                if cand:
                    out.append(cand)
            if out:
                return out
        except Exception:
            continue

    return out


async def enrich_full_name(raw_line: str, cleaned_name: str, expanded_name: str, store_hint: str, deadline: float) -> Tuple[str, str, float]:
    """
    Returns (final_name, source, score).
    Must respect the absolute deadline; if time is too low, skip.
    """
    if not ENABLE_NAME_ENRICH:
        return expanded_name, "none", 0.0

    cache_key = dedupe_key(f"{store_hint}:{expanded_name}" if store_hint else expanded_name)
    if cache_key:
        cached = _enrich_cache_get(cache_key)
        if cached:
            return cached[0], cached[1], cached[2]

    # Learned map first (fast)
    learned = _learned_map_lookup(raw_line, expanded_name, store_hint=store_hint)
    if learned:
        if cache_key:
            _enrich_cache_set(cache_key, learned, "learned_map", 1.0)
        return learned, "learned_map", 1.0

    # Deadline guard
    if time.time() + 0.15 >= deadline:
        # Not enough time; skip network
        _pending_add(store_hint, raw_line, cleaned_name, expanded_name)
        if cache_key:
            _enrich_cache_set(cache_key, expanded_name, "none", 0.0)
        return expanded_name, "none", 0.0

    # OFF best match (fast/limited)
    key = dedupe_key(expanded_name)
    if len(key) < 5:
        _pending_add(store_hint, raw_line, cleaned_name, expanded_name)
        if cache_key:
            _enrich_cache_set(cache_key, expanded_name, "none", 0.0)
        return expanded_name, "none", 0.0

    # Build a small set of variants (avoid too many queries)
    variants: list[str] = [expanded_name]
    if store_hint and dedupe_key(store_hint) not in key:
        variants.append(f"{store_hint} {expanded_name}")
    if "publix" in key:
        variants.append(re.sub(r"\bpublix\b", "", key).strip())

    seen = set()
    query_variants: list[str] = []
    for v in variants:
        v = re.sub(r"\s+", " ", (v or "").strip())
        if not v:
            continue
        vl = v.lower()
        if vl in seen:
            continue
        seen.add(vl)
        query_variants.append(v)

    best_name: Optional[str] = None
    best_score = 0.0

    for q in query_variants:
        # deadline guard
        if time.time() + 0.20 >= deadline:
            break

        try:
            # enforce per-query timeout using fail_after
            remaining = max(0.05, min(ENRICH_PER_QUERY_TIMEOUT, deadline - time.time() - 0.05))
            async with anyio.fail_after(remaining):
                candidates = await _off_search(q)
        except TimeoutError:
            candidates = []
        except Exception:
            candidates = []

        for cand in candidates:
            s = _score_candidate(q, cand, store_hint=store_hint)
            if s > best_score:
                best_score = s
                best_name = cand

        # early exit if already strong
        if best_score >= 0.80:
            break

    if best_name:
        if best_score >= ENRICH_MIN_CONF:
            if cache_key:
                _enrich_cache_set(cache_key, best_name, "openfoodfacts", best_score)
            return best_name, "openfoodfacts", best_score

        if ENRICH_FORCE_BEST_EFFORT and best_score >= ENRICH_FORCE_SCORE_FLOOR:
            if cache_key:
                _enrich_cache_set(cache_key, best_name, "openfoodfacts_forced", best_score)
            return best_name, "openfoodfacts_forced", best_score

    _pending_add(store_hint, raw_line, cleaned_name, expanded_name)
    if cache_key:
        _enrich_cache_set(cache_key, expanded_name, "none", 0.0)
    return expanded_name, "none", 0.0


# ============================================================
# OCR (fast + non-blocking)
# ============================================================

def _preprocess_image_bytes(data: bytes) -> bytes:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")

    # Downscale aggressively for speed (receipts don’t need 4k)
    w, h = img.size
    m = max(w, h)
    if m > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / float(m)
        nw, nh = int(w * scale), int(h * scale)
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)

    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)

    out = io.BytesIO()
    # PNG is fine; JPEG is often faster to upload, but Vision accepts both.
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


def _ocr_text_google_vision_sync(image_bytes: bytes) -> str:
    if not VISION_CLIENT:
        raise RuntimeError("Google Vision client not initialized (check credentials).")

    image = vision.Image(content=image_bytes)

    if OCR_MODE == "text":
        resp = VISION_CLIENT.text_detection(image=image)
    else:
        resp = VISION_CLIENT.document_text_detection(image=image)

    if resp.error and resp.error.message:
        raise RuntimeError(resp.error.message)

    if getattr(resp, "full_text_annotation", None) and resp.full_text_annotation and resp.full_text_annotation.text:
        return resp.full_text_annotation.text

    if resp.text_annotations:
        return resp.text_annotations[0].description or ""

    return ""


async def ocr_text_google_vision(image_bytes: bytes) -> str:
    if OCR_THREADPOOL:
        return await anyio.to_thread.run_sync(_ocr_text_google_vision_sync, image_bytes)
    return _ocr_text_google_vision_sync(image_bytes)


# ============================================================
# Routes
# ============================================================

@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True}


class ParsedItem(BaseModel):
    name: str
    quantity: int
    category: str
    image_url: str


@app.post("/parse-receipt", response_model=List[ParsedItem])
@app.post("/parse-receipt/", response_model=List[ParsedItem])
async def parse_receipt(
    request: Request,
    file: UploadFile | None = File(None),
    image: UploadFile | None = File(None),
    only_food: bool = Query(True, description="If true, return only Food items; if false, return Food + Household"),
    debug: bool = Query(False, description="accepted but does not change response shape (use /parse-receipt-debug for wrapper)"),
):
    start = time.time()
    deadline = start + REQUEST_BUDGET_SECONDS

    upload = file or image
    if upload is None:
        raise HTTPException(status_code=422, detail="Missing receipt file field (expected multipart 'file' or 'image').")

    raw = await upload.read()
    if not raw:
        return JSONResponse(status_code=400, content={"error": "Empty file"})

    # Preprocess + OCR (time budget)
    try:
        # preprocess can be CPU heavy -> threadpool
        pre = await anyio.to_thread.run_sync(_preprocess_image_bytes, raw)
        # OCR
        text = await ocr_text_google_vision(pre)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": f"OCR failed: {str(e)}",
                "hint": "Check GOOGLE_APPLICATION_CREDENTIALS_JSON / GOOGLE_APPLICATION_CREDENTIALS",
            },
        )

    raw_lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    store_hint = detect_store_hint(raw_lines)

    # Find earliest subtotal/total marker (safe boundary for item zone)
    total_marker_idx: int | None = None
    for idx, ln in enumerate(raw_lines):
        if re.search(r"\b(sub\s*total|subtotal|total)\b", ln or "", flags=re.IGNORECASE):
            total_marker_idx = idx
            break

    dropped_lines: list[dict[str, Any]] = []

    # Filtering pass
    keep: list[tuple[int, str]] = []
    skip_next_value = False
    store_hint_lc = (store_hint or "").lower().strip()

    for i, ln in enumerate(raw_lines):
        s = (ln or "").strip()
        if not s:
            continue

        # drop store name line if detected (prevents PUBLIX becoming item)
        if store_hint_lc and store_hint_lc in s.lower() and len(s) <= 40:
            dropped_lines.append({"line": s, "stage": "store_hint"})
            continue

        if skip_next_value:
            if ONLY_MONEYISH_RE.match(s) or PRICE_FLAG_RE.match(s):
                dropped_lines.append({"line": s, "stage": "noise_value"})
                skip_next_value = False
                continue
            skip_next_value = False

        if _is_header_or_address(s):
            dropped_lines.append({"line": s, "stage": "junk_gate"})
            continue

        if NOISE_RE.search(s):
            dropped_lines.append({"line": s, "stage": "junk_gate"})
            skip_next_value = True
            continue

        keep.append((i, s))

    # Candidate extraction with lookahead pairing
    kept: list[tuple[str, str]] = []  # (raw_item_line, cleaned_item_line)
    i = 0
    while i < len(keep):
        raw_idx, cur = keep[i]
        cur = cur.strip()

        # never treat pure price/flag/weight lines as items
        if _is_price_like_line(cur) or WEIGHT_ONLY_RE.match(cur) or _is_weight_or_unit_price_line(cur):
            i += 1
            continue

        if not _looks_like_item(cur):
            dropped_lines.append({"line": cur, "stage": "looks_like_item_raw"})
            i += 1
            continue

        has_hint = _raw_line_has_price_or_qty_hint(cur)
        consume = 0

        n1 = keep[i + 1][1].strip() if i + 1 < len(keep) else ""
        n2 = keep[i + 2][1].strip() if i + 2 < len(keep) else ""

        if not has_hint:
            if n1 and (_is_price_like_line(n1) or _is_weight_or_unit_price_line(n1) or _raw_line_has_price_or_qty_hint(n1)):
                has_hint = True
                consume = 1
            elif n1 and _is_weight_or_unit_price_line(n1) and n2 and _is_price_like_line(n2):
                has_hint = True
                consume = 2
            elif n2 and (_is_price_like_line(n2) or _is_weight_or_unit_price_line(n2)):
                has_hint = True
                consume = 2

        if not has_hint:
            before_total = (total_marker_idx is None) or (raw_idx < total_marker_idx)
            if not before_total:
                dropped_lines.append({"line": cur, "stage": "price_hint"})
                i += 1
                continue

        cleaned = _clean_line(cur)
        if not cleaned:
            dropped_lines.append({"line": cur, "stage": "clean_line_empty"})
            i += 1 + consume
            continue

        if not _looks_like_item(cleaned):
            dropped_lines.append({"line": cur, "stage": "looks_like_item_clean", "cleaned": cleaned})
            i += 1 + consume
            continue

        kept.append((cur, cleaned))
        i += 1 + consume

    # Parse initial items (fast path)
    base_items: list[dict[str, Any]] = []
    for raw_ln, cleaned_ln in kept:
        qty, name = _parse_quantity(cleaned_ln)
        name_cleaned = _clean_line(name)
        expanded = expand_abbreviations(name_cleaned)

        if not expanded:
            continue
        if NOISE_RE.search(expanded) or _is_header_or_address(expanded):
            continue
        if not _has_valid_item_words(expanded):
            continue

        base_items.append(
            {
                "raw_line": raw_ln,
                "cleaned_name": name_cleaned,
                "expanded_name": expanded,
                "quantity": int(qty),
            }
        )

    # Enrich names (time-budgeted) – if we’re out of time, skip and return expanded names.
    async def _enrich_one(it: dict[str, Any], enrich_deadline: float) -> dict[str, Any]:
        final = it["expanded_name"]
        source = "none"
        score = 0.0
        try:
            final, source, score = await enrich_full_name(
                raw_line=it["raw_line"],
                cleaned_name=it["cleaned_name"],
                expanded_name=it["expanded_name"],
                store_hint=store_hint,
                deadline=enrich_deadline,
            )
        except Exception:
            final = it["expanded_name"]
            source = "none"
            score = 0.0

        it["final_name"] = (final or "").strip()
        it["enrich_source"] = source
        it["enrich_score"] = score
        return it

    # Compute enrichment deadline (separate budget inside request)
    enrich_deadline = min(deadline, time.time() + ENRICH_TOTAL_BUDGET_SECONDS)

    if ENABLE_NAME_ENRICH and time.time() + 0.25 < enrich_deadline and base_items:
        # only enrich top N to avoid blowing up latency on big receipts
        to_enrich = base_items[: max(0, ENRICH_MAX_ITEMS)]
        rest = base_items[len(to_enrich):]

        sem = anyio.Semaphore(6)

        async def sem_task(it: dict[str, Any]) -> dict[str, Any]:
            async with sem:
                # If we’re too close to deadline, don’t start network
                if time.time() + 0.20 >= enrich_deadline:
                    it["final_name"] = it["expanded_name"]
                    it["enrich_source"] = "none"
                    it["enrich_score"] = 0.0
                    return it
                return await _enrich_one(it, enrich_deadline)

        enriched_results: list[dict[str, Any]] = []
        async with anyio.create_task_group() as tg:
            results_container: list[dict[str, Any]] = []

            async def run_and_collect(item: dict[str, Any]):
                res = await sem_task(item)
                results_container.append(res)

            for it in to_enrich:
                tg.start_soon(run_and_collect, it)

            # task_group waits

            enriched_results = results_container

        # Preserve original order approximately
        enriched_results.sort(key=lambda x: base_items.index(x) if x in base_items else 10**9)

        # Items we didn’t enrich fall back to expanded
        for it in rest:
            it["final_name"] = it["expanded_name"]
            it["enrich_source"] = "skipped"
            it["enrich_score"] = 0.0

        base_items = enriched_results + rest
    else:
        for it in base_items:
            it["final_name"] = it["expanded_name"]
            it["enrich_source"] = "none"
            it["enrich_score"] = 0.0

    # Build final parsed list
    parsed: list[dict[str, Any]] = []
    for it in base_items:
        final_name = (it.get("final_name") or "").strip()
        if not final_name:
            continue
        if ONLY_MONEYISH_RE.match(final_name) or PRICE_FLAG_RE.match(final_name) or WEIGHT_ONLY_RE.match(final_name):
            continue
        if NOISE_RE.search(final_name) or _is_header_or_address(final_name):
            continue
        if not _has_valid_item_words(final_name):
            continue

        category = _classify(final_name)
        if only_food and category != "Food":
            continue

        parsed.append(
            {
                "name": title_case(final_name),
                "quantity": int(it["quantity"]),
                "category": category,
            }
        )

    parsed = _dedupe_and_merge(parsed)

    base_url = _public_base_url(request)
    for it in parsed:
        it["image_url"] = _image_url_for_item(base_url, it["name"])

    return parsed


@app.post("/parse-receipt-debug")
@app.post("/parse-receipt-debug/")
async def parse_receipt_debug(
    request: Request,
    file: UploadFile | None = File(None),
    image: UploadFile | None = File(None),
    only_food: bool = Query(True),
):
    start = time.time()
    items = await parse_receipt(request, file=file, image=image, only_food=only_food, debug=True)

    # Provide lightweight debug info without exploding response size
    return {
        "items": items,
        "timing": {
            "request_budget_seconds": REQUEST_BUDGET_SECONDS,
            "enrich_total_budget_seconds": ENRICH_TOTAL_BUDGET_SECONDS,
            "elapsed_seconds": round(time.time() - start, 3),
        },
        "enrich": {
            "enabled": ENABLE_NAME_ENRICH,
            "max_items": ENRICH_MAX_ITEMS,
            "min_conf": ENRICH_MIN_CONF,
            "forced": ENRICH_FORCE_BEST_EFFORT,
            "forced_floor": ENRICH_FORCE_SCORE_FLOOR,
            "cache_size": len(_ENRICH_CACHE),
            "learned_map_entries": len(_LEARNED_MAP),
            "pending_entries": len(_PENDING),
        },
        "ocr": {
            "mode": OCR_MODE,
            "max_image_dim": MAX_IMAGE_DIM,
        },
    }


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
    except Exception:
        return None


@app.get("/image")
async def get_product_image(
    name: str = Query(...),
    upc: str | None = Query(None),
    product_id: str | None = Query(None),
):
    ck = _cache_key(name, upc, product_id)

    if ck in _IMAGE_CACHE and ck in _IMAGE_CONTENT_TYPE_CACHE:
        return Response(content=_IMAGE_CACHE[ck], media_type=_IMAGE_CONTENT_TYPE_CACHE[ck])

    # Packshot service first
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

    # Fallback map
    key = dedupe_key(name)
    img_url = PRODUCT_IMAGE_MAP.get(key, FALLBACK_PRODUCT_IMAGE)

    result = await fetch_image(img_url)
    if result:
        img_bytes, ctype = result
        _IMAGE_CACHE[ck] = img_bytes
        _IMAGE_CONTENT_TYPE_CACHE[ck] = ctype
        _trim_caches_if_needed()
        return Response(content=img_bytes, media_type=ctype)

    # 1x1 fallback
    tiny_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
    )
    _IMAGE_CACHE[ck] = tiny_bytes
    _IMAGE_CONTENT_TYPE_CACHE[ck] = "image/png"
    _trim_caches_if_needed()
    return Response(content=tiny_bytes, media_type="image/png")


# ============================================================
# Admin endpoints
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
