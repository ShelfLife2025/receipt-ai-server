from __future__ import annotations

import asyncio
import base64
import difflib
import io
import json
import os
import re
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any, Optional, Tuple, List, Dict

import httpx
from fastapi import FastAPI, File, UploadFile, Query, Request, HTTPException
from fastapi.responses import JSONResponse, Response
from google.cloud import vision
from PIL import Image, ImageOps, ImageFilter
from pydantic import BaseModel

# ============================================================
# FastAPI app
# ============================================================

app = FastAPI()

# ============================================================
# Clients / concurrency controls
# ============================================================

OFF_CLIENT: Optional[httpx.AsyncClient] = None
IMG_CLIENT: Optional[httpx.AsyncClient] = None

OFF_SEM: Optional[asyncio.Semaphore] = None
ENRICH_SEM: Optional[asyncio.Semaphore] = None
OFF_BUDGET_LOCK: Optional[asyncio.Lock] = None

# ============================================================
# Config
# ============================================================

# Overall request time budget (prevents platform timeouts).
# Keep this BELOW your platform gateway timeout.
REQUEST_DEADLINE_SECONDS = float(os.getenv("REQUEST_DEADLINE_SECONDS", "12.0"))

# OpenFoodFacts/OpenProductsFacts per-call timeout.
ENRICH_TIMEOUT_SECONDS = float(os.getenv("ENRICH_TIMEOUT_SECONDS", "1.8"))

# Enrichment behavior
ENABLE_NAME_ENRICH = (os.getenv("ENABLE_NAME_ENRICH", "1").strip() == "1")
ENRICH_MIN_CONF = float(os.getenv("ENRICH_MIN_CONF", "0.58"))

# If true, allow a best-effort OFF/OPF/OBF result below min conf (still bounded by budgets)
ENRICH_FORCE_BEST_EFFORT = (os.getenv("ENRICH_FORCE_BEST_EFFORT", "1").strip() == "1")
ENRICH_FORCE_SCORE_FLOOR = float(os.getenv("ENRICH_FORCE_SCORE_FLOOR", "0.50"))

# Hard cap: how many items per request are allowed to hit external catalogs
MAX_OFF_LOOKUPS_PER_REQUEST = int(os.getenv("MAX_OFF_LOOKUPS_PER_REQUEST", "12"))

# Concurrency
OFF_CONCURRENCY = int(os.getenv("OFF_CONCURRENCY", "4"))
ENRICH_CONCURRENCY = int(os.getenv("ENRICH_CONCURRENCY", "6"))

# Catalog endpoints & query sizing
OFF_SEARCH_URL_US = (os.getenv("OFF_SEARCH_URL_US") or "https://us.openfoodfacts.org/cgi/search.pl").strip()
OFF_SEARCH_URL_WORLD = (os.getenv("OFF_SEARCH_URL_WORLD") or "https://world.openfoodfacts.org/cgi/search.pl").strip()

# Non-food fallbacks (same API shape as OFF)
OPF_SEARCH_URL_WORLD = (os.getenv("OPF_SEARCH_URL_WORLD") or "https://world.openproductsfacts.org/cgi/search.pl").strip()
OBF_SEARCH_URL_WORLD = (os.getenv("OBF_SEARCH_URL_WORLD") or "https://world.openbeautyfacts.org/cgi/search.pl").strip()

OFF_PAGE_SIZE = int(os.getenv("OFF_PAGE_SIZE", "12"))
OFF_FIELDS = (os.getenv("OFF_FIELDS") or "product_name,product_name_en,brands,quantity").strip()

# Learned map / pending
NAME_MAP_JSON = (os.getenv("NAME_MAP_JSON") or "").strip()
NAME_MAP_PATH = (os.getenv("NAME_MAP_PATH") or "/tmp/name_map.json").strip()

PENDING_PATH = (os.getenv("PENDING_PATH") or "/tmp/pending_map.json").strip()
PENDING_ENABLED = (os.getenv("PENDING_ENABLED", "1").strip() == "1")

ADMIN_KEY = (os.getenv("ADMIN_KEY") or "").strip()

# Google Vision creds
VISION_TMP_PATH = "/tmp/gcloud_key.json"

# Image packshot proxy (optional)
PACKSHOT_SERVICE_URL = (os.getenv("PACKSHOT_SERVICE_URL") or "").strip().rstrip("/")
PACKSHOT_SERVICE_KEY = (os.getenv("PACKSHOT_SERVICE_KEY") or "").strip()

# ============================================================
# Startup / shutdown
# ============================================================

@app.on_event("startup")
async def _startup():
    global OFF_CLIENT, IMG_CLIENT, OFF_SEM, ENRICH_SEM, OFF_BUDGET_LOCK
    OFF_CLIENT = httpx.AsyncClient(
        timeout=httpx.Timeout(ENRICH_TIMEOUT_SECONDS),
        follow_redirects=True,
        headers={"User-Agent": "ShelfLife/1.0"},
    )
    IMG_CLIENT = httpx.AsyncClient(
        timeout=httpx.Timeout(10.0),
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    OFF_SEM = asyncio.Semaphore(max(1, OFF_CONCURRENCY))
    ENRICH_SEM = asyncio.Semaphore(max(1, ENRICH_CONCURRENCY))
    OFF_BUDGET_LOCK = asyncio.Lock()

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
# Helpers: safe time budgeting
# ============================================================

@dataclass
class ReqBudget:
    started: float
    deadline: float
    off_used: int = 0

    def remaining(self) -> float:
        return self.deadline - time.monotonic()

    def expired(self) -> bool:
        return self.remaining() <= 0


# ============================================================
# Google Vision setup
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


def _vision_client() -> vision.ImageAnnotatorClient:
    return vision.ImageAnnotatorClient()


def _preprocess_image_bytes(data: bytes, variant: int = 0) -> bytes:
    """
    Reliability-focused preprocessing (receipt OCR):
    Variant 0: balanced (default)
    Variant 1: stronger binarization (fallback)
    """
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")

    # Downscale less aggressively (too much downscale loses small receipt text)
    max_dim = int(os.getenv("OCR_MAX_DIM", "2400"))
    w, h = img.size
    scale = max(w, h) / max_dim if max(w, h) > max_dim else 1.0
    if scale > 1.0:
        img = img.resize((max(1, int(w / scale)), max(1, int(h / scale))), Image.Resampling.LANCZOS)

    img = ImageOps.grayscale(img)

    if variant == 0:
        img = ImageOps.autocontrast(img)
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=160, threshold=3))
    else:
        # Stronger: contrast + median + simple threshold
        img = ImageOps.autocontrast(img)
        img = img.filter(ImageFilter.MedianFilter(size=3))
        img = img.point(lambda p: 255 if p > 160 else 0)

    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


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
# Parsing helpers (noise filters, cleaning, quantity)
# ============================================================

NOISE_PATTERNS = [
    # ---- TARGET / big-box promo/meta blocks (critical) ----
    r"\btarget\s*circle\b", r"\bcircle\b\s*\d+", r"\bregular\s+price\b",
    r"\bexpect\s+more\b", r"\bpay\s+less\b", r"\bexpect\s+more\s+pay\s+less\b",
    r"\bbottle\s+deposit\b", r"\bdeposit\s+fee\b", r"\bbottle\s+deposit\s+fee\b",
    r"^\s*o?target\s*$", r"^\s*otarget\s*$", r"^\s*target\.com\s*$",

    # totals/tenders
    r"\bsub\s*total\b", r"\bsubtotal\b",
    r"\bamount\s+due\b", r"\bbalance\s+due\b", r"\bchange\s+due\b",
    r"\btax\b", r"\bsales\s+tax\b",
    r"\btender\b", r"\bcash\b", r"\bcash\s*back\b", r"\bcashback\b",
    r"\bpayment\b", r"\bdebit\b", r"\bcredit\b",
    r"\bvisa\b", r"\bmastercard\b", r"\bamex\b", r"\bdiscover\b",
    r"\bauth\b", r"\bapproval\b", r"\bapproved\b",
    r"\border\s*total\b", r"\btotal\s+due\b", r"\bgrand\s+total\b",
    r"^\s*total\s*$", r"\bshoppe(?:s)?\b",

    # store/meta / POS metadata
    r"\bregister\b", r"\breg\b", r"\blane\b", r"\bterminal\b", r"\bterm\b", r"\bpos\b",
    r"\bcashier\b", r"\bmanager\b", r"\boperator\b", r"\bop\s*#\b",
    r"\bstore\s*#\b", r"\bst\s*#\b", r"\btrx\b", r"\btransaction\b",
    r"\btrace\b", r"\btrace\s*#\b", r"\bacct\b", r"\bacct\s*#\b",
    r"\binvoice\b", r"\binv\b", r"\border\s*#\b",

    # receipt/footer
    r"\breceipt\b", r"\bserved\b", r"\bguest\b", r"\bvisit\b",
    r"\bthank you\b", r"\bthanks\b", r"\bcome again\b",
    r"\breturn\b", r"\brefund\b", r"\bpolicy\b",

    # coupons/discounts (hard terms only)
    r"\bcoupon\b", r"\bdiscount\b", r"\bpromo\b", r"\bpromotion\b", r"\byou saved\b", r"\bsavings\b",

    # loyalty / points
    r"\bpoints\b", r"\bmember\b", r"\bloyalty\b", r"\bclub\b",

    # contact/web
    r"\bphone\b", r"\btel\b", r"\bwww\.", r"\.com\b",

    # common “items count” footer/header
    r"\bitems?\b\s*\d+\b",
    r"^\s*#\s*\d+\s*$",

    # plaza/location headers
    r"\bshopping\s+center\b",
    r"\bshopping\s+ctr\b",

    # junk token
    r"\bvov\b",

    # Target loyalty / offers (do not treat as items)
    r"\btarget\s*circle\b",
    r"\bcircle\s*(?:offer|deal|rewards?)\b",
    r"\btarget\s*circle\s*\d+\s*%\b",
    r"\b\d+\s*%\s*(?:off|discount)\b",
    r"\b(?:percent|pct)\s*off\b",r"\b(?:target\s*)?circle\s*\d+\b",
r"\b(?:target\s*)?circle\s*\d+\s*%\b",
]
NOISE_RE = re.compile("|".join(f"(?:{p})" for p in NOISE_PATTERNS), re.IGNORECASE)

ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
PHONEISH_RE = re.compile(r"\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b")

ADDR_SUFFIX_RE = re.compile(
    r"\b(st|street|ave|avenue|rd|road|blvd|boulevard|dr|drive|ln|lane|ct|court|"
    r"pkwy|parkway|hwy|highway|trl|trail|pl|place|cir|circle|way)\b",
    re.IGNORECASE,
)
DIRECTION_RE = re.compile(r"\b(north|south|east|west|ne|nw|se|sw)\b", re.IGNORECASE)
STATE_ABBR_RE = re.compile(
    r"\b(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)\b",
    re.IGNORECASE,
)

# Lines like: "ORLANDO FL" or "ALTAMONTE SPRINGS, FL" or "ORLANDO FL 32801"
CITY_STATE_LINE_RE = re.compile(
    r"^\s*[A-Za-z][A-Za-z\s\.'-]{2,}\s*,?\s*"
    r"(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)"
    r"(?:\s+\d{5}(?:-\d{4})?)?\s*$",
    re.IGNORECASE,
)

ADDR_META_RE = re.compile(r"\b(suite|ste|unit|apt|po\s*box|p\.?\s*o\.?\s*box)\b", re.IGNORECASE)

# Allow dot or comma decimals for OCR
UNIT_PRICE_RE = re.compile(r"\b(\d+)\s*@\s*\$?\s*\d+(?:[.,]\s*\d{1,2})?\b", re.IGNORECASE)

# money tokens show up in item lines AND standalone price lines
# NOTE: include OCR-mangled decimals where '.' becomes 'o'/'O' (e.g., "3o5", "3o 5")
MONEY_TOKEN_RE = re.compile(
    r"(?:\$?\s*)\b\d{1,6}(?:[.,]\s*\d{2})\b|(?:\$?\s*)\b\d{1,6}\s*[oO]\s*\d{1,2}\b|\b\d{1,6}\s+\d{2}\b"
)
ONLY_MONEYISH_RE = re.compile(
    r"^\s*(?:\$?\s*)\d+(?:[.,]\s*\d{2})?\s*$|^\s*(?:\$?\s*)\d+\s*[oO]\s*\d{1,2}\s*$|^\s*\d+\s+\d{2}\s*$"
)
PRICE_FLAG_RE = re.compile(
    r"^\s*(?:\$?\s*)\d+(?:[.,]\s*\d{2})\s*[A-Za-z]{1,2}\s*$|^\s*(?:\$?\s*)\d+\s*[oO]\s*\d{1,2}\s*[A-Za-z]{1,2}\s*$|^\s*\d+\s+\d{2}\s*[A-Za-z]{1,2}\s*$"
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

GENERIC_HEADER_WORDS_RE = re.compile(
    r"\b(super\s*markets?|supercenter|market|stores?|wholesale|pharmacy)\b",
    re.IGNORECASE,
)

# Publix-style leading "price + flags" (rare; some OCR layouts flip columns)
LEADING_PRICE_RE = re.compile(r"^\s*(?:\$?\s*)?(?:\d+[.,]\s*\d{2}|\d+\s*[oO]\s*\d{1,2}|\d+\s+\d{2})\s+", re.IGNORECASE)
LEADING_FLAGS_RE = re.compile(r"^\s*(?:(?:[tTfF]\b)\s*){1,4}", re.IGNORECASE)

# Leading item-code / DPCI / internal id prefixes (Target/Walmart, etc.)
LEADING_ITEM_CODE_RE = re.compile(r"^\s*\d{6,14}\s+(?=\S)")
# Long numeric tokens anywhere in the line that are almost certainly internal ids (not sizes like 12pk)
LONG_NUM_TOKEN_RE = re.compile(r"\b\d{6,14}\b")

# Harder store/header suppression anywhere (must be "header-like": no money/qty hints and short)
STORE_WORDS_RE = re.compile(
    r"\b(publix|wal[-\s]*mart|walmart|target|costco|kroger|aldi|whole\s+foods|trader\s+joe'?s)\b",
    re.IGNORECASE,
)

# -------------------------------
# FIX 1: prevent dropping store-brand grocery items.
# If a line contains a store word AND also contains other meaningful product words,
# treat it as an item (unless it's basically JUST a store header).
# -------------------------------
_STORE_TOKENS = {
    "publix", "walmart", "wal", "mart", "target", "costco", "kroger", "aldi", "whole", "foods", "trader", "joe", "joes"
}
_GENERIC_HEADER_TOKENS = {
    "super", "markets", "market", "stores", "store", "wholesale", "pharmacy", "supercenter"
}

_JUNK_EXACT_LINES = {
    "t", "f", "tf", "t f", "ft", "tt", "ff",
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

    # Phone / ZIP / date / time
    if PHONEISH_RE.search(s):
        return True
    if ZIP_RE.search(s):
        return True
    if DATE_RE.search(s) or TIME_RE.search(s):
        return True

    # City + State (+ optional ZIP) lines
    if CITY_STATE_LINE_RE.match(s):
        return True

    # Address meta tokens
    if ADDR_META_RE.search(s):
        return True

    # Street-number lines like "1234 W SOME RD"
    if re.search(r"^\s*\d{2,6}\b", s) and (ADDR_SUFFIX_RE.search(s) or DIRECTION_RE.search(s)):
        return True

    return False


def _is_junk_line(s: str) -> bool:
    key = dedupe_key(s)
    if not key:
        return True
    if key in _JUNK_EXACT_LINES:
        return True
    # A single tiny alpha token like "Tf" should not become an item
    toks = key.split()
    if len(toks) == 1 and toks[0].isalpha() and len(toks[0]) <= 2:
        return True
    return False


def _has_valid_item_words(line: str) -> bool:
    if _is_junk_line(line):
        return False
    words = [w.lower() for w in re.findall(r"[A-Za-z]{2,}", line or "")]
    words = [w for w in words if w not in STOP_ITEM_WORDS]
    return len(words) > 0


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
    if re.match(r"^\s*-\s*\d+\s*[oO]\s*\d{1,2}\s*[A-Za-z]{0,2}\s*$", ss):
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


def _is_store_or_header_line_anywhere(s: str) -> bool:
    """
    Strong filter to remove store-name/header lines wherever they occur,
    but ONLY when they look like headers (no price/qty hints and short).

    FIX: Do NOT drop store-brand grocery lines like "Publix Milk" / "Publix Eggs"
    that contain a store token + additional product tokens.
    """
    if not s:
        return False
    ss = (s or "").strip()
    if not ss:
        return True

    # If there's any price/qty hint, it is not a header
    if _raw_line_has_price_or_qty_hint(ss):
        return False

    key = dedupe_key(ss)
    if not key:
        return True

    toks = key.split()
    if not toks:
        return True

    # If it contains store words, only treat as header if it is basically ONLY store/header tokens.
    if STORE_WORDS_RE.search(ss):
        remaining = [t for t in toks if (t not in _STORE_TOKENS and t not in _GENERIC_HEADER_TOKENS)]
        # If there are meaningful product tokens left, it's an item line, not a header.
        if len(remaining) >= 1:
            return False

        # Otherwise, it is likely a real header line ("publix", "walmart", "publix super markets", etc.)
        if len(toks) <= 8:
            return True

    # Generic header words + very short
    if GENERIC_HEADER_WORDS_RE.search(ss) and len(toks) <= 8:
        return True

    return False


def _strip_long_numeric_tokens(s: str) -> str:
    """
    Remove internal ids like Target DPCI / POS codes from names.
    Keeps normal "size" digits like 12pk / 16oz because those are alnum tokens.
    """
    if not s:
        return ""
    # remove digit-only tokens length 6-14
    s2 = LONG_NUM_TOKEN_RE.sub("", s)
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2


def _clean_line(line: str) -> str:
    """
    Cleans an OCR line into a candidate item name.
    Removes embedded/trailing prices and flags so we return ONLY the name.
    Also strips big-box internal item codes (e.g., Target DPCI prefixes).
    """
    s = (line or "").strip()
    if not s:
        return ""

    # Early discard of obvious junk tokens (prevents "Tf" from surviving later stages)
    if _is_junk_line(s):
        return ""

    # Strip leading price + flags if OCR produced a left-column price.
    if LEADING_PRICE_RE.match(s):
        s = LEADING_PRICE_RE.sub("", s).strip()
        s = LEADING_FLAGS_RE.sub("", s).strip()

    # Strip leading internal item codes (Target/Walmart style)
    s = LEADING_ITEM_CODE_RE.sub("", s).strip()

    # Strip trailing prices (Publix often puts price at end)
    s = re.sub(r"(?:\s+\$?\s*\d{1,6}(?:[.,]\s*\d{2})\s*)\s*$", "", s)
    s = re.sub(r"(?:\s+\$?\s*\d{1,6}\s*[oO]\s*\d{1,2}\s*)\s*$", "", s)
    s = re.sub(r"(?:\s+\d{1,6}\s+\d{2}\s*)\s*$", "", s)

    # Strip possible trailing flags after price (e.g., "2.69 T F" or "6.70 F")
    s = re.sub(r"(?:\s+[tTfF]){1,4}\s*$", "", s)

    # Strip trailing cents-only token (common OCR tail: "... 99")
    s = re.sub(r"\s+\d{2}\s*$", "", s)

    # Strip trailing tax/flag junk
    s = re.sub(r"\s+\b(?:T|F|TF|TX|TAX)\b\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+\b(?:T|F|TF|TX|TAX)\b\s*$", "", s, flags=re.IGNORECASE)

    # Remove unit price artifacts
    s = UNIT_PRICE_RE.sub("", s).strip()
    s = re.sub(r"\b\d+(?:[.,]\s*\d+)?\s*/\s*(lb|lbs|oz|g|kg|ea|each|ct)\b", "", s, flags=re.IGNORECASE).strip()

    s = s.replace("—", "-")
    s = re.sub(r"\s+", " ", s).strip()

    # Remove any remaining internal numeric id tokens anywhere
    s = _strip_long_numeric_tokens(s)

    # Final junk check
    if _is_junk_line(s):
        return ""

    return s


def _parse_quantity(line: str) -> Tuple[int, str]:
    s = (line or "").strip()

    # trailing: "Item x2"
    m = re.search(r"(.*?)\b[xX]\s*(\d+)\s*$", s)
    if m:
        return max(int(m.group(2)), 1), m.group(1).strip()

    # leading: "2x Item"
    m = re.match(r"^\s*(\d+)\s*[xX]\s+(.*)$", s)
    if m:
        return max(int(m.group(1)), 1), m.group(2).strip()

    # leading bare qty: "2 Item"
    m = re.match(r"^\s*(\d+)\s+(.*)$", s)
    if m:
        qty = int(m.group(1))
        if 1 <= qty <= 50:
            return qty, m.group(2).strip()

    return 1, s


def _parse_qty_hint_from_attached_line(s: str) -> int:
    if not s:
        return 1
    m = UNIT_PRICE_RE.search(s)
    if m:
        try:
            q = int(m.group(1))
            if 1 <= q <= 50:
                return q
        except Exception:
            return 1
    return 1


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
# Store hint (used only as a hint; do not drop store-brand items)
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

STORE_HEADER_PATTERNS: dict[str, re.Pattern] = {
    "publix": re.compile(r"^\s*publix(?:\s+super\s*markets?)?\s*$", re.IGNORECASE),
    "walmart": re.compile(r"^\s*wal[-\s]*mart(?:\s+stores?)?\s*$", re.IGNORECASE),
    "target": re.compile(r"^\s*o?target\s*$", re.IGNORECASE),
    "costco": re.compile(r"^\s*costco(?:\s+wholesale)?\s*$", re.IGNORECASE),
    "kroger": re.compile(r"^\s*kroger\s*$", re.IGNORECASE),
    "aldi": re.compile(r"^\s*aldi\s*$", re.IGNORECASE),
    "whole foods": re.compile(r"^\s*whole\s+foods(?:\s+market)?\s*$", re.IGNORECASE),
    "trader joe's": re.compile(r"^\s*trader\s+joe'?s\s*$", re.IGNORECASE),
}


def detect_store_hint(raw_lines: list[str]) -> str:
    blob = " \n ".join(raw_lines[:160]).lower()
    for name, pat in STORE_HINTS:
        if pat.search(blob):
            return name
    return ""


def _is_probable_store_header_line(s: str, store_hint: str, position_idx: int) -> bool:
    if not s:
        return False
    if position_idx > 18:
        return False

    ss = (s or "").strip()
    key = dedupe_key(ss)
    if not key:
        return False

    if store_hint:
        sh = dedupe_key(store_hint)
        if sh and (key == sh or (sh in key and GENERIC_HEADER_WORDS_RE.search(ss) and len(key.split()) <= 6)):
            if _raw_line_has_price_or_qty_hint(ss):
                return False
            return True

    if GENERIC_HEADER_WORDS_RE.search(ss) and not _raw_line_has_price_or_qty_hint(ss) and len(key.split()) <= 7:
        return True

    return False


# ============================================================
# Abbrev expansion + phrase preservation (aim: no abbreviations)
# ============================================================

ABBREV_TOKEN_MAP: dict[str, str] = {
    # store-brand tokens
    "pblx": "publix",
    "publx": "publix",
    "pub": "publix",
    "pbl": "publix",
    "pbx": "publix",
    "Gg": "good and gather"
    "chees": "cheese",

    # common brand/product abbreviations
    "sar": "sargento",
    "arts": "artisan",
    "blnd": "blends",
    "parm": "parmesan",
    "nesp": "nespresso",
    "pke": "pike",
    "sprng": "spring",
    "wht": "white",
    "sdl": "seedless",

    # bakery
    "bg": "bagels",
    "swr": "swirl",
    "swrl": "swirl",

    # dairy
    "crm": "cream",
    "chs": "cheese",
    "whp": "whipping",
    "hvy": "heavy",
    "bttr": "butter",
    "marg": "margarine",
    "yog": "yogurt",

    # staples
    "veg": "vegetable",
    "org": "organic",
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

    # misc receipt shorthand
    "prm": "premium",
    "grl": "grilled",
    "pta": "potato",
    "rst": "roast",
    "rstd": "roasted",
    "ov": "oven",

    # Jack Daniel's style
    "jd": "jack daniel's",
    "tenn": "tennessee",
    "hny": "honey",

    # Publix cheese line
    "sh": "sharp",
    "chd": "cheddar",
    "cut": "cut",
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
    """
    Goal: expand as many receipt abbreviations as possible so returned names
    are not abbreviated even when catalog enrichment cannot run in time.
    """
    if not name:
        return ""
    raw = (name or "").strip()
    raw = raw.replace("&", " and ")
    raw = raw.replace("-", " ")
    raw = re.sub(r"[^\w\s']", " ", raw)  # keep apostrophes for "daniel's"
    raw = re.sub(r"\s+", " ", raw).strip()

    toks = [t for t in raw.split(" ") if t]
    expanded: list[str] = []

    for i, t in enumerate(toks):
        tl = t.lower()
        nxt = toks[i + 1].lower() if i + 1 < len(toks) else ""

        if tl == "cr" and nxt in {"cut", "cr", "ctr"}:
            expanded.append("cracker")
            continue

        expanded.append(ABBREV_TOKEN_MAP.get(tl, tl))

    joined = " ".join(expanded).strip()
    norm = _normalize_for_phrase_match(joined)

    for k in sorted(PHRASE_MAP.keys(), key=lambda x: len(x.split()), reverse=True):
        if k in norm:
            norm = re.sub(rf"\b{re.escape(k)}\b", PHRASE_MAP[k], norm)

    return re.sub(r"\s+", " ", norm).strip()


def post_name_cleanup(name: str) -> str:
    """
    Final, conservative cleanup for OCR leftovers/truncations.
    This runs after expand_abbreviations and after enrichment, and should only
    touch obvious garbage so we don't accidentally change real product names.
    """
    s = (name or "").strip()
    if not s:
        return ""

    low = s.lower().strip()

    if low == "mojo oven roasted h" or (low.startswith("mojo oven roasted") and low.endswith(" h")):
        return "mojo oven roasted chicken"

    toks = [t for t in low.split() if t]

    if toks and toks[-1] in {"sn"}:
        toks = toks[:-1]
        low = " ".join(toks).strip()

    toks = [t for t in low.split() if t]
    if toks and len(toks[-1]) == 1 and toks[-1].isalpha():
        toks = toks[:-1]
        low = " ".join(toks).strip()

    # remove internal digit-only ids that slipped through
    low = _strip_long_numeric_tokens(low)

    return re.sub(r"\s+", " ", low).strip()


def _looks_abbreviated(name: str) -> bool:
    s = (name or "").strip()
    if not s:
        return False
    toks = re.findall(r"[A-Za-z0-9']+", s)
    if not toks:
        return False
    short = sum(1 for t in toks if len(t) <= 3 and t.isalpha())
    letters = sum(1 for ch in s if ch.isalpha())
    if letters == 0:
        return False
    return short >= 2 or (len(s) <= 10)


def _needs_official_name(name: str) -> bool:
    s = (name or "").strip()
    if not s:
        return False
    toks = [t for t in re.findall(r"[A-Za-z']+", s) if t]
    if len(toks) <= 1:
        return True
    if _looks_abbreviated(s):
        return True
    letters = sum(1 for ch in s if ch.isalpha())
    uppers = sum(1 for ch in s if ch.isalpha() and ch.isupper())
    if letters >= 4 and (uppers / max(letters, 1)) > 0.85 and len(s) <= 18:
        return True
    return False


# ============================================================
# Name enrichment (OFF + OPF/OBF + Learned Map + Pending collector)
# ============================================================

_ENRICH_CACHE: dict[str, Tuple[float, str, str, float]] = {}
_ENRICH_CACHE_TTL = int(os.getenv("ENRICH_CACHE_TTL_SECONDS", "86400"))

_LEARNED_MAP: dict[str, str] = {}
_PENDING: dict[str, dict[str, Any]] = {}

_ENRICH_SCORE_STOPWORDS = {
    "the", "and", "or", "of", "a", "an", "with", "for",
    "pack", "ct", "count", "oz", "lb", "lbs", "g", "kg", "ml", "l",
    "publix", "walmart", "wal", "mart", "target", "costco", "kroger", "aldi",
}

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
            score += 0.03
        if sh and sh in q and sh in c:
            score += 0.02

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


async def _off_get(url: str, params: dict[str, Any], timeout_s: float) -> Optional[dict[str, Any]]:
    global OFF_CLIENT, OFF_SEM
    if not OFF_CLIENT or not OFF_SEM:
        return None
    try:
        async with OFF_SEM:
            r = await OFF_CLIENT.get(url, params=params, timeout=httpx.Timeout(timeout_s))
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _catalog_urls_for_category(category: str) -> list[tuple[str, str]]:
    cat = (category or "").strip().lower()
    if cat == "household":
        out: list[tuple[str, str]] = []
        if OPF_SEARCH_URL_WORLD:
            out.append(("openproductsfacts", OPF_SEARCH_URL_WORLD))
        if OBF_SEARCH_URL_WORLD:
            out.append(("openbeautyfacts", OBF_SEARCH_URL_WORLD))
        return out

    urls: list[tuple[str, str]] = []
    if OFF_SEARCH_URL_US:
        urls.append(("openfoodfacts_us", OFF_SEARCH_URL_US))
    if OFF_SEARCH_URL_WORLD:
        urls.append(("openfoodfacts_world", OFF_SEARCH_URL_WORLD))
    return urls


async def _catalog_best_match(
    name: str,
    store_hint: str,
    category: str,
    budget: ReqBudget
) -> Optional[Tuple[str, float, str, str]]:
    key = dedupe_key(name)
    if len(key) < 5:
        return None
    if budget.expired():
        return None

    variants: list[str] = [name]

    if store_hint and dedupe_key(store_hint) not in key:
        variants.append(f"{store_hint} {name}")

    if "publix" in key:
        variants.append(re.sub(r"\bpublix\b", "", key).strip())

    seen: set[str] = set()
    qvars: list[str] = []
    for v in variants:
        v = re.sub(r"\s+", " ", (v or "").strip())
        if not v:
            continue
        vl = v.lower()
        if vl in seen:
            continue
        seen.add(vl)
        qvars.append(v)

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
    best_query = ""
    best_source = ""

    catalogs = _catalog_urls_for_category(category)
    if not catalogs:
        return None

    for q in qvars:
        if budget.expired():
            break

        params = dict(params_base)
        params["search_terms"] = q

        for source_name, url in catalogs:
            if budget.expired():
                break

            remaining = budget.remaining()
            if remaining <= 0.35:
                break
            timeout_s = min(ENRICH_TIMEOUT_SECONDS, max(0.35, remaining - 0.15))

            data = await _off_get(url, params=params, timeout_s=timeout_s)
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
                    best_query = q
                    best_source = source_name
                    if best_score >= 0.90:
                        return best_name, best_score, best_query, best_source

    if best_name:
        return best_name, best_score, best_query, best_source
    return None


async def enrich_full_name(
    raw_line: str,
    cleaned_name: str,
    expanded_name: str,
    store_hint: str,
    category: str,
    budget: ReqBudget,
) -> Tuple[str, str, float, str]:
    global OFF_BUDGET_LOCK

    if not ENABLE_NAME_ENRICH:
        return expanded_name, "none", 0.0, ""

    if budget.expired():
        return expanded_name, "budget_expired", 0.0, ""

    cache_key = dedupe_key(f"{store_hint}:{category}:{expanded_name}" if store_hint else f"{category}:{expanded_name}")
    if cache_key:
        cached = _enrich_cache_get(cache_key)
        if cached:
            return cached[0], cached[1], cached[2], ""

    learned = _learned_map_lookup(raw_line, expanded_name, store_hint=store_hint)
    if learned:
        if _is_header_or_address(learned) or _is_store_or_header_line_anywhere(learned) or NOISE_RE.search(learned) or _is_junk_line(learned):
            _pending_add(store_hint=store_hint, raw_line=raw_line, cleaned=cleaned_name, expanded=expanded_name)
            return expanded_name, "learned_map_rejected_header", 0.0, ""
        if cache_key:
            _enrich_cache_set(cache_key, learned, "learned_map", 1.0)
        return learned, "learned_map", 1.0, ""

    always_off = (os.getenv("ALWAYS_OFF_ENRICH", "1").strip() == "1")
    should_try_catalog = always_off or _needs_official_name(expanded_name)

    if not should_try_catalog:
        _pending_add(store_hint=store_hint, raw_line=raw_line, cleaned=cleaned_name, expanded=expanded_name)
        if cache_key:
            _enrich_cache_set(cache_key, expanded_name, "skipped_catalog", 0.0)
        return expanded_name, "skipped_catalog", 0.0, ""

    if budget.remaining() <= 0.8:
        _pending_add(store_hint=store_hint, raw_line=raw_line, cleaned=cleaned_name, expanded=expanded_name)
        if cache_key:
            _enrich_cache_set(cache_key, expanded_name, "low_budget", 0.0)
        return expanded_name, "low_budget", 0.0, ""

    if OFF_BUDGET_LOCK is None:
        OFF_BUDGET_LOCK = asyncio.Lock()

    async with OFF_BUDGET_LOCK:
        if budget.off_used >= MAX_OFF_LOOKUPS_PER_REQUEST:
            _pending_add(store_hint=store_hint, raw_line=raw_line, cleaned=cleaned_name, expanded=expanded_name)
            if cache_key:
                _enrich_cache_set(cache_key, expanded_name, "catalog_cap", 0.0)
            return expanded_name, "catalog_cap", 0.0, ""
        budget.off_used += 1

    hit = await _catalog_best_match(expanded_name, store_hint=store_hint, category=category, budget=budget)
    if hit:
        candidate, score, query_used, source_name = hit

        if _is_header_or_address(candidate) or _is_store_or_header_line_anywhere(candidate) or NOISE_RE.search(candidate) or _is_junk_line(candidate):
            _pending_add(store_hint=store_hint, raw_line=raw_line, cleaned=cleaned_name, expanded=expanded_name)
            if cache_key:
                _enrich_cache_set(cache_key, expanded_name, f"{source_name}_rejected_header", 0.0)
            return expanded_name, f"{source_name}_rejected_header", 0.0, ""

        if score >= ENRICH_MIN_CONF:
            if cache_key:
                _enrich_cache_set(cache_key, candidate, source_name, score)
            return candidate, source_name, score, query_used

        if ENRICH_FORCE_BEST_EFFORT and score >= ENRICH_FORCE_SCORE_FLOOR:
            if cache_key:
                _enrich_cache_set(cache_key, candidate, f"{source_name}_forced", score)
            return candidate, f"{source_name}_forced", score, query_used

    _pending_add(store_hint=store_hint, raw_line=raw_line, cleaned=cleaned_name, expanded=expanded_name)
    if cache_key:
        _enrich_cache_set(cache_key, expanded_name, "none", 0.0)
    return expanded_name, "none", 0.0, ""


# ============================================================
# Totals marker / items-zone detection
# ============================================================

TOTAL_MARKER_RE = re.compile(
    r"\b(sub\s*total|subtotal|tax|balance|amount\s+due|total\s+due|grand\s+total)\b",
    re.IGNORECASE,
)

def find_totals_marker_index(raw_lines: list[str]) -> Optional[int]:
    n = len(raw_lines)
    if n == 0:
        return None

    tail_start = max(0, n - max(65, n // 3))
    candidates: list[int] = []
    for idx in range(tail_start, n):
        ln = raw_lines[idx] or ""
        if TOTAL_MARKER_RE.search(ln):
            if MONEY_TOKEN_RE.search(ln) or len(dedupe_key(ln).split()) <= 6:
                candidates.append(idx)

    return candidates[-1] if candidates else None


def _detect_item_zone_indices(raw_lines: list[str]) -> Tuple[int, int]:
    n = len(raw_lines)
    if n == 0:
        return 0, 0

    totals_idx = find_totals_marker_index(raw_lines)
    end = totals_idx if totals_idx is not None else n

    start = 0
    scan_limit = min(end, 180)
    for i in range(0, scan_limit):
        s = (raw_lines[i] or "").strip()
        if not s:
            continue
        if _is_header_or_address(s) or NOISE_RE.search(s) or _is_store_or_header_line_anywhere(s) or _is_junk_line(s):
            continue
        if _has_valid_item_words(s) and (_raw_line_has_price_or_qty_hint(s) or len(dedupe_key(s).split()) >= 2):
            start = max(0, i - 3)
            break

    end = max(start, end)
    return start, end


# ============================================================
# Candidate extraction
# ============================================================

class Candidate(BaseModel):
    raw_line: str
    cleaned_line: str
    qty_hint: int


def _next_nonempty(raw_lines: list[str], idx: int, zone_end: int) -> Tuple[str, int]:
    j = idx
    while j < zone_end:
        s = (raw_lines[j] or "").strip()
        if s:
            return s, j
        j += 1
    return "", zone_end


def _line_has_embedded_price(s: str) -> bool:
    if not s:
        return False
    if _is_price_like_line(s) or WEIGHT_ONLY_RE.match(s):
        return False
    return bool(MONEY_TOKEN_RE.search(s))


def _is_descriptionish_line(s: str) -> bool:
    if not s:
        return False
    if _is_header_or_address(s):
        return False
    if NOISE_RE.search(s):
        return False
    if _is_store_or_header_line_anywhere(s):
        return False
    if _is_junk_line(s):
        return False
    if _is_price_like_line(s) or WEIGHT_ONLY_RE.match(s):
        return False
    if _is_weight_or_unit_price_line(s):
        return False
    cl = _clean_line(s)
    return bool(cl) and _has_valid_item_words(cl)


def _extract_candidates_from_lines(
    raw_lines: list[str],
    store_hint: str,
) -> Tuple[list[Candidate], list[dict[str, Any]]]:
    dropped_lines: list[dict[str, Any]] = []

    zone_start, zone_end = _detect_item_zone_indices(raw_lines)
    store_header_pat = STORE_HEADER_PATTERNS.get(store_hint) if store_hint else None

    candidates: list[Candidate] = []

    pending_raw: Optional[str] = None
    pending_clean: Optional[str] = None
    pending_qty_hint: int = 1

    def _finalize_pending(reason: str) -> None:
        nonlocal pending_raw, pending_clean, pending_qty_hint
        if pending_raw and pending_clean and _has_valid_item_words(pending_clean):
            if not _is_header_or_address(pending_clean) and not _is_store_or_header_line_anywhere(pending_clean) and not NOISE_RE.search(pending_clean) and not _is_junk_line(pending_clean):
                candidates.append(Candidate(raw_line=pending_raw, cleaned_line=pending_clean, qty_hint=max(1, pending_qty_hint)))
            else:
                dropped_lines.append({"line": pending_raw, "stage": f"pending_drop:{reason}:header_guard", "cleaned": pending_clean})
        else:
            if pending_raw:
                dropped_lines.append({"line": pending_raw, "stage": f"pending_drop:{reason}", "cleaned": pending_clean})
        pending_raw = None
        pending_clean = None
        pending_qty_hint = 1

    i = zone_start
    while i < zone_end:
        s = (raw_lines[i] or "").strip()
        if not s:
            i += 1
            continue

        if _is_junk_line(s):
            if pending_raw:
                _finalize_pending("hit_junk_line")
            dropped_lines.append({"line": s, "stage": "junk_line"})
            i += 1
            continue

        if store_header_pat and store_header_pat.match(s):
            dropped_lines.append({"line": s, "stage": "store_header_exact"})
            i += 1
            continue
        if _is_probable_store_header_line(s, store_hint=store_hint, position_idx=i - zone_start):
            dropped_lines.append({"line": s, "stage": "store_header_fuzzy"})
            i += 1
            continue

        if _is_store_or_header_line_anywhere(s):
            if pending_raw:
                _finalize_pending("hit_store_header_anywhere")
            dropped_lines.append({"line": s, "stage": "store_header_anywhere"})
            i += 1
            continue

        if _is_header_or_address(s):
            dropped_lines.append({"line": s, "stage": "header_or_address"})
            i += 1
            continue
        if NOISE_RE.search(s):
            if pending_raw:
                _finalize_pending("hit_noise")
            dropped_lines.append({"line": s, "stage": "noise"})
            i += 1
            continue

        if _is_weight_or_unit_price_line(s):
            qh = _parse_qty_hint_from_attached_line(s)
            if pending_raw:
                pending_qty_hint = max(pending_qty_hint, qh)
            elif candidates:
                last = candidates[-1]
                last.qty_hint = max(int(last.qty_hint), qh)
            else:
                dropped_lines.append({"line": s, "stage": "orphan_unit_price"})
            i += 1
            continue

        if _is_price_like_line(s) or WEIGHT_ONLY_RE.match(s):
            if pending_raw:
                _finalize_pending("price_line")
            else:
                dropped_lines.append({"line": s, "stage": "price_only_no_pending"})
            i += 1
            continue

        cleaned = _clean_line(s)
        if not cleaned:
            dropped_lines.append({"line": s, "stage": "clean_empty"})
            i += 1
            continue

        next1, next1_idx = _next_nonempty(raw_lines, i + 1, zone_end)
        if next1 and _is_descriptionish_line(next1):
            key1 = dedupe_key(s)
            key2 = dedupe_key(next1)
            toks1 = key1.split()
            toks2 = key2.split()

            is_store_only = bool(toks1) and all(t in _STORE_TOKENS for t in toks1)
            short_lead = (len(toks1) <= 2 and len(key1) <= 10) or is_store_only

            next2, _ = _next_nonempty(raw_lines, next1_idx + 1, zone_end)
            next2_supports = bool(next2) and (
                _is_weight_or_unit_price_line(next2) or _is_price_like_line(next2) or WEIGHT_ONLY_RE.match(next2) or _line_has_embedded_price(next1)
            )

            if short_lead and (len(toks2) >= 1) and next2_supports:
                combined_raw = f"{s} {next1}".strip()
                combined_clean = _clean_line(combined_raw)
                if combined_clean and _has_valid_item_words(combined_clean) and not _is_header_or_address(combined_clean) and not _is_store_or_header_line_anywhere(combined_clean) and not NOISE_RE.search(combined_clean) and not _is_junk_line(combined_clean):
                    s = combined_raw
                    cleaned = combined_clean
                    i = next1_idx + 1
                else:
                    i += 1
            else:
                i += 1
        else:
            i += 1

        if pending_raw:
            _finalize_pending("new_desc")

        if not _has_valid_item_words(cleaned):
            dropped_lines.append({"line": s, "stage": "no_item_words", "cleaned": cleaned})
            continue

        if _is_header_or_address(cleaned) or _is_store_or_header_line_anywhere(cleaned) or NOISE_RE.search(cleaned) or _is_junk_line(cleaned):
            dropped_lines.append({"line": s, "stage": "desc_rejected_header_guard", "cleaned": cleaned})
            continue

        if _line_has_embedded_price(s):
            candidates.append(Candidate(raw_line=s, cleaned_line=cleaned, qty_hint=1))
            continue

        next1_after, _ = _next_nonempty(raw_lines, i, zone_end)
        next2_after, _ = _next_nonempty(raw_lines, i + 1, zone_end)

        if next1_after and (_is_weight_or_unit_price_line(next1_after) or _is_price_like_line(next1_after) or WEIGHT_ONLY_RE.match(next1_after)):
            pending_raw = s
            pending_clean = cleaned
            pending_qty_hint = 1
            continue

        if next2_after and (_is_weight_or_unit_price_line(next2_after) or _is_price_like_line(next2_after) or WEIGHT_ONLY_RE.match(next2_after)):
            pending_raw = s
            pending_clean = cleaned
            pending_qty_hint = 1
            continue

        candidates.append(Candidate(raw_line=s, cleaned_line=cleaned, qty_hint=1))

    if pending_raw:
        _finalize_pending("zone_end")

    return candidates, dropped_lines


# ============================================================
# Models
# ============================================================

class ParsedItem(BaseModel):
    name: str
    quantity: int
    category: str
    image_url: str


# ============================================================
# Routes
# ============================================================

@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True}


@app.post("/parse-receipt", response_model=List[ParsedItem])
@app.post("/parse-receipt/", response_model=List[ParsedItem])
async def parse_receipt(
    request: Request,
    file: UploadFile | None = File(None),
    image: UploadFile | None = File(None),
    only_food: bool = Query(True, description="If true, return only Food items; if false, return Food + Household"),
    debug: bool = Query(False, description="accepted but does not change response shape (use /parse-receipt-debug for wrapper)"),
):
    upload = file or image
    if upload is None:
        raise HTTPException(status_code=422, detail="Missing receipt file field (expected multipart 'file' or 'image').")

    raw = await upload.read()
    if not raw:
        return JSONResponse(status_code=400, content={"error": "Empty file"})

    budget = ReqBudget(started=time.monotonic(), deadline=time.monotonic() + REQUEST_DEADLINE_SECONDS)

    try:
        pre0 = _preprocess_image_bytes(raw, variant=0)
        text0 = ocr_text_google_vision(pre0)
        raw_lines0 = [ln.strip() for ln in (text0 or "").splitlines() if ln and ln.strip()]
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"OCR failed: {str(e)}", "hint": "Check GOOGLE_APPLICATION_CREDENTIALS_JSON / GOOGLE_APPLICATION_CREDENTIALS"},
        )

    raw_lines = raw_lines0
    text = text0

    if len(raw_lines0) < 22 and budget.remaining() > 4.0:
        try:
            pre1 = _preprocess_image_bytes(raw, variant=1)
            text1 = ocr_text_google_vision(pre1)
            raw_lines1 = [ln.strip() for ln in (text1 or "").splitlines() if ln and ln.strip()]
            if len(raw_lines1) > len(raw_lines0):
                raw_lines = raw_lines1
                text = text1
        except Exception:
            pass

    store_hint = detect_store_hint(raw_lines)

    candidates, _dropped_lines = _extract_candidates_from_lines(raw_lines, store_hint=store_hint)

    base_url = _public_base_url(request)

    items: list[dict[str, Any]] = []
    for c in candidates:
        qty, nm = _parse_quantity(c.cleaned_line)
        if qty == 1 and int(c.qty_hint) > 1:
            qty = int(c.qty_hint)

        name_cleaned = _clean_line(nm)
        expanded = expand_abbreviations(name_cleaned)
        final_name = post_name_cleanup(expanded).strip()

        if not final_name:
            continue
        if ONLY_MONEYISH_RE.match(final_name) or PRICE_FLAG_RE.match(final_name) or WEIGHT_ONLY_RE.match(final_name):
            continue
        if NOISE_RE.search(final_name) or _is_header_or_address(final_name) or _is_store_or_header_line_anywhere(final_name) or _is_junk_line(final_name):
            continue
        if not _has_valid_item_words(final_name):
            continue

        category = _classify(final_name)
        if only_food and category != "Food":
            continue

        pretty = title_case(final_name)
        items.append(
            {
                "name": pretty,
                "quantity": int(max(1, qty)),
                "category": category,
                "image_url": _image_url_for_item(base_url, pretty),
                "_raw_line": c.raw_line,
                "_name_cleaned": name_cleaned,
                "_expanded": expanded,
            }
        )

    items = _dedupe_and_merge(items)

    if ENABLE_NAME_ENRICH and items and budget.remaining() > 1.0:
        always_off = (os.getenv("ALWAYS_OFF_ENRICH", "1").strip() == "1")
        for it in items:
            if budget.remaining() <= 0.75:
                break
            expanded = (it.get("_expanded") or it["name"] or "").strip()
            raw_line = (it.get("_raw_line") or it["name"] or "").strip()
            name_cleaned = (it.get("_name_cleaned") or expanded).strip()
            category = (it.get("category") or "Food").strip()

            if not (always_off or _needs_official_name(expanded)):
                continue

            if ENRICH_SEM:
                async with ENRICH_SEM:
                    enriched, _src, _score, _q = await enrich_full_name(
                        raw_line=raw_line,
                        cleaned_name=name_cleaned,
                        expanded_name=expanded,
                        store_hint=store_hint,
                        category=category,
                        budget=budget,
                    )
            else:
                enriched, _src, _score, _q = await enrich_full_name(
                    raw_line=raw_line,
                    cleaned_name=name_cleaned,
                    expanded_name=expanded,
                    store_hint=store_hint,
                    category=category,
                    budget=budget,
                )

            enriched = post_name_cleanup(enriched).strip()

            if (
                enriched
                and _has_valid_item_words(enriched)
                and not NOISE_RE.search(enriched)
                and not _is_header_or_address(enriched)
                and not _is_store_or_header_line_anywhere(enriched)
                and not _is_junk_line(enriched)
            ):
                pretty = title_case(enriched)
                it["name"] = pretty
                it["image_url"] = _image_url_for_item(base_url, pretty)

    for it in items:
        it.pop("_raw_line", None)
        it.pop("_name_cleaned", None)
        it.pop("_expanded", None)

    return items


@app.post("/parse-receipt-debug")
@app.post("/parse-receipt-debug/")
async def parse_receipt_debug(
    request: Request,
    file: UploadFile | None = File(None),
    image: UploadFile | None = File(None),
    only_food: bool = Query(True, description="If true, return only Food items; if false, return Food + Household"),
    debug: bool = Query(True, description="kept for compatibility; this endpoint always returns wrapper"),
):
    upload = file or image
    if upload is None:
        raise HTTPException(status_code=422, detail="Missing receipt file field (expected multipart 'file' or 'image').")

    raw = await upload.read()
    if not raw:
        return JSONResponse(status_code=400, content={"error": "Empty file"})

    budget = ReqBudget(started=time.monotonic(), deadline=time.monotonic() + REQUEST_DEADLINE_SECONDS)

    try:
        pre0 = _preprocess_image_bytes(raw, variant=0)
        text0 = ocr_text_google_vision(pre0)
        raw_lines0 = [ln.strip() for ln in (text0 or "").splitlines() if ln and ln.strip()]
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"OCR failed: {str(e)}", "hint": "Check GOOGLE_APPLICATION_CREDENTIALS_JSON / GOOGLE_APPLICATION_CREDENTIALS"},
        )

    raw_lines = raw_lines0
    text = text0
    ocr_variant_used = 0

    if len(raw_lines0) < 22 and budget.remaining() > 4.0:
        try:
            pre1 = _preprocess_image_bytes(raw, variant=1)
            text1 = ocr_text_google_vision(pre1)
            raw_lines1 = [ln.strip() for ln in (text1 or "").splitlines() if ln and ln.strip()]
            if len(raw_lines1) > len(raw_lines0):
                raw_lines = raw_lines1
                text = text1
                ocr_variant_used = 1
        except Exception:
            pass

    store_hint = detect_store_hint(raw_lines)
    totals_idx = find_totals_marker_index(raw_lines)

    candidates, dropped_lines = _extract_candidates_from_lines(raw_lines, store_hint=store_hint)

    base_url = _public_base_url(request)

    parsed: list[dict[str, Any]] = []
    enrich_debug: list[dict[str, Any]] = []

    always_off = (os.getenv("ALWAYS_OFF_ENRICH", "1").strip() == "1")

    for c in candidates:
        qty, nm = _parse_quantity(c.cleaned_line)
        if qty == 1 and int(c.qty_hint) > 1:
            qty = int(c.qty_hint)

        name_cleaned = _clean_line(nm)
        expanded = expand_abbreviations(name_cleaned)
        expanded = post_name_cleanup(expanded)

        category_guess = _classify(expanded)

        enriched = expanded
        source = "none"
        score = 0.0
        query_used = ""

        if ENABLE_NAME_ENRICH and budget.remaining() > 0.9 and (always_off or _needs_official_name(expanded)):
            enriched, source, score, query_used = await enrich_full_name(
                raw_line=c.raw_line,
                cleaned_name=name_cleaned,
                expanded_name=expanded,
                store_hint=store_hint,
                category=category_guess,
                budget=budget,
            )
            enriched = post_name_cleanup(enriched)

        enrich_debug.append(
            {
                "raw_line": c.raw_line,
                "cleaned_line": c.cleaned_line,
                "qty": qty,
                "qty_hint": c.qty_hint,
                "name_cleaned": name_cleaned,
                "name_expanded": expanded,
                "name_enriched": enriched,
                "category": category_guess,
                "enrich_source": source,
                "enrich_score": score,
                "enrich_query_used": query_used,
            }
        )

        final_name = (enriched or "").strip()
        if not final_name:
            continue
        if ONLY_MONEYISH_RE.match(final_name) or PRICE_FLAG_RE.match(final_name) or WEIGHT_ONLY_RE.match(final_name):
            continue
        if NOISE_RE.search(final_name) or _is_header_or_address(final_name) or _is_store_or_header_line_anywhere(final_name) or _is_junk_line(final_name):
            continue
        if not _has_valid_item_words(final_name):
            continue

        category = _classify(final_name)
        if only_food and category != "Food":
            continue

        nm2 = title_case(final_name)
        parsed.append(
            {
                "name": nm2,
                "quantity": int(max(1, qty)),
                "category": category,
                "image_url": _image_url_for_item(base_url, nm2),
            }
        )

    parsed = _dedupe_and_merge(parsed)

    return {
        "items": parsed,
        "raw_line_count": len(raw_lines),
        "ocr_variant_used": ocr_variant_used,
        "store_hint": store_hint,
        "totals_idx": totals_idx,
        "candidate_count": len(candidates),
        "candidates": [c.cleaned_line for c in candidates][:400],
        "enrichment_debug": enrich_debug[:300],
        "enrich_enabled": ENABLE_NAME_ENRICH,
        "enrich_min_conf": ENRICH_MIN_CONF,
        "enrich_force_best_effort": ENRICH_FORCE_BEST_EFFORT,
        "enrich_force_score_floor": ENRICH_FORCE_SCORE_FLOOR,
        "off_page_size": OFF_PAGE_SIZE,
        "catalog_urls_food": [OFF_SEARCH_URL_US, OFF_SEARCH_URL_WORLD],
        "catalog_urls_household": [OPF_SEARCH_URL_WORLD, OBF_SEARCH_URL_WORLD],
        "learned_map_entries": len(_LEARNED_MAP),
        "pending_entries": len(_PENDING),
        "budget_seconds": REQUEST_DEADLINE_SECONDS,
        "max_off_lookups_per_request": MAX_OFF_LOOKUPS_PER_REQUEST,
        "debug": {"dropped_lines": dropped_lines[:600]},
    }


# ============================================================
# Instacart list link (unchanged)
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
# Image delivery (cached)
# ============================================================

_IMAGE_CACHE: dict[str, bytes] = {}
_IMAGE_CONTENT_TYPE_CACHE: dict[str, str] = {}
_MAX_CACHE_ITEMS = 2000

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
# Admin endpoints (pending -> learned map)
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
