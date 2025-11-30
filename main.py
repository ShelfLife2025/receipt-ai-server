# main.py
# FastAPI receipt parser with noise filtering, canonicalization, and deduplication.
# Google Cloud Vision OCR with env var creds. Includes optional debug mode.

from __future__ import annotations

import os
import base64
import io
import re
import urllib.parse
from typing import List, Tuple, Optional, Dict, Any

import httpx
from fastapi import FastAPI, File, UploadFile, Query, Request
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageOps, ImageFilter

# ---------------- GOOGLE VISION SETUP ----------------
#
# Render provides GOOGLE_APPLICATION_CREDENTIALS_JSON as an env var (literal JSON).
# We write it to disk and set GOOGLE_APPLICATION_CREDENTIALS so Vision SDK can read it.

VISION_TMP_PATH = "/tmp/gcloud_key.json"


def _init_google_credentials_file() -> None:
    creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not creds_json:
        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            print("Google creds: using GOOGLE_APPLICATION_CREDENTIALS already set.")
        else:
            print("WARNING: GOOGLE_APPLICATION_CREDENTIALS_JSON is not set (Vision may be unavailable).")
        return

    try:
        with open(VISION_TMP_PATH, "w") as f:
            f.write(creds_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = VISION_TMP_PATH
        print("Google Vision creds wrote to /tmp and env var set.")
    except Exception as e:
        print("Failed to write Vision creds:", e)


_init_google_credentials_file()

try:
    from google.cloud import vision  # type: ignore

    _VISION_AVAILABLE = True
    try:
        _vision_client = vision.ImageAnnotatorClient()
        print("Google Vision client initialized.")
    except Exception as e:
        _vision_client = None
        print("Google Vision client init failed:", e)
except Exception as e:
    print("Google Vision import failed:", e)
    _VISION_AVAILABLE = False
    _vision_client = None

# ---------------- APP ----------------
app = FastAPI(title="FastAPI", version="0.1.0")

# ---------------- PUBLIC BASE URL ----------------
PUBLIC_BASE_URL = (os.getenv("PUBLIC_BASE_URL") or "").strip().rstrip("/")

# ===================== Parsing Dictionaries / Rules =====================

# Store/vendor words to REMOVE if they appear
STORE_WORDS = {
    "publix", "kroger", "walmart", "target", "costco", "aldi", "trader", "traderjoes", "trader joe",
    "store", "location", "shopping", "center", "plaza",
}

# Helpful brand words to KEEP (used for images / phrase reconstruction)
BRAND_WORDS = {
    "starbucks", "nespresso", "thomas", "stacy's", "stacys", "sargento", "rao", "rao's", "panera",
    "philadelphia", "dr", "pepper", "celentano", "banderita", "king's", "hawaiian",
    "kraft", "chobani", "bounty", "charmin", "clorox", "tide", "gain",
}

# Junk tokens we should strip out of item names
JUNK_TOKENS = {
    # units / receipt markers
    "lb", "lbs", "oz", "ct", "pk", "pkg", "ea", "qty", "unit",
    "fl", "floz", "ml", "l", "g", "kg", "qt", "pt", "gal",
    # publix flags / OCR cruft
    "f", "e", "t", "x",
    # common junk
    "reg", "sale", "save", "savings", "promo",
}

# More aggressive abbreviation expansion
TOKEN_MAP = {
    # cheese / dairy
    "parm": "parmesan",
    "parma": "parmesan",
    "prm": "parmesan",
    "moz": "mozzarella",
    "mozz": "mozzarella",
    "ched": "cheddar",
    "chs": "cheese",
    "chs.": "cheese",
    "ches": "cheese",
    "chz": "cheese",

    # produce / basics
    "org": "organic",
    "grn": "green",
    "grns": "greens",
    "wht": "white",
    "sdls": "seedless",
    "sdls.": "seedless",
    "grps": "grapes",
    "tmt": "tomatoes",
    "grlc": "garlic",
    "grl": "garlic",  # often "grl" -> garlic on snacks

    # bakery
    "bg": "bagels",
    "bgl": "bagels",
    "buns": "buns",

    # snacks
    "pta": "pita",
    "ptas": "pita",
    "chp": "chips",
    "chps": "chips",
    "ptato": "potato",

    # coffee
    "nesp": "nespresso",
    "pke": "pike",
    "plc": "place",

    # misc
    "rstd": "roasted",
    "rstd.": "roasted",
    "ovn": "oven",
    "rst": "roast",
    "sn": "snap",
}

# Quantity patterns: "x2", "2 x", "qty:2" etc
QTY_TOKEN_REGEX = re.compile(
    r"(?:\bqty[:=]?\s*(\d+)\b)|(?:\b(\d+)\s*x\b)|(?:x\s*(\d+)\b)",
    re.IGNORECASE,
)

PRICE_REGEX = re.compile(r"\$?\d+\.\d{2}")
PRICE_ONLY_LINE_RE = re.compile(r"^\s*\$?\d{1,6}([.,]\d{2})?\s*[A-Za-z]?\s*$")
TRAILING_PRICE_RE = re.compile(r"(\$?\d{1,6}\.\d{2})\s*$")
TRAILING_LONG_CODE_RE = re.compile(r"\b\d{4,}\b$")

SIZE_REGEX = re.compile(
    r"(?<!\w)("
    r"\d+(\.\d+)?\s?(fl\s?oz|oz|lb|lbs|ct|pk|pkg|g|kg|ml|l|qt|pt|gal)|"
    r"\d+\s?(pack|pk|ct|lb|oz)"
    r")(?!\w)",
    re.IGNORECASE,
)

CODEY_REGEX = re.compile(
    r"(^|[^A-Za-z])([bq]?(upc|plu|sku)\b[: ]?\d+|\d{6,})($|[^A-Za-z])",
    re.IGNORECASE,
)

# HARD noise: headers/footers/people/address/payment/etc
NOISE_PATTERNS = [
    # people / roles
    r"\bstore\s+manager\b",
    r"\bmanager\b",
    r"\bcashier\b",
    r"\bassociate\b",
    r"\bclerk\b",

    # common junk single words
    r"^\s*(you|vou)\s*$",

    # store header / address-ish
    r"^\s*publix\b",
    r"\bshopping\s+center\b",
    r"^\s*\d{1,6}\s+[a-z0-9\s]+\b(ave|avenue|st|street|rd|road|blvd|boulevard|drive|dr|ln|lane|pkwy|parkway|hwy|highway)\b",
    r"^\s*[a-z\s]+,\s*[a-z]{2}\s*$",
    r"^\s*[a-z\s]+\s+[a-z]{2}\s*$",
    r"^\s*[a-z\s]+\s+[a-z]{2}\s+\d{5}(-\d{4})?\s*$",
    r"^\s*promotion\b",

    # payment totals
    r"^payment", r"^entry\s+method", r"^you\s?saved\b", r"^savings\b",
    r"\bsubtotal\b", r"\bsub\s*total\b", r"\btax\b", r"\btotal\b", r"\bchange\b", r"\bbalance\b",
    r"\bdebit\b", r"\bcredit\b", r"\bvisa\b", r"\bmastercard\b", r"\bdiscover\b", r"\bamex\b",
    r"\bthank you\b|\bthanks\b",
    r"\bpresto!?$",
    r"https?://", r"\bwww\.", r"@[A-Za-z0-9_]+",
    r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
    r"\btrace\s*#?:?\s*\d+\b", r"\bacct\s*#?:?\s*\w+\b",
    r"\bcontactless\b|\btap\b|\bchip\b|\bmerchant\b|\bterminal\b|\bpos\b",
    r"\bgrocery\s?item\b$",

    # price+flag-only lines like "13.99 F"
    r"^\s*\$?\d{1,6}([.,]\d{2})?\s*[a-z]\s*$",
]

HOUSEHOLD_HINTS = {
    "detergent", "laundry", "pods", "dish", "dishwashing", "soap", "bleach", "cleaner",
    "toilet", "bath tissue", "paper towel", "paper towels", "towel",
    "wipes", "disinfecting", "foil", "baggies", "bags", "trash", "liners", "sponges"
}

# Common phrase upgrades (turn abbreviations into “full government names” 😂)
PHRASE_RULES: List[Tuple[re.Pattern, str]] = [
    # Stacy's chips
    (re.compile(r"\bstac(y|ys|y's)\b.*\bparmesan\b.*\bgarlic\b.*\bpita\b.*\bchips?\b", re.I),
     "Stacy's Parmesan Garlic Pita Chips"),
    (re.compile(r"\bstac(y|ys|y's)\b.*\bparmesan\b.*\bgarlic\b.*\bpita\b", re.I),
     "Stacy's Parmesan Garlic Pita Chips"),

    # Thomas bagels
    (re.compile(r"\bthomas\b.*\bcinnamon\b.*\bswirl\b.*\bbagels?\b", re.I),
     "Thomas' Cinnamon Swirl Bagels"),
    (re.compile(r"\bthomas\b.*\bcin\b.*\bswirl\b.*\bbagels?\b", re.I),
     "Thomas' Cinnamon Swirl Bagels"),

    # Starbucks Nespresso Pike Place
    (re.compile(r"\bnespresso\b.*\bpike\b.*\bplace\b", re.I),
     "Starbucks Nespresso Pike Place Roast"),
    (re.compile(r"\bnespresso\b.*\bpike\b", re.I),
     "Starbucks Nespresso Pike Place Roast"),

    # Publix shorthand
    (re.compile(r"\bpbx\b.*\bsharp\b.*\bcheddar\b.*\bcut\b", re.I),
     "Publix Sharp Cheddar Cheese (Cracker Cut)"),
]


def _base_url(request: Request) -> str:
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL
    return str(request.base_url).rstrip("/")


def guess_image_url(request: Request, display_name: str) -> str:
    encoded = urllib.parse.quote((display_name or "").strip())
    return f"{_base_url(request)}/image?name={encoded}"


def guess_branded_image_url(request: Request, original_line: str, cleaned_display_name: str) -> str:
    line_lower = (original_line or "").lower()
    BRAND_IMAGE_MAP = {
        "sargento": "https://images.unsplash.com/photo-1601004890684-d8cbf643f5f2?w=512&q=80",
        "kraft": "https://images.unsplash.com/photo-1600166898747-96f2ef749acd?w=512&q=80",
        "chobani": "https://images.unsplash.com/photo-1589302168068-964664d93dc0?w=512&q=80",
        "bounty": "https://images.unsplash.com/photo-1581579186981-5f3f0c612e75?w=512&q=80",
        "charmin": "https://images.unsplash.com/photo-1584559582151-fb1dfa8b33a5?w=512&q=80",
        "clorox": "https://images.unsplash.com/photo-1581579187080-9f31fa8c7c53?w=512&q=80",
        "tide": "https://images.unsplash.com/photo-1581579187080-9f31fa8c7c53?w=512&q=80",
        "gain": "https://images.unsplash.com/photo-1581579187080-9f31fa8c7c53?w=512&q=80",
        "starbucks": "https://images.unsplash.com/photo-1541167760496-1628856ab772?w=512&q=80",
        "panera": "https://images.unsplash.com/photo-1604908177071-6c2b7b66010c?w=512&q=80",
        "rao": "https://images.unsplash.com/photo-1611075389455-2f43fa462446?w=512&q=80",
    }
    for brand, url in BRAND_IMAGE_MAP.items():
        if brand in line_lower:
            return url
    return guess_image_url(request, cleaned_display_name)


def is_noise_line(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True

    # super short = almost always junk (we’ll allow 3+ through later if meaningful)
    if len(t) <= 2:
        return True

    for pat in NOISE_PATTERNS:
        if re.search(pat, t):
            return True

    # if it's mostly digits or symbols, drop
    letters = sum(c.isalpha() for c in t)
    digits = sum(c.isdigit() for c in t)
    if letters == 0 and digits > 0:
        return True

    return False


def extract_qty(text: str) -> Tuple[str, int]:
    qty = 1

    def repl(m: re.Match) -> str:
        nonlocal qty
        for g in m.groups():
            if g and g.isdigit():
                qty = max(qty, int(g))
                break
        return " "

    cleaned = QTY_TOKEN_REGEX.sub(repl, text)
    return cleaned, qty


def strip_price_and_codes(text: str) -> str:
    t = PRICE_REGEX.sub(" ", text)
    t = TRAILING_PRICE_RE.sub("", t).strip()
    t = TRAILING_LONG_CODE_RE.sub("", t).strip()
    t = CODEY_REGEX.sub(" ", t)
    t = SIZE_REGEX.sub(" ", t)
    return t


def _tokenize_expand(text: str) -> List[str]:
    # normalize punctuation
    t = (text or "").lower()
    t = t.replace("&", " and ")
    t = re.sub(r"[^a-z0-9'\s]+", " ", t)

    tokens: List[str] = []
    for w in t.split():
        w = w.strip()
        if not w:
            continue

        # strip store/vendor words
        if w in STORE_WORDS:
            continue

        # expand abbreviations
        w2 = TOKEN_MAP.get(w, w)

        # drop junk/unit tokens and single letters
        if w2 in JUNK_TOKENS:
            continue
        if len(w2) == 1:
            continue

        tokens.append(w2)

    # kill lines that are basically just units/flags repeated (ex: "lb lb", "ff")
    meaningful = [x for x in tokens if x not in JUNK_TOKENS and len(x) > 1]
    if len(meaningful) == 0:
        return []

    return tokens


def _apply_phrase_rules(raw_line: str, candidate: str) -> str:
    s = candidate.strip()
    blob = f"{raw_line} {candidate}".strip()
    for pat, repl in PHRASE_RULES:
        if pat.search(blob):
            return repl
    return s


def _title_case(s: str) -> str:
    # smart title case with “small words” rules
    small = {"and", "or", "of", "the", "with", "in", "on", "a", "an"}
    words = s.split()
    out: List[str] = []
    for i, w in enumerate(words):
        wl = w.lower()

        # keep some brand punctuation
        if wl in {"stacy's", "rao's"}:
            out.append(wl.title().replace("S", "S").replace("R", "R"))
            continue

        if i != 0 and wl in small:
            out.append(wl)
        else:
            # handle things like "dr" -> "Dr"
            if wl == "dr":
                out.append("Dr")
            else:
                out.append(wl[:1].upper() + wl[1:])
    return " ".join(out)


def canonical_name(raw_line: str, cleaned_tokens: List[str]) -> str:
    # final candidate string
    base = " ".join(cleaned_tokens).strip()
    if not base:
        return ""

    # phrase rules can override into “full-ish” names
    base = _apply_phrase_rules(raw_line, base)

    # normalize some common forms
    b = base.lower()
    if "parmesan" in b and "cheese" in b and "sargento" in b and "artisan" in b:
        base = "Sargento Artisan Blends Parmesan Cheese"
    if b == "dr pepper":
        base = "Dr Pepper"
    if "philadelphia" in b and "cream" in b and "cheese" in b:
        base = "Philadelphia Cream Cheese"

    return _title_case(base)


def dedupe_key(display_name: str) -> str:
    # more aggressive dedupe: drop tiny words + normalize whitespace
    t = (display_name or "").lower()
    t = re.sub(r"[^a-z0-9\s']+", " ", t)
    toks = [x for x in t.split() if x and x not in {"and", "with", "the", "of"}]
    return " ".join(toks).strip()


def categorize_item(name: str) -> str:
    t = (name or "").lower()
    for h in HOUSEHOLD_HINTS:
        if h in t:
            return "Household"
    return "Food"


def looks_like_item(display: str, raw_line: str) -> bool:
    d = (display or "").strip()
    r = (raw_line or "").strip().lower()

    if not d or len(d) < 4:
        return False

    # prevent names/roles slipping through even after cleaning
    if "manager" in r or "cashier" in r or "store manager" in r:
        return False

    # must include letters
    if not re.search(r"[A-Za-z]", d):
        return False

    # avoid extremely short token-y results like "Tf", "Ff"
    if len(d.split()) == 1 and len(d) <= 4:
        return False

    return True


def _merge_name_and_price_lines(lines: List[str]) -> List[str]:
    merged: List[str] = []
    i = 0
    while i < len(lines):
        cur = (lines[i] or "").strip()
        nxt = (lines[i + 1] or "").strip() if i + 1 < len(lines) else ""
        if cur and nxt:
            if re.search(r"[A-Za-z]", cur) and PRICE_ONLY_LINE_RE.match(nxt):
                merged.append(f"{cur} {nxt}".strip())
                i += 2
                continue
        merged.append(cur)
        i += 1
    return merged

# ---------------- GOOGLE VISION OCR HELPERS ----------------

def run_google_vision_ocr(jpeg_bytes: bytes) -> List[str]:
    if not _VISION_AVAILABLE or _vision_client is None:
        print("Vision client not available, returning [].")
        return []

    try:
        image = vision.Image(content=jpeg_bytes)
        response = _vision_client.text_detection(image=image)

        if getattr(response, "error", None) and response.error.message:
            print("Vision API error:", response.error.message)
            return []

        full_text = (response.full_text_annotation.text or "") if response.full_text_annotation else ""
        lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]
        return lines
    except Exception as e:
        print("Vision OCR failed:", e)
        return []


def parse_lines_to_items(
    request: Request,
    lines: List[str],
    debug: bool = False
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    raw_lines = [l.strip() for l in lines if (l or "").strip()]
    raw_lines = _merge_name_and_price_lines(raw_lines)

    items: Dict[str, Dict[str, Any]] = {}
    dropped: List[Dict[str, Any]] = []
    kept: List[Dict[str, Any]] = []

    for raw in raw_lines:
        raw0 = (raw or "").strip()
        if not raw0:
            if debug:
                dropped.append({"line": raw, "stage": "empty", "reason": "blank"})
            continue

        if is_noise_line(raw0):
            if debug:
                dropped.append({"line": raw0, "stage": "is_noise_line", "reason": "matched_noise"})
            continue

        no_price = strip_price_and_codes(raw0)
        no_price2, qty = extract_qty(no_price)

        tokens = _tokenize_expand(no_price2)
        display = canonical_name(raw0, tokens)

        if not looks_like_item(display, raw0):
            if debug:
                dropped.append({
                    "line": raw0,
                    "stage": "looks_like_item",
                    "reason": "failed_item_check",
                    "no_price": no_price,
                    "no_price_qty_stripped": no_price2,
                    "tokens": tokens,
                    "display": display,
                    "qty": qty,
                })
            continue

        category = categorize_item(display)
        key = dedupe_key(display)

        img_url = guess_branded_image_url(
            request=request,
            original_line=raw0,
            cleaned_display_name=display
        )

        if debug:
            kept.append({
                "line": raw0,
                "no_price": no_price,
                "no_price_qty_stripped": no_price2,
                "tokens": tokens,
                "display": display,
                "dedupe_key": key,
                "qty": qty,
                "category": category,
                "image_url": img_url,
            })

        if key in items:
            items[key]["quantity"] = int(items[key]["quantity"]) + qty
        else:
            items[key] = {
                "name": display,
                "quantity": qty,
                "category": category,
                "image_url": img_url,
            }

    if not debug:
        return list(items.values()), None

    dbg = {
        "line_count_in": len(lines),
        "line_count_after_merge": len(raw_lines),
        "kept_count": len(kept),
        "dropped_count": len(dropped),
        "kept": kept[:250],
        "dropped": dropped[:500],
    }
    return list(items.values()), dbg

# ===================== API ROUTES =====================

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/parse-receipt")
async def parse_receipt(
    request: Request,
    file: UploadFile = File(...),
    debug: bool = Query(False),
):
    jpeg_bytes = await file.read()

    # light pre-process
    try:
        pil_img = Image.open(io.BytesIO(jpeg_bytes)).convert("L")
        pil_img = ImageOps.autocontrast(pil_img)
        pil_img = pil_img.filter(ImageFilter.SHARPEN)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=90)
        processed_bytes = buf.getvalue()
    except Exception as e:
        print("Preprocess failed, using raw bytes:", e)
        processed_bytes = jpeg_bytes

    # OCR via Google Vision
    ocr_lines = run_google_vision_ocr(processed_bytes)

    # Parse lines -> items
    items_list, dbg = parse_lines_to_items(request, ocr_lines, debug=debug)

    if not debug:
        return JSONResponse(items_list)

    return JSONResponse({
        "items": items_list,
        "debug": dbg,
        "ocr_line_count": len(ocr_lines),
        "ocr_text_preview": "\n".join(ocr_lines[:120]),
    })

# ===================== IMAGE DELIVERY =====================

_IMAGE_CACHE: Dict[str, bytes] = {}

PRODUCT_IMAGE_MAP: Dict[str, str] = {
    "sargento artisan blends parmesan cheese": "https://images.unsplash.com/photo-1601004890684-d8cbf643f5f2?w=512&q=80",
    "philadelphia cream cheese": "https://images.unsplash.com/photo-1589301763197-9713a1e1e5c0?w=512&q=80",
    "dr pepper": "https://images.unsplash.com/photo-1621451532593-49f463c06d65?w=512&q=80",
    "starbucks nespresso pike place roast": "https://images.unsplash.com/photo-1541167760496-1628856ab772?w=512&q=80",
    "rao's marinara sauce": "https://images.unsplash.com/photo-1611075389455-2f43fa462446?w=512&q=80",
    "bread": "https://images.unsplash.com/photo-1608198093002-de0e3580bb67?w=512&q=80",
    "garlic": "https://images.unsplash.com/photo-1506806732259-39c2d0268443?w=512&q=80",
    "tomatoes": "https://images.unsplash.com/photo-1567306226416-28f0efdc88ce?w=512&q=80",
    "grapes": "https://images.unsplash.com/photo-1601004890211-3f3d02dd3c10?w=512&q=80",
}

FALLBACK_PRODUCT_IMAGE = "https://images.unsplash.com/photo-1604908177071-6c2b7b66010c?w=512&q=80"


def _canonical_lookup_key(raw_name: str) -> str:
    return dedupe_key((raw_name or "").strip())


async def fetch_bytes(url: str) -> Optional[bytes]:
    try:
        async with httpx.AsyncClient(
            timeout=10.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"},
        ) as client:
            r = await client.get(url)
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type") or "").lower()
        if "image" not in ctype:
            return None
        return r.content
    except Exception as e:
        print("fetch_bytes error:", e)
        return None


@app.get("/image")
async def get_product_image(name: str = Query(..., description="product name to fetch image for")):
    key = _canonical_lookup_key(name)

    if key in _IMAGE_CACHE:
        return Response(content=_IMAGE_CACHE[key], media_type="image/jpeg")

    img_url = PRODUCT_IMAGE_MAP.get(key, FALLBACK_PRODUCT_IMAGE)
    img_bytes = await fetch_bytes(img_url)
    if img_bytes:
        _IMAGE_CACHE[key] = img_bytes
        return Response(content=img_bytes, media_type="image/jpeg")

    TINY_PNG_BASE64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMA"
        "ASsJTYQAAAAASUVORK5CYII="
    )
    tiny_bytes = base64.b64decode(TINY_PNG_BASE64)
    _IMAGE_CACHE[key] = tiny_bytes
    return Response(content=tiny_bytes, media_type="image/png")
