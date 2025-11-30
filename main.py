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
        # If user already configured GOOGLE_APPLICATION_CREDENTIALS another way, that's fine.
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
# Keep title/version matching your deployed openapi
app = FastAPI(title="FastAPI", version="0.1.0")

# ---------------- PUBLIC BASE URL (NO MORE HARDCODED OLD DOMAIN) ----------------
# If set, this wins. Otherwise we’ll fall back to the incoming request base URL.
PUBLIC_BASE_URL = (os.getenv("PUBLIC_BASE_URL") or "").strip().rstrip("/")

# ===================== Canonicalization / Dictionaries =====================

BRAND_WORDS = {
    "publix", "kroger", "target", "walmart", "costco", "aldi", "trader", "joe", "traderjoes",
    "great value", "market pantry", "signature", "members mark", "member's mark", "kirkland",
    "kell", "kellogg", "kelloggs", "kellogg's", "nabisco", "ritz", "cheez it", "general mills",
    "frito lay", "lays", "pringles", "campbell", "progresso", "annies", "annie's", "yoplait",
    "chobani", "silk", "oatly", "fairlife", "starbucks", "dunkin", "folgers", "maxwell house",
    "sargento", "kraft", "classico", "barilla", "ronzoni", "buitoni", "goya", "old el paso",
    "ben & jerry", "ben and jerry", "haagen dazs", "breyers", "blue bell", "blue bunny",
    "tyson", "perdue", "jennie o", "oscar mayer", "hormel", "boar", "boar's head", "boars head",
    "pillsbury", "betty crocker", "land o lakes", "challenge", "tide", "gain", "downy", "dawn",
    "clorox", "lysol", "charmin", "bounty", "pampers", "huggies"
}

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

# Expand common receipt abbreviations (safe, grocery-oriented).
# Add more here as you see new patterns.
TOKEN_MAP = {
    # cheese / dairy
    "parm": "parmesan",
    "parma": "parmesan",
    "ches": "cheese",
    "chz": "cheese",
    "mozz": "mozzarella",
    "mzz": "mozzarella",
    "cr": "cracker",
    "crkr": "crackers",
    "crkrs": "crackers",
    "chs": "cheese",
    "crm": "cream",
    "whp": "whipping",
    "hvy": "heavy",

    # produce / pantry
    "org": "organic",
    "grnd": "ground",
    "grlc": "garlic",
    "tmt": "tomato",
    "spn": "spinach",
    "bn": "bean",
    "bns": "beans",

    # meats
    "chk": "chicken",
    "brst": "breast",
    "rst": "roasted",
    "ov": "oven",

    # store-ish / OCR weirdness
    "pbx": "publix",     # often shows up as "Pbx ..."
    "sh": "sharp",       # "Sh Chd" -> sharp cheddar
    "chd": "cheddar",
    "cut": "cuts",
    "pke": "pike",       # if needed (rare)
    "nesp": "nespresso", # if needed

    # beverage
    "dr": "dr",
}

HOUSEHOLD_HINTS = {
    "detergent", "laundry", "pods", "dish", "dishwashing", "soap", "bleach", "cleaner",
    "toilet", "bath tissue", "paper towel", "paper towels", "towel",
    "wipes", "disinfecting", "foil", "baggies", "bags", "trash", "liners", "sponges"
}

PRICE_REGEX = re.compile(r"\$?\d+\.\d{2}")
# "x2", "2 x", "qty:2" etc
QTY_TOKEN_REGEX = re.compile(
    r"(?:\bqty[:=]?\s*(\d+)\b)|(?:\b(\d+)\s*x\b)|(?:x\s*(\d+)\b)",
    re.IGNORECASE,
)

# Publix often OCRs item then price-only line under it (ex: "ORG BEA" then "5.29 E")
PRICE_ONLY_LINE_RE = re.compile(r"^\s*\$?\d{1,6}\.\d{2}\s*[A-Za-z]?\s*$")
TRAILING_PRICE_RE = re.compile(r"(\$?\d{1,6}\.\d{2})\s*$")
TRAILING_LONG_CODE_RE = re.compile(r"\b\d{4,}\b$")

# ===================== Noise / Quality Gates =====================

STATE_ABBR = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME",
    "MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA",
    "RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY","DC"
}

NOISE_PATTERNS = [
    # obvious receipt non-items
    r"\bthank you\b|\bthanks\b",
    r"\bapply\b|\bjobs\b|publix\.?jobs|\bcareer\b",
    r"\bsubtotal\b|\bsub\s*total\b|\btax\b|\btotal\b|\bbalance\b|\bamount due\b|\bchange\b",
    r"\bpayment\b|\bpaid\b|\bdebit\b|\bcredit\b|\bvisa\b|\bmastercard\b|\bdiscover\b|\bamex\b",
    r"\bauth\b|\bapproval\b|\bref\b|\btrace\b|\bacct\b|\baccount\b",
    r"\bcontactless\b|\btap\b|\bchip\b",
    r"\bcashier\b|\bregister\b|\bterminal\b|\bpos\b|\bmerchant\b",
    r"\bpromo\b|\bpromotion\b|\bdiscount\b|\bsavings\b|\byou\s?saved\b|\bcoupon\b",
    r"https?://|\bwww\.",
    # phone numbers
    r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
    # address-ish
    r"^\s*\d{1,6}\s+[a-z0-9\s]+\b(ave|avenue|st|street|rd|road|blvd|boulevard|drive|dr|ln|lane|pkwy|parkway|hwy|highway)\b",
    r"\bshopping\s+center\b|\bplaza\b|\bmall\b",
    # “price + flag” lines
    r"^\s*\$?\d{1,6}([.,]\d{2})?\s*[a-z]\s*$",
]

def _base_url(request: Request) -> str:
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL
    return str(request.base_url).rstrip("/")

def guess_image_url(request: Request, display_name: str) -> str:
    encoded = urllib.parse.quote((display_name or "").strip())
    return f"{_base_url(request)}/image?name={encoded}"

def _looks_like_location(s: str) -> bool:
    # City ST, City ST ZIP, etc.
    t = (s or "").strip()
    if not t:
        return False
    # e.g. "Winter Park, FL"
    if re.fullmatch(r"[A-Za-z .'-]+,\s*[A-Za-z]{2}\s*", t):
        return True
    parts = t.split()
    if len(parts) >= 2 and parts[-1].upper() in STATE_ABBR:
        return True
    if re.search(r"\b[A-Z]{2}\s*\d{5}(-\d{4})?\b", t):
        return True
    return False

def _strip_money_and_codes(s: str) -> str:
    s = PRICE_REGEX.sub(" ", s)
    s = re.sub(r"@\s*\d+(\.\d{2})?\b", " ", s)
    s = CODEY_REGEX.sub(" ", s)
    s = TRAILING_PRICE_RE.sub("", s).strip()
    s = TRAILING_LONG_CODE_RE.sub("", s).strip()
    return re.sub(r"\s+", " ", s).strip()

def _drop_trailing_flag_tokens(s: str) -> str:
    # Publix often has trailing single letters like F/H/E after price lines
    s = (s or "").strip()
    s = re.sub(r"\s+[A-Za-z]\b$", "", s).strip()
    # sometimes multiple flags: " ... F E"
    s = re.sub(r"(\s+[A-Za-z]\b)+$", "", s).strip()
    return s

def _is_two_letter_garbage(s: str) -> bool:
    t = re.sub(r"[^A-Za-z ]+", " ", (s or ""))
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return True
    toks = t.split()
    # Allow “Dr” (Dr Pepper)
    allow = {"dr"}
    if all((len(x) <= 2 and x.lower() not in allow) for x in toks) and len(toks) <= 3:
        return True
    return False

def is_noise_line(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True

    low = t.lower()
    if _looks_like_location(t):
        return True

    for pat in NOISE_PATTERNS:
        if re.search(pat, low):
            return True

    # if the line is basically only numbers/prices
    letters = sum(c.isalpha() for c in t)
    digits = sum(c.isdigit() for c in t)
    if letters == 0 and digits > 0:
        return True

    # kill "TF", "FF", "LB LB", etc
    cleaned = _drop_trailing_flag_tokens(_strip_money_and_codes(t))
    if _is_two_letter_garbage(cleaned):
        return True

    return False

def normalize(text: str) -> str:
    """
    Goal:
    - remove prices, codes, sizes
    - expand common abbreviations
    - remove trailing flag letters (F/H/E/etc)
    - keep meaningful multi-word names
    """
    t = (text or "").strip().lower()
    t = t.replace("&", " and ")

    t = _strip_money_and_codes(t)
    t = _drop_trailing_flag_tokens(t)

    # remove brands as "noise" for canonical comparison (we still keep them in display where possible)
    for bw in BRAND_WORDS:
        t = re.sub(rf"\b{re.escape(bw)}\b", " ", t, flags=re.IGNORECASE)

    t = SIZE_REGEX.sub(" ", t)

    words: List[str] = []
    for w in re.split(r"[^a-z0-9']+", t):
        if not w:
            continue
        w2 = TOKEN_MAP.get(w, w)
        # drop lone flag letters anywhere
        if len(w2) == 1 and w2.isalpha():
            continue
        words.append(w2)

    out = " ".join(words)
    out = re.sub(r"\s+", " ", out).strip()

    # final trailing clean
    out = _drop_trailing_flag_tokens(out)
    return out

def title_case(s: str) -> str:
    small = {"and", "or", "the", "of", "a", "an", "with", "in", "to"}
    # preserve some brand punctuation
    special = {
        "raos": "Rao's",
        "dr": "Dr",
        "mac": "Mac",
    }
    parts = (s or "").split()
    out: List[str] = []
    for i, p in enumerate(parts):
        core = p.strip()
        low = core.lower().strip("'")
        if low in special:
            out.append(special[low])
            continue
        if i > 0 and low in small:
            out.append(low)
        else:
            out.append(low[:1].upper() + low[1:])
    return " ".join(out).strip()

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

def looks_like_item(raw: str, normalized: str) -> bool:
    """
    We want to keep real items, but drop junk.
    Rules:
    - not noise
    - must contain at least one 3+ letter chunk after cleaning
    - reject mostly 1–2 letter tokens
    """
    if is_noise_line(raw):
        return False

    cand = normalized.strip()
    if not cand:
        return False

    if _is_two_letter_garbage(cand):
        return False

    # must contain a real word (3+ letters)
    if not re.search(r"[a-zA-Z]{3,}", cand):
        return False

    return True

def categorize_item(name: str) -> str:
    t = (name or "").lower()
    for h in HOUSEHOLD_HINTS:
        if h in t:
            return "Household"
    return "Food"

def canonical_key(name: str) -> str:
    """
    Strong dedupe key:
    - lowercase
    - remove punctuation/spaces
    - remove trailing flag letters
    """
    n = (name or "").lower()
    n = _drop_trailing_flag_tokens(n)
    n = re.sub(r"[^a-z0-9]+", "", n)
    return n

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

def guess_branded_image_url(request: Request, original_line: str, cleaned_display_name: str) -> str:
    """
    Try to return a more brand-specific product image if we can guess a brand
    from the raw receipt line. Fallback to our generic /image route.
    """
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
    }

    for brand, url in BRAND_IMAGE_MAP.items():
        if brand in line_lower:
            return url

    return guess_image_url(request, cleaned_display_name)

# ---------------- GOOGLE VISION OCR HELPERS ----------------

def run_google_vision_ocr(jpeg_bytes: bytes) -> List[str]:
    """
    Send the image bytes to Google Vision and get raw text lines back.
    If Vision setup failed (client is None), return [].
    """
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
    """
    Take OCR text lines, clean them, group duplicates, guess qty/category/image.
    If debug=True, also returns dropped/kept diagnostics.
    """
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

        # hard noise gate
        if is_noise_line(raw0):
            if debug:
                dropped.append({"line": raw0, "stage": "is_noise_line", "reason": "noise"})
            continue

        # remove prices + qty tokens early
        no_price = PRICE_REGEX.sub(" ", raw0)
        no_price = _drop_trailing_flag_tokens(no_price)
        no_price2, qty = extract_qty(no_price)

        norm = normalize(no_price2)

        if not looks_like_item(raw0, norm):
            if debug:
                dropped.append({
                    "line": raw0,
                    "stage": "looks_like_item",
                    "reason": "failed_quality",
                    "no_price": no_price,
                    "no_price_qty_stripped": no_price2,
                    "normalized": norm,
                    "qty": qty,
                })
            continue

        display = title_case(norm)
        if not display or len(display) < 3:
            if debug:
                dropped.append({"line": raw0, "stage": "display", "reason": "empty_display"})
            continue

        category = categorize_item(display)
        img_url = guess_branded_image_url(request=request, original_line=raw0, cleaned_display_name=display)

        key = canonical_key(display)
        if not key:
            if debug:
                dropped.append({"line": raw0, "stage": "dedupe_key", "reason": "empty_key"})
            continue

        if debug:
            kept.append({
                "line": raw0,
                "no_price": no_price,
                "no_price_qty_stripped": no_price2,
                "normalized": norm,
                "display": display,
                "qty": qty,
                "category": category,
                "key": key,
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
        # NOTE: we only truncate the debug arrays for response size; we do NOT cap items.
        "kept": kept[:250],
        "dropped": dropped[:400],
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
    """
    1. Read the uploaded image (JPEG basically).
    2. Run Google Vision OCR to get all text lines.
    3. Parse those lines into structured items.
    4. Return items to the iOS app.

    If debug=true, also return drop reasons and transformations.
    """
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
        "ocr_text_preview": "\n".join(ocr_lines[:80]),
    })

# ===================== IMAGE DELIVERY =====================

_IMAGE_CACHE: Dict[str, bytes] = {}

PRODUCT_IMAGE_MAP: Dict[str, str] = {
    "parmesan cheese": "https://images.unsplash.com/photo-1601004890684-d8cbf643f5f2?w=512&q=80",
    "mozzarella cheese": "https://images.unsplash.com/photo-1600166898747-96f2ef749acd?w=512&q=80",
    "cheddar cheese": "https://images.unsplash.com/photo-1601004890684-d8cbf643f5f2?w=512&q=80",

    "bread": "https://images.unsplash.com/photo-1608198093002-de0e3580bb67?w=512&q=80",
    "bagels": "https://images.unsplash.com/photo-1509440159596-0249088772ff?w=512&q=80",

    "yogurt": "https://images.unsplash.com/photo-1589302168068-964664d93dc0?w=512&q=80",
    "cream cheese": "https://images.unsplash.com/photo-1589301763197-9713a1e1e5c0?w=512&q=80",

    "garlic": "https://images.unsplash.com/photo-1506806732259-39c2d0268443?w=512&q=80",
    "tomatoes": "https://images.unsplash.com/photo-1567306226416-28f0efdc88ce?w=512&q=80",
    "grapes": "https://images.unsplash.com/photo-1601004890211-3f3d02dd3c10?w=512&q=80",

    "dr pepper": "https://images.unsplash.com/photo-1621451532593-49f463c06d65?w=512&q=80",
    "panera mac and cheese": "https://images.unsplash.com/photo-1604908177071-6c2b7b66010c?w=512&q=80",
    "raos marinara sauce": "https://images.unsplash.com/photo-1611075389455-2f43fa462446?w=512&q=80",

    "paper towels": "https://images.unsplash.com/photo-1581579186981-5f3f0c612e75?w=512&q=80",
    "toilet paper": "https://images.unsplash.com/photo-1584559582151-fb1dfa8b33a5?w=512&q=80",
    "laundry detergent": "https://images.unsplash.com/photo-1581579187080-9f31fa8c7c53?w=512&q=80",
    "laundry pods": "https://images.unsplash.com/photo-1581579187080-9f31fa8c7c53?w=512&q=80",

    "water": "https://images.unsplash.com/photo-1561043433-aaf687c4cf4e?w=512&q=80",
    "ketchup": "https://images.unsplash.com/photo-1604908177071-6c2b7b66010c?w=512&q=80",
}

FALLBACK_PRODUCT_IMAGE = "https://images.unsplash.com/photo-1604908177071-6c2b7b66010c?w=512&q=80"


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
    """
    Stable image endpoint:
    1. Canonicalize the name.
    2. If cached bytes exist, return them.
    3. Otherwise, try PRODUCT_IMAGE_MAP (Unsplash-ish brand-style photo).
    4. If that fails, return a 1x1 transparent PNG so we never crash.
    """
    key = (name or "").strip().lower()

    if key in _IMAGE_CACHE:
        return Response(content=_IMAGE_CACHE[key], media_type="image/jpeg")

    img_url = PRODUCT_IMAGE_MAP.get(key, FALLBACK_PRODUCT_IMAGE)
    img_bytes = await fetch_bytes(img_url)
    if img_bytes:
        _IMAGE_CACHE[key] = img_bytes
        return Response(content=img_bytes, media_type="image/jpeg")

    # 1x1 transparent PNG fallback
    TINY_PNG_BASE64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMA"
        "ASsJTYQAAAAASUVORK5CYII="
    )
    tiny_bytes = base64.b64decode(TINY_PNG_BASE64)
    _IMAGE_CACHE[key] = tiny_bytes
    return Response(content=tiny_bytes, media_type="image/png")
