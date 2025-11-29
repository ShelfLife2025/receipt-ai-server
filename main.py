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

TOKEN_MAP = {
    "parm": "parmesan", "parma": "parmesan", "parmesan": "parmesan",
    "ches": "cheese", "chz": "cheese", "cheez": "cheese",
    "mozz": "mozzarella",
    "snk": "snack",
    "crkr": "crackers", "crkrs": "crackers",
    "grnd": "ground", "org": "organic", "bn": "bean", "bns": "beans",
    "chk": "chicken", "brst": "breast",
    "grlc": "garlic", "grl": "garlic", "tmt": "tomato", "toma": "tomato", "spn": "spinach",
    "strwb": "strawberry",
    "wtr": "water", "blk": "black", "whl": "whole"
}

HOUSEHOLD_HINTS = {
    "detergent", "laundry", "pods", "dish", "dishwashing", "soap", "bleach", "cleaner",
    "toilet", "bath tissue", "paper towel", "paper towels", "towel",
    "wipes", "disinfecting", "foil", "baggies", "bags", "trash", "liners", "sponges"
}

FOOD_HINTS = {
    "cheese", "parmesan", "mozzarella", "cheddar", "milk", "yogurt", "butter", "cream",
    "eggs", "bread", "loaf", "bagel", "tortilla", "pasta", "spaghetti", "macaroni", "noodles",
    "rice", "cereal", "oatmeal", "granola", "cracker", "crackers", "chips", "snack", "cookies",
    "tomato", "tomatoes", "lettuce", "spinach", "greens", "kale", "broccoli", "carrot", "onion",
    "garlic", "pepper", "cucumber", "apple", "banana", "strawberry", "berries", "lemon", "lime",
    "orange", "grapes", "avocado", "potato", "sweet potato", "mushroom",
    "chicken", "beef", "pork", "turkey", "sausage", "bacon", "ham",
    "fish", "salmon", "tuna", "shrimp",
    "hummus", "salsa", "guacamole", "ketchup", "mustard", "mayo", "mayonnaise", "pesto",
    "yoghurt", "coffee", "tea", "juice", "water", "soda", "sparkling", "broth", "stock",
    "flour", "sugar", "salt", "pepper", "spice", "seasoning", "oil", "olive oil", "vinegar",
    "ice cream", "frozen", "pizza", "waffle", "pancake", "waffles", "pancakes"
}

PRICE_REGEX = re.compile(r"\$?\d+\.\d{2}")
# "x2", "2 x", "qty:2" etc
QTY_TOKEN_REGEX = re.compile(
    r"(?:\bqty[:=]?\s*(\d+)\b)|(?:\b(\d+)\s*x\b)|(?:x\s*(\d+)\b)",
    re.IGNORECASE,
)

# Publix often OCRs item then price-only line under it (ex: "ORG BEA" then "5.29 E")
PRICE_ONLY_LINE_RE = re.compile(r"^\s*\$?\d{1,4}\.\d{2}\s*[A-Za-z]?\s*$")
TRAILING_PRICE_RE = re.compile(r"(\$?\d{1,4}\.\d{2})\s*$")
TRAILING_LONG_CODE_RE = re.compile(r"\b\d{4,}\b$")

NOISE_PATTERNS = [
    r"^apply\b", r"jobs\b", r"publix\.?jobs", r"\bcareer\b",
    r"^payment", r"^entry\s+method", r"^you\s?saved\b", r"^savings\b",
    r"^acct[:# ]", r"^account\b", r"^trace[:# ]", r"\bapproval\b", r"\bauth\b",
    r"\bsubtotal\b", r"\bsub\s*total\b", r"\btax\b", r"\btotal\b", r"\bchange\b", r"\bbalance\b",
    r"\bdebit\b", r"\bcredit\b", r"\bvisa\b", r"\bmastercard\b", r"\bdiscover\b", r"\bamex\b",
    r"\bthank you\b|\bthanks\b", r"\bcashier\b|\bmanager\b|\bstore\b\s*#",
    r"\bpresto!?$", r"\bpresto!\b", r"\bplaza\b|\bmall\b|\bmillenia\b",
    r"https?://", r"\bwww\.", r"@[A-Za-z0-9_]+",
    r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
    r"\b\d{1,4}\s+[A-Za-z0-9]+\s+(ave|avenue|st|street|rd|road|blvd|drive|dr)\b",
    r"\btrace\s*#?:?\s*\d+\b", r"\bacct\s*#?:?\s*\w+\b",
    r"\bcontactless\b|\btap\b|\bchip\b", r"\bmerchant\b|\bterminal\b|\bpos\b",
    r"\bgrocery\s?item\b$",
]

# ===================== Canonical helpers =====================

def canonical_name(name: str) -> str:
    t = (name or "").strip().lower()
    if "parmesan" in t and "cheese" in t:
        return "Parmesan cheese"
    if "mozzarella" in t and "cheese" in t:
        return "Mozzarella cheese"
    if "cheddar" in t and "cheese" in t:
        return "Cheddar cheese"
    if "cracker" in t:
        return "Crackers"
    if "bread" in t:
        return "Bread"
    if "yogurt" in t:
        return "Yogurt"
    if "hummus" in t:
        return "Hummus"
    if "ketchup" in t:
        return "Ketchup"
    if "spinach" in t:
        return "Spinach"
    if "tomato" in t:
        return "Tomatoes"
    if "garlic" in t:
        return "Garlic"
    if "detergent" in t and "pod" in t:
        return "Laundry pods"
    if "detergent" in t:
        return "Laundry detergent"
    if "dish" in t and "soap" in t:
        return "Dish soap"
    if "paper" in t and "towel" in t:
        return "Paper towels"
    if "toilet" in t or "bath tissue" in t:
        return "Toilet paper"
    if "wipe" in t:
        return "Wipes"
    return t[:1].upper() + t[1:] if t else ""


def _canonical_lookup_key(raw_name: str) -> str:
    return canonical_name(raw_name).strip().lower()


def _base_url(request: Request) -> str:
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL
    # e.g. "https://receipt-ai-server.onrender.com/"
    return str(request.base_url).rstrip("/")


def guess_image_url(request: Request, display_name: str) -> str:
    encoded = urllib.parse.quote((display_name or "").strip())
    return f"{_base_url(request)}/image?name={encoded}"


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

# ===================== Parsing helpers =====================

def is_noise_line(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t or len(t) < 3:
        return True
    for pat in NOISE_PATTERNS:
        if re.search(pat, t):
            return True

    # price-only / mostly numeric lines = noise
    letters = sum(c.isalpha() for c in t)
    digits = sum(c.isdigit() for c in t)
    if letters == 0 and digits > 0:
        return True

    return False


def normalize(text: str) -> str:
    t = (text or "").lower()
    t = t.replace("&", " and ")

    for bw in BRAND_WORDS:
        t = re.sub(rf"\b{re.escape(bw)}\b", " ", t, flags=re.IGNORECASE)

    t = SIZE_REGEX.sub(" ", t)
    t = CODEY_REGEX.sub(" ", t)

    words: List[str] = []
    for w in re.split(r"[^a-z0-9]+", t):
        if not w:
            continue
        words.append(TOKEN_MAP.get(w, w))

    t = " ".join(words)
    t = re.sub(r"\s+", " ", t).strip()

    # strip trailing price/codes that sneak through OCR merges
    t = TRAILING_PRICE_RE.sub("", t).strip()
    t = TRAILING_LONG_CODE_RE.sub("", t).strip()
    return t


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


def strip_price(text: str) -> str:
    return PRICE_REGEX.sub(" ", text)


def categorize_item(name: str) -> str:
    t = (name or "").lower()
    for h in HOUSEHOLD_HINTS:
        if h in t:
            return "Household"
    return "Food"


def looks_like_item(raw_or_cleaned: str) -> bool:
    """
    Make this permissive so we stop losing real items:
    - must have letters
    - not noise
    - not super short
    """
    t = (raw_or_cleaned or "").strip()
    if len(t) < 3:
        return False
    if is_noise_line(t):
        return False
    if not re.search(r"[A-Za-z]", t):
        return False
    # Avoid weird single-token mega-strings
    if len(t) > 64 and " " not in t:
        return False
    return True


def _merge_name_and_price_lines(lines: List[str]) -> List[str]:
    merged: List[str] = []
    i = 0
    while i < len(lines):
        cur = (lines[i] or "").strip()
        nxt = (lines[i + 1] or "").strip() if i + 1 < len(lines) else ""
        if cur and nxt:
            # merge item line with next price-only line
            if re.search(r"[A-Za-z]", cur) and PRICE_ONLY_LINE_RE.match(nxt):
                merged.append(f"{cur} {nxt}".strip())
                i += 2
                continue
        merged.append(cur)
        i += 1
    return merged


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
    # Clean + merge
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

        # remove obvious prices
        without_price = strip_price(raw0)
        without_price2, qty = extract_qty(without_price)

        norm = normalize(without_price2)

        # IMPORTANT: check item-likeness on the raw-ish line too, not only hint tokens
        if not looks_like_item(raw0) and not looks_like_item(norm):
            if debug:
                dropped.append({
                    "line": raw0,
                    "stage": "looks_like_item",
                    "reason": "failed_permissive_check",
                    "without_price": without_price,
                    "without_price_qty_stripped": without_price2,
                    "normalized": norm,
                    "qty": qty,
                })
            continue

        display = canonical_name(norm if norm else without_price2)
        if not display or len(display) < 3:
            if debug:
                dropped.append({"line": raw0, "stage": "canonical_name", "reason": "empty_display"})
            continue

        category = categorize_item(display)
        key = display.lower()

        img_url = guess_branded_image_url(
            request=request,
            original_line=raw0,
            cleaned_display_name=display
        )

        if debug:
            kept.append({
                "line": raw0,
                "without_price": without_price,
                "without_price_qty_stripped": without_price2,
                "normalized": norm,
                "display": display,
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
    "panera mac & cheese": "https://images.unsplash.com/photo-1604908177071-6c2b7b66010c?w=512&q=80",
    "rao's marinara sauce": "https://images.unsplash.com/photo-1611075389455-2f43fa462446?w=512&q=80",

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
    key = _canonical_lookup_key(name)

    if key in _IMAGE_CACHE:
        # could be jpg or png; this keeps behavior simple/stable
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
