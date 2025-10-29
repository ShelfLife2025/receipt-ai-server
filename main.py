# main.py
# FastAPI receipt parser with noise filtering, canonicalization, and deduplication.
# Now using Google Cloud Vision instead of pytesseract, and using env var creds.

import os
import json
import base64
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse, Response
from typing import List, Tuple, Optional, Dict
from PIL import Image, ImageOps, ImageFilter
import io
import re
import urllib.parse  # for safe URL encoding
import httpx

# ---------------- GOOGLE VISION SETUP ----------------
#
# We expect Render to provide GOOGLE_APPLICATION_CREDENTIALS_JSON as an env var.
# We'll write that JSON to a temp file on disk so the Vision SDK can read it.

VISION_TMP_PATH = "/tmp/gcloud_key.json"

def _init_google_credentials_file():
    creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not creds_json:
        print("WARNING: GOOGLE_APPLICATION_CREDENTIALS_JSON is not set")
        return

    try:
        # creds_json is literal JSON text. Write it out.
        with open(VISION_TMP_PATH, "w") as f:
            f.write(creds_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = VISION_TMP_PATH
        print("Google Vision creds wrote to /tmp and env var set.")
    except Exception as e:
        print("Failed to write Vision creds:", e)

_init_google_credentials_file()

# Import Vision only AFTER creds are prepared
try:
    from google.cloud import vision
    _VISION_AVAILABLE = True
    _vision_client = vision.ImageAnnotatorClient()
    print("Google Vision client initialized.")
except Exception as e:
    print("Google Vision init failed:", e)
    _VISION_AVAILABLE = False
    _vision_client = None

# =====================================================

app = FastAPI()

# ===================== Canonicalization / Dictionaries =====================

BRAND_WORDS = {
    "publix","kroger","target","walmart","costco","aldi","trader","joe","traderjoes",
    "great value","market pantry","signature","members mark","member's mark","kirkland",
    "kell","kellogg","kelloggs","kellogg's","nabisco","ritz","cheez it","general mills",
    "frito lay","lays","pringles","campbell","progresso","annies","annie's","yoplait",
    "chobani","silk","oatly","fairlife","starbucks","dunkin","folgers","maxwell house",
    "sargento","kraft","classico","barilla","ronzoni","buitoni","goya","old el paso",
    "ben & jerry","ben and jerry","haagen dazs","breyers","blue bell","blue bunny",
    "tyson","perdue","jennie o","oscar mayer","hormel","boar","boar's head","boars head",
    "pillsbury","betty crocker","land o lakes","challenge","tide","gain","downy","dawn",
    "clorox","lysol","charmin","bounty","pampers","huggies"
}

SIZE_REGEX = re.compile(
    r"(?<!\w)("
    r"\d+(\.\d+)?\s?(fl\s?oz|oz|lb|lbs|ct|pk|pkg|g|kg|ml|l|qt|pt|gal)|"
    r"\d+\s?(pack|pk|ct|lb|oz)"
    r")(?!\w)",
    re.IGNORECASE,
)

CODEY_REGEX = re.compile(r"(^|[^A-Za-z])([bq]?(upc|plu|sku)\b[: ]?\d+|\d{6,})($|[^A-Za-z])", re.IGNORECASE)

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
    "detergent","laundry","pods","dish","dishwashing","soap","bleach","cleaner",
    "toilet","bath tissue","paper towel","paper towels","towel",
    "wipes","disinfecting","foil","baggies","bags","trash","liners","sponges"
}

FOOD_HINTS = {
    "cheese","parmesan","mozzarella","cheddar","milk","yogurt","butter","cream",
    "eggs","bread","loaf","bagel","tortilla","pasta","spaghetti","macaroni","noodles",
    "rice","cereal","oatmeal","granola","cracker","crackers","chips","snack","cookies",
    "tomato","tomatoes","lettuce","spinach","greens","kale","broccoli","carrot","onion",
    "garlic","pepper","cucumber","apple","banana","strawberry","berries","lemon","lime",
    "orange","grapes","avocado","potato","sweet potato","mushroom",
    "chicken","beef","pork","turkey","sausage","bacon","ham",
    "fish","salmon","tuna","shrimp",
    "hummus","salsa","guacamole","ketchup","mustard","mayo","mayonnaise","pesto",
    "yoghurt","coffee","tea","juice","water","soda","sparkling","broth","stock",
    "flour","sugar","salt","pepper","spice","seasoning","oil","olive oil","vinegar",
    "ice cream","frozen","pizza","waffle","pancake","waffles","pancakes"
}

PRICE_REGEX = re.compile(r"\$?\d+\.\d{2}")
QTY_TOKEN_REGEX = re.compile(r"(?:\bqty[:=]?\s*(\d+)\b)|(?:\b(\d+)\s*x\b)|(?:x\s*(\d+)\b)", re.IGNORECASE)

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
    r"\bgrocery\s?item\b$"
]

def canonical_name(name: str) -> str:
    t = name.lower()
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
    return t[:1].upper() + t[1:]

def _canonical_lookup_key(raw_name: str) -> str:
    return canonical_name(raw_name).strip().lower()

def guess_image_url(display_name: str) -> str:
    encoded = urllib.parse.quote(display_name.strip())
    return f"https://receiptai-server.onrender.com/image?name={encoded}"

def guess_branded_image_url(original_line: str, cleaned_display_name: str) -> str:
    """
    Try to return a more brand-specific product image if we can guess a brand
    from the raw receipt line (e.g. 'SARGENTO SHRED PARM 8OZ').
    Fallback to our generic /image route.
    """
    line_lower = (original_line or "").lower()

    BRAND_IMAGE_MAP = {
        # Cheese / dairy brands
        "sargento": "https://images.unsplash.com/photo-1601004890684-d8cbf643f5f2?w=512&q=80",
        "kraft": "https://images.unsplash.com/photo-1600166898747-96f2ef749acd?w=512&q=80",
        "chobani": "https://images.unsplash.com/photo-1589302168068-964664d93dc0?w=512&q=80",

        # Paper goods
        "bounty": "https://images.unsplash.com/photo-1581579186981-5f3f0c612e75?w=512&q=80",
        "charmin": "https://images.unsplash.com/photo-1584559582151-fb1dfa8b33a5?w=512&q=80",

        # Cleaning / laundry
        "clorox": "https://images.unsplash.com/photo-1581579187080-9f31fa8c7c53?w=512&q=80",
        "tide": "https://images.unsplash.com/photo-1581579187080-9f31fa8c7c53?w=512&q=80",
        "gain": "https://images.unsplash.com/photo-1581579187080-9f31fa8c7c53?w=512&q=80",
    }

    for brand, url in BRAND_IMAGE_MAP.items():
        if brand in line_lower:
            return url

    # generic fallback -> this hits our /image route
    return guess_image_url(cleaned_display_name)

def is_noise_line(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t or len(t) < 3:
        return True
    for pat in NOISE_PATTERNS:
        if re.search(pat, t):
            return True
    letters = sum(c.isalpha() for c in t)
    if letters < 2:
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
    return t

def extract_qty(text: str) -> Tuple[str, int]:
    qty = 1
    def repl(m):
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

def _tokens(s: str) -> List[str]:
    return [w for w in re.split(r"[^a-z0-9]+", s.lower()) if w]

def tokens_have_hint(tokens: List[str]) -> bool:
    if not tokens:
        return False
    joined = " ".join(tokens)
    for h in HOUSEHOLD_HINTS:
        if h in joined:
            return True
    for f in FOOD_HINTS:
        if f in joined:
            return True
    last = tokens[-1]
    if len(last) >= 4 and (last.endswith("s") or last in {"bread","chicken","beef","pork","fish"}):
        return True
    return False

def looks_like_item(cleaned: str) -> bool:
    t = cleaned.strip().lower()
    if not t:
        return False
    if sum(c.isalpha() for c in t) < 3:
        return False
    toks = _tokens(t)
    if not tokens_have_hint(toks):
        return False
    if re.fullmatch(r"[a-z]{0,2}\d{4,}", t):
        return False
    return True

def categorize_item(name: str) -> str:
    t = name.lower()
    for h in HOUSEHOLD_HINTS:
        if h in t:
            return "Household"
    return "Food"

# ---------------- GOOGLE VISION OCR HELPERS ----------------

def run_google_vision_ocr(jpeg_bytes: bytes) -> List[str]:
    """
    Send the image bytes to Google Vision and get raw text lines back.
    If Vision setup failed (client is None), return [].
    """
    if not _VISION_AVAILABLE or _vision_client is None:
        print("Vision client not available, returning [].")
        return []

    image = vision.Image(content=jpeg_bytes)
    response = _vision_client.text_detection(image=image)

    if response.error.message:
        print("Vision API error:", response.error.message)
        return []

    full_text = response.full_text_annotation.text or ""
    lines = [ln.strip() for ln in full_text.splitlines()]
    return lines

def parse_lines_to_items(lines: List[str]) -> List[Dict[str, object]]:
    """
    Take OCR text lines, clean them, group duplicates, guess qty/category/image.
    Now we also call guess_branded_image_url() so we can attach
    brand-looking images (Sargento, Tide, etc) instead of just generic photos.
    """
    items: Dict[str, Dict[str, object]] = {}

    for raw in lines:
        raw = (raw or "").strip()
        if not raw:
            continue

        if is_noise_line(raw):
            continue

        without_price = strip_price(raw)
        without_price, qty = extract_qty(without_price)
        norm = normalize(without_price)

        if not looks_like_item(norm):
            continue

        display = canonical_name(norm)
        category = categorize_item(display)
        key = display.lower()

        # choose image
        img_url = guess_branded_image_url(original_line=raw, cleaned_display_name=display)

        if key in items:
            items[key]["quantity"] = int(items[key]["quantity"]) + qty
        else:
            items[key] = {
                "name": display,
                "quantity": qty,
                "category": category,
                "image_url": img_url,
            }

    return list(items.values())

# ===================== API ROUTES =====================

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/parse-receipt")
async def parse_receipt(file: UploadFile = File(...)):
    """
    1. Read the uploaded image (JPEG basically).
    2. Run Google Vision OCR to get all text lines.
    3. Parse those lines into structured items.
    4. Return items to the iOS app.
    """
    jpeg_bytes = await file.read()

    # light pre-process just like before
    try:
        pil_img = Image.open(io.BytesIO(jpeg_bytes)).convert("L")
        pil_img = ImageOps.autocontrast(pil_img)
        pil_img = pil_img.filter(ImageFilter.SHARPEN) if hasattr(Image, "filter") else pil_img
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=90)
        processed_bytes = buf.getvalue()
    except Exception as e:
        print("Preprocess failed, using raw bytes:", e)
        processed_bytes = jpeg_bytes

    # OCR via Google Vision
    lines = run_google_vision_ocr(processed_bytes)

    # Parse lines -> items
    items_list = parse_lines_to_items(lines)

    # If Vision failed or returned nothing, send back empty list (not 500)
    return JSONResponse(items_list)

# ===================== IMAGE SEARCH / DELIVERY =====================

# in-memory cache: product key -> final JPEG bytes
_IMAGE_CACHE: Dict[str, bytes] = {}

# nice manual fallbacks, like before
PRODUCT_IMAGE_MAP = {
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
    """Download bytes from a URL (jpg, png, webp allowed)."""
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            r = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return None
        ctype = r.headers.get("Content-Type", "").lower()
        if "image" not in ctype:
            return None
        return r.content
    except Exception as e:
        print("fetch_bytes error:", e)
        return None

async def search_google_shopping_image(query: str) -> Optional[bytes]:
    """
    Try to grab a product-looking thumbnail by hitting Google Images heuristically.
    This is best-effort and may fail sometimes. If it fails, we'll fall back.
    """
    try:
        q = urllib.parse.quote_plus(query + " product photo")
        url = (
            "https://www.google.com/search"
            "?tbm=isch&safe=active&hl=en&ijn=0&"
            f"q={q}"
        )

        async with httpx.AsyncClient(
            timeout=10.0,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept-Language": "en-US,en;q=0.9",
            },
        ) as client:
            resp = await client.get(url)

        if resp.status_code != 200:
            print("google image search bad status:", resp.status_code)
            return None

        html = resp.text

        # naive scrape: find first likely direct image URL
        m = re.search(r"https://[^\"']+\.(?:jpg|jpeg|png|webp)", html, re.IGNORECASE)
        if not m:
            print("no image url match in google html")
            return None

        img_url = m.group(0)
        print("image candidate:", img_url)

        return await fetch_bytes(img_url)

    except Exception as e:
        print("search_google_shopping_image error:", e)
        return None

@app.get("/image")
async def get_product_image(name: str = Query(..., description="product name to fetch image for")):
    """
    1. If we already cached bytes for this product name, return them.
    2. Else try live product-style image (search_google_shopping_image).
    3. Else fall back to our curated Unsplash-style photo.
    4. Else return a 1x1 pixel jpeg so the UI never breaks.
    """
    key = _canonical_lookup_key(name)

    # 1. cache hit?
    if key in _IMAGE_CACHE:
        return Response(content=_IMAGE_CACHE[key], media_type="image/jpeg")

    # 2. try live "product" image
    live_bytes = await search_google_shopping_image(name)
    if live_bytes:
        _IMAGE_CACHE[key] = live_bytes
        return Response(content=live_bytes, media_type="image/jpeg")

    # 3. fallback to our manual list / Unsplash
    fallback_url = PRODUCT_IMAGE_MAP.get(key, FALLBACK_PRODUCT_IMAGE)
    fallback_bytes = await fetch_bytes(fallback_url)
    if fallback_bytes:
        _IMAGE_CACHE[key] = fallback_bytes
        return Response(content=fallback_bytes, media_type="image/jpeg")

    # 4. absolute last resort: tiny 1x1 so the app UI has *something*
    tiny_jpeg = base64.b64decode(
        b"/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDABALDA4MChAODQ4SERATGCgaGBYW"
        b"GTEiJCQmLjQxND8+Q0RFRUZDXFtcYGNleXp4eXyDhIWGh4iJiouPkZCX/2wBD"
        b"ARATGBcaICTCjI+Pj5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+fn5+"
        b"fn5+fn5+fn5+fn5+fn5+fn5+/wAARCAABAAEDAREAAhEBAxEB/8QAFQABAQAAA"
        b"AAAAAAAAAAAAAAAAf/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAA"
        b"AAAAAAAAAAAP/EABQRAQAAAAAAAAAAAAAAAAAAAAD/2gAMAwEAAhEDEQA/AMf/"
        b"2Q=="
    )
    _IMAGE_CACHE[key] = tiny_jpeg
    return Response(content=tiny_jpeg, media_type="image/jpeg")
