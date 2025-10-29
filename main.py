# main.py
# FastAPI receipt parser with noise filtering, canonicalization, and deduplication.

from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse, Response
from typing import List, Tuple, Optional, Dict
from PIL import Image, ImageOps, ImageFilter
import io
import re
import urllib.parse  # for safe URL encoding
import httpx
import os

# Tell Google client libraries where to find your service account credentials.
# key.json must exist in the same folder as this file on the server.
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"

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

# ===================== API =====================

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/parse-receipt")
async def parse_receipt(file: UploadFile = File(...)):
    """
    TEMP VERSION:
    - We ignore OCR completely (no pytesseract)
    - We just return a few fake but realistic items, using guess_image_url
    - This keeps the iOS app happy and unblocks you
    """

    # read uploaded image so FastAPI doesn't complain about unused await
    _ = await file.read()

    fake_items = [
        {
            "name": "Parmesan cheese",
            "quantity": 1,
            "category": "Food",
            "image_url": guess_image_url("Parmesan cheese"),
        },
        {
            "name": "Bread",
            "quantity": 1,
            "category": "Food",
            "image_url": guess_image_url("Bread"),
        },
        {
            "name": "Laundry detergent",
            "quantity": 1,
            "category": "Household",
            "image_url": guess_image_url("Laundry detergent"),
        },
    ]

    return JSONResponse(fake_items)

# ===================== IMAGE ROUTE =====================

_IMAGE_CACHE: Dict[str, bytes] = {}

PRODUCT_IMAGE_MAP = {
    "parmesan cheese": "https://images.unsplash.com/photo-1589301760014-d929f3979dbc?w=512&q=80",
    "bread": "https://images.unsplash.com/photo-1608198093002-de0e3580bb67?w=512&q=80",
    "yogurt": "https://images.unsplash.com/photo-1589302168068-964664d93dc0?w=512&q=80",
    "garlic": "https://images.unsplash.com/photo-1506806732259-39c2d0268443?w=512&q=80",
    "tomatoes": "https://images.unsplash.com/photo-1567306226416-28f0efdc88ce?w=512&q=80",
    "dr pepper": "https://images.unsplash.com/photo-1621451532593-49f463c06d65?w=512&q=80",
    "panera mac & cheese": "https://images.unsplash.com/photo-1604908177071-6c2b7b66010c?w=512&q=80",
    "mozzarella cheese": "https://images.unsplash.com/photo-1600166898747-96f2ef749acd?w=512&q=80",
    "cheddar cheese": "https://images.unsplash.com/photo-1601004890684-d8cbf643f5f2?w=512&q=80",
    "crackers": "https://images.unsplash.com/photo-1603048297340-5e05ad3e3b80?w=512&q=80",
    "paper towels": "https://images.unsplash.com/photo-1581579186981-5f3f0c612e75?w=512&q=80",
    "toilet paper": "https://images.unsplash.com/photo-1584559582151-fb1dfa8b33a5?w=512&q=80",
    "laundry detergent": "https://images.unsplash.com/photo-1581579187080-9f31fa8c7c53?w=512&q=80",
    "laundry pods": "https://images.unsplash.com/photo-1581579187080-9f31fa8c7c53?w=512&q=80",
    "water": "https://images.unsplash.com/photo-1561043433-aaf687c4cf4e?w=512&q=80",
    "rao marinara sauce": "https://images.unsplash.com/photo-1611075389455-2f43fa462446?w=512&q=80",
    "cream cheese": "https://images.unsplash.com/photo-1589301763197-9713a1e1e5c0?w=512&q=80",
    "bagels": "https://images.unsplash.com/photo-1509440159596-0249088772ff?w=512&q=80",
    "grapes": "https://images.unsplash.com/photo-1601004890211-3f3d02dd3c10?w=512&q=80",
}

FALLBACK_PRODUCT_IMAGE = "https://images.unsplash.com/photo-1604908177071-6c2b7b66010c?w=512&q=80"

@app.get("/image")
async def get_product_image(name: str = Query(..., description="product name to fetch image for")):
    key = _canonical_lookup_key(name)
    img_url = PRODUCT_IMAGE_MAP.get(key, FALLBACK_PRODUCT_IMAGE)

    if key in _IMAGE_CACHE:
        return Response(content=_IMAGE_CACHE[key], media_type="image/jpeg")

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(img_url)
        if r.status_code != 200:
            return Response(status_code=502)
        data = r.content

    _IMAGE_CACHE[key] = data
    return Response(content=data, media_type="image/jpeg")
