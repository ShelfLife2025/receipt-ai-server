"""
main.py — ShelfLife receipt + image service (Render-ready)

- Deploy-safe
- /parse-receipt (Google Vision OCR)
- /image (packshot proxy → fallback map → tiny png)
- /health

Filtering:
- Blocks random pulls hard (price-only, weight-only, short junk tokens, addresses/headers/totals/meta)
- Filters to only grocery (Food) items
- Expands common receipt abbreviations into full words/phrases

Instacart:
- /instacart/create-list creates an Instacart shopping-list link for a set of items
"""

from __future__ import annotations

import os
import re
import io
import json
import base64
import urllib.parse
from typing import Any

import httpx
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, Query, Request, HTTPException
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageOps, ImageFilter

from google.cloud import vision

app = FastAPI()

# ---------------- GOOGLE VISION SETUP ----------------
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


_init_google_credentials_file()

# ---------------- BASIC HELPERS ----------------

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

UNIT_PRICE_RE = re.compile(r"\b\d+\s*@\s*\$?\d+(?:\.\d{1,2})?\b", re.IGNORECASE)
MONEY_TOKEN_RE = re.compile(r"\$?\d{1,6}(?:\.\d{2})\b")
ONLY_MONEYISH_RE = re.compile(r"^\s*\$?\d+(?:\.\d{2})?\s*$")
PRICE_FLAG_RE = re.compile(r"^\s*\$?\d+(?:\.\d{2})\s*[A-Za-z]{1,2}\s*$")

# weight/unit-only line killers
WEIGHT_ONLY_RE = re.compile(
    r"^\s*\$?\d+(?:\.\d+)?\s*(lb|lbs|oz|g|kg|ct)\s*$",
    re.IGNORECASE,
)
PER_UNIT_PRICE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*/\s*(lb|lbs|oz|g|kg|ea|each|ct)\b",
    re.IGNORECASE,
)

# word token that looks like a real item word (>=2 letters)
REAL_WORD_RE = re.compile(r"[A-Za-z]{2,}")

# “words” that should NOT count as an item word
STOP_ITEM_WORDS = {
    "lb", "lbs", "oz", "g", "kg", "ct", "ea", "each",
    "w", "wt", "weight",
    "at", "x",
    "vov",
}

# short junk codes: kill if line is ONLY a short token like "VOV"
SHORT_CODE_ONLY_RE = re.compile(r"^\s*[A-Za-z]{2,5}\s*$")

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
    small = {"and", "or", "of", "the", "a", "an", "to", "in", "on", "for"}
    words = [w for w in re.split(r"\s+", (s or "").strip()) if w]
    out: list[str] = []
    for i, w in enumerate(words):
        lw = w.lower()
        if i != 0 and lw in small:
            out.append(lw)
        else:
            out.append(lw[:1].upper() + lw[1:])
    return " ".join(out).strip()


# ---------------- ABBREVIATION EXPANSION ----------------

ABBREV_TOKEN_MAP: dict[str, str] = {
    # dairy
    "crm": "cream",
    "chs": "cheese",
    "whp": "whipping",
    "whp.": "whipping",
    "hvy": "heavy",
    "hvy.": "heavy",
    "sour": "sour",
    "cr": "cream",
    "chs.": "cheese",
    "bttr": "butter",
    "marg": "margarine",
    "yog": "yogurt",

    # pantry / staples
    "veg": "vegetable",
    "org": "organic",
    "wb": "whole",
    "grd": "ground",
    "bf": "beef",
    "chk": "chicken",
    "brst": "breast",
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

    norm = re.sub(r"\s+", " ", norm).strip()
    return norm


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
    sl = s.lower().strip()

    if ONLY_MONEYISH_RE.match(s):
        return False
    if PRICE_FLAG_RE.match(s):
        return False
    if WEIGHT_ONLY_RE.match(s):
        return False
    if SHORT_CODE_ONLY_RE.match(s) and sl in STOP_ITEM_WORDS:
        return False

    if PER_UNIT_PRICE_RE.search(s) and not _has_valid_item_words(s):
        return False

    if NOISE_RE.search(s):
        return False
    if _is_header_or_address(s):
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

    s = re.sub(r"\s+\$?\d+(?:\.\d{2})\s*$", "", s)
    s = UNIT_PRICE_RE.sub("", s).strip()
    s = re.sub(r"(?:\s+\$?\d+(?:\.\d{2}))+\s*$", "", s).strip()
    s = re.sub(r"\b\d+(?:\.\d+)?\s*/\s*(lb|lbs|oz|g|kg|ea|each|ct)\b", "", s, flags=re.IGNORECASE).strip()

    s = s.replace("—", "-")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_quantity(line: str) -> tuple[int, str]:
    s = (line or "").strip()

    m = re.search(r"(.*?)\b[xX]\s*(\d+)\s*$", s)
    if m:
        name = m.group(1).strip()
        qty = int(m.group(2))
        return max(qty, 1), name

    m = re.match(r"^\s*(\d+)\s*[xX]\s+(.*)$", s)
    if m:
        qty = int(m.group(1))
        name = m.group(2).strip()
        return max(qty, 1), name

    m = re.match(r"^\s*(\d+)\s+(.*)$", s)
    if m:
        qty = int(m.group(1))
        name = m.group(2).strip()
        if 1 <= qty <= 50:
            return qty, name

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


# ---------------- OCR ----------------

def _preprocess_image_bytes(data: bytes) -> bytes:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = ImageOps.exif_transpose(img)
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


# ---------------- ROUTES ----------------

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

    lines = [ln.strip() for ln in (text or "").splitlines()]
    lines = [ln for ln in lines if ln]

    kept: list[str] = []
    for ln in lines:
        if _looks_like_item(ln):
            cleaned = _clean_line(ln)
            if cleaned and _looks_like_item(cleaned):
                kept.append(cleaned)

    parsed: list[dict[str, Any]] = []
    for ln in kept:
        qty, name = _parse_quantity(ln)
        name = _clean_line(name)

        name = expand_abbreviations(name)

        if not name:
            continue
        if ONLY_MONEYISH_RE.match(name) or PRICE_FLAG_RE.match(name) or WEIGHT_ONLY_RE.match(name):
            continue
        if NOISE_RE.search(name) or _is_header_or_address(name):
            continue
        if not _has_valid_item_words(name):
            continue

        category = _classify(name)

        if category != "Food":
            continue

        parsed.append(
            {
                "name": title_case(name),
                "quantity": int(qty),
                "category": category,
            }
        )

    parsed = _dedupe_and_merge(parsed)

    base_url = _public_base_url(request)
    for it in parsed:
        it["image_url"] = _image_url_for_item(base_url, it["name"])

    if debug:
        return {
            "items": parsed,
            "raw_line_count": len(lines),
            "kept_line_count": len(kept),
            "kept_lines": kept[:200],
            "base_url": base_url,
        }

    return parsed


# ---------------- INSTACART LIST LINK ----------------

class InstacartLineItem(BaseModel):
    name: str
    quantity: float = 1.0
    unit: str = "each"


class InstacartCreateListRequest(BaseModel):
    title: str = "ShelfLife Shopping List"
    items: list[InstacartLineItem]


@app.post("/instacart/create-list")
async def instacart_create_list(req: InstacartCreateListRequest):
    api_key = (os.getenv("INSTACART_API_KEY") or "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing INSTACART_API_KEY env var on server")

    payload = {
        "title": req.title,
        "link_type": "shopping_list",
        "line_items": [
            {"name": i.name, "quantity": i.quantity, "unit": i.unit}
            for i in req.items
        ],
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


# ===================== IMAGE DELIVERY =====================

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


async def fetch_image(url: str, headers: dict[str, str] | None = None) -> tuple[bytes, str] | None:
    try:
        async with httpx.AsyncClient(
            timeout=12.0,
            follow_redirects=True,
            headers=headers or {"User-Agent": "Mozilla/5.0"},
        ) as client:
            r = await client.get(url)
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
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMA"
        "ASsJTYQAAAAASUVORK5CYII="
    )
    _IMAGE_CACHE[ck] = tiny_bytes
    _IMAGE_CONTENT_TYPE_CACHE[ck] = "image/png"
    _trim_caches_if_needed()
    return Response(content=tiny_bytes, media_type="image/png")


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    try:
        resp = await call_next(request)
        return resp
    finally:
        print(f"{request.method} {request.url.path}")
