"""
main.py — ShelfLife receipt + image service (Render-ready)

✅ Deploy-safe (no missing Dict/Optional/etc.)
✅ /parse-receipt works (Google Vision OCR)
✅ /image works (packshot proxy first → fallback map → tiny png)
✅ /health for quick checks

Start command on Render:
uvicorn main:app --host 0.0.0.0 --port $PORT
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
from fastapi import FastAPI, File, UploadFile, Query, Request
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageOps, ImageFilter

# Google Vision
from google.cloud import vision


app = FastAPI()

# ---------------- GOOGLE VISION SETUP ----------------
# Render provides GOOGLE_APPLICATION_CREDENTIALS_JSON as an env var (literal JSON).
# We write it to disk and set GOOGLE_APPLICATION_CREDENTIALS so Vision SDK can read it.
VISION_TMP_PATH = "/tmp/gcloud_key.json"


def _init_google_credentials_file() -> None:
    creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    # If user already set a file path, don't override
    if not creds_json:
        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            print("Google creds: using GOOGLE_APPLICATION_CREDENTIALS already set.")
        else:
            print("Google creds: GOOGLE_APPLICATION_CREDENTIALS_JSON not set (OCR will fail).")
        return

    try:
        # Accept either a raw JSON string or something already JSON-encoded
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
    r"\btotal\b", r"\bsub\s*total\b", r"\btax\b", r"\bbalance\b",
    r"\bchange\b", r"\bamount\b", r"\bpayment\b", r"\bdebit\b", r"\bcredit\b",
    r"\bvisa\b", r"\bmastercard\b", r"\bamex\b", r"\bdiscover\b",
    r"\baccount\b", r"\bacct\b", r"\btrace\b", r"\bauth\b", r"\bapproval\b",
    r"\bregister\b", r"\bcashier\b", r"\bmanager\b", r"\bstore\b",
    r"\bthank you\b", r"\bthanks\b", r"\breturn\b", r"\brefund\b",
    r"\btransaction\b", r"\bterminal\b", r"\bcard\b", r"\bchip\b",
    r"\bmerchant\b", r"\bpin\b", r"\bsignature\b",
    r"\bphone\b", r"\bwww\.", r"\.com\b",
    r"\baddress\b", r"\bcity\b", r"\bst\b", r"\bave\b", r"\broad\b",
    r"\breceipt\b",
    r"\bitems?\b\s*\d+\b",
    r"^\s*#\s*\d+\s*$",
]

NOISE_RE = re.compile("|".join(f"(?:{p})" for p in NOISE_PATTERNS), re.IGNORECASE)

# A light household keyword list (tune over time)
HOUSEHOLD_WORDS = {
    "paper", "towel", "towels", "toilet", "tissue", "napkin", "napkins",
    "detergent", "bleach", "cleaner", "wipes", "wipe", "soap", "dish", "dawn",
    "shampoo", "conditioner", "deodorant", "toothpaste", "floss", "razor",
    "trash", "garbage", "bag", "bags", "foil", "wrap", "parchment",
    "rubbing", "alcohol", "isopropyl", "cotton", "swab", "swabs",
    "battery", "batteries", "lightbulb", "lighter", "matches",
    "pet", "litter",
}

UNIT_PRICE_RE = re.compile(r"\b\d+\s*@\s*\$?\d+(?:\.\d{1,2})?\b", re.IGNORECASE)
MONEY_RE = re.compile(r"\$?\d+(?:\.\d{1,2})")


def dedupe_key(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def title_case(s: str) -> str:
    # simple title-case with small-word handling
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


def _looks_like_item(line: str) -> bool:
    if not line:
        return False
    if NOISE_RE.search(line):
        return False

    # Must have at least one letter
    if not re.search(r"[A-Za-z]", line):
        return False

    # Exclude "pure money" lines
    stripped = re.sub(r"\s+", "", line)
    if re.fullmatch(r"[$0-9\.,]+", stripped):
        return False

    # Avoid very long header-ish lines
    if len(line) > 64:
        return False

    return True


def _clean_line(line: str) -> str:
    s = (line or "").strip()

    # Remove common trailing price tokens (keep name; quantities handled separately)
    # Examples: "MILK 3.99" or "MILK $3.99"
    s = re.sub(r"\s+\$?\d+(?:\.\d{1,2})\s*$", "", s)

    # Remove "3 @ 2.99" patterns (unit price; keep qty parsing separately)
    s = UNIT_PRICE_RE.sub("", s).strip()

    # Remove extraneous symbols
    s = s.replace("—", "-")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_quantity(line: str) -> tuple[int, str]:
    """
    Returns (quantity, remaining_name).
    Handles:
    - "2x MILK", "2 x MILK", "MILK x2"
    - leading number: "3 BANANAS"
    - patterns like "3 @ 2.99" (already removed in clean_line; but still handle)
    """
    s = (line or "").strip()

    # "x2" at end
    m = re.search(r"(.*?)\b[xX]\s*(\d+)\s*$", s)
    if m:
        name = m.group(1).strip()
        qty = int(m.group(2))
        return max(qty, 1), name

    # "2x" at start
    m = re.match(r"^\s*(\d+)\s*[xX]\s+(.*)$", s)
    if m:
        qty = int(m.group(1))
        name = m.group(2).strip()
        return max(qty, 1), name

    # leading integer qty: "3 BANANAS"
    m = re.match(r"^\s*(\d+)\s+(.*)$", s)
    if m:
        qty = int(m.group(1))
        name = m.group(2).strip()
        # Avoid treating years/receipt numbers as qty
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

            # If either version says Household, keep it Household
            if merged[k].get("category") == "Household" or it.get("category") == "Household":
                merged[k]["category"] = "Household"

    # stable output
    return sorted(merged.values(), key=lambda x: x["name"].lower())


# ---------------- OCR ----------------

def _preprocess_image_bytes(data: bytes) -> bytes:
    """
    Small preprocessing to help OCR quality.
    Returns PNG bytes.
    """
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

    # document_text_detection is usually better for receipts than text_detection
    resp = client.document_text_detection(image=image)

    if resp.error and resp.error.message:
        raise RuntimeError(resp.error.message)

    if resp.full_text_annotation and resp.full_text_annotation.text:
        return resp.full_text_annotation.text

    # Fallback
    if resp.text_annotations:
        return resp.text_annotations[0].description or ""

    return ""


# ---------------- ROUTES ----------------

@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True}


@app.post("/parse-receipt")
async def parse_receipt(
    file: UploadFile = File(...),
    debug: bool = Query(False, description="include debug fields"),
):
    raw = await file.read()
    if not raw:
        return JSONResponse(status_code=400, content={"error": "Empty file"})

    try:
        pre = _preprocess_image_bytes(raw)
        text = ocr_text_google_vision(pre)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"OCR failed: {str(e)}", "hint": "Check GOOGLE_APPLICATION_CREDENTIALS_JSON / GOOGLE_APPLICATION_CREDENTIALS"},
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
        if not name or not re.search(r"[A-Za-z]", name):
            continue

        item = {
            "name": title_case(name),
            "quantity": int(qty),
            "category": _classify(name),
        }
        parsed.append(item)

    parsed = _dedupe_and_merge(parsed)

    if debug:
        return {
            "items": parsed,
            "raw_line_count": len(lines),
            "kept_line_count": len(kept),
            "kept_lines": kept[:200],
        }

    return parsed


# ===================== IMAGE DELIVERY =====================

_IMAGE_CACHE: dict[str, bytes] = {}
_IMAGE_CONTENT_TYPE_CACHE: dict[str, str] = {}

# optional: avoid memory ballooning on long runs
_MAX_CACHE_ITEMS = 2000

# ✅ Your Instacart-style packshot proxy (your service)
# Example: https://shelflife-packshots.onrender.com
PACKSHOT_SERVICE_URL = (os.getenv("PACKSHOT_SERVICE_URL") or "").strip().rstrip("/")
PACKSHOT_SERVICE_KEY = (os.getenv("PACKSHOT_SERVICE_KEY") or "").strip()

# Keep your existing fallback map for now (until packshot service is live)
PRODUCT_IMAGE_MAP: dict[str, str] = {
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
    """
    Fetch image bytes + content-type. Return correct media_type so SwiftUI
    won’t choke if upstream is PNG/WEBP/etc.
    """
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

    # serve from cache
    if ck in _IMAGE_CACHE and ck in _IMAGE_CONTENT_TYPE_CACHE:
        return Response(content=_IMAGE_CACHE[ck], media_type=_IMAGE_CONTENT_TYPE_CACHE[ck])

    # 1) ✅ Try your packshot service first (this becomes “Instacart-crisp”)
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

    # 2) Fallback to your local map (name-based)
    key = dedupe_key(name)
    img_url = PRODUCT_IMAGE_MAP.get(key, FALLBACK_PRODUCT_IMAGE)

    result = await fetch_image(img_url)
    if result:
        img_bytes, ctype = result
        _IMAGE_CACHE[ck] = img_bytes
        _IMAGE_CONTENT_TYPE_CACHE[ck] = ctype
        _trim_caches_if_needed()
        return Response(content=img_bytes, media_type=ctype)

    # 3) last-resort 1x1 png
    tiny_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMA"
        "ASsJTYQAAAAASUVORK5CYII="
    )
    _IMAGE_CACHE[ck] = tiny_bytes
    _IMAGE_CONTENT_TYPE_CACHE[ck] = "image/png"
    _trim_caches_if_needed()
    return Response(content=tiny_bytes, media_type="image/png")


# Optional: request logging helps when debugging deploys
@app.middleware("http")
async def _log_requests(request: Request, call_next):
    try:
        resp = await call_next(request)
        return resp
    finally:
        print(f"{request.method} {request.url.path}")
