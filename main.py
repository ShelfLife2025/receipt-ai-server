# ===================== IMAGE DELIVERY =====================

_IMAGE_CACHE: Dict[str, bytes] = {}
_IMAGE_CONTENT_TYPE_CACHE: Dict[str, str] = {}

# optional: avoid memory ballooning on long runs
_MAX_CACHE_ITEMS = 2000

# ✅ Your Instacart-style packshot proxy (your service)
# Example: https://shelflife-packshots.onrender.com
PACKSHOT_SERVICE_URL = (os.getenv("PACKSHOT_SERVICE_URL") or "").strip().rstrip("/")
PACKSHOT_SERVICE_KEY = (os.getenv("PACKSHOT_SERVICE_KEY") or "").strip()

# Keep your existing fallback map for now (until packshot service is live)
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


def _trim_caches_if_needed() -> None:
    if len(_IMAGE_CACHE) <= _MAX_CACHE_ITEMS:
        return
    _IMAGE_CACHE.clear()
    _IMAGE_CONTENT_TYPE_CACHE.clear()


async def fetch_image(url: str, headers: Optional[Dict[str, str]] = None) -> Optional[Tuple[bytes, str]]:
    """
    Fetch image bytes + content-type. We return the correct media_type so SwiftUI
    doesn’t choke if upstream is PNG/WEBP/etc.
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
        if "image" not in ctype:
            return None
        return r.content, ctype
    except Exception as e:
        print("fetch_image error:", e)
        return None


def _cache_key(name: str, upc: Optional[str], product_id: Optional[str]) -> str:
    # Prefer stable IDs if present
    if product_id:
        return f"pid:{product_id.strip()}"
    if upc:
        return f"upc:{re.sub(r'[^0-9]', '', upc)}"
    return f"name:{_canonical_lookup_key(name)}"


@app.get("/image")
async def get_product_image(
    name: str = Query(..., description="product name to fetch image for"),
    upc: Optional[str] = Query(None, description="UPC/GTIN if available"),
    product_id: Optional[str] = Query(None, description="catalog product id if available"),
):
    ck = _cache_key(name, upc, product_id)

    # serve from cache
    if ck in _IMAGE_CACHE and ck in _IMAGE_CONTENT_TYPE_CACHE:
        return Response(content=_IMAGE_CACHE[ck], media_type=_IMAGE_CONTENT_TYPE_CACHE[ck])

    # 1) ✅ Try your packshot service first (this is what becomes “Instacart-crisp”)
    if PACKSHOT_SERVICE_URL:
        qp = []
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
    key = _canonical_lookup_key(name)
    img_url = PRODUCT_IMAGE_MAP.get(key, FALLBACK_PRODUCT_IMAGE)

    result = await fetch_image(img_url)
    if result:
        img_bytes, ctype = result
        _IMAGE_CACHE[ck] = img_bytes
        _IMAGE_CONTENT_TYPE_CACHE[ck] = ctype
        _trim_caches_if_needed()
        return Response(content=img_bytes, media_type=ctype)

    # last-resort 1x1 png
    TINY_PNG_BASE64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMA"
        "ASsJTYQAAAAASUVORK5CYII="
    )
    tiny_bytes = base64.b64decode(TINY_PNG_BASE64)
    _IMAGE_CACHE[ck] = tiny_bytes
    _IMAGE_CONTENT_TYPE_CACHE[ck] = "image/png"
    _trim_caches_if_needed()
    return Response(content=tiny_bytes, media_type="image/png")
