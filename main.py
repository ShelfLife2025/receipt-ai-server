# main.py
# FastAPI receipt parser with noise filtering, canonicalization, and deduplication.

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List, Tuple, Optional, Dict
from PIL import Image, ImageOps, ImageFilter
import pytesseract
import io
import re
from difflib import get_close_matches

app = FastAPI()

# ===================== Canonicalization / Dictionaries =====================

# Common store/brand words we want to strip if they appear in line items.
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

# Size/pack tokens to remove: "8 oz", "12oz", "1 lb", "2ct", "3 pk", etc.
SIZE_REGEX = re.compile(
    r"(?<!\w)("
    r"\d+(\.\d+)?\s?(fl\s?oz|oz|lb|lbs|ct|pk|pkg|g|kg|ml|l|qt|pt|gal)|"
    r"\d+\s?(pack|pk|ct|lb|oz)"
    r")(?!\w)",
    re.IGNORECASE,
)

# Product/Coupon/PLU/long-code-ish chunks to remove
CODEY_REGEX = re.compile(r"(^|[^A-Za-z])([bq]?(upc|plu|sku)\b[: ]?\d+|\d{6,})($|[^A-Za-z])", re.IGNORECASE)

# Abbreviation/keyword expansions → canonical fragments
TOKEN_MAP = {
    # cheese
    "parm": "parmesan", "parma": "parmesan", "parmesan": "parmesan",
    "ches": "cheese", "chz": "cheese", "cheez": "cheese",
    "mozz": "mozzarella",
    # snacks / crackers
    "snk": "snack",
    "crkr": "crackers", "crkrs": "crackers",
    # proteins / basics
    "grnd": "ground", "org": "organic", "bn": "bean", "bns": "beans",
    "chk": "chicken", "brst": "breast",
    # produce
    "grlc": "garlic", "grl": "garlic", "tmt": "tomato", "toma": "tomato", "spn": "spinach",
    "strwb": "strawberry",
    # beverages / misc
    "wtr": "water", "blk": "black", "whl": "whole"
}

# Household keyword hinting
HOUSEHOLD_HINTS = {
    "detergent","laundry","pods","dish","dishwashing","soap","bleach","cleaner",
    "toilet","bath tissue","paper towel","paper towels","towel",
    "wipes","disinfecting","foil","baggies","bags","trash","liners","sponges"
}

# A broader list of FOOD-ish tokens. A line must contain at least one of these
# (or contain a HOUSEHOLD_HINT) after normalization to count as an item.
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

# Price / quantity patterns
PRICE_REGEX = re.compile(r"\$?\d+\.\d{2}")
QTY_TOKEN_REGEX = re.compile(r"(?:\bqty[:=]?\s*(\d+)\b)|(?:\b(\d+)\s*x\b)|(?:x\s*(\d+)\b)", re.IGNORECASE)

# ======= Extra noise patterns (headers, payments, addresses, ads, system lines) =======
NOISE_PATTERNS = [
    r"^apply\b", r"jobs\b", r"publix\.?jobs", r"\bcareer\b",
    r"^payment", r"^entry\s+method", r"^you\s?saved\b", r"^savings\b",
    r"^acct[:# ]", r"^account\b", r"^trace[:# ]", r"\bapproval\b", r"\bauth\b",
    r"\bsubtotal\b", r"\bsub\s*total\b", r"\btax\b", r"\btotal\b", r"\bchange\b", r"\bbalance\b",
    r"\bdebit\b", r"\bcredit\b", r"\bvisa\b", r"\bmastercard\b", r"\bdiscover\b", r"\bamex\b",
    r"\bthank you\b|\bthanks\b", r"\bcashier\b|\bmanager\b|\bstore\b\s*#",
    r"\bpresto!?$", r"\bpresto!\b", r"\bplaza\b|\bmall\b|\bmillenia\b",
    r"https?://", r"\bwww\.", r"@[A-Za-z0-9_]+",                         # urls/handles
    r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",                                # phone numbers
    r"\b\d{1,4}\s+[A-Za-z0-9]+\s+(ave|avenue|st|street|rd|road|blvd|drive|dr)\b",  # addresses
    r"\btrace\s*#?:?\s*\d+\b", r"\bacct\s*#?:?\s*\w+\b",
    r"\bcontactless\b|\btap\b|\bchip\b", r"\bmerchant\b|\bterminal\b|\bpos\b",
    r"\bgrocery\s?item\b$"  # fallback junk label we sometimes produce
]

# ===================== Helpers =====================

def is_noise_line(text: str) -> bool:
    """Filter out obvious non-item lines before any normalization."""
    t = (text or "").strip().lower()
    if not t or len(t) < 3:
        return True

    # Reject if matches any explicit noise pattern
    for pat in NOISE_PATTERNS:
        if re.search(pat, t):
            return True

    # Too few letters → likely code/blank/noise
    letters = sum(c.isalpha() for c in t)
    if letters < 2:
        return True

    return False

def normalize(text: str) -> str:
    """Lowercase, remove brands/sizes/codes, expand tokens, collapse spaces."""
    t = (text or "").lower()
    t = t.replace("&", " and ")

    # remove brand words (whole-word)
    for bw in BRAND_WORDS:
        t = re.sub(rf"\b{re.escape(bw)}\b", " ", t, flags=re.IGNORECASE)

    # remove sizes (8 oz, 2pk, etc.)
    t = SIZE_REGEX.sub(" ", t)

    # remove long numeric/code-ish chunks
    t = CODEY_REGEX.sub(" ", t)

    # expand common tokens and rebuild
    words: List[str] = []
    for w in re.split(r"[^a-z0-9]+", t):
        if not w:
            continue
        words.append(TOKEN_MAP.get(w, w))

    t = " ".join(words)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def extract_qty(text: str) -> Tuple[str, int]:
    """Extract and remove a quantity if present. Defaults to 1."""
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
    """Require at least one FOOD or HOUSEHOLD hint to consider it an item."""
    if not tokens:
        return False
    joined = " ".join(tokens)
    # Household?
    for h in HOUSEHOLD_HINTS:
        if h in joined:
            return True
    # Food?
    for f in FOOD_HINTS:
        if f in joined:
            return True
    # Light fallback: if it ends with a common food-y plural/singular noun pattern
    # like 'crackers', 'tomatoes', 'apples', 'chicken', 'bread'
    # (but avoid super-short 1-2 char tokens)
    last = tokens[-1]
    if len(last) >= 4 and (last.endswith("s") or last in {"bread","chicken","beef","pork","fish"}):
        return True
    return False

def looks_like_item(cleaned: str) -> bool:
    """Heuristic to decide if the cleaned line is an actual product."""
    t = cleaned.strip().lower()
    if not t:
        return False
    if sum(c.isalpha() for c in t) < 3:
        return False
    toks = _tokens(t)
    if not tokens_have_hint(toks):
        return False
    # Avoid still-codey strings
    if re.fullmatch(r"[a-z]{0,2}\d{4,}", t):
        return False
    return True

def categorize_item(name: str) -> str:
    """Very simple Food/Household split."""
    t = name.lower()
    for h in HOUSEHOLD_HINTS:
        if h in t:
            return "Household"
    return "Food"

def canonical_name(name: str) -> str:
    """
    Map common abbreviations to a friendly display. Lightweight so it generalizes.
    """
    t = name.lower()

    # Common grocery names
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

    # Household simplifications
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

    # Fallback: Title-case cleaned string
    return t[:1].upper() + t[1:]

def parse_line(raw: str) -> Optional[Tuple[str, int, str]]:
    """
    Return (display_name, qty, category) for one OCR line, or None if not a grocery item.
    """
    if is_noise_line(raw):
        return None

    s = strip_price(raw)
    s, qty = extract_qty(s)
    s = normalize(s)

    if not looks_like_item(s):
        return None

    # Last sanity filters: drop obvious stray fragments
    bad_fragments = {"payment", "entry", "method", "apply", "trace", "you", "saved", "presto"}
    if any(frag in s.split() for frag in bad_fragments):
        return None

    display = canonical_name(s)
    category = categorize_item(display)
    return display, max(1, qty), category

# ===================== API =====================

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/parse-receipt")
async def parse_receipt(file: UploadFile = File(...)):
    # 1) Load and lightly enhance image for OCR
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale
    image = ImageOps.autocontrast(image)
    image = Image.filter(ImageFilter.SHARPEN) if hasattr(Image, "filter") else image

    # 2) OCR
    # psm 6 = assume a single uniform block of text; good default for receipts
    text = pytesseract.image_to_string(image, config="--psm 6")

    # 3) Split lines and parse
    items: Dict[str, Dict[str, object]] = {}
    for raw in text.splitlines():
        raw = (raw or "").strip()
        if not raw:
            continue
        parsed = parse_line(raw)
        if not parsed:
            continue

        name, qty, category = parsed
        key = name.lower()
        if key in items:
            items[key]["quantity"] = int(items[key]["quantity"]) + qty
        else:
            items[key] = {
                "name": name,
                "quantity": qty,
                "category": category,
            }

    # 4) Return grouped items as a list
    return JSONResponse(list(items.values()))

