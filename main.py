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
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from google.cloud import vision
import google.generativeai as genai
from PIL import Image, ImageFilter, ImageOps
from pydantic import BaseModel

app = FastAPI()

OFF_CLIENT: Optional[httpx.AsyncClient] = None
IMG_CLIENT: Optional[httpx.AsyncClient] = None

OFF_SEM: Optional[asyncio.Semaphore] = None
ENRICH_SEM: Optional[asyncio.Semaphore] = None
OFF_BUDGET_LOCK: Optional[asyncio.Lock] = None

REQUEST_DEADLINE_SECONDS = float(os.getenv("REQUEST_DEADLINE_SECONDS", "12.0"))
ENRICH_TIMEOUT_SECONDS = float(os.getenv("ENRICH_TIMEOUT_SECONDS", "1.8"))

ENABLE_NAME_ENRICH = (os.getenv("ENABLE_NAME_ENRICH", "1").strip() == "1")
ENRICH_MIN_CONF = float(os.getenv("ENRICH_MIN_CONF", "0.58"))

ENRICH_FORCE_BEST_EFFORT = (os.getenv("ENRICH_FORCE_BEST_EFFORT", "1").strip() == "1")
ENRICH_FORCE_SCORE_FLOOR = float(os.getenv("ENRICH_FORCE_SCORE_FLOOR", "0.50"))

MAX_OFF_LOOKUPS_PER_REQUEST = int(os.getenv("MAX_OFF_LOOKUPS_PER_REQUEST", "12"))

OFF_CONCURRENCY = int(os.getenv("OFF_CONCURRENCY", "4"))
ENRICH_CONCURRENCY = int(os.getenv("ENRICH_CONCURRENCY", "6"))

OFF_SEARCH_URL_US = (os.getenv("OFF_SEARCH_URL_US") or "https://us.openfoodfacts.org/cgi/search.pl").strip()
OFF_SEARCH_URL_WORLD = (os.getenv("OFF_SEARCH_URL_WORLD") or "https://world.openfoodfacts.org/cgi/search.pl").strip()

OPF_SEARCH_URL_WORLD = (os.getenv("OPF_SEARCH_URL_WORLD") or "https://world.openproductsfacts.org/cgi/search.pl").strip()
OBF_SEARCH_URL_WORLD = (os.getenv("OBF_SEARCH_URL_WORLD") or "https://world.openbeautyfacts.org/cgi/search.pl").strip()

OFF_PAGE_SIZE = int(os.getenv("OFF_PAGE_SIZE", "12"))
OFF_FIELDS = (os.getenv("OFF_FIELDS") or "product_name,product_name_en,brands,quantity").strip()

NAME_MAP_JSON = (os.getenv("NAME_MAP_JSON") or "").strip()
NAME_MAP_PATH = (os.getenv("NAME_MAP_PATH") or "/tmp/name_map.json").strip()

PENDING_PATH = (os.getenv("PENDING_PATH") or "/tmp/pending_map.json").strip()
PENDING_ENABLED = (os.getenv("PENDING_ENABLED", "1").strip() == "1")

ADMIN_KEY = (os.getenv("ADMIN_KEY") or "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

VISION_TMP_PATH = "/tmp/gcloud_key.json"
FOOD_KNOWLEDGE_FALLBACK: Dict[str, Dict] = {
    "chicken breast":    {"expires_in_days": 2,    "storage": "fridge",   "category": "Food"},
    "chicken":           {"expires_in_days": 2,    "storage": "fridge",   "category": "Food"},
    "ground beef":       {"expires_in_days": 2,    "storage": "fridge",   "category": "Food"},
    "beef":              {"expires_in_days": 3,    "storage": "fridge",   "category": "Food"},
    "steak":             {"expires_in_days": 3,    "storage": "fridge",   "category": "Food"},
    "pork":              {"expires_in_days": 3,    "storage": "fridge",   "category": "Food"},
    "bacon":             {"expires_in_days": 7,    "storage": "fridge",   "category": "Food"},
    "salmon":            {"expires_in_days": 2,    "storage": "fridge",   "category": "Food"},
    "shrimp":            {"expires_in_days": 2,    "storage": "fridge",   "category": "Food"},
    "fish":              {"expires_in_days": 2,    "storage": "fridge",   "category": "Food"},
    "deli meat":         {"expires_in_days": 5,    "storage": "fridge",   "category": "Food"},
    "hot dogs":          {"expires_in_days": 7,    "storage": "fridge",   "category": "Food"},
    "sausage":           {"expires_in_days": 3,    "storage": "fridge",   "category": "Food"},
    "milk":              {"expires_in_days": 7,    "storage": "fridge",   "category": "Food"},
    "eggs":              {"expires_in_days": 21,   "storage": "fridge",   "category": "Food"},
    "butter":            {"expires_in_days": 30,   "storage": "fridge",   "category": "Food"},
    "cheese":            {"expires_in_days": 14,   "storage": "fridge",   "category": "Food"},
    "cream cheese":      {"expires_in_days": 10,   "storage": "fridge",   "category": "Food"},
    "sour cream":        {"expires_in_days": 14,   "storage": "fridge",   "category": "Food"},
    "yogurt":            {"expires_in_days": 14,   "storage": "fridge",   "category": "Food"},
    "heavy cream":       {"expires_in_days": 7,    "storage": "fridge",   "category": "Food"},
    "bread":             {"expires_in_days": 5,    "storage": "pantry",   "category": "Food"},
    "bagels":            {"expires_in_days": 5,    "storage": "pantry",   "category": "Food"},
    "tortillas":         {"expires_in_days": 7,    "storage": "pantry",   "category": "Food"},
    "garlic bread":      {"expires_in_days": 90,   "storage": "freezer",  "category": "Food"},
    "apples":            {"expires_in_days": 21,   "storage": "fridge",   "category": "Food"},
    "bananas":           {"expires_in_days": 5,    "storage": "pantry",   "category": "Food"},
    "oranges":           {"expires_in_days": 14,   "storage": "fridge",   "category": "Food"},
    "strawberries":      {"expires_in_days": 5,    "storage": "fridge",   "category": "Food"},
    "blueberries":       {"expires_in_days": 7,    "storage": "fridge",   "category": "Food"},
    "grapes":            {"expires_in_days": 7,    "storage": "fridge",   "category": "Food"},
    "lettuce":           {"expires_in_days": 7,    "storage": "fridge",   "category": "Food"},
    "spinach":           {"expires_in_days": 5,    "storage": "fridge",   "category": "Food"},
    "broccoli":          {"expires_in_days": 5,    "storage": "fridge",   "category": "Food"},
    "carrots":           {"expires_in_days": 21,   "storage": "fridge",   "category": "Food"},
    "tomatoes":          {"expires_in_days": 5,    "storage": "pantry",   "category": "Food"},
    "onions":            {"expires_in_days": 30,   "storage": "pantry",   "category": "Food"},
    "garlic":            {"expires_in_days": 90,   "storage": "pantry",   "category": "Food"},
    "potatoes":          {"expires_in_days": 30,   "storage": "pantry",   "category": "Food"},
    "avocado":           {"expires_in_days": 4,    "storage": "pantry",   "category": "Food"},
    "mushrooms":         {"expires_in_days": 5,    "storage": "fridge",   "category": "Food"},
    "ravioli":           {"expires_in_days": 3,    "storage": "fridge",   "category": "Food"},
    "fresh pasta":       {"expires_in_days": 3,    "storage": "fridge",   "category": "Food"},
    "hummus":            {"expires_in_days": 7,    "storage": "fridge",   "category": "Food"},
    "salsa":             {"expires_in_days": 14,   "storage": "fridge",   "category": "Food"},
    "frozen pizza":      {"expires_in_days": 180,  "storage": "freezer",  "category": "Food"},
    "frozen vegetables": {"expires_in_days": 365,  "storage": "freezer",  "category": "Food"},
    "ice cream":         {"expires_in_days": 180,  "storage": "freezer",  "category": "Food"},
    "frozen chicken":    {"expires_in_days": 270,  "storage": "freezer",  "category": "Food"},
    "canned beans":      {"expires_in_days": 730,  "storage": "pantry",   "category": "Food"},
    "canned soup":       {"expires_in_days": 730,  "storage": "pantry",   "category": "Food"},
    "canned tuna":       {"expires_in_days": 1095, "storage": "pantry",   "category": "Food"},
    "pasta":             {"expires_in_days": 730,  "storage": "pantry",   "category": "Food"},
    "rice":              {"expires_in_days": 730,  "storage": "pantry",   "category": "Food"},
    "cereal":            {"expires_in_days": 180,  "storage": "pantry",   "category": "Food"},
    "crackers":          {"expires_in_days": 90,   "storage": "pantry",   "category": "Food"},
    "chips":             {"expires_in_days": 60,   "storage": "pantry",   "category": "Food"},
    "peanut butter":     {"expires_in_days": 180,  "storage": "pantry",   "category": "Food"},
    "olive oil":         {"expires_in_days": 365,  "storage": "pantry",   "category": "Food"},
    "flour":             {"expires_in_days": 365,  "storage": "pantry",   "category": "Food"},
    "sugar":             {"expires_in_days": 3650, "storage": "pantry",   "category": "Food"},
    "salt":              {"expires_in_days": 3650, "storage": "pantry",   "category": "Food"},
    "coffee":            {"expires_in_days": 180,  "storage": "pantry",   "category": "Food"},
    "juice":             {"expires_in_days": 7,    "storage": "fridge",   "category": "Food"},
    "paper towels":      {"expires_in_days": 3650, "storage": "pantry",   "category": "Household"},
    "toilet paper":      {"expires_in_days": 3650, "storage": "pantry",   "category": "Household"},
    "dish soap":         {"expires_in_days": 730,  "storage": "pantry",   "category": "Household"},
    "laundry detergent": {"expires_in_days": 730,  "storage": "pantry",   "category": "Household"},
    "trash bags":        {"expires_in_days": 3650, "storage": "pantry",   "category": "Household"},
    "shampoo":           {"expires_in_days": 730,  "storage": "pantry",   "category": "Household"},
    "toothpaste":        {"expires_in_days": 730,  "storage": "pantry",   "category": "Household"},
    "tide":              {"expires_in_days": 730,  "storage": "pantry",   "category": "Household"},
    "fresh step":        {"expires_in_days": 730,  "storage": "pantry",   "category": "Household"},
    "cat litter":        {"expires_in_days": 730,  "storage": "pantry",   "category": "Household"},
    "popcorn":           {"expires_in_days": 90,   "storage": "pantry",   "category": "Food"},
    "kettle corn":       {"expires_in_days": 90,   "storage": "pantry",   "category": "Food"},
    "cheddar":           {"expires_in_days": 30,   "storage": "fridge",   "category": "Food"},
    "sharp cheddar":     {"expires_in_days": 30,   "storage": "fridge",   "category": "Food"},
    "marinade":          {"expires_in_days": 365,  "storage": "pantry",   "category": "Food"},
    "dinner rolls":      {"expires_in_days": 5,    "storage": "pantry",   "category": "Food"},
    "hamburger buns":    {"expires_in_days": 5,    "storage": "pantry",   "category": "Food"},
    "ranch":             {"expires_in_days": 60,   "storage": "fridge",   "category": "Food"},
    "vodka":             {"expires_in_days": 3650, "storage": "pantry",   "category": "Food"},
    "butter":            {"expires_in_days": 30,   "storage": "fridge",   "category": "Food"},
}

_ai_enrichment_cache: Dict[str, Dict] = {}

def _fallback_for_item(name: str) -> Dict:
    name_lower = name.lower().strip()
    if name_lower in FOOD_KNOWLEDGE_FALLBACK:
        return FOOD_KNOWLEDGE_FALLBACK[name_lower].copy()
    for key in FOOD_KNOWLEDGE_FALLBACK:
        if key in name_lower:
            return FOOD_KNOWLEDGE_FALLBACK[key].copy()
    return {"expires_in_days": 14, "storage": "fridge", "category": "Food"}

async def enrich_items_with_ai(items: List[Dict]) -> List[Dict]:
    if not items:
        return []

    results: List[Optional[Dict]] = [None] * len(items)
    uncached_indices: List[int] = []

    for i, item in enumerate(items):
        key = item["name"].lower().strip()
        if key in _ai_enrichment_cache:
            results[i] = _ai_enrichment_cache[key].copy()
        else:
            uncached_indices.append(i)

    if not uncached_indices:
        return results  # type: ignore

    uncached_items = [items[i] for i in uncached_indices]
    gemini_succeeded = False

    if GEMINI_API_KEY:
        items_json = json.dumps(
            [{"name": it["name"], "category": it["category"]} for it in uncached_items],
            indent=2
        )
        prompt = f"""You are an expert grocery assistant. For each item below, return a JSON array with expires_in_days (int), storage ("fridge"/"freezer"/"pantry"), and category ("Food"/"Household"). Same order as input. No markdown, just JSON.

Reference: chicken=2d fridge, ground beef=2d fridge, fish=2d fridge, fresh pasta/ravioli=3d fridge, milk=7d fridge, bread=5d pantry, eggs=21d fridge, butter=30d fridge, garlic bread=90d freezer, frozen pizza=180d freezer, frozen meat=270d freezer, chips=60d pantry, canned goods=730d pantry, dry pasta/rice=730d pantry, sugar/salt=3650d pantry.

Items:
{items_json}

Respond with ONLY a valid JSON array."""

        for model_name in ("gemini-2.0-flash-lite", "gemini-1.5-flash"):
            try:
                model = genai.GenerativeModel(model_name)
                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: model.generate_content(prompt)
                )
                raw_text = response.text.strip()
                raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                raw_text = re.sub(r"\s*```$", "", raw_text)
                gemini_results = json.loads(raw_text)

                if not isinstance(gemini_results, list) or len(gemini_results) != len(uncached_items):
                    raise ValueError("Bad response from Gemini")

                valid_storages = {"fridge", "freezer", "pantry"}
                valid_categories = {"Food", "Household"}

                for j, (item, result) in enumerate(zip(uncached_items, gemini_results)):
                    key = item["name"].lower().strip()
                    enrichment = {
                        "expires_in_days": int(result.get("expires_in_days", 14)),
                        "storage": result.get("storage", "fridge") if result.get("storage") in valid_storages else "fridge",
                        "category": result.get("category", "Food") if result.get("category") in valid_categories else "Food",
                    }
                    _ai_enrichment_cache[key] = enrichment
                    results[uncached_indices[j]] = enrichment.copy()

                gemini_succeeded = True
                break
            except Exception as exc:
                print(f"[ai_enrichment] {model_name} failed: {exc}")
                continue

    if not gemini_succeeded:
        for j, i in enumerate(uncached_indices):
            key = items[i]["name"].lower().strip()
            enrichment = _fallback_for_item(items[i]["name"])
            _ai_enrichment_cache[key] = enrichment
            results[i] = enrichment.copy()

    for i in range(len(results)):
        if results[i] is None:
            results[i] = _fallback_for_item(items[i]["name"])

    return results  # type: ignore

PACKSHOT_SERVICE_URL = (os.getenv("PACKSHOT_SERVICE_URL") or "").strip().rstrip("/")
PACKSHOT_SERVICE_KEY = (os.getenv("PACKSHOT_SERVICE_KEY") or "").strip()

INSTACART_PRODUCTS_LINK_URL = (
    os.getenv("INSTACART_PRODUCTS_LINK_URL") or "https://connect.instacart.com/idp/v1/products/products_link"
).strip()

MAX_MULTI_SPLIT_PARTS = int(os.getenv("MAX_MULTI_SPLIT_PARTS", "4"))
MERGE_HEAD_MAX_TOKENS = int(os.getenv("MERGE_HEAD_MAX_TOKENS", "2"))
MERGE_HEAD_MAX_LEN = int(os.getenv("MERGE_HEAD_MAX_LEN", "12"))

_ENRICH_CACHE: Dict[str, Tuple[float, str, str, float]] = {}
_ENRICH_CACHE_TTL = int(os.getenv("ENRICH_CACHE_TTL_SECONDS", "86400"))
_LEARNED_MAP: Dict[str, str] = {}
_PENDING: Dict[str, Dict[str, Any]] = {}

_IMAGE_CACHE: Dict[str, bytes] = {}
_IMAGE_CONTENT_TYPE_CACHE: Dict[str, str] = {}
_MAX_CACHE_ITEMS = 2000

# =========================
# OCR / scanner constants
# =========================

WM_CODE_TOKEN_RE = re.compile(
    r"\b(?:0{4,}\d{3,}[A-Za-z]{0,4}|[A-Za-z]{2,20}\d{6,14}[A-Za-z]{0,4}|\d{7,14}[A-Za-z]{1,4})\b"
)
WM_DEPT_CODE_LINE_RE = re.compile(r"^\s*[A-Za-z]{2,24}\s*\d{6,14}\s*$")
WM_ID_LINE_RE = re.compile(r"^\s*id\s+[a-z0-9]{6,}\s*$", re.IGNORECASE)
WM_AT_FOR_LINE_RE = re.compile(r"\bat\s+\d+\s+for\b", re.IGNORECASE)

ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?\b", re.IGNORECASE)
DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
PHONEISH_RE = re.compile(r"\b\d{3}[-\s.]?\d{3}[-\s.]?\d{4}\b")
ADDR_SUFFIX_RE = re.compile(
    r"\b(st|street|ave|avenue|rd|road|blvd|boulevard|dr|drive|ln|lane|ct|court|"
    r"pkwy|parkway|hwy|highway|trl|trail|pl|place|cir|circle|way)\b",
    re.IGNORECASE,
)
DIRECTION_RE = re.compile(r"\b(north|south|east|west|ne|nw|se|sw|n|s|e|w)\b", re.IGNORECASE)
CITY_STATE_LINE_RE = re.compile(
    r"^\s*[A-Za-z][A-Za-z\s\.'-]{2,}\s*,?\s*"
    r"(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)"
    r"(?:\s+\d{5}(?:-\d{4})?)?\s*$",
    re.IGNORECASE,
)
ADDR_META_RE = re.compile(r"\b(suite|ste|unit|apt|po\s*box|p\.?\s*o\.?\s*box)\b", re.IGNORECASE)

PRICE_RE = re.compile(r"\$?\s*\d+(?:[.,]\s*\d{2})")
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
UNIT_PRICE_RE = re.compile(r"\b(\d+)\s*@\s*\$?\s*\d+(?:[.,]\s*\d{1,2})?\b", re.IGNORECASE)
WEIGHT_X_UNIT_PRICE_RE = re.compile(
    r"\b\d+(?:[.,]\s*\d+)?\s*(?:lb|lbs|oz|g|kg)\s*[xX]\s*\$?\s*\d+(?:[.,]\s*\d+)?\s*/\s*(?:lb|lbs|oz|g|kg)\b",
    re.IGNORECASE,
)
PER_UNIT_PRICE_RE = re.compile(r"\b\d+(?:[.,]\s*\d+)?\s*/\s*(lb|lbs|oz|g|kg|ea|each|ct)\b", re.IGNORECASE)
PRICE_SPAN_RE = re.compile(r"(?:\$?\s*)\d+(?:[.,]\s*\d{2})|(?:\$?\s*)\d+\s*[oO]\s*\d{1,2}|\b\d+\s+\d{2}\b")

QTY_X_SUFFIX_RE = re.compile(r"\b[xX]\s*(\d{1,3})\b")
QTY_X_PREFIX_RE = re.compile(r"\b(\d{1,3})\s*[xX]\b")
QTY_AT_RE = re.compile(r"\b(\d{1,3})\s*@\s*\$?\s*\d", re.IGNORECASE)
QTY_FOR_RE = re.compile(r"\b(\d{1,3})\s+for\s+\$?\s*\d", re.IGNORECASE)
LEADING_QTY_RE = re.compile(r"^\s*(\d{1,3})\s+(.*)$")
PACK_QTY_RE = re.compile(r"\b(\d{1,3})\s*(?:ct|count|pk|pack)\b", re.IGNORECASE)

LONG_NUM_TOKEN_RE = re.compile(r"\b\d{6,14}\b")
LEADING_ITEM_CODE_RE = re.compile(r"^\s*\d{6,14}\s+(?=\S)")
LEADING_PRICE_RE = re.compile(r"^\s*(?:\$?\s*)?(?:\d+[.,]\s*\d{2}|\d+\s*[oO]\s*\d{1,2}|\d+\s+\d{2})\s+", re.IGNORECASE)
LEADING_FLAGS_RE = re.compile(r"^\s*(?:(?:[tTfF]\b)\s*){1,4}", re.IGNORECASE)

STORE_WORDS_RE = re.compile(
    r"\b(publix|wal[-\s]*mart|walmart|target|costco|kroger|aldi|whole\s+foods|trader\s+joe'?s|wm\s*supercenter)\b",
    re.IGNORECASE,
)
GENERIC_HEADER_WORDS_RE = re.compile(r"\b(super\s*markets?|supercenter|market|stores?|wholesale|pharmacy)\b", re.IGNORECASE)

TOTAL_MARKER_RE = re.compile(
    r"\b(sub\s*total|subtotal|tax|balance|amount\s+due|total\s+due|grand\s+total|order\s*total|total)\b",
    re.IGNORECASE,
)

_STOPWORDS_SCORE = {
    "the", "and", "or", "of", "a", "an", "with", "for",
    "pack", "ct", "count", "oz", "lb", "lbs", "g", "kg", "ml", "l",
    "publix", "walmart", "wal", "mart", "target", "costco", "kroger", "aldi", "grocery", "groceries",
}

STORE_HINTS: List[Tuple[str, re.Pattern]] = [
    ("publix", re.compile(r"\bpublix\b", re.IGNORECASE)),
    ("walmart", re.compile(r"\bwalmart\b|\bwal[-\s]*mart\b|\bwm\s*supercenter\b", re.IGNORECASE)),
    ("target", re.compile(r"\btarget\b", re.IGNORECASE)),
    ("costco", re.compile(r"\bcostco\b", re.IGNORECASE)),
    ("kroger", re.compile(r"\bkroger\b", re.IGNORECASE)),
    ("aldi", re.compile(r"\baldi\b", re.IGNORECASE)),
    ("whole foods", re.compile(r"\bwhole\s+foods\b", re.IGNORECASE)),
    ("trader joe's", re.compile(r"\btrader\s+joe'?s\b", re.IGNORECASE)),
]

STORE_HEADER_PATTERNS: Dict[str, re.Pattern] = {
    "publix": re.compile(r"^\s*publix(?:\s+super\s*markets?)?\s*$", re.IGNORECASE),
    "walmart": re.compile(r"^\s*(?:wm\s*supercenter|wal[-\s]*mart(?:\s+stores?)?|walmart(?:\s+supercenter)?)\s*$", re.IGNORECASE),
    "target": re.compile(r"^\s*o?target\s*$", re.IGNORECASE),
    "costco": re.compile(r"^\s*costco(?:\s+wholesale)?\s*$", re.IGNORECASE),
    "kroger": re.compile(r"^\s*kroger\s*$", re.IGNORECASE),
    "aldi": re.compile(r"^\s*aldi\s*$", re.IGNORECASE),
    "whole foods": re.compile(r"^\s*whole\s+foods(?:\s+market)?\s*$", re.IGNORECASE),
    "trader joe's": re.compile(r"^\s*trader\s+joe'?s\s*$", re.IGNORECASE),
}

_STORE_TOKENS = {"publix", "walmart", "wal", "mart", "target", "costco", "kroger", "aldi", "whole", "foods", "trader", "joe", "joes", "wm"}
_GENERIC_HEADER_TOKENS = {"super", "markets", "market", "stores", "store", "wholesale", "pharmacy", "supercenter"}
_JUNK_EXACT_LINES = {"t", "f", "tf", "t f", "ft", "tt", "ff", "grocery", "groceries", "visa", "debit", "credit", "for"}

STOP_ITEM_WORDS = {"lb", "lbs", "oz", "g", "kg", "ct", "ea", "each", "w", "wt", "weight", "at", "x", "vov"}

HOUSEHOLD_WORDS = {
    "paper", "towel", "towels", "toilet", "tissue", "napkin", "napkins",
    "detergent", "bleach", "cleaner", "wipes", "wipe", "soap", "dish", "dawn",
    "shampoo", "conditioner", "deodorant", "toothpaste", "floss", "razor",
    "trash", "garbage", "bag", "bags", "foil", "wrap", "parchment",
    "rubbing", "alcohol", "isopropyl", "cotton", "swab", "swabs",
    "battery", "batteries", "lightbulb", "lighter", "matches", "pet", "litter",
    "toothbrush", "mouthwash", "cleaning", "laundry", "sponge", "sponges",
    "tide", "febreze", "lysol", "cascade", "downy", "bounce", "gain",
    "cremo", "barber", "bergamot", "extreme", "odor", "renewal", "freshstep",
}

NOISE_PATTERNS = [
    r"\btarget\s*circle\b",
    r"\bcircle\b\s*\d+",
    r"\bregular\s+price\b",
    r"\bexpect\s+more\b",
    r"\bpay\s+less\b",
    r"\bexpect\s+more\s+pay\s+less\b",
    r"\bbottle\s+deposit\b",
    r"\bdeposit\s+fee\b",
    r"\bbottle\s+deposit\s+fee\b",
    r"^\s*o?target\s*$",
    r"^\s*otarget\s*$",
    r"^\s*target\.com\s*$",
    r"\bcircle\b\s*[iIlL]\s*[oO0]\b",
    r"\btarget\s*circle\b\s*[iIlL]\s*[oO0]\b",
    r"\btarget\s*circle\b\s*\d+\s*%?\b",
    r"\bwm\s*supercenter\b",
    r"\bwal\s*mart\s*supercenter\b",
    r"\bwalmart\s*supercenter\b",
    r"\bwalmart\s*pay\b",
    r"\bscan\s*&\s*go\b",
    r"\bspark\b\s*(?:driver|shopper)?\b",
    r"\breturns?\b\s*(?:policy|center)?\b",
    r"\bbarcode\b",
    r"\bmerchant\s+copy\b",
    r"\bcustomer\s+copy\b",
    r"\bitem\s*#\b",
    r"\bdept\b",
    r"\btc\s*#\b",
    r"\bst\s*#\b",
    r"\btr\s*#\b",
    r"\bcard\s*#\b",
    r"\bref\s*#\b",
    r"\baid\b",
    r"^\s*id\s+[a-z0-9]{6,}\s*$",
    r"\bat\s+\d+\s+for\b",
    r"^\s*[A-Za-z]{2,24}\s*\d{6,14}\s*$",
    r"\b0{4,}\d{3,}[A-Za-z]{0,4}\b",
    r"^\s*grocery\s*$",
    r"^\s*groceries\s*$",
    r"\bsub\s*total\b",
    r"\bsubtotal\b",
    r"\bamount\s+due\b",
    r"\bbalance\s+due\b",
    r"\bchange\s+due\b",
    r"\btax\b",
    r"\bsales\s+tax\b",
    r"\btender\b",
    r"\bcash\b",
    r"\bcash\s*back\b",
    r"\bcashback\b",
    r"\bpayment\b",
    r"\bdebit\b",
    r"\bcredit\b",
    r"\bvisa\b",
    r"\bmastercard\b",
    r"\bamex\b",
    r"\bdiscover\b",
    r"\bauth\b",
    r"\bapproval\b",
    r"\bapproved\b",
    r"\border\s*total\b",
    r"\btotal\s+due\b",
    r"\bgrand\s+total\b",
    r"^\s*total\s*$",
    r"\bregister\b",
    r"\breg\b",
    r"\blane\b",
    r"\bterminal\b",
    r"\bterm\b",
    r"\bpos\b",
    r"\bcashier\b",
    r"\bmanager\b",
    r"\boperator\b",
    r"\bop\s*#\b",
    r"\bstore\s*#\b",
    r"\btrx\b",
    r"\btransaction\b",
    r"\btrace\b",
    r"\btrace\s*#\b",
    r"\bacct\b",
    r"\bacct\s*#\b",
    r"\binvoice\b",
    r"\binv\b",
    r"\border\s*#\b",
    r"\breceipt\b",
    r"\bserved\b",
    r"\bguest\b",
    r"\bvisit\b",
    r"\bthank you\b",
    r"\bthanks\b",
    r"\bcome again\b",
    r"\breturn\b",
    r"\brefund\b",
    r"\bpolicy\b",
    r"\bcoupon\b",
    r"\bdiscount\b",
    r"\bpromo\b",
    r"\bpromotion\b",
    r"\byou saved\b",
    r"\bsavings\b",
    r"\btotal\s+savings\b",
    r"\byour\s+savings\b",
    r"\bpoints\b",
    r"\bmember\b",
    r"\bloyalty\b",
    r"\bclub\b",
    r"\bphone\b",
    r"\btel\b",
    r"\bwww\.",
    r"\.com\b",
    r"\bitems?\b\s*\d+\b",
    r"^\s*#\s*\d+\s*$",
    r"\bshopping\s+center\b",
    r"\bshopping\s+ctr\b",
    r"\b(?:target\s*)?circle\s*\d+\s*%\b",
    r"\b\d+\s*%\s*(?:off|discount)\b",
    r"\b(?:percent|pct)\s*off\b",
]
NOISE_RE = re.compile("|".join(f"(?:{p})" for p in NOISE_PATTERNS), re.IGNORECASE)

ABBREV_TOKEN_MAP: Dict[str, str] = {
    "pblx": "publix",
    "publx": "publix",
    "pub": "publix",
    "pbl": "publix",
    "pbx": "publix",

    "gg": "good and gather",
    "ggd": "good and gather",

    "chees": "cheese",
    "ches": "cheese",
    "chs": "cheese",
    "chz": "cheese",

    "rte": "ready to eat",
    "sar": "sargento",
    "sarg": "sargento",
    "arts": "artisan",
    "artis": "artisan",
    "blnd": "blends",
    "parm": "parmesan",
    "nesp": "nespresso",
    "pke": "pike",
    "sprng": "spring",
    "wht": "white",
    "whl": "whole",
    "sdl": "seedless",
    "sdls": "seedless",

    "bg": "bagels",
    "ba": "bagels",
    "bgl": "bagel",
    "bgls": "bagels",

    "swr": "swirl",
    "swrl": "swirl",

    "crm": "cream",
    "cr": "cream",
    "chsd": "cheesed",

    "whp": "whipping",
    "hvy": "heavy",
    "bttr": "butter",
    "btr": "butter",
    "marg": "margarine",

    "yog": "yogurt",
    "yg": "yogurt",
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
    "powd": "powdered",
    "sug": "sugar",
    "brn": "brown",

    "van": "vanilla",
    "ext": "extract",
    "vngr": "vinegar",
    "alc": "alcohol",
    "iso": "isopropyl",
    "isoprop": "isopropyl",
    "isopropyl": "isopropyl",

    "prm": "premium",
    "grl": "grilled",
    "pta": "potato",
    "pot": "potato",
    "rst": "roast",
    "rstd": "roasted",
    "ov": "oven",

    "jd": "jack daniel's",
    "tenn": "tennessee",
    "hny": "honey",
    "sh": "sharp",
    "chd": "cheddar",
    "cut": "cut",

    "swt": "sweet",
    "pln": "plain",
    "frsh": "fresh",
    "orig": "original",
    "blk": "black",
    "grn": "green",
    "pk": "pack",
    "pck": "pack",
    "pks": "packs",

    "moz": "mozzarella",
    "mzzrl": "mozzarella",
    "mnst": "muenster",
    "mnstr": "muenster",

    "hf": "half",
    "hnh": "half and half",
    "hh": "half and half",
    "avdo": "avocado",
    "avo": "avocado",

    "ppr": "paper",
    "twls": "towels",
    "twl": "towel",
    "tp": "toilet paper",
    "pt": "paper towels",

    "rub": "rubbing",
    "rubb": "rubbing",

    "veg": "vegetable",
    "oo": "olive oil",
    "apflr": "all purpose flour",

    # Wishbone Ranch
    "wb": "wishbone",

    # AMC Theatres popcorn brand — keep as-is, do NOT expand
    "amc": "amc",

    # Sara Lee Buffalo Wing Marinade
    "sbr": "sara lee",
    "buff": "buffalo",
    "wg": "wing",
    "marind": "marinade",

    # King's Hawaiian
    "kh": "king's hawaiian",
    "sav": "savory",
    "din": "dinner",
    "rol": "rolls",

    # Fresh Step Extreme
    "xtrm": "extreme",
    "stp": "step",

    # Tide renewal / HE
    "ren": "renewal",

    # Butter sticks
    "stk": "sticks",
    "stks": "sticks",
}

PHRASE_MAP: Dict[str, str] = {
    "half and half": "half-and-half",
    "h and h": "half-and-half",
    "half n half": "half-and-half",
    "halfnhalf": "half-and-half",
    "good and gather": "good & gather",

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

    "paper towels": "paper towels",
    "toilet paper": "toilet paper",
    "dish soap": "dish soap",
    "rubbing alcohol": "rubbing alcohol",
    "cream of chicken": "cream of chicken",
    "mac and cheese": "mac and cheese",
}

STORE_TOKEN_MAPS: Dict[str, Dict[str, str]] = {
    "publix": {
        "wdg": "wedge",
        "wdge": "wedge",
        "dcd": "diced",
        "slt": "salted",
        "unslt": "unsalted",
        "it": "italian",
        "itl": "italian",
        "frsh": "fresh",
        "pk": "pack",
        "pck": "pack",
        "pks": "packs",
        "swt": "sweet",
        "pln": "plain",
        "sprd": "spread",
        "mnst": "muenster",
        "mnstr": "muenster",
        "moz": "mozzarella",
        "mzzrl": "mozzarella",
        "parm": "parmesan",
        "rg": "regular",
        "lg": "large",
        "sm": "small",
    },
    "target": {
        "gg": "good and gather",
        "gg": "good and gather",
    },
    "walmart": {},
    "aldi": {},
    "costco": {},
    "kroger": {},
    "whole foods": {},
    "trader joe's": {},
}

GLOBAL_TOKEN_MAP: Dict[str, str] = {
    "hvy": "heavy",
    "whp": "whipping",
    "crm": "cream",
    "chs": "cheese",
    "chees": "cheese",
    "bttr": "butter",
    "marg": "margarine",
    "yog": "yogurt",
    "org": "organic",
    "veg": "vegetable",
    "grd": "ground",
    "bnls": "boneless",
    "sknls": "skinless",
    "brst": "breast",
    "chk": "chicken",
    "ckn": "chicken",
    "chkn": "chicken",
    "bf": "beef",
    "blk": "black",
    "pepr": "pepper",
    "grn": "green",
    "chp": "chip",
    "mffn": "muffin",
    "sdls": "seedless",
    "orig": "original",
    "prog": "progresso",
    "tus": "tuscan",
    "bea": "beans",
    "ches": "cheese",
    "ex": "extra",
    "srp": "sharp",
    "ba": "bagels",
    "belg": "belgioioso",
    "alc": "alcohol",
    "iso": "isopropyl",
    "isoprop": "isopropyl",
    "jd": "jack daniel's",
    "tenn": "tennessee",
    "rte": "ready to eat",
}

FORCED_FALLBACK_MAP: Dict[str, str] = {
    "snk": "snack",
    "sn": "snack",
    "chs": "cheese",
    "chz": "cheese",
    "wt": "weight",
    "ct": "count",
    "pk": "pack",
    "pck": "pack",
    "reg": "regular",
    "lrg": "large",
    "lg": "large",
    "sm": "small",
    "frz": "freezer",
    "frsh": "fresh",
    "dcd": "diced",
    "swt": "sweet",
    "pln": "plain",
    "brd": "bread",
    "twls": "towels",
    "twl": "towel",
    "avo": "avocado",
    "avdo": "avocado",
}

ALLOWED_SHORT_TOKENS = {"oz", "lb", "lbs", "g", "kg", "ml", "l", "xl", "xxl", "bf", "ea", "ct"}
_ABBR_TOKEN_RE = re.compile(r"^[A-Za-z]{2,4}$")
_ALLCAPS_RE = re.compile(r"^[A-Z]{2,6}$")

# Brand names that the catalog enrichment should NEVER overwrite.
# If the expanded name starts with one of these, skip catalog lookup.
_PROTECTED_BRAND_PREFIXES = {
    "amc", "jack daniel's", "rao", "panera", "publix", "king's hawaiian",
    "nutrl", "buitoni", "thomas", "snyder's", "ben's", "great value",
    "sara lee", "wishbone", "cremo", "mojo",
}

PRODUCT_IMAGE_MAP: Dict[str, str] = {
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


# =========================
# Lifecycle
# =========================

@app.on_event("startup")
async def _startup() -> None:
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
async def _shutdown() -> None:
    global OFF_CLIENT, IMG_CLIENT
    if OFF_CLIENT:
        await OFF_CLIENT.aclose()
    if IMG_CLIENT:
        await IMG_CLIENT.aclose()
    OFF_CLIENT = None
    IMG_CLIENT = None


# =========================
# Models
# =========================

@dataclass
class ReqBudget:
    started: float
    deadline: float
    off_used: int = 0

    def remaining(self) -> float:
        return self.deadline - time.monotonic()

    def expired(self) -> bool:
        return self.remaining() <= 0


class Candidate(BaseModel):
    raw_line: str
    cleaned_line: str
    qty_hint: int = 1
    source: str = "direct"


class ParsedItem(BaseModel):
    name: str
    quantity: int
    category: str
    image_url: str
    expires_in_days: Optional[int] = None
    storage: Optional[str] = None


class InstacartLineItem(BaseModel):
    name: str
    quantity: float = 1.0
    unit: str = "each"


class InstacartCreateListRequest(BaseModel):
    title: str = "ShelfLife Shopping List"
    items: Optional[List[InstacartLineItem]] = None
    line_items: Optional[List[InstacartLineItem]] = None
    link_type: Optional[str] = None


class FeedbackBody(BaseModel):
    store_hint: str = ""
    expanded: str
    full_name: str
    raw_example: str = ""


# =========================
# Helpers
# =========================

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
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")

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


def dedupe_key(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def title_case(s: str) -> str:
    small = {"and", "or", "of", "the", "a", "an", "to", "in", "on", "for", "with"}

    def _cap_token(raw: str, is_first: bool) -> str:
        if not raw:
            return raw

        if raw.lower() == "i":
            return "I"

        if raw == "&":
            return "&"

        if raw.isalpha() and raw.upper() == raw and 2 <= len(raw) <= 5:
            return raw

        lw = raw.lower()

        if not is_first and lw in small:
            return lw

        if "-" in raw:
            parts = [p for p in raw.split("-") if p]
            out = []
            for idx, p in enumerate(parts):
                pl = p.lower()
                if idx != 0 and pl in small:
                    out.append(pl)
                else:
                    out.append(pl[:1].upper() + pl[1:])
            return "-".join(out)

        if "'" in raw:
            bits = raw.split("'")
            first = bits[0].lower()
            first = first[:1].upper() + first[1:] if first else first
            rest = "'".join(bits[1:]).lower()
            return first + ("'" + rest if rest else "")

        return lw[:1].upper() + lw[1:]

    words = [w for w in re.split(r"\s+", (s or "").strip()) if w]
    return " ".join(_cap_token(w, i == 0) for i, w in enumerate(words)).strip()


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


def _public_base_url(request: Request) -> str:
    xf_proto = request.headers.get("x-forwarded-proto")
    xf_host = request.headers.get("x-forwarded-host")
    if xf_host:
        scheme = xf_proto or request.url.scheme or "https"
        return f"{scheme}://{xf_host}".rstrip("/")
    return str(request.base_url).rstrip("/")


def _image_url_for_item(base_url: str, name: str) -> str:
    return f"{base_url}/image?name={urllib.parse.quote((name or '').strip())}"


def detect_store_hint(raw_lines: List[str]) -> str:
    blob = " \n ".join(raw_lines[:180]).lower()
    for name, pat in STORE_HINTS:
        if pat.search(blob):
            return name
    return ""


def _normalized_instacart_items(req: InstacartCreateListRequest) -> List[InstacartLineItem]:
    return req.items or req.line_items or []


def _extract_instacart_link(data: Dict[str, Any]) -> str:
    if not isinstance(data, dict):
        return ""
    for k in ("products_link_url", "url", "link", "share_url", "productsLinkUrl"):
        v = data.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    for k in ("products_link", "productsLink", "result", "data"):
        v = data.get(k)
        if isinstance(v, dict):
            lk = _extract_instacart_link(v)
            if lk:
                return lk
    return ""


# =========================
# Scanner filters
# =========================

def _is_walmart_code_or_meta_line(s: str) -> bool:
    if not s:
        return False
    ss = (s or "").strip()
    if not ss:
        return False
    if WM_ID_LINE_RE.match(ss):
        return True
    if WM_DEPT_CODE_LINE_RE.match(ss):
        return True
    if WM_AT_FOR_LINE_RE.search(ss):
        return True
    key = re.sub(r"[^A-Za-z0-9\s]+", " ", ss)
    key = re.sub(r"\s+", " ", key).strip()
    toks = key.split()
    if toks and all(WM_CODE_TOKEN_RE.fullmatch(t) for t in toks):
        return True
    return False


def _noise_normalize(s: str) -> str:
    ss = (s or "").strip()
    if not ss:
        return ""
    ss = ss.lower()
    ss = re.sub(r"(?<=\d)\s*[oO]\s*(?=\d)", "0", ss)
    ss = re.sub(r"(?<=\d)\s*[iIlL]\s*(?=\d)", "1", ss)
    ss = re.sub(r"\b([iIlL])\s*([oO0])\b", "10", ss)
    ss = re.sub(r"[^a-z0-9\s]+", " ", ss)
    ss = re.sub(r"\s+", " ", ss).strip()
    return ss


def _is_noise_line(s: str) -> bool:
    if not s:
        return False

    if _is_walmart_code_or_meta_line(s):
        return True

    if NOISE_RE.search(s):
        return True

    norm = _noise_normalize(s)
    if norm and NOISE_RE.search(norm):
        return True

    toks = norm.split()
    if not toks:
        return False

    if "circle" in toks and ("target" in toks or len(toks) <= 3):
        extras = []
        for t in toks:
            if t in {"target", "circle", "offer", "deal", "rewards", "discount", "off", "percent", "pct", "save", "savings"}:
                continue
            if re.fullmatch(r"\d+", t):
                continue
            extras.append(t)
        if not extras:
            return True

    return False


def _is_header_or_address(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return True

    if PHONEISH_RE.search(s) or ZIP_RE.search(s) or DATE_RE.search(s) or TIME_RE.search(s):
        return True

    if CITY_STATE_LINE_RE.match(s) or ADDR_META_RE.search(s):
        return True

    if re.search(r"^\s*\d{2,6}\b", s) and (ADDR_SUFFIX_RE.search(s) or DIRECTION_RE.search(s)):
        return True

    if re.search(r"^\s*\d{2,6}\b", s) and re.search(r"\b(camino|avenida|boulevard|blvd)\b", s, flags=re.IGNORECASE):
        return True

    return False


def _is_junk_line(s: str) -> bool:
    if _is_walmart_code_or_meta_line(s):
        return True
    key = dedupe_key(s)
    if not key:
        return True
    if key in _JUNK_EXACT_LINES:
        return True
    toks = key.split()
    if len(toks) == 1 and toks[0].isalpha() and len(toks[0]) <= 2:
        return True
    return False


def _raw_line_has_price_or_qty_hint(s: str) -> bool:
    if not s:
        return False
    if MONEY_TOKEN_RE.search(s) or UNIT_PRICE_RE.search(s):
        return True
    if QTY_X_SUFFIX_RE.search(s) or QTY_X_PREFIX_RE.search(s) or QTY_AT_RE.search(s) or QTY_FOR_RE.search(s):
        return True
    return False


def _is_price_like_line(s: str) -> bool:
    if not s:
        return False
    ss = (s or "").strip()
    if ONLY_MONEYISH_RE.match(ss) or PRICE_FLAG_RE.match(ss):
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
    if WEIGHT_X_UNIT_PRICE_RE.search(ss) or UNIT_PRICE_RE.search(ss) or PER_UNIT_PRICE_RE.search(ss):
        return True
    if re.search(r"\blb\b", ss, flags=re.IGNORECASE) and ("@" in ss or "/" in ss):
        return True
    return False


def _is_store_or_header_line_anywhere(s: str) -> bool:
    if not s:
        return False

    ss = (s or "").strip()
    if not ss:
        return True

    if _is_walmart_code_or_meta_line(ss):
        return True

    # If the line has a price or qty hint, it's an item line — never a header
    if _raw_line_has_price_or_qty_hint(ss):
        return False

    key = dedupe_key(ss)
    if not key:
        return True

    toks = key.split()
    if not toks:
        return True

    if STORE_WORDS_RE.search(ss):
        remaining = [t for t in toks if t not in _STORE_TOKENS and t not in _GENERIC_HEADER_TOKENS]
        # If there are non-store words remaining, it's a product (e.g. "Publix Spring Water")
        # Only kill it if it's ONLY store words
        if remaining:
            return False
        if len(toks) <= 8:
            return True

    if GENERIC_HEADER_WORDS_RE.search(ss) and len(toks) <= 8:
        return True

    return False


def _is_probable_store_header_line(s: str, store_hint: str, position_idx: int) -> bool:
    if not s or position_idx > 18:
        return False

    ss = (s or "").strip()
    key = dedupe_key(ss)

    if not key or _is_walmart_code_or_meta_line(ss):
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


def _strip_long_numeric_tokens(s: str) -> str:
    if not s:
        return ""
    s2 = LONG_NUM_TOKEN_RE.sub("", s)
    s2 = WM_CODE_TOKEN_RE.sub("", s2)
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2


def _clean_line(line: str) -> str:
    s = (line or "").strip()
    if not s:
        return ""

    if _is_junk_line(s) or _is_walmart_code_or_meta_line(s):
        return ""

    s = s.replace("—", "-")
    s = s.replace("|", " ")
    s = re.sub(r"\s+", " ", s).strip()

    s = LEADING_ITEM_CODE_RE.sub("", s).strip()
    s = LEADING_PRICE_RE.sub("", s).strip()
    s = LEADING_FLAGS_RE.sub("", s).strip()

    s = re.sub(r"(?:\s+\$?\s*\d{1,6}(?:[.,]\s*\d{2})\s*)\s*$", "", s)
    s = re.sub(r"(?:\s+\$?\s*\d{1,6}\s*[oO]\s*\d{1,2}\s*)\s*$", "", s)
    s = re.sub(r"(?:\s+\d{1,6}\s+\d{2}\s*)\s*$", "", s)
    s = re.sub(r"(?:\s+[tTfF]){1,4}\s*$", "", s)
    s = re.sub(r"\s+\d{2}\s*$", "", s)
    s = re.sub(r"\s+\b(?:T|F|TF|TX|TAX)\b\s*$", "", s, flags=re.IGNORECASE)

    s = UNIT_PRICE_RE.sub("", s).strip()
    s = PER_UNIT_PRICE_RE.sub("", s).strip()
    s = re.sub(r"\b\d+(?:[.,]\s*\d+)?\s*/\s*(lb|lbs|oz|g|kg|ea|each|ct)\b", "", s, flags=re.IGNORECASE).strip()

    s = re.sub(r"\b\d+\s+for\s+\$?\s*\d+(?:[.,]\s*\d{2})?\b", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\b\d+\s*@\s*\$?\s*\d+(?:[.,]\s*\d{2})?\b", "", s, flags=re.IGNORECASE).strip()

    s = WM_CODE_TOKEN_RE.sub("", s).strip()
    s = _strip_long_numeric_tokens(s)

    s = re.sub(r"[^\w\s&'/-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    if _is_junk_line(s) or _is_walmart_code_or_meta_line(s):
        return ""

    return s


def _has_valid_item_words(line: str) -> bool:
    if _is_junk_line(line):
        return False

    words = [w.lower() for w in re.findall(r"[A-Za-z]{2,}", line or "")]
    words = [w for w in words if w not in STOP_ITEM_WORDS]

    if not words:
        return False

    bad = {"subtotal", "total", "tax", "change", "payment", "approved", "auth", "receipt"}
    if all(w in bad for w in words):
        return False

    return True


def _next_nonempty(raw_lines: List[str], idx: int, zone_end: int) -> Tuple[str, int]:
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

    if _is_walmart_code_or_meta_line(s) or _is_header_or_address(s) or _is_noise_line(s) or _is_store_or_header_line_anywhere(s) or _is_junk_line(s):
        return False

    if _is_price_like_line(s) or WEIGHT_ONLY_RE.match(s) or _is_weight_or_unit_price_line(s):
        return False

    cl = _clean_line(s)
    return bool(cl) and _has_valid_item_words(cl)


def _has_multiple_price_tokens(s: str) -> bool:
    if not s:
        return False
    return len(list(PRICE_SPAN_RE.finditer(s))) >= 2


def _should_merge_with_next(head: str, tail: str, next_support: str) -> bool:
    if not head or not tail:
        return False

    head_key = dedupe_key(head)
    head_toks = head_key.split()
    if not head_toks:
        return False

    short_lead = len(head_toks) <= MERGE_HEAD_MAX_TOKENS and len(head_key) <= MERGE_HEAD_MAX_LEN
    if not short_lead:
        return False

    if not _is_descriptionish_line(tail):
        return False

    if not next_support:
        return False

    return bool(
        _is_weight_or_unit_price_line(next_support)
        or _is_price_like_line(next_support)
        or WEIGHT_ONLY_RE.match(next_support)
        or _line_has_embedded_price(tail)
    )


def _split_embedded_multi_item_line(s: str) -> List[str]:
    ss = (s or "").strip()
    if not ss or not _has_multiple_price_tokens(ss):
        return [ss] if ss else []

    parts: List[str] = []
    last_end = 0
    matches = list(PRICE_SPAN_RE.finditer(ss))

    for m in matches[:MAX_MULTI_SPLIT_PARTS]:
        segment = ss[last_end:m.end()].strip(" -")
        if segment:
            parts.append(segment)
        last_end = m.end()

    tail = ss[last_end:].strip(" -")
    if tail and parts:
        parts[-1] = f"{parts[-1]} {tail}".strip()
    elif tail:
        parts.append(tail)

    clean_parts = []
    for p in parts:
        pc = p.strip()
        if pc and _line_has_embedded_price(pc):
            clean_parts.append(pc)

    return clean_parts or [ss]


def find_totals_marker_index(raw_lines: List[str]) -> Optional[int]:
    n = len(raw_lines)
    if n == 0:
        return None

    tail_start = max(0, n - max(65, n // 3))
    candidates: List[int] = []

    for idx in range(tail_start, n):
        ln = raw_lines[idx] or ""
        if TOTAL_MARKER_RE.search(ln):
            if MONEY_TOKEN_RE.search(ln) or len(dedupe_key(ln).split()) <= 6:
                candidates.append(idx)

    if not candidates:
        return None

    best = candidates[-1]

    # Walk backwards from the totals line to make sure we don't cut off
    # weight-priced items (e.g. "POTATOES RUSSET" followed by "2.83 lb @ 1.79/lb")
    # that appear just before the total. Include up to 4 extra lines.
    for lookback in range(1, 5):
        check_idx = best - lookback
        if check_idx < 0:
            break
        ln = (raw_lines[check_idx] or "").strip()
        if not ln:
            continue
        if _is_weight_or_unit_price_line(ln) or _is_price_like_line(ln) or WEIGHT_ONLY_RE.match(ln):
            # The line before this might be the item name — push zone end past it
            best = max(best, check_idx + 1)
            break

    return best


def _detect_item_zone_indices(raw_lines: List[str], store_hint: str) -> Tuple[int, int]:
    n = len(raw_lines)
    if n == 0:
        return 0, 0

    totals_idx = find_totals_marker_index(raw_lines)
    end = totals_idx if totals_idx is not None else n
    start = 0

    store_header_pat = STORE_HEADER_PATTERNS.get(store_hint) if store_hint else None
    scan_limit = min(end, 200)

    for i in range(0, scan_limit):
        s = (raw_lines[i] or "").strip()
        if not s:
            continue

        if store_header_pat and store_header_pat.match(s):
            continue
        if _is_probable_store_header_line(s, store_hint=store_hint, position_idx=i):
            continue
        if _is_header_or_address(s) or _is_noise_line(s) or _is_store_or_header_line_anywhere(s) or _is_junk_line(s):
            continue

        cleaned = _clean_line(s)
        if not cleaned:
            continue

        if _has_valid_item_words(cleaned) and (_raw_line_has_price_or_qty_hint(s) or len(dedupe_key(cleaned).split()) >= 1):
            start = max(0, i - 3)
            break

    end = max(start, end)
    return start, end


def _extract_candidates_from_lines(raw_lines: List[str], store_hint: str) -> Tuple[List[Candidate], List[Dict[str, Any]]]:
    dropped_lines: List[Dict[str, Any]] = []
    zone_start, zone_end = _detect_item_zone_indices(raw_lines, store_hint=store_hint)
    store_header_pat = STORE_HEADER_PATTERNS.get(store_hint) if store_hint else None

    candidates: List[Candidate] = []
    pending_raw: Optional[str] = None
    pending_clean: Optional[str] = None
    pending_qty_hint: int = 1

    def finalize_pending(reason: str) -> None:
        nonlocal pending_raw, pending_clean, pending_qty_hint

        if pending_raw and pending_clean and _has_valid_item_words(pending_clean):
            if not _is_header_or_address(pending_clean) and not _is_store_or_header_line_anywhere(pending_clean) and not _is_noise_line(pending_clean) and not _is_junk_line(pending_clean):
                candidates.append(
                    Candidate(
                        raw_line=pending_raw,
                        cleaned_line=pending_clean,
                        qty_hint=max(1, pending_qty_hint),
                        source=f"pending:{reason}",
                    )
                )
            else:
                dropped_lines.append({"line": pending_raw, "stage": f"pending_drop:{reason}:header_guard", "cleaned": pending_clean})
        elif pending_raw:
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

        if _is_walmart_code_or_meta_line(s):
            if pending_raw:
                finalize_pending("hit_walmart_code_meta")
            dropped_lines.append({"line": s, "stage": "walmart_code_meta"})
            i += 1
            continue

        if _is_junk_line(s):
            if pending_raw:
                finalize_pending("hit_junk_line")
            dropped_lines.append({"line": s, "stage": "junk_line"})
            i += 1
            continue

        if store_header_pat and store_header_pat.match(s):
            if pending_raw:
                finalize_pending("hit_store_header_exact")
            dropped_lines.append({"line": s, "stage": "store_header_exact"})
            i += 1
            continue

        if _is_probable_store_header_line(s, store_hint=store_hint, position_idx=i - zone_start):
            if pending_raw:
                finalize_pending("hit_store_header_fuzzy")
            dropped_lines.append({"line": s, "stage": "store_header_fuzzy"})
            i += 1
            continue

        if _is_store_or_header_line_anywhere(s):
            if pending_raw:
                finalize_pending("hit_store_header_anywhere")
            dropped_lines.append({"line": s, "stage": "store_header_anywhere"})
            i += 1
            continue

        if _is_header_or_address(s):
            if pending_raw:
                finalize_pending("hit_header_or_address")
            dropped_lines.append({"line": s, "stage": "header_or_address"})
            i += 1
            continue

        if _is_noise_line(s):
            if pending_raw:
                finalize_pending("hit_noise")
            dropped_lines.append({"line": s, "stage": "noise"})
            i += 1
            continue

        if _is_weight_or_unit_price_line(s):
            qty_hint = _parse_qty_hint_from_attached_line(s)
            if pending_raw:
                pending_qty_hint = max(pending_qty_hint, qty_hint)
            elif candidates:
                last = candidates[-1]
                last.qty_hint = max(int(last.qty_hint), qty_hint)
            else:
                dropped_lines.append({"line": s, "stage": "orphan_unit_price"})
            i += 1
            continue

        if _is_price_like_line(s) or WEIGHT_ONLY_RE.match(s):
            if pending_raw:
                finalize_pending("price_line")
            else:
                dropped_lines.append({"line": s, "stage": "price_only_no_pending"})
            i += 1
            continue

        split_parts = _split_embedded_multi_item_line(s)
        if len(split_parts) > 1:
            for part in split_parts:
                part_clean = _clean_line(part)
                if not part_clean or not _has_valid_item_words(part_clean):
                    dropped_lines.append({"line": part, "stage": "split_part_invalid", "parent": s})
                    continue
                candidates.append(Candidate(raw_line=part, cleaned_line=part_clean, qty_hint=1, source="split_multi"))
            i += 1
            continue

        cleaned = _clean_line(s)
        if not cleaned:
            dropped_lines.append({"line": s, "stage": "clean_empty"})
            i += 1
            continue

        next1, next1_idx = _next_nonempty(raw_lines, i + 1, zone_end)
        next2, _ = _next_nonempty(raw_lines, next1_idx + 1, zone_end) if next1 else ("", zone_end)

        if _should_merge_with_next(s, next1, next2):
            combined_raw = f"{s} {next1}".strip()
            combined_clean = _clean_line(combined_raw)
            if combined_clean and _has_valid_item_words(combined_clean) and not _is_header_or_address(combined_clean) and not _is_store_or_header_line_anywhere(combined_clean) and not _is_noise_line(combined_clean) and not _is_junk_line(combined_clean):
                s = combined_raw
                cleaned = combined_clean
                i = next1_idx + 1
            else:
                i += 1
        else:
            i += 1

        if pending_raw:
            finalize_pending("new_desc")

        if not _has_valid_item_words(cleaned):
            dropped_lines.append({"line": s, "stage": "no_item_words", "cleaned": cleaned})
            continue

        if _is_header_or_address(cleaned) or _is_store_or_header_line_anywhere(cleaned) or _is_noise_line(cleaned) or _is_junk_line(cleaned):
            dropped_lines.append({"line": s, "stage": "desc_rejected_header_guard", "cleaned": cleaned})
            continue

        if _line_has_embedded_price(s):
            candidates.append(Candidate(raw_line=s, cleaned_line=cleaned, qty_hint=1, source="embedded_price"))
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

        candidates.append(Candidate(raw_line=s, cleaned_line=cleaned, qty_hint=1, source="direct"))

    if pending_raw:
        finalize_pending("zone_end")

    return candidates, dropped_lines


# =========================
# Name normalization
# =========================

def _normalize_for_phrase_match(s: str) -> str:
    s = (s or "").lower().strip().replace("&", " and ")
    s = re.sub(r"[^a-z0-9\s'-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _split_tokens_preserve_numbers(s: str) -> List[str]:
    raw = (s or "").strip().replace("&", " and ")
    raw = re.sub(r"[^\w\s'-]+", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return [t for t in raw.split(" ") if t]


def _looks_like_bad_abbrev_token(tok: str) -> bool:
    if not tok or re.search(r"\d", tok):
        return False
    tl = tok.lower()
    if tl in ALLOWED_SHORT_TOKENS:
        return False
    return bool(_ALLCAPS_RE.fullmatch(tok) or _ABBR_TOKEN_RE.fullmatch(tok))


def _force_expand_remaining_abbrevs(tokens: List[str]) -> List[str]:
    out: List[str] = []
    for t in tokens:
        if not t:
            continue
        tl = t.lower()
        if _looks_like_bad_abbrev_token(t) and tl in FORCED_FALLBACK_MAP:
            out.append(FORCED_FALLBACK_MAP[tl])
        else:
            out.append(tl)
    return out


def _token_expand(token: str, store_hint: str) -> str:
    tl = token.lower()
    if store_hint and store_hint in STORE_TOKEN_MAPS:
        sm = STORE_TOKEN_MAPS.get(store_hint) or {}
        if tl in sm:
            return sm[tl]
    if tl in GLOBAL_TOKEN_MAP:
        return GLOBAL_TOKEN_MAP[tl]
    if tl in ABBREV_TOKEN_MAP:
        return ABBREV_TOKEN_MAP[tl]
    return tl


def expand_abbreviations(name: str) -> str:
    if not name:
        return ""

    raw = (name or "").strip().replace("&", " and ").replace("-", " ")
    # Replace W/B (Wishbone) before the slash is stripped
    raw = re.sub(r"\bw/b\b", "wishbone", raw, flags=re.IGNORECASE)
    raw = re.sub(r"[^\w\s']", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()

    toks = [t for t in raw.split(" ") if t]
    expanded: List[str] = []

    for i, t in enumerate(toks):
        tl = t.lower()
        nxt = toks[i + 1].lower() if i + 1 < len(toks) else ""

        if tl == "cr" and nxt in {"cut", "cr", "ctr"}:
            expanded.append("cracker")
            continue

        if tl == "h" and nxt == "h":
            expanded.append("half and half")
            continue

        expanded.append(ABBREV_TOKEN_MAP.get(tl, tl))

    joined = " ".join(expanded).strip()
    norm = _normalize_for_phrase_match(joined)

    for phrase in sorted(PHRASE_MAP.keys(), key=lambda x: len(x.split()), reverse=True):
        norm = re.sub(rf"\b{re.escape(phrase)}\b", PHRASE_MAP[phrase], norm)

    return re.sub(r"\s+", " ", norm).strip()


def post_name_cleanup(name: str) -> str:
    s = (name or "").strip()
    if not s:
        return ""

    low = s.lower().strip()

    if low == "mojo oven roasted h" or (low.startswith("mojo oven roasted") and low.endswith(" h")):
        return "mojo oven roasted chicken"

    # PUBLIX STICKS SALT -> Publix Salted Butter Sticks
    if re.search(r"\bsticks?\b", low) and re.search(r"\bsalted?\b", low) and "butter" not in low:
        low = re.sub(r"\bsticks?\b", "butter sticks", low)
        low = re.sub(r"\bsalted?\b", "salted", low)
        low = re.sub(r"\s+", " ", low).strip()

    toks = [t for t in low.split() if t]
    if toks and toks[-1] in {"sn"}:
        toks = toks[:-1]
        low = " ".join(toks).strip()

    toks = [t for t in low.split() if t]
    if toks and len(toks[-1]) == 1 and toks[-1].isalpha():
        toks = toks[:-1]
        low = " ".join(toks).strip()

    low = _strip_long_numeric_tokens(low)
    low = re.sub(r"\s+", " ", low).strip()
    return low


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
    if len(toks) <= 1 or _looks_abbreviated(s):
        return True
    letters = sum(1 for ch in s if ch.isalpha())
    uppers = sum(1 for ch in s if ch.isalpha() and ch.isupper())
    return letters >= 4 and (uppers / max(letters, 1)) > 0.85 and len(s) <= 18


def normalize_display_name(name: str, store_hint: str = "") -> Tuple[str, Dict[str, Any]]:
    dbg: Dict[str, Any] = {}
    if not name:
        return "", dbg

    base = post_name_cleanup(expand_abbreviations(_clean_line(name))).strip()
    base = re.sub(r"\s+", " ", base).strip()
    dbg["base"] = base
    if not base:
        return "", dbg

    toks = _split_tokens_preserve_numbers(base)
    dbg["tokens_in"] = toks[:]

    expanded_tokens = [_token_expand(t, store_hint) for t in toks]
    joined = " ".join(expanded_tokens).strip()

    norm = _normalize_for_phrase_match(joined)
    for phrase in sorted(PHRASE_MAP.keys(), key=lambda x: len(x.split()), reverse=True):
        norm = re.sub(rf"\b{re.escape(phrase)}\b", PHRASE_MAP[phrase], norm)
    norm = re.sub(r"\s+", " ", norm).strip()
    dbg["after_store_global_expand"] = norm

    toks2 = _split_tokens_preserve_numbers(norm)
    forced = _force_expand_remaining_abbrevs(toks2)
    dbg["forced_tokens"] = forced[:]

    norm2 = re.sub(r"\s+", " ", " ".join(forced).strip()).strip()
    norm2 = _strip_long_numeric_tokens(norm2)
    norm2 = re.sub(r"\s+", " ", norm2).strip()
    dbg["after_forced"] = norm2
    dbg["still_looks_abbrev"] = bool(_looks_abbreviated(norm2))

    pretty = title_case(norm2)
    pretty = pretty.replace("And Gather", "& Gather") if pretty.startswith("Good And Gather") else pretty
    dbg["pretty"] = pretty
    return pretty, dbg


# =========================
# Quantity parsing
# =========================

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
            pass

    m = QTY_AT_RE.search(s)
    if m:
        try:
            q = int(m.group(1))
            if 1 <= q <= 50:
                return q
        except Exception:
            pass

    m = QTY_FOR_RE.search(s)
    if m:
        try:
            q = int(m.group(1))
            if 1 <= q <= 50:
                return q
        except Exception:
            pass

    return 1


def _parse_quantity(line: str) -> Tuple[int, str]:
    s = (line or "").strip()
    if not s:
        return 1, ""

    # "x2" at end
    m = re.search(r"(.*?)\b[xX]\s*(\d{1,3})\s*$", s)
    if m:
        return max(int(m.group(2)), 1), m.group(1).strip()

    # "2x milk"
    m = re.match(r"^\s*(\d{1,3})\s*[xX]\s+(.*)$", s)
    if m:
        return max(int(m.group(1)), 1), m.group(2).strip()

    # "milk x2"
    m = re.match(r"^(.*)\s+[xX]\s*(\d{1,3})\s*$", s)
    if m:
        return max(int(m.group(2)), 1), m.group(1).strip()

    # "2 @ 1.99 milk" or "milk 2 @ 1.99"
    m = re.match(r"^\s*(\d{1,3})\s*@\s*\$?\s*\d+(?:[.,]\s*\d{1,2})?\s+(.*)$", s, flags=re.IGNORECASE)
    if m:
        return max(int(m.group(1)), 1), m.group(2).strip()

    m = re.match(r"^(.*)\b(\d{1,3})\s*@\s*\$?\s*\d+(?:[.,]\s*\d{1,2})?\s*$", s, flags=re.IGNORECASE)
    if m:
        return max(int(m.group(2)), 1), m.group(1).strip()

    # "2 for 5.00 milk"
    m = re.match(r"^\s*(\d{1,3})\s+for\s+\$?\s*\d+(?:[.,]\s*\d{1,2})?\s+(.*)$", s, flags=re.IGNORECASE)
    if m:
        return max(int(m.group(1)), 1), m.group(2).strip()

    m = re.match(r"^(.*)\b(\d{1,3})\s+for\s+\$?\s*\d+(?:[.,]\s*\d{1,2})?\s*$", s, flags=re.IGNORECASE)
    if m:
        return max(int(m.group(2)), 1), m.group(1).strip()

    # leading quantity
    m = LEADING_QTY_RE.match(s)
    if m:
        qty = int(m.group(1))
        tail = m.group(2).strip()
        # avoid treating "16 OZ MILK" as qty 16
        if 1 <= qty <= 24 and tail and not re.match(r"^(oz|lb|lbs|g|kg|ml|l)\b", tail, flags=re.IGNORECASE):
            return qty, tail

    # count/pack hints
    m = PACK_QTY_RE.search(s)
    if m:
        try:
            qty = int(m.group(1))
            if 1 <= qty <= 50:
                cleaned = PACK_QTY_RE.sub("", s).strip()
                cleaned = re.sub(r"\s+", " ", cleaned)
                if cleaned:
                    return qty, cleaned
        except Exception:
            pass

    return 1, s


# =========================
# Classification / merge
# =========================

def _classify(name: str) -> str:
    tokens = set(dedupe_key(name).split())
    if tokens & HOUSEHOLD_WORDS:
        return "Household"
    return "Food"


def _safe_merge_key(name: str) -> str:
    s = dedupe_key(name)
    s = re.sub(r"\b\d+(?:\.\d+)?\s*(oz|fl oz|lb|lbs|g|kg|ct|pack|pk)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _token_set(name: str) -> set[str]:
    return set(t for t in dedupe_key(name).split() if t)


def _should_merge_items(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    if (a.get("category") or "Food") != (b.get("category") or "Food"):
        return False

    a_name = (a.get("name") or "").strip()
    b_name = (b.get("name") or "").strip()
    if not a_name or not b_name:
        return False

    a_key = _safe_merge_key(a_name)
    b_key = _safe_merge_key(b_name)

    if a_key == b_key:
        return True

    a_tokens = _token_set(a_name)
    b_tokens = _token_set(b_name)

    if not a_tokens or not b_tokens:
        return False

    overlap = len(a_tokens & b_tokens)

    return overlap == len(a_tokens) == len(b_tokens)


def _dedupe_and_merge(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []

    for it in items:
        found = False

        for existing in merged:
            if _should_merge_items(existing, it):
                existing["quantity"] = int(existing["quantity"]) + int(it["quantity"])
                found = True
                break

        if not found:
            merged.append(dict(it))

    return sorted(merged, key=lambda x: (x.get("category") or "", x["name"].lower()))


# =========================
# Enrichment
# =========================

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


def _learned_map_lookup(raw_line: str, expanded_name: str, store_hint: str = "") -> Optional[str]:
    raw = (raw_line or "").strip()
    if not raw:
        return None

    base_keys = [raw, dedupe_key(raw), expanded_name, dedupe_key(expanded_name)]
    keys: List[str] = []

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


def _tokenize_for_score(s: str) -> List[str]:
    return [t for t in dedupe_key(s).split() if t and t not in _STOPWORDS_SCORE]


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


def _build_off_candidate(p: Dict[str, Any]) -> str:
    product_name = (p.get("product_name_en") or p.get("product_name") or "").strip()
    brands = (p.get("brands") or "").strip()
    qty = (p.get("quantity") or "").strip()

    if "," in brands:
        brands = brands.split(",")[0].strip()

    return " ".join(x for x in [brands, product_name, qty] if x).strip()


async def _off_get(url: str, params: Dict[str, Any], timeout_s: float) -> Optional[Dict[str, Any]]:
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


def _catalog_urls_for_category(category: str) -> List[Tuple[str, str]]:
    cat = (category or "").strip().lower()
    if cat == "household":
        out: List[Tuple[str, str]] = []
        if OPF_SEARCH_URL_WORLD:
            out.append(("openproductsfacts", OPF_SEARCH_URL_WORLD))
        if OBF_SEARCH_URL_WORLD:
            out.append(("openbeautyfacts", OBF_SEARCH_URL_WORLD))
        return out

    urls: List[Tuple[str, str]] = []
    if OFF_SEARCH_URL_US:
        urls.append(("openfoodfacts_us", OFF_SEARCH_URL_US))
    if OFF_SEARCH_URL_WORLD:
        urls.append(("openfoodfacts_world", OFF_SEARCH_URL_WORLD))
    return urls


async def _catalog_best_match(name: str, store_hint: str, category: str, budget: ReqBudget) -> Optional[Tuple[str, float, str, str]]:
    key = dedupe_key(name)
    if len(key) < 5 or budget.expired():
        return None

    variants: List[str] = [name]
    if store_hint and dedupe_key(store_hint) not in key:
        variants.append(f"{store_hint} {name}")
    if "publix" in key:
        variants.append(re.sub(r"\bpublix\b", "", key).strip())

    seen: set[str] = set()
    qvars: List[str] = []
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

                score = _score_candidate(q, cand, store_hint=store_hint)
                if score > best_score:
                    best_score = score
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
        if _is_header_or_address(learned) or _is_store_or_header_line_anywhere(learned) or _is_noise_line(learned) or _is_junk_line(learned):
            _pending_add(store_hint=store_hint, raw_line=raw_line, cleaned=cleaned_name, expanded=expanded_name)
            return expanded_name, "learned_map_rejected_header", 0.0, ""

        if cache_key:
            _enrich_cache_set(cache_key, learned, "learned_map", 1.0)
        return learned, "learned_map", 1.0, ""

    always_off = (os.getenv("ALWAYS_OFF_ENRICH", "1").strip() == "1")
    should_try_catalog = always_off or _needs_official_name(expanded_name)

    # Skip catalog lookup if expanded name starts with a protected brand
    _exp_lower = expanded_name.lower()
    if any(_exp_lower.startswith(brand) for brand in _PROTECTED_BRAND_PREFIXES):
        _pending_add(store_hint=store_hint, raw_line=raw_line, cleaned=cleaned_name, expanded=expanded_name)
        if cache_key:
            _enrich_cache_set(cache_key, expanded_name, "protected_brand", 0.0)
        return expanded_name, "protected_brand", 0.0, ""

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

        if _is_header_or_address(candidate) or _is_store_or_header_line_anywhere(candidate) or _is_noise_line(candidate) or _is_junk_line(candidate):
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


# =========================
# Image helpers
# =========================

def _cache_key(name: str, upc: Optional[str], product_id: Optional[str]) -> str:
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


async def fetch_image(url: str, headers: Optional[Dict[str, str]] = None) -> Optional[Tuple[bytes, str]]:
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


# =========================
# Routes
# =========================

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True}


@app.post("/parse-receipt", response_model=List[ParsedItem])
@app.post("/parse-receipt/", response_model=List[ParsedItem])
async def parse_receipt(
    request: Request,
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    only_food: bool = Query(False),
    merge_duplicates: bool = Query(False),
    debug: bool = Query(False),
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
    if len(raw_lines0) < 22 and budget.remaining() > 4.0:
        try:
            pre1 = _preprocess_image_bytes(raw, variant=1)
            text1 = ocr_text_google_vision(pre1)
            raw_lines1 = [ln.strip() for ln in (text1 or "").splitlines() if ln and ln.strip()]
            if len(raw_lines1) > len(raw_lines0):
                raw_lines = raw_lines1
        except Exception:
            pass

    store_hint = detect_store_hint(raw_lines)
    candidates, _dropped_lines = _extract_candidates_from_lines(raw_lines, store_hint=store_hint)
    base_url = _public_base_url(request)

    items: List[Dict[str, Any]] = []

    for c in candidates:
        qty, nm = _parse_quantity(c.cleaned_line)
        if qty == 1 and int(c.qty_hint) > 1:
            qty = int(c.qty_hint)

        pretty, dbg = normalize_display_name(nm, store_hint=store_hint)
        final_name = (pretty or "").strip()
        if not final_name:
            continue

        if ONLY_MONEYISH_RE.match(final_name) or PRICE_FLAG_RE.match(final_name) or WEIGHT_ONLY_RE.match(final_name):
            continue

        if _is_noise_line(final_name) or _is_header_or_address(final_name) or _is_store_or_header_line_anywhere(final_name) or _is_junk_line(final_name):
            continue

        if not _has_valid_item_words(final_name):
            continue

        category = _classify(final_name)
        if only_food and category != "Food":
            continue

        items.append({
            "name": final_name,
            "quantity": int(max(1, qty)),
            "category": category,
            "image_url": _image_url_for_item(base_url, final_name),
            "_raw_line": c.raw_line,
            "_name_cleaned": _clean_line(nm),
            "_expanded": dbg.get("base", ""),
        })

    if merge_duplicates:
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

            pretty2, _dbg2 = normalize_display_name(enriched, store_hint=store_hint)
            enriched_final = (pretty2 or "").strip()

            if enriched_final and _has_valid_item_words(enriched_final) and not _is_noise_line(enriched_final) and not _is_header_or_address(enriched_final) and not _is_store_or_header_line_anywhere(enriched_final) and not _is_junk_line(enriched_final):
                it["name"] = enriched_final
                it["image_url"] = _image_url_for_item(base_url, enriched_final)

        for it in items:
            it.pop("_raw_line", None)
            it.pop("_name_cleaned", None)
            it.pop("_expanded", None)

    if items:
        enriched = await enrich_items_with_ai(items)
        for it, info in zip(items, enriched):
            it["expires_in_days"] = info.get("expires_in_days")
            it["storage"] = info.get("storage")
            if info.get("category"):
                it["category"] = info["category"]

    return items


@app.post("/parse-receipt-debug")
@app.post("/parse-receipt-debug/")
async def parse_receipt_debug(
    request: Request,
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    only_food: bool = Query(False),
    merge_duplicates: bool = Query(False),
    debug: bool = Query(True),
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
    ocr_variant_used = 0

    if len(raw_lines0) < 22 and budget.remaining() > 4.0:
        try:
            pre1 = _preprocess_image_bytes(raw, variant=1)
            text1 = ocr_text_google_vision(pre1)
            raw_lines1 = [ln.strip() for ln in (text1 or "").splitlines() if ln and ln.strip()]
            if len(raw_lines1) > len(raw_lines0):
                raw_lines = raw_lines1
                ocr_variant_used = 1
        except Exception:
            pass

    store_hint = detect_store_hint(raw_lines)
    totals_idx = find_totals_marker_index(raw_lines)
    zone_start, zone_end = _detect_item_zone_indices(raw_lines, store_hint=store_hint)
    candidates, dropped_lines = _extract_candidates_from_lines(raw_lines, store_hint=store_hint)

    base_url = _public_base_url(request)
    parsed: List[Dict[str, Any]] = []
    enrich_debug: List[Dict[str, Any]] = []

    always_off = (os.getenv("ALWAYS_OFF_ENRICH", "1").strip() == "1")

    for c in candidates:
        qty, nm = _parse_quantity(c.cleaned_line)
        if qty == 1 and int(c.qty_hint) > 1:
            qty = int(c.qty_hint)

        pretty0, dbg0 = normalize_display_name(nm, store_hint=store_hint)
        expanded = dbg0.get("base", "")
        category_guess = _classify(pretty0)

        enriched = expanded
        source = "none"
        score = 0.0
        query_used = ""

        if ENABLE_NAME_ENRICH and budget.remaining() > 0.9 and (always_off or _needs_official_name(expanded)):
            enriched, source, score, query_used = await enrich_full_name(
                raw_line=c.raw_line,
                cleaned_name=_clean_line(nm),
                expanded_name=expanded,
                store_hint=store_hint,
                category=category_guess,
                budget=budget,
            )

        pretty1, dbg1 = normalize_display_name(enriched, store_hint=store_hint)

        enrich_debug.append({
            "raw_line": c.raw_line,
            "cleaned_line": c.cleaned_line,
            "candidate_source": c.source,
            "qty": qty,
            "qty_hint": c.qty_hint,
            "display_base": pretty0,
            "display_after_enrich": pretty1,
            "normalize_dbg_base": dbg0,
            "normalize_dbg_enriched": dbg1,
            "category": category_guess,
            "enrich_source": source,
            "enrich_score": score,
            "enrich_query_used": query_used,
        })

        final_name = (pretty1 or "").strip()
        if not final_name:
            continue

        if ONLY_MONEYISH_RE.match(final_name) or PRICE_FLAG_RE.match(final_name) or WEIGHT_ONLY_RE.match(final_name):
            continue
        if _is_noise_line(final_name) or _is_header_or_address(final_name) or _is_store_or_header_line_anywhere(final_name) or _is_junk_line(final_name):
            continue
        if not _has_valid_item_words(final_name):
            continue

        category = _classify(final_name)
        if only_food and category != "Food":
            continue

        parsed.append({
            "name": final_name,
            "quantity": int(max(1, qty)),
            "category": category,
            "image_url": _image_url_for_item(base_url, final_name),
        })

    if merge_duplicates:
        parsed = _dedupe_and_merge(parsed)

    return {
        "items": parsed,
        "raw_line_count": len(raw_lines),
        "raw_lines": raw_lines[:800],
        "ocr_variant_used": ocr_variant_used,
        "store_hint": store_hint,
        "zone_start": zone_start,
        "zone_end": zone_end,
        "totals_idx": totals_idx,
        "candidate_count": len(candidates),
        "candidates": [
            {
                "raw_line": c.raw_line,
                "cleaned_line": c.cleaned_line,
                "qty_hint": c.qty_hint,
                "source": c.source,
            }
            for c in candidates[:500]
        ],
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


@app.post("/instacart/create-list")
@app.post("/instacart/create-list/")
@app.post("/instacart/create_list")
@app.post("/instacart/create_list/")
async def instacart_create_list(req: InstacartCreateListRequest) -> Dict[str, str]:
    api_key = (os.getenv("INSTACART_API_KEY") or "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing INSTACART_API_KEY env var on server")

    items = _normalized_instacart_items(req)
    if not items:
        raise HTTPException(status_code=422, detail="Field required: items")

    payload = {
        "title": req.title,
        "link_type": "shopping_list",
        "line_items": [{"name": i.name, "quantity": i.quantity, "unit": i.unit} for i in items],
    }

    url = INSTACART_PRODUCTS_LINK_URL
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
            r = await client.post(url, headers=headers, json=payload)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Instacart request failed: {e}")

    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    try:
        data = r.json()
    except Exception:
        raise HTTPException(status_code=502, detail=f"Instacart returned non-JSON: {r.text[:500]}")

    link = _extract_instacart_link(data)
    if not link:
        raise HTTPException(status_code=502, detail=f"Instacart response missing link: {data}")

    if not (link.startswith("http://") or link.startswith("https://") or link.startswith("instacart://")):
        raise HTTPException(status_code=502, detail=f"Instacart returned unexpected link: {link}")

    return {"url": link}


@app.get("/image")
async def get_product_image(name: str = Query(...), upc: Optional[str] = Query(None), product_id: Optional[str] = Query(None)):
    ck = _cache_key(name, upc, product_id)

    if ck in _IMAGE_CACHE and ck in _IMAGE_CONTENT_TYPE_CACHE:
        return Response(content=_IMAGE_CACHE[ck], media_type=_IMAGE_CONTENT_TYPE_CACHE[ck])

    if PACKSHOT_SERVICE_URL:
        qp: List[str] = []
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


def _require_admin(key: Optional[str]) -> None:
    if not ADMIN_KEY:
        raise HTTPException(status_code=500, detail="ADMIN_KEY not set on server")
    if not key or key.strip() != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/admin/learned-map")
async def admin_learned_map(key: Optional[str] = Query(None)):
    _require_admin(key)
    return {"entries": len(_LEARNED_MAP), "path": NAME_MAP_PATH}


@app.get("/admin/pending")
async def admin_pending(key: Optional[str] = Query(None), limit: int = Query(200)):
    _require_admin(key)

    items = list(_PENDING.items())
    items.sort(key=lambda kv: int((kv[1] or {}).get("count", 0)), reverse=True)
    out = [{"key": k, **v} for k, v in items[:max(1, min(limit, 2000))]]

    return {"pending": out, "total": len(_PENDING), "path": PENDING_PATH}


@app.post("/admin/feedback")
async def admin_feedback(body: FeedbackBody, key: Optional[str] = Query(None)):
    _require_admin(key)

    expanded = (body.expanded or "").strip()
    full_name = (body.full_name or "").strip()
    store_hint = (body.store_hint or "").strip()

    if not expanded or not full_name:
        raise HTTPException(status_code=400, detail="expanded and full_name required")

    updates: Dict[str, str] = {}
    updates[expanded] = full_name
    updates[dedupe_key(expanded)] = full_name

    if store_hint:
        updates[f"{store_hint}:{expanded}"] = full_name
        updates[f"{dedupe_key(store_hint)}:{expanded}"] = full_name
        updates[f"{store_hint}:{dedupe_key(expanded)}"] = full_name
        updates[f"{dedupe_key(store_hint)}:{dedupe_key(expanded)}"] = full_name

    current: Dict[str, str] = {}
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


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    try:
        resp = await call_next(request)
        return resp
    finally:
        print(f"{request.method} {request.url.path}")
