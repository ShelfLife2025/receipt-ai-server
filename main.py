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
from fastapi import FastAPI, File,  HTTPException, Query, Request, UploadFile
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

SPOONACULAR_API = (os.getenv("SPOONACULAR_API") or "").strip()

VISION_TMP_PATH = "/tmp/gcloud_key.json"
# ── Multi-storage shelf life table (USDA / FDA / StillTasty sourced) ──────────
# Each entry has fridge, freezer, and pantry days.
# Use None where that storage method is not recommended.
# expires_in_days = the DEFAULT storage method's days (what gets applied on scan)
SHELF_LIFE_DB: Dict[str, Dict] = {
    # format: "key": {"fridge": days, "freezer": days, "pantry": days, "default_storage": "fridge"|"freezer"|"pantry", "category": "Food"|"Household"}
    # ── MEAT: Raw ─────────────────────────────────────────────────────────────
    "chicken breast":       {"fridge": 2,    "freezer": 270,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "chicken thigh":        {"fridge": 2,    "freezer": 270,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "chicken wings":        {"fridge": 2,    "freezer": 270,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "chicken drumstick":    {"fridge": 2,    "freezer": 270,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "whole chicken":        {"fridge": 2,    "freezer": 365,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "chicken":              {"fridge": 2,    "freezer": 270,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "turkey":               {"fridge": 2,    "freezer": 365,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "ground beef":          {"fridge": 2,    "freezer": 120,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "ground turkey":        {"fridge": 2,    "freezer": 120,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "ground pork":          {"fridge": 2,    "freezer": 120,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "ground chicken":       {"fridge": 2,    "freezer": 120,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "beef":                 {"fridge": 4,    "freezer": 270,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "steak":                {"fridge": 4,    "freezer": 270,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "pork chop":            {"fridge": 4,    "freezer": 150,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "pork loin":            {"fridge": 4,    "freezer": 180,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "pulled pork":                {"fridge": 45,   "freezer": 90,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "jack daniel's pulled pork":  {"fridge": 45,   "freezer": 90,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "jack daniel's pork":         {"fridge": 45,   "freezer": 90,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "jack daniel's":              {"fridge": 45,   "freezer": 90,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "cooked pulled pork":         {"fridge": 45,   "freezer": 90,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "pre-cooked pork":            {"fridge": 45,   "freezer": 90,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "rotisserie chicken":         {"fridge": 4,    "freezer": 90,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "fully cooked chicken":       {"fridge": 5,    "freezer": 90,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "grilled chicken strips":     {"fridge": 5,    "freezer": 90,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "fully cooked bacon":         {"fridge": 14,   "freezer": 30,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "precooked bacon":            {"fridge": 14,   "freezer": 30,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "vacuum sealed meat":         {"fridge": 30,   "freezer": 180,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "smoked sausage":             {"fridge": 14,   "freezer": 60,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "kielbasa":                   {"fridge": 14,   "freezer": 60,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "andouille":                  {"fridge": 14,   "freezer": 60,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "bratwurst":                  {"fridge": 4,    "freezer": 60,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "pork":                 {"fridge": 4,    "freezer": 150,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "lamb":                 {"fridge": 4,    "freezer": 270,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "veal":                 {"fridge": 4,    "freezer": 270,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "ribs":                 {"fridge": 4,    "freezer": 180,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "roast":                {"fridge": 4,    "freezer": 365,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    # ── MEAT: Processed ───────────────────────────────────────────────────────
    "bacon":                {"fridge": 7,    "freezer": 30,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "pancetta":             {"fridge": 7,    "freezer": 30,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "ham":                  {"fridge": 5,    "freezer": 60,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "prosciutto":           {"fridge": 5,    "freezer": 60,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "salami":               {"fridge": 5,    "freezer": 60,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "pepperoni":            {"fridge": 7,    "freezer": 60,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "deli meat":            {"fridge": 5,    "freezer": 60,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "lunch meat":           {"fridge": 5,    "freezer": 60,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "lunchmeat":            {"fridge": 5,    "freezer": 60,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "deli turkey":          {"fridge": 5,    "freezer": 60,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "deli ham":             {"fridge": 5,    "freezer": 60,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "hot dogs":             {"fridge": 7,    "freezer": 60,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "hot dog":              {"fridge": 7,    "freezer": 60,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "bratwurst":            {"fridge": 3,    "freezer": 60,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "sausage":              {"fridge": 3,    "freezer": 60,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "italian sausage":      {"fridge": 3,    "freezer": 60,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "chorizo":              {"fridge": 3,    "freezer": 60,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "rotisserie chicken":   {"fridge": 4,    "freezer": 120,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "cooked chicken":       {"fridge": 4,    "freezer": 120,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    # ── SEAFOOD ──────────────────────────────────────────────────────────────
    "salmon":               {"fridge": 2,    "freezer": 180,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "tuna steak":           {"fridge": 2,    "freezer": 90,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "tilapia":              {"fridge": 2,    "freezer": 240,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "cod":                  {"fridge": 2,    "freezer": 240,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "halibut":              {"fridge": 2,    "freezer": 240,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "shrimp":               {"fridge": 2,    "freezer": 180,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "scallops":             {"fridge": 2,    "freezer": 180,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "lobster":              {"fridge": 2,    "freezer": 180,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "crab":                 {"fridge": 2,    "freezer": 180,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "mussels":              {"fridge": 2,    "freezer": 90,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "clams":                {"fridge": 2,    "freezer": 90,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "fish":                 {"fridge": 2,    "freezer": 180,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "smoked salmon":        {"fridge": 14,   "freezer": 60,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "smoked fish":          {"fridge": 14,   "freezer": 60,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "canned tuna":          {"fridge": None, "freezer": None, "pantry": 1095, "default_storage": "pantry",  "category": "Food"},
    "canned salmon":        {"fridge": None, "freezer": None, "pantry": 1095, "default_storage": "pantry",  "category": "Food"},
    # ── DAIRY ────────────────────────────────────────────────────────────────
    "whole milk":           {"fridge": 7,    "freezer": 90,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "skim milk":            {"fridge": 7,    "freezer": 90,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "2% milk":              {"fridge": 7,    "freezer": 90,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "milk":                 {"fridge": 7,    "freezer": 90,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "oat milk":             {"fridge": 7,    "freezer": 180,  "pantry": 365,  "default_storage": "fridge", "category": "Food"},
    "almond milk":          {"fridge": 7,    "freezer": 180,  "pantry": 365,  "default_storage": "fridge", "category": "Food"},
    "soy milk":             {"fridge": 7,    "freezer": 180,  "pantry": 365,  "default_storage": "fridge", "category": "Food"},
    "buttermilk":           {"fridge": 14,   "freezer": 90,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "heavy cream":          {"fridge": 10,   "freezer": 120,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "heavy whipping cream": {"fridge": 10,   "freezer": 120,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "half and half":        {"fridge": 10,   "freezer": 120,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "half & half":          {"fridge": 10,   "freezer": 120,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "whipping cream":       {"fridge": 10,   "freezer": 120,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "sour cream":           {"fridge": 14,   "freezer": 180,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "cottage cheese":       {"fridge": 14,   "freezer": 90,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "greek yogurt":         {"fridge": 14,   "freezer": 60,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "yogurt":               {"fridge": 14,   "freezer": 60,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "eggs":                 {"fridge": 35,   "freezer": 365,  "pantry": 14,   "default_storage": "fridge", "category": "Food"},
    "egg":                  {"fridge": 35,   "freezer": 365,  "pantry": 14,   "default_storage": "fridge", "category": "Food"},
    "salted butter":        {"fridge": 30,   "freezer": 365,  "pantry": 14,   "default_storage": "fridge", "category": "Food"},
    "butter":               {"fridge": 30,   "freezer": 365,  "pantry": 14,   "default_storage": "fridge", "category": "Food"},
    "unsalted butter":      {"fridge": 14,   "freezer": 365,  "pantry": 5,    "default_storage": "fridge", "category": "Food"},
    "eggnog":               {"fridge": 5,    "freezer": 180,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    # ── CHEESE ───────────────────────────────────────────────────────────────
    "fresh mozzarella":     {"fridge": 5,    "freezer": 60,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "burrata":              {"fridge": 5,    "freezer": 90,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "ricotta":              {"fridge": 7,    "freezer": 90,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "brie":                 {"fridge": 7,    "freezer": 60,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "camembert":            {"fridge": 7,    "freezer": 60,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "cream cheese":         {"fridge": 10,   "freezer": 60,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "mascarpone":           {"fridge": 10,   "freezer": 60,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "shredded mozzarella":  {"fridge": 14,   "freezer": 90,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "shredded cheese":      {"fridge": 14,   "freezer": 90,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "mozzarella":           {"fridge": 14,   "freezer": 90,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "provolone":            {"fridge": 14,   "freezer": 90,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "swiss cheese":         {"fridge": 14,   "freezer": 90,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "muenster":             {"fridge": 14,   "freezer": 90,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "american cheese":      {"fridge": 14,   "freezer": 90,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "sliced cheese":        {"fridge": 14,   "freezer": 90,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "cheddar":              {"fridge": 30,   "freezer": 180,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "sharp cheddar":        {"fridge": 30,   "freezer": 180,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "colby jack":           {"fridge": 30,   "freezer": 180,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "monterey jack":        {"fridge": 30,   "freezer": 180,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "gouda":                {"fridge": 30,   "freezer": 180,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "havarti":              {"fridge": 30,   "freezer": 180,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "cheese":               {"fridge": 14,   "freezer": 90,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "parmesan":             {"fridge": 60,   "freezer": 180,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "romano":               {"fridge": 60,   "freezer": 180,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "asiago":               {"fridge": 60,   "freezer": 180,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "feta":                 {"fridge": 14,   "freezer": 90,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    "blue cheese":          {"fridge": 14,   "freezer": 90,   "pantry": None, "default_storage": "fridge", "category": "Food"},
    # ── PRODUCE: Fruit ───────────────────────────────────────────────────────
    "strawberries":         {"fridge": 3,    "freezer": 365,  "pantry": 1,    "default_storage": "fridge", "category": "Food"},
    "raspberries":          {"fridge": 3,    "freezer": 365,  "pantry": 1,    "default_storage": "fridge", "category": "Food"},
    "blackberries":         {"fridge": 3,    "freezer": 365,  "pantry": 1,    "default_storage": "fridge", "category": "Food"},
    "blueberries":          {"fridge": 7,    "freezer": 365,  "pantry": 2,    "default_storage": "fridge", "category": "Food"},
    "grapes":               {"fridge": 7,    "freezer": 365,  "pantry": 2,    "default_storage": "fridge", "category": "Food"},
    "cherries":             {"fridge": 5,    "freezer": 365,  "pantry": 2,    "default_storage": "fridge", "category": "Food"},
    "peaches":              {"fridge": 5,    "freezer": 365,  "pantry": 3,    "default_storage": "fridge", "category": "Food"},
    "plums":                {"fridge": 5,    "freezer": 365,  "pantry": 3,    "default_storage": "fridge", "category": "Food"},
    "nectarines":           {"fridge": 5,    "freezer": 365,  "pantry": 3,    "default_storage": "fridge", "category": "Food"},
    "mango":                {"fridge": 5,    "freezer": 365,  "pantry": 3,    "default_storage": "pantry",  "category": "Food"},
    "pineapple":            {"fridge": 7,    "freezer": 365,  "pantry": 5,    "default_storage": "pantry",  "category": "Food"},
    "watermelon":           {"fridge": 14,   "freezer": 365,  "pantry": 10,   "default_storage": "pantry",  "category": "Food"},
    "cantaloupe":           {"fridge": 5,    "freezer": 365,  "pantry": 3,    "default_storage": "fridge",  "category": "Food"},
    "honeydew":             {"fridge": 5,    "freezer": 365,  "pantry": 3,    "default_storage": "fridge",  "category": "Food"},
    "apples":               {"fridge": 42,   "freezer": 365,  "pantry": 7,    "default_storage": "fridge", "category": "Food"},
    "apple":                {"fridge": 42,   "freezer": 365,  "pantry": 7,    "default_storage": "fridge", "category": "Food"},
    "oranges":              {"fridge": 21,   "freezer": 365,  "pantry": 7,    "default_storage": "fridge", "category": "Food"},
    "orange":               {"fridge": 21,   "freezer": 365,  "pantry": 7,    "default_storage": "fridge", "category": "Food"},
    "lemons":               {"fridge": 21,   "freezer": 365,  "pantry": 7,    "default_storage": "fridge", "category": "Food"},
    "limes":                {"fridge": 21,   "freezer": 365,  "pantry": 7,    "default_storage": "fridge", "category": "Food"},
    "grapefruit":           {"fridge": 21,   "freezer": 365,  "pantry": 7,    "default_storage": "fridge", "category": "Food"},
    "avocado":              {"fridge": 4,    "freezer": 120,  "pantry": 4,    "default_storage": "pantry",  "category": "Food"},
    "bananas":              {"fridge": 7,    "freezer": 90,   "pantry": 5,    "default_storage": "pantry",  "category": "Food"},
    "banana":               {"fridge": 7,    "freezer": 90,   "pantry": 5,    "default_storage": "pantry",  "category": "Food"},
    # ── PRODUCE: Vegetables ──────────────────────────────────────────────────
    "spinach":              {"fridge": 5,    "freezer": 365,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "arugula":              {"fridge": 5,    "freezer": 365,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "mixed greens":         {"fridge": 5,    "freezer": None, "pantry": None, "default_storage": "fridge", "category": "Food"},
    "salad mix":            {"fridge": 5,    "freezer": None, "pantry": None, "default_storage": "fridge", "category": "Food"},
    "spring mix":           {"fridge": 5,    "freezer": None, "pantry": None, "default_storage": "fridge", "category": "Food"},
    "lettuce":              {"fridge": 7,    "freezer": None, "pantry": None, "default_storage": "fridge", "category": "Food"},
    "romaine":              {"fridge": 7,    "freezer": None, "pantry": None, "default_storage": "fridge", "category": "Food"},
    "broccoli":             {"fridge": 5,    "freezer": 365,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "cauliflower":          {"fridge": 5,    "freezer": 365,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "brussels sprouts":     {"fridge": 5,    "freezer": 365,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "asparagus":            {"fridge": 3,    "freezer": 365,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "mushrooms":            {"fridge": 5,    "freezer": 365,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "zucchini":             {"fridge": 5,    "freezer": 365,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "summer squash":        {"fridge": 5,    "freezer": 365,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "bell pepper":          {"fridge": 7,    "freezer": 365,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "bell peppers":         {"fridge": 7,    "freezer": 365,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "cucumber":             {"fridge": 7,    "freezer": None, "pantry": None, "default_storage": "fridge", "category": "Food"},
    "green beans":          {"fridge": 5,    "freezer": 365,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "snap peas":            {"fridge": 5,    "freezer": 365,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "corn":                 {"fridge": 2,    "freezer": 365,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "corn on the cob":      {"fridge": 2,    "freezer": 365,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "kale":                 {"fridge": 7,    "freezer": 365,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "cabbage":              {"fridge": 14,   "freezer": 365,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "celery":               {"fridge": 14,   "freezer": 365,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "carrots":              {"fridge": 21,   "freezer": 365,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "baby carrots":         {"fridge": 21,   "freezer": 365,  "pantry": None, "default_storage": "fridge", "category": "Food"},
    "tomatoes":             {"fridge": 7,    "freezer": 180,  "pantry": 5,    "default_storage": "pantry",  "category": "Food"},
    "tomato":               {"fridge": 7,    "freezer": 180,  "pantry": 5,    "default_storage": "pantry",  "category": "Food"},
    "cherry tomatoes":      {"fridge": 7,    "freezer": 180,  "pantry": 5,    "default_storage": "pantry",  "category": "Food"},
    "onions":               {"fridge": 60,   "freezer": 180,  "pantry": 30,   "default_storage": "pantry",  "category": "Food"},
    "onion":                {"fridge": 60,   "freezer": 180,  "pantry": 30,   "default_storage": "pantry",  "category": "Food"},
    "shallots":             {"fridge": 60,   "freezer": 180,  "pantry": 30,   "default_storage": "pantry",  "category": "Food"},
    "garlic":               {"fridge": 180,  "freezer": 365,  "pantry": 90,   "default_storage": "pantry",  "category": "Food"},
    "potatoes":             {"fridge": 60,   "freezer": 365,  "pantry": 30,   "default_storage": "pantry",  "category": "Food"},
    "potato":               {"fridge": 60,   "freezer": 365,  "pantry": 30,   "default_storage": "pantry",  "category": "Food"},
    "sweet potato":         {"fridge": 30,   "freezer": 365,  "pantry": 21,   "default_storage": "pantry",  "category": "Food"},
    "sweet potatoes":       {"fridge": 30,   "freezer": 365,  "pantry": 21,   "default_storage": "pantry",  "category": "Food"},
    "beets":                {"fridge": 14,   "freezer": 365,  "pantry": None, "default_storage": "fridge",  "category": "Food"},
    "radishes":             {"fridge": 14,   "freezer": None, "pantry": None, "default_storage": "fridge",  "category": "Food"},
    "eggplant":             {"fridge": 5,    "freezer": 365,  "pantry": None, "default_storage": "fridge",  "category": "Food"},
    "artichoke":            {"fridge": 5,    "freezer": 180,  "pantry": None, "default_storage": "fridge",  "category": "Food"},
    "leeks":                {"fridge": 7,    "freezer": 180,  "pantry": None, "default_storage": "fridge",  "category": "Food"},
    # ── BREAD & BAKERY ───────────────────────────────────────────────────────
    "bread":                {"fridge": 14,   "freezer": 90,   "pantry": 7,    "default_storage": "pantry",  "category": "Food"},
    "sandwich bread":       {"fridge": 14,   "freezer": 90,   "pantry": 10,   "default_storage": "pantry",  "category": "Food"},
    "sourdough":            {"fridge": 14,   "freezer": 90,   "pantry": 7,    "default_storage": "pantry",  "category": "Food"},
    "bagels":               {"fridge": 14,   "freezer": 90,   "pantry": 7,    "default_storage": "pantry",  "category": "Food"},
    "english muffins":      {"fridge": 14,   "freezer": 90,   "pantry": 7,    "default_storage": "pantry",  "category": "Food"},
    "tortillas":            {"fridge": 30,   "freezer": 180,  "pantry": 14,   "default_storage": "pantry",  "category": "Food"},
    "flour tortillas":      {"fridge": 30,   "freezer": 180,  "pantry": 14,   "default_storage": "pantry",  "category": "Food"},
    "corn tortillas":       {"fridge": 30,   "freezer": 180,  "pantry": 14,   "default_storage": "pantry",  "category": "Food"},
    "dinner rolls":         {"fridge": 14,   "freezer": 90,   "pantry": 7,    "default_storage": "pantry",  "category": "Food"},
    "sweet rolls":          {"fridge": 14,   "freezer": 90,   "pantry": 14,   "default_storage": "pantry",  "category": "Food"},
    "hawaiian rolls":       {"fridge": 21,   "freezer": 90,   "pantry": 14,   "default_storage": "pantry",  "category": "Food"},
    "king's hawaiian":      {"fridge": 21,   "freezer": 90,   "pantry": 14,   "default_storage": "pantry",  "category": "Food"},
    "savory rolls":         {"fridge": 14,   "freezer": 90,   "pantry": 14,   "default_storage": "pantry",  "category": "Food"},
    "hamburger buns":       {"fridge": 14,   "freezer": 90,   "pantry": 10,   "default_storage": "pantry",  "category": "Food"},
    "hot dog buns":         {"fridge": 14,   "freezer": 90,   "pantry": 10,   "default_storage": "pantry",  "category": "Food"},
    "pita bread":           {"fridge": 14,   "freezer": 90,   "pantry": 7,    "default_storage": "pantry",  "category": "Food"},
    "croissants":           {"fridge": 7,    "freezer": 30,   "pantry": 3,    "default_storage": "pantry",  "category": "Food"},
    "muffins":              {"fridge": 7,    "freezer": 90,   "pantry": 5,    "default_storage": "pantry",  "category": "Food"},
    "donuts":               {"fridge": 5,    "freezer": 30,   "pantry": 3,    "default_storage": "pantry",  "category": "Food"},
    "cake":                 {"fridge": 7,    "freezer": 90,   "pantry": 5,    "default_storage": "pantry",  "category": "Food"},
    "garlic bread":         {"fridge": 5,    "freezer": 90,   "pantry": None, "default_storage": "freezer", "category": "Food"},
    "pretzel buns":         {"fridge": 14,   "freezer": 90,   "pantry": 7,    "default_storage": "pantry",  "category": "Food"},
    "brioche":              {"fridge": 14,   "freezer": 90,   "pantry": 7,    "default_storage": "pantry",  "category": "Food"},
    # ── PREPARED / FRESH FOODS ───────────────────────────────────────────────
    "tortellini":           {"fridge": 4,    "freezer": 60,   "pantry": None, "default_storage": "fridge",  "category": "Food"},
    "ravioli":              {"fridge": 4,    "freezer": 60,   "pantry": None, "default_storage": "fridge",  "category": "Food"},
    "fresh pasta":          {"fridge": 3,    "freezer": 60,   "pantry": None, "default_storage": "fridge",  "category": "Food"},
    "gnocchi":              {"fridge": 3,    "freezer": 60,   "pantry": None, "default_storage": "fridge",  "category": "Food"},
    "hummus":               {"fridge": 7,    "freezer": 90,   "pantry": None, "default_storage": "fridge",  "category": "Food"},
    "guacamole":            {"fridge": 3,    "freezer": 90,   "pantry": None, "default_storage": "fridge",  "category": "Food"},
    "salsa":                {"fridge": 14,   "freezer": 180,  "pantry": 730,  "default_storage": "fridge",  "category": "Food"},
    "dip":                  {"fridge": 7,    "freezer": None, "pantry": None, "default_storage": "fridge",  "category": "Food"},
    "dips":                 {"fridge": 7,    "freezer": None, "pantry": None, "default_storage": "fridge",  "category": "Food"},
    "pesto":                {"fridge": 7,    "freezer": 90,   "pantry": None, "default_storage": "fridge",  "category": "Food"},
    "pizza dough":          {"fridge": 3,    "freezer": 90,   "pantry": None, "default_storage": "fridge",  "category": "Food"},
    "soup":                 {"fridge": 4,    "freezer": 90,   "pantry": None, "default_storage": "fridge",  "category": "Food"},
    "tofu":                 {"fridge": 7,    "freezer": 90,   "pantry": None, "default_storage": "fridge",  "category": "Food"},
    "tempeh":               {"fridge": 7,    "freezer": 90,   "pantry": None, "default_storage": "fridge",  "category": "Food"},
    # ── FROZEN ───────────────────────────────────────────────────────────────
    "frozen pizza":         {"fridge": None, "freezer": 180,  "pantry": None, "default_storage": "freezer", "category": "Food"},
    "frozen vegetables":    {"fridge": None, "freezer": 365,  "pantry": None, "default_storage": "freezer", "category": "Food"},
    "frozen fruit":         {"fridge": None, "freezer": 365,  "pantry": None, "default_storage": "freezer", "category": "Food"},
    "frozen chicken":       {"fridge": 2,    "freezer": 270,  "pantry": None, "default_storage": "freezer", "category": "Food"},
    "frozen beef":          {"fridge": 2,    "freezer": 180,  "pantry": None, "default_storage": "freezer", "category": "Food"},
    "frozen shrimp":        {"fridge": 2,    "freezer": 180,  "pantry": None, "default_storage": "freezer", "category": "Food"},
    "frozen fish":          {"fridge": 2,    "freezer": 180,  "pantry": None, "default_storage": "freezer", "category": "Food"},
    "frozen meal":          {"fridge": None, "freezer": 180,  "pantry": None, "default_storage": "freezer", "category": "Food"},
    "frozen burrito":       {"fridge": None, "freezer": 180,  "pantry": None, "default_storage": "freezer", "category": "Food"},
    "frozen waffles":       {"fridge": None, "freezer": 60,   "pantry": None, "default_storage": "freezer", "category": "Food"},
    "frozen pancakes":      {"fridge": None, "freezer": 60,   "pantry": None, "default_storage": "freezer", "category": "Food"},
    "ice cream":            {"fridge": None, "freezer": 180,  "pantry": None, "default_storage": "freezer", "category": "Food"},
    "frozen yogurt":        {"fridge": None, "freezer": 180,  "pantry": None, "default_storage": "freezer", "category": "Food"},
    "ice pops":             {"fridge": None, "freezer": 180,  "pantry": None, "default_storage": "freezer", "category": "Food"},
    # ── PANTRY: Dry Goods ────────────────────────────────────────────────────
    "pasta":                {"fridge": None, "freezer": 730,  "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "spaghetti":            {"fridge": None, "freezer": 730,  "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "penne":                {"fridge": None, "freezer": 730,  "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "rice":                 {"fridge": None, "freezer": 730,  "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "white rice":           {"fridge": None, "freezer": 730,  "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "brown rice":           {"fridge": None, "freezer": 730,  "pantry": 365,  "default_storage": "pantry",  "category": "Food"},
    "quinoa":               {"fridge": None, "freezer": 730,  "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "oats":                 {"fridge": None, "freezer": 730,  "pantry": 365,  "default_storage": "pantry",  "category": "Food"},
    "oatmeal":              {"fridge": None, "freezer": 730,  "pantry": 365,  "default_storage": "pantry",  "category": "Food"},
    "cereal":               {"fridge": None, "freezer": None, "pantry": 180,  "default_storage": "pantry",  "category": "Food"},
    "granola":              {"fridge": None, "freezer": 365,  "pantry": 180,  "default_storage": "pantry",  "category": "Food"},
    "flour":                {"fridge": 365,  "freezer": 730,  "pantry": 365,  "default_storage": "pantry",  "category": "Food"},
    "bread crumbs":         {"fridge": None, "freezer": 365,  "pantry": 180,  "default_storage": "pantry",  "category": "Food"},
    "cornmeal":             {"fridge": None, "freezer": 365,  "pantry": 365,  "default_storage": "pantry",  "category": "Food"},
    "baking powder":        {"fridge": None, "freezer": None, "pantry": 365,  "default_storage": "pantry",  "category": "Food"},
    "baking soda":                    {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "arm and hammer baking soda":     {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "arm & hammer baking soda":       {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "arm hammer baking soda":         {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "sugar":                {"fridge": None, "freezer": None, "pantry": 3650, "default_storage": "pantry",  "category": "Food"},
    "brown sugar":          {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "powdered sugar":       {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "salt":                 {"fridge": None, "freezer": None, "pantry": 3650, "default_storage": "pantry",  "category": "Food"},
    "honey":                {"fridge": None, "freezer": None, "pantry": 3650, "default_storage": "pantry",  "category": "Food"},
    "maple syrup":          {"fridge": 365,  "freezer": None, "pantry": 365,  "default_storage": "pantry",  "category": "Food"},
    "spices":               {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "pepper":               {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "cinnamon":             {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "cumin":                {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "paprika":              {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "cayenne":              {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "oregano":              {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "basil":                {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "thyme":                {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "rosemary":             {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    # ── PANTRY: Canned / Jarred ──────────────────────────────────────────────
    "canned beans":         {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "black beans":          {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "chickpeas":            {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "lentils":              {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "canned tomatoes":      {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "tomato sauce":         {"fridge": 5,    "freezer": 180,  "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "tomato paste":         {"fridge": 5,    "freezer": 180,  "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "canned corn":          {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "canned soup":          {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "chicken broth":        {"fridge": 5,    "freezer": 180,  "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "beef broth":           {"fridge": 5,    "freezer": 180,  "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "vegetable broth":      {"fridge": 5,    "freezer": 180,  "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "broth":                {"fridge": 5,    "freezer": 180,  "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    # ── CONDIMENTS / SAUCES ──────────────────────────────────────────────────
    "ketchup":              {"fridge": 180,  "freezer": None, "pantry": 365,  "default_storage": "fridge",  "category": "Food"},
    "mustard":              {"fridge": 365,  "freezer": None, "pantry": 730,  "default_storage": "fridge",  "category": "Food"},
    "mayonnaise":           {"fridge": 60,   "freezer": None, "pantry": None, "default_storage": "fridge",  "category": "Food"},
    "mayo":                 {"fridge": 60,   "freezer": None, "pantry": None, "default_storage": "fridge",  "category": "Food"},
    "ranch":                {"fridge": 60,   "freezer": None, "pantry": None, "default_storage": "fridge",  "category": "Food"},
    "hot sauce":            {"fridge": 365,  "freezer": None, "pantry": 365,  "default_storage": "pantry",  "category": "Food"},
    "soy sauce":            {"fridge": 730,  "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "worcestershire":       {"fridge": 365,  "freezer": None, "pantry": 365,  "default_storage": "pantry",  "category": "Food"},
    "bbq sauce":            {"fridge": 180,  "freezer": None, "pantry": 365,  "default_storage": "pantry",  "category": "Food"},
    "buffalo sauce":        {"fridge": 180,  "freezer": None, "pantry": None, "default_storage": "fridge",  "category": "Food"},
    "marinade":             {"fridge": 14,   "freezer": None, "pantry": 365,  "default_storage": "pantry",  "category": "Food"},
    "pasta sauce":          {"fridge": 5,    "freezer": 180,  "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "olive oil":            {"fridge": None, "freezer": None, "pantry": 365,  "default_storage": "pantry",  "category": "Food"},
    "vegetable oil":        {"fridge": None, "freezer": None, "pantry": 365,  "default_storage": "pantry",  "category": "Food"},
    "coconut oil":          {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Food"},
    "vinegar":              {"fridge": None, "freezer": None, "pantry": 3650, "default_storage": "pantry",  "category": "Food"},
    "salad dressing":       {"fridge": 90,   "freezer": None, "pantry": 365,  "default_storage": "fridge",  "category": "Food"},
    "pickle":               {"fridge": 90,   "freezer": None, "pantry": None, "default_storage": "fridge",  "category": "Food"},
    "pickles":              {"fridge": 90,   "freezer": None, "pantry": None, "default_storage": "fridge",  "category": "Food"},
    "jam":                  {"fridge": 180,  "freezer": None, "pantry": 365,  "default_storage": "fridge",  "category": "Food"},
    "jelly":                {"fridge": 180,  "freezer": None, "pantry": 365,  "default_storage": "fridge",  "category": "Food"},
    "peanut butter":        {"fridge": 180,  "freezer": None, "pantry": 180,  "default_storage": "pantry",  "category": "Food"},
    "almond butter":        {"fridge": 180,  "freezer": None, "pantry": 180,  "default_storage": "pantry",  "category": "Food"},
    "nutella":              {"fridge": None, "freezer": None, "pantry": 365,  "default_storage": "pantry",  "category": "Food"},
    # ── SNACKS ───────────────────────────────────────────────────────────────
    "chips":                {"fridge": None, "freezer": None, "pantry": 60,   "default_storage": "pantry",  "category": "Food"},
    "potato chips":         {"fridge": None, "freezer": None, "pantry": 60,   "default_storage": "pantry",  "category": "Food"},
    "tortilla chips":       {"fridge": None, "freezer": None, "pantry": 60,   "default_storage": "pantry",  "category": "Food"},
    "crackers":             {"fridge": None, "freezer": None, "pantry": 90,   "default_storage": "pantry",  "category": "Food"},
    "pretzels":             {"fridge": None, "freezer": None, "pantry": 90,   "default_storage": "pantry",  "category": "Food"},
    "popcorn":              {"fridge": None, "freezer": None, "pantry": 90,   "default_storage": "pantry",  "category": "Food"},
    "kettle corn":          {"fridge": None, "freezer": None, "pantry": 90,   "default_storage": "pantry",  "category": "Food"},
    "nuts":                 {"fridge": 180,  "freezer": 365,  "pantry": 180,  "default_storage": "pantry",  "category": "Food"},
    "almonds":              {"fridge": 365,  "freezer": 730,  "pantry": 365,  "default_storage": "pantry",  "category": "Food"},
    "cashews":              {"fridge": 180,  "freezer": 365,  "pantry": 180,  "default_storage": "pantry",  "category": "Food"},
    "walnuts":              {"fridge": 180,  "freezer": 365,  "pantry": 180,  "default_storage": "pantry",  "category": "Food"},
    "pecans":               {"fridge": 180,  "freezer": 365,  "pantry": 90,   "default_storage": "pantry",  "category": "Food"},
    "peanuts":              {"fridge": 180,  "freezer": 365,  "pantry": 180,  "default_storage": "pantry",  "category": "Food"},
    "pistachios":           {"fridge": 180,  "freezer": 365,  "pantry": 90,   "default_storage": "pantry",  "category": "Food"},
    "trail mix":            {"fridge": None, "freezer": None, "pantry": 180,  "default_storage": "pantry",  "category": "Food"},
    "granola bar":          {"fridge": None, "freezer": 365,  "pantry": 180,  "default_storage": "pantry",  "category": "Food"},
    "granola bars":         {"fridge": None, "freezer": 365,  "pantry": 180,  "default_storage": "pantry",  "category": "Food"},
    "protein bar":          {"fridge": None, "freezer": 180,  "pantry": 180,  "default_storage": "pantry",  "category": "Food"},
    "cookies":              {"fridge": None, "freezer": 90,   "pantry": 90,   "default_storage": "pantry",  "category": "Food"},
    "candy":                {"fridge": None, "freezer": None, "pantry": 365,  "default_storage": "pantry",  "category": "Food"},
    "chocolate":            {"fridge": None, "freezer": None, "pantry": 365,  "default_storage": "pantry",  "category": "Food"},
    # ── BEVERAGES ────────────────────────────────────────────────────────────
    "orange juice":         {"fridge": 7,    "freezer": 365,  "pantry": 365,  "default_storage": "fridge",  "category": "Food"},
    "apple juice":          {"fridge": 7,    "freezer": 365,  "pantry": 365,  "default_storage": "fridge",  "category": "Food"},
    "juice":                {"fridge": 7,    "freezer": 365,  "pantry": None, "default_storage": "fridge",  "category": "Food"},
    "soda":                 {"fridge": None, "freezer": None, "pantry": 270,  "default_storage": "pantry",  "category": "Food"},
    "sparkling water":      {"fridge": None, "freezer": None, "pantry": 270,  "default_storage": "pantry",  "category": "Food"},
    "water":                {"fridge": None, "freezer": None, "pantry": 3650, "default_storage": "pantry",  "category": "Food"},
    "coffee":               {"fridge": None, "freezer": 730,  "pantry": 180,  "default_storage": "pantry",  "category": "Food"},
    "tea":                  {"fridge": None, "freezer": None, "pantry": 365,  "default_storage": "pantry",  "category": "Food"},
    "wine":                 {"fridge": 3,    "freezer": None, "pantry": 3,    "default_storage": "pantry",  "category": "Food"},
    "beer":                 {"fridge": 180,  "freezer": None, "pantry": 180,  "default_storage": "pantry",  "category": "Food"},
    "vodka":                {"fridge": None, "freezer": None, "pantry": 3650, "default_storage": "pantry",  "category": "Food"},
    "whiskey":              {"fridge": None, "freezer": None, "pantry": 3650, "default_storage": "pantry",  "category": "Food"},
    "sports drink":                  {"fridge": None, "freezer": None, "pantry": 270,  "default_storage": "pantry",  "category": "Food"},
    "energy drink":                  {"fridge": None, "freezer": None, "pantry": 270,  "default_storage": "pantry",  "category": "Food"},
    "hard seltzer":                  {"fridge": None, "freezer": None, "pantry": 365,  "default_storage": "pantry",  "category": "Food"},
    "nutrl vodka seltzer":           {"fridge": None, "freezer": None, "pantry": 365,  "default_storage": "pantry",  "category": "Food"},
    "nutrl":                         {"fridge": None, "freezer": None, "pantry": 365,  "default_storage": "pantry",  "category": "Food"},
    "white claw":                    {"fridge": None, "freezer": None, "pantry": 365,  "default_storage": "pantry",  "category": "Food"},
    "truly hard seltzer":            {"fridge": None, "freezer": None, "pantry": 365,  "default_storage": "pantry",  "category": "Food"},
    "truly":                         {"fridge": None, "freezer": None, "pantry": 365,  "default_storage": "pantry",  "category": "Food"},
    "bud light seltzer":             {"fridge": None, "freezer": None, "pantry": 365,  "default_storage": "pantry",  "category": "Food"},
    "canned cocktail":               {"fridge": None, "freezer": None, "pantry": 365,  "default_storage": "pantry",  "category": "Food"},
    "hard lemonade":                 {"fridge": None, "freezer": None, "pantry": 365,  "default_storage": "pantry",  "category": "Food"},
    "brown sugar bacon":             {"fridge": 7,    "freezer": 30,   "pantry": None, "default_storage": "fridge",  "category": "Food"},
    "great value brown sugar bacon": {"fridge": 7,    "freezer": 30,   "pantry": None, "default_storage": "fridge",  "category": "Food"},
    # ── HOUSEHOLD ────────────────────────────────────────────────────────────
    "paper towels":         {"fridge": None, "freezer": None, "pantry": 3650, "default_storage": "pantry",  "category": "Household"},
    "toilet paper":         {"fridge": None, "freezer": None, "pantry": 3650, "default_storage": "pantry",  "category": "Household"},
    "napkins":              {"fridge": None, "freezer": None, "pantry": 3650, "default_storage": "pantry",  "category": "Household"},
    "plastic bags":         {"fridge": None, "freezer": None, "pantry": 3650, "default_storage": "pantry",  "category": "Household"},
    "trash bags":           {"fridge": None, "freezer": None, "pantry": 3650, "default_storage": "pantry",  "category": "Household"},
    "aluminum foil":        {"fridge": None, "freezer": None, "pantry": 3650, "default_storage": "pantry",  "category": "Household"},
    "plastic wrap":         {"fridge": None, "freezer": None, "pantry": 3650, "default_storage": "pantry",  "category": "Household"},
    "dish soap":            {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Household"},
    "laundry detergent":    {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Household"},
    "fabric softener":      {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Household"},
    "dishwasher pods":      {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Household"},
    "cleaning spray":       {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Household"},
    "bleach":               {"fridge": None, "freezer": None, "pantry": 365,  "default_storage": "pantry",  "category": "Household"},
    "shampoo":              {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Household"},
    "conditioner":          {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Household"},
    "body wash":            {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Household"},
    "toothpaste":           {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Household"},
    "deodorant":            {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Household"},
    "hand soap":            {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Household"},
    "lotion":               {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Household"},
    "sunscreen":            {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Household"},
    "tide":                 {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Household"},
    "fresh step":           {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Household"},
    "cat litter":           {"fridge": None, "freezer": None, "pantry": 730,  "default_storage": "pantry",  "category": "Household"},
    "dog food":             {"fridge": None, "freezer": None, "pantry": 365,  "default_storage": "pantry",  "category": "Household"},
    "cat food":             {"fridge": None, "freezer": None, "pantry": 365,  "default_storage": "pantry",  "category": "Household"},
    "pet food":             {"fridge": None, "freezer": None, "pantry": 365,  "default_storage": "pantry",  "category": "Household"},
}

# Keep old name for compatibility
FOOD_KNOWLEDGE_FALLBACK = {
    k: {
        "expires_in_days": v[v["default_storage"]] or 14,
        "storage": v["default_storage"],
        "category": v["category"]
    }
    for k, v in SHELF_LIFE_DB.items()
}

_ai_enrichment_cache: Dict[str, Dict] = {}

# ── Keyword overrides ────────────────────────────────────────────────────────
_KEYWORD_OVERRIDES: Dict[str, str] = {
    "frozen":     "freezer",
    "canned":     "pantry",
}

def _rules_lookup(name: str, storage: Optional[str] = None) -> Optional[Dict]:
    """
    Look up shelf life for an item, optionally for a specific storage location.
    Returns dict with expires_in_days, storage, category, is_estimated, confidence.
    storage: 'fridge' | 'freezer' | 'pantry' | None (use default)
    """
    name_lower = name.lower().strip()

    # 1. Check keyword overrides (frozen → freezer, canned → pantry)
    override_storage = None
    for kw, forced_storage in _KEYWORD_OVERRIDES.items():
        if kw in name_lower:
            override_storage = forced_storage
            break

    # 2. Find best matching entry — exact first, then longest substring
    entry = None
    confidence = "low"
    if name_lower in SHELF_LIFE_DB:
        entry = SHELF_LIFE_DB[name_lower]
        confidence = "high"
    else:
        best_key = None
        best_len = 0
        for key in SHELF_LIFE_DB:
            if key in name_lower and len(key) > best_len:
                best_key = key
                best_len = len(key)
        if best_key:
            entry = SHELF_LIFE_DB[best_key]
            confidence = "medium"

    if entry is None:
        return None

    # 3. Determine storage to use
    target_storage = storage or override_storage or entry["default_storage"]
    days = entry.get(target_storage)

    # If requested storage has no data, fall back to default
    if days is None:
        target_storage = entry["default_storage"]
        days = entry.get(target_storage) or 14

    return {
        "expires_in_days": days,
        "storage": target_storage,
        "category": entry["category"],
        "is_estimated": True,
        "confidence": confidence,
        # Include all storage options so iOS app can recalculate instantly
        "shelf_life_by_storage": {
            k: entry[k] for k in ["fridge", "freezer", "pantry"]
        }
    }

def _fallback_for_item(name: str, storage: Optional[str] = None) -> Dict:
    result = _rules_lookup(name, storage)
    if result:
        return result
    target = storage or "fridge"
    return {
        "expires_in_days": 14,
        "storage": target,
        "category": "Food",
        "is_estimated": True,
        "confidence": "low",
        "shelf_life_by_storage": {"fridge": 14, "freezer": None, "pantry": None}
    }

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

    print(f"[GEMINI] KEY present={bool(GEMINI_API_KEY)} key_preview={str(GEMINI_API_KEY)[:8] if GEMINI_API_KEY else 'MISSING'}", flush=True)
    if GEMINI_API_KEY:
        print(f"[GEMINI] Starting enrichment for {len(uncached_items)} items: {[it['name'] for it in uncached_items]}", flush=True)
        items_json = json.dumps(
            [{"name": it["name"], "category": it["category"]} for it in uncached_items],
            indent=2
        )
        prompt = f"""You are the world's leading expert in food science, grocery retail, and USDA/FDA food safety guidelines. Your job is to analyze grocery items and return precise shelf life data. You have encyclopedic knowledge of every food product, brand, and prepared item sold in American grocery stores.

For each item below, return a JSON array in the SAME ORDER as the input.

For each item return EXACTLY these fields:
- full_name: The complete, properly capitalized brand name and product name as it would appear on a store shelf. Expand ALL receipt abbreviations. You must know every grocery store's abbreviation style. Use the examples and store brand key below.

STORE BRAND CODES (critical — these appear at the start of receipt line items):
  PUBLIX: PBX/PUB/PBLX = Publix store brand
  WALMART: GV/GW/GRTVAL = Great Value (Walmart store brand), SC/SAMC = Sam's Choice (Walmart premium), EQ/EQT = Equate (Walmart health), MKTS/MKTSD = Marketside (Walmart fresh deli)
  TARGET: GG/G&G = Good & Gather (Target store brand), MP/MKT = Market Pantry (Target), UP&UP = Up & Up (Target health)
  COSTCO: KS/KSL/KIRK/KIRKL/KRKL = Kirkland Signature (Costco store brand)
  KROGER: ST/STR = Simple Truth (Kroger organic brand), PS/PRSEL = Private Selection (Kroger premium), KR/KRG = Kroger store brand
  ALDI: SN/SIMNAT = Simply Nature (ALDI organic), SS/SPSLCT = Specially Selected (ALDI premium), EG = Earth Grown (ALDI plant-based), FA = Fit & Active (ALDI health), BHC/BB = Bake House Creations (ALDI bakery), CC/CH = Countryside Creamery (ALDI dairy)
  WHOLE FOODS: WFM = Whole Foods Market, 365 = 365 by Whole Foods Market
  TRADER JOE'S: TJS/TJ = Trader Joe's
  FRESH MARKET: TFM/FM = The Fresh Market

UNIVERSAL RECEIPT ABBREVIATIONS (used by all stores):
  Meat: BNLS=Boneless, SKNLS=Skinless, BRST=Breast, CHKN/CKN/CHK=Chicken, TRKY=Turkey, GRND/GRD=Ground, THGH=Thigh, SMKD=Smoked, RSTD=Roasted, RST=Roast, GRL=Grilled, SLCD=Sliced, SHRDD=Shredded
  Dairy: CRM=Cream, BTTR/BTR=Butter, YOG=Yogurt, GRK=Greek, MLK=Milk, CHS/CHZ=Cheese, PARM=Parmesan, MOZ=Mozzarella
  Produce: ORG=Organic, FRSH=Fresh, FRZN=Frozen, AVO/AVDO=Avocado, STRBRY=Strawberry, BLBRY=Blueberry, ORNG=Orange, CIN/CINN=Cinnamon
  Modifiers: LF=Low Fat, RF=Reduced Fat, FF=Fat Free, NF=Nonfat, UNSLT=Unsalted, SLT=Salted, ORIG=Original, NAT=Natural, PREM=Premium, ASST=Assorted, SPCY=Spicy, HNY=Honey, DK=Dark, VAN=Vanilla, CHOC=Chocolate, WHL=Whole, WHT=Wheat, MLT=Multi, BRD=Bread, SNK=Snack, SDWCH=Sandwich
  Nuts: ALND/ALM=Almond, CSH=Cashew, WLT=Walnut, PST=Pistachio, MXD=Mixed
  Diet: GF=Gluten Free, VGN=Vegan, KSH=Kosher

EXAMPLES:
  "BNLS CHICK BRST" -> "Boneless Chicken Breast"
  "GV GRND BEEF 80/20" -> "Great Value Ground Beef 80/20"
  "GW BRWN SGR BAC" -> "Great Value Brown Sugar Bacon"
  "KS ORG CHKN BRST" -> "Kirkland Signature Organic Chicken Breast"
  "KS KIRKL FRZN SALM" -> "Kirkland Signature Frozen Salmon"
  "ST ORG WHOLE MILK" -> "Simple Truth Organic Whole Milk"
  "GG VAN GREEK YOG" -> "Good & Gather Vanilla Greek Yogurt"
  "GG ORG CHKN BRST" -> "Good & Gather Organic Chicken Breast"
  "SN ORG BABY SPIN" -> "Simply Nature Organic Baby Spinach"
  "BUIT 5 CHSE TORTEL" -> "Buitoni 5 Cheese Tortellini"
  "SBR BUFF WG MARIND" -> "Sweet Baby Ray's Buffalo Wing Marinade"
  "KH PRETZEL BUNS" -> "King's Hawaiian Pretzel Buns"
  "PBX PARMESAN WEDGE" -> "Publix Parmesan Wedge"
  "FRSH STP XTRM ODOR" -> "Fresh Step Extreme Odor Control Cat Litter"
  "KH SAVORY DIN ROLL" -> "King's Hawaiian Savory Dinner Rolls"
  "PUB MILK RF GRN PT 2%" -> "Publix 2% Milk Green Cap"
  "CHOC DPD STRAW" -> "Chocolate Dipped Strawberries"
  "DELI ROAST CHKN" -> "Deli Rotisserie Chicken"
  "TJS ORG FRZN STRBRY" -> "Trader Joe's Organic Frozen Strawberries"
  "365 ORG WHOLE MLK" -> "365 by Whole Foods Market Organic Whole Milk"
  "EQ VITMN D3" -> "Equate Vitamin D3"
  "PS SLCD SMKD SALM" -> "Private Selection Sliced Smoked Salmon"
  "MKTS GRND BEEF 85/15" -> "Marketside Ground Beef 85/15"
If the name is already clean, return it as-is. CRITICAL: Each input row is exactly ONE product. Never combine two products. Never mix food and non-food words in a single name. "rf" on Publix milk = reduced fat. "grn" on milk = green cap (2% fat). "pt" = pint or half-gallon.
- expires_in_days: integer — days from purchase until expiration using the recommended storage method
- storage: "fridge", "freezer", or "pantry" — the best storage method for this item as purchased
- fridge: integer days in fridge, or null if fridge is unsafe/not applicable
- freezer: integer days in freezer, or null if freezer is unsafe/not applicable  
- pantry: integer days in pantry, or null if pantry is unsafe/not applicable
- category: "Food" or "Household"
- food_category: the specific display subcategory for this item. Must be EXACTLY one of these values:
  Food items: "Produce", "Meat & Seafood", "Dairy & Eggs", "Bread & Bakery", "Deli", "Beverages", "Snacks & Candy", "Cereal & Breakfast", "Other"
  Non-food items: "Household"
  RULES: "Produce" = ALL fresh fruits and vegetables — strawberries, raspberries, blueberries, apples, bananas, oranges, spinach, broccoli, carrots, avocado, lettuce, tomatoes, etc. "Meat & Seafood" = raw or cooked meat, poultry, seafood. "Dairy & Eggs" = milk, cheese, yogurt, butter, eggs, cream. "Bread & Bakery" = bread, tortillas, rolls, bagels, pastries, cakes, pies, cookies. "Deli" = deli meats, prepared hot bar items, rotisserie chicken, deli sides. "Beverages" = drinkable liquids: juice, soda (the drink), water, beer, wine, spirits, energy drinks, coffee, tea. CRITICAL: "baking soda" is NOT a beverage — it is "Other". "Snacks & Candy" = chips, crackers, nuts, candy, granola bars, popcorn, pretzels. "Cereal & Breakfast" = cereal, oatmeal, pancake mix, breakfast bars, granola. "Other" = pantry staples (baking soda, flour, sugar, oil, vinegar, spices), canned goods, condiments, sauces, frozen meals, supplements, baby food, international foods, anything that doesn't clearly fit above. "Household" = ONLY for non-food, non-drinkable items (cleaning supplies, paper goods, personal care, diapers, pet supplies, batteries).
  TRICKY CASES: "Arm & Hammer Baking Soda" = Other (pantry, not a drink). "Organic Strawberries" = Produce. "Organic Raspberries" = Produce. "Pull-Ups Training Pants" = Household. "Signature Diapers" = Household. "Liv Sugar Free Soda" = Beverages (it IS a drink).
- photo_query: the single best Unsplash search query to find a professional product photo. This must be a 2-5 word phrase describing the PHYSICAL OBJECT as it would appear in a grocery store. RULES:
  - Use the generic food name, NOT the brand name ("sliced white bread" not "Wonder Bread")
  - Be specific about the FORM of the product ("sour cream container" not "sour cream", "pasta shells box" not "pasta", "chicken breast package" not "chicken")
  - Always describe the CONTAINER or PHYSICAL FORM: box, bag, container, carton, bottle, jar, can, package, bunch, roll
  - For household items: describe the product object ("laundry detergent bottle", "paper towel roll", "deodorant stick")
  - For produce: describe fresh raw form ("fresh strawberries", "bunch of bananas", "raw broccoli florets")
  - NEVER describe a plated dish, cooked meal, or bowl of food — always the uncooked product or its package
  - EXAMPLES: "sour cream container dairy", "pasta shells box grocery", "raw chicken breast package", "shredded cheddar cheese bag", "greek yogurt cup", "baby spinach bag", "orange juice carton", "fresh strawberries", "laundry detergent bottle", "diapers package", "paper towel roll", "baking soda box"

════════════════════════════════════════
CATEGORY RULES
════════════════════════════════════════
HOUSEHOLD — not edible, not drinkable:
  Paper: Bounty, Charmin, Scott, Kleenex, paper towels, toilet paper, napkins, tissues, paper plates
  Cleaning: Tide, Gain, Downy, Bounce, Lysol, Febreze, Dawn, Cascade, Windex, Clorox, Swiffer, Fabuloso, Mr. Clean, bleach, dish soap, laundry detergent, dryer sheets, fabric softener, sponges, scrub brushes, Pine-Sol, Comet, Bona, Seventh Generation cleaner
  Personal care: Cremo, Old Spice, Dove, Pantene, Head & Shoulders, Axe, Degree, Secret, Gillette, shampoo, conditioner, body wash, deodorant, toothpaste, toothbrush, floss, mouthwash, razors, shaving cream, lotion, soap bars, face wash, moisturizer, sunscreen, nail polish, makeup
  Pet: Fresh Step, Tidy Cats, Arm & Hammer litter, cat litter, dog food, cat food, pet treats, flea treatment, puppy pads
  Misc: trash bags, garbage bags, aluminum foil, plastic wrap, Ziploc bags, batteries, light bulbs, matches, cotton balls, cotton swabs, first aid items, bandages
FOOD — anything consumed by eating or drinking:
  All meats, seafood, dairy, eggs, produce, bread, bakery, cereal, snacks, candy, beverages, condiments, sauces, frozen meals, canned goods, deli items, prepared foods, supplements, protein powder, baby formula, alcohol of any kind
  When in doubt: if a human eats or drinks it, it is Food

════════════════════════════════════════
SHELF LIFE REFERENCE — USDA/FDA STANDARDS
Always use the MOST SPECIFIC match. Read every word of the item name.
════════════════════════════════════════

── COMPOSITE / PREPARED ITEMS (CRITICAL RULE) ──
When an item is made FROM perishable ingredients, the shelf life is governed by the MOST PERISHABLE component — not the coating or flavoring.
Examples:
- Chocolate dipped strawberries = 2d fridge (strawberries spoil in 2-3d, chocolate coating irrelevant)
- Strawberry cheesecake = 4d fridge (cream cheese + fresh fruit)
- Fruit tart, fruit flan = 2d fridge (fresh fruit on custard)
- Chocolate covered cherries (fresh) = 3d fridge
- Dipped fruit of any kind = 2d fridge
- Cream-filled pastry, eclair, cream puff = 2d fridge
- Tiramisu = 3d fridge
- Cheesecake (plain) = 5d fridge
- Key lime pie, lemon meringue pie = 4d fridge
- Pumpkin pie = 4d fridge
- Pecan pie, apple pie (store bought) = 4d pantry
- Birthday cake, layer cake with frosting = 4d pantry or 7d fridge
- Brownies, fudge = 5d pantry
- Deli pasta salad, potato salad, macaroni salad = 4d fridge
- Deli coleslaw = 4d fridge
- Deli chicken salad, tuna salad, egg salad = 3d fridge
- Hummus (opened) = 7d fridge
- Guacamole (fresh, opened) = 2d fridge
- Salsa (fresh/refrigerated) = 7d fridge
- Queso dip (refrigerated) = 10d fridge
- Spinach artichoke dip = 5d fridge
- Buffalo chicken dip = 4d fridge
- Bean dip = 5d fridge

── DELI / HOT BAR / PREPARED FOODS ──
- Rotisserie chicken, deli roast chicken = 4d fridge
- Deli fried chicken = 3d fridge
- Deli meatloaf, meatballs = 4d fridge
- Deli macaroni & cheese = 4d fridge
- Deli mashed potatoes, deli sides = 4d fridge
- Deli soup (refrigerated) = 4d fridge
- Deli pizza slice = 4d fridge
- Deli sushi = 1d fridge (same day)
- Prepared sandwiches (deli) = 2d fridge
- Deli stuffed peppers, stuffed chicken = 4d fridge

── CHEESE ──
- Fresh mozzarella balls, burrata, fresh ricotta = 5d fridge
- Brie, camembert, triple cream, soft-ripened = 7d fridge
- Cream cheese block or spread = 10d fridge
- Mascarpone = 7d fridge
- Whipped cream cheese = 10d fridge
- Shredded mozzarella, shredded pizza blend = 14d fridge
- Shredded cheddar, shredded Mexican blend = 14d fridge
- Sliced American, Velveeta slices = 21d fridge
- Sliced provolone, muenster, havarti = 14d fridge
- Sliced swiss, sliced pepper jack = 14d fridge
- Cheddar block (mild, medium, sharp) = 30d fridge
- Colby jack block, Monterey jack block = 30d fridge
- Gouda (semi-aged block) = 30d fridge
- Feta (crumbled or block in brine) = 30d fridge
- Blue cheese, gorgonzola = 21d fridge
- Parmesan shredded or grated (refrigerated) = 60d fridge
- Parmesan wedge or block, Pecorino Romano wedge = 90d fridge
- Aged gouda, aged cheddar (2yr+) = 90d fridge
- String cheese, cheese sticks = 21d fridge
- Cottage cheese = 14d fridge
- Velveeta block = 60d pantry (unopened), 8 weeks fridge (opened)

── MEAT & POULTRY (RAW) ──
- Raw chicken breast, thighs, drumsticks, wings = 2d fridge, 270d freezer
- Raw whole chicken, whole turkey = 2d fridge, 365d freezer
- Raw ground beef, ground turkey, ground pork, ground chicken = 2d fridge, 120d freezer
- Raw beef steak (ribeye, NY strip, sirloin, filet) = 3d fridge, 270d freezer
- Raw pork chops, pork tenderloin = 3d fridge, 180d freezer
- Raw lamb chops, veal = 3d fridge, 270d freezer
- Raw ribs (beef or pork) = 3d fridge, 180d freezer
- Raw beef roast, chuck roast, brisket = 4d fridge, 365d freezer
- Raw liver, organ meats = 2d fridge, 90d freezer

── MEAT & POULTRY (PROCESSED/COOKED) ──
- Bacon (raw, unopened) = 7d fridge, 30d freezer
- Bacon (opened package) = 7d fridge
- Pancetta = 7d fridge, 30d freezer
- Prosciutto, serrano ham (sliced) = 5d fridge, 60d freezer
- Salami, pepperoni (sliced deli) = 5d fridge, 60d freezer
- Deli turkey, deli ham, deli chicken (sliced) = 5d fridge, 60d freezer
- Deli roast beef = 5d fridge
- Bologna, liverwurst = 5d fridge
- Hot dogs (opened) = 7d fridge, 60d freezer
- Hot dogs (unopened) = 14d fridge, 60d freezer
- Smoked sausage, kielbasa, andouille (cooked) = 14d fridge, 60d freezer
- Bratwurst (raw) = 3d fridge, 60d freezer
- Italian sausage (raw links) = 3d fridge, 60d freezer
- Cooked chicken, cooked turkey = 4d fridge, 120d freezer
- Cooked beef, cooked pork = 4d fridge, 90d freezer
- Rotisserie chicken = 4d fridge, 90d freezer
- Fully cooked bacon, precooked bacon = 14d fridge
- Pulled pork (cooked, vacuum sealed) = 14d fridge, 90d freezer
- Corned beef (cooked) = 4d fridge
- Pastrami (deli sliced) = 5d fridge

── SEAFOOD ──
- Raw salmon, tuna, halibut, cod, tilapia, mahi-mahi = 2d fridge, 180d freezer
- Raw shrimp, scallops, clams, mussels = 2d fridge, 180d freezer
- Raw lobster, crab (live or fresh) = 1d fridge
- Canned tuna, canned salmon, canned sardines = 1095d pantry
- Canned crab, canned clams = 1095d pantry
- Smoked salmon (vacuum sealed, refrigerated) = 14d fridge, 60d freezer
- Smoked oysters (canned) = 1095d pantry
- Imitation crab meat = 5d fridge, 90d freezer
- Cooked shrimp = 3d fridge, 90d freezer
- Fish sticks, breaded fish (frozen) = 180d freezer
- Lox (opened) = 5d fridge

── DAIRY ──
- Whole milk, 2% milk, skim milk, 1% milk = 7d fridge
- Chocolate milk = 7d fridge
- Buttermilk = 14d fridge
- Oat milk, almond milk, soy milk, coconut milk (opened carton) = 7d fridge
- Oat milk, almond milk, soy milk (unopened shelf-stable) = 365d pantry
- Heavy cream, heavy whipping cream = 10d fridge
- Half and half, half & half = 10d fridge
- Whipping cream (regular) = 10d fridge
- Coffee creamer (liquid, refrigerated) = 14d fridge
- Coffee creamer (powdered) = 180d pantry
- Sour cream = 14d fridge
- Crème fraîche = 14d fridge
- Greek yogurt (plain or flavored) = 14d fridge
- Regular yogurt, skyr = 14d fridge
- Drinkable yogurt, Kefir = 14d fridge
- Butter (salted block) = 30d fridge, 365d freezer
- Butter (unsalted block) = 14d fridge, 365d freezer
- Whipped butter = 14d fridge
- Ghee (clarified butter) = 90d pantry
- Eggs (large, white, brown — any) = 35d fridge
- Hard boiled eggs = 7d fridge
- Egg whites (carton) = 7d fridge
- Eggnog = 5d fridge
- Whipped cream (can, Cool Whip tub) = 14d fridge
- Condensed milk (unopened can) = 730d pantry
- Evaporated milk (unopened can) = 730d pantry

── PRODUCE — FRUIT ──
- Strawberries = 3d fridge
- Raspberries, blackberries = 2d fridge
- Blueberries = 7d fridge
- Grapes (red, green, cotton candy) = 5d fridge
- Cherries = 5d fridge
- Ripe bananas = 3d pantry
- Unripe bananas = 7d pantry
- Apples (bagged or loose) = 21d fridge
- Pears (ripe) = 5d fridge
- Peaches, nectarines, plums (ripe) = 3d fridge
- Peaches, nectarines, plums (unripe) = 5d pantry
- Mango (ripe) = 3d fridge
- Mango (unripe) = 5d pantry
- Papaya (ripe) = 3d fridge
- Kiwi (ripe) = 5d fridge
- Oranges, mandarins, clementines = 14d fridge
- Grapefruits = 14d fridge
- Lemons, limes = 21d fridge
- Avocado (ripe, soft) = 2d pantry or 3d fridge
- Avocado (firm, unripe) = 5d pantry
- Cantaloupe, honeydew (whole) = 7d pantry
- Cantaloupe, honeydew (cut) = 4d fridge
- Watermelon (whole) = 10d pantry
- Watermelon (cut) = 4d fridge
- Pineapple (whole) = 5d pantry
- Pineapple (cut) = 4d fridge
- Pomegranate (whole) = 14d fridge
- Dates, dried figs = 365d pantry
- Raisins, dried cranberries, dried apricots = 365d pantry

── PRODUCE — VEGETABLES ──
- Spinach, baby spinach, arugula = 5d fridge
- Spring mix, mesclun, mixed greens = 5d fridge
- Romaine hearts, romaine head = 7d fridge
- Iceberg lettuce (head) = 14d fridge
- Butter lettuce, Bibb lettuce = 5d fridge
- Kale, Swiss chard, collard greens = 7d fridge
- Broccoli (head or florets) = 5d fridge
- Cauliflower = 7d fridge
- Brussels sprouts = 5d fridge
- Broccolini, broccoflower = 5d fridge
- Mushrooms (white button, cremini, portobello) = 5d fridge
- Asparagus = 3d fridge
- Green beans, haricots verts, snap peas = 5d fridge
- Edamame (fresh) = 3d fridge
- Zucchini, yellow squash = 5d fridge
- Bell peppers (whole) = 7d fridge
- Jalapeños, serranos, poblanos = 7d fridge
- Cucumber = 7d fridge
- Corn on the cob = 2d fridge
- Tomatoes (ripe, whole) = 5d pantry
- Cherry tomatoes, grape tomatoes = 7d pantry
- Carrots (whole or baby, bagged) = 21d fridge
- Celery = 14d fridge
- Radishes = 14d fridge
- Fennel bulb = 7d fridge
- Artichokes = 5d fridge
- Leeks = 7d fridge
- Green onions, scallions = 5d fridge
- Onions (yellow, white, red) = 30d pantry
- Shallots = 30d pantry
- Garlic (whole head) = 90d pantry
- Garlic (minced, refrigerated jar) = 90d fridge
- Potatoes (russet, Yukon gold, red) = 30d pantry
- Sweet potatoes, yams = 21d pantry
- Butternut squash, acorn squash (whole) = 60d pantry
- Cabbage (whole head) = 14d fridge
- Brussels sprouts (on stalk) = 5d fridge
- Bok choy = 5d fridge
- Bean sprouts = 3d fridge
- Pre-cut vegetable medley = 4d fridge
- Bagged salad kit (Caesar, etc.) = 5d fridge

── FRESH HERBS ──
- Basil (fresh bunch) = 3d pantry (room temp in water)
- Cilantro, parsley, dill, mint (fresh) = 5d fridge
- Rosemary, thyme, sage (fresh) = 7d fridge
- Chives = 5d fridge

── BREAD & BAKERY ──
- Sliced sandwich bread, white or wheat = 5d pantry, 90d freezer
- Whole grain bread, multigrain = 5d pantry
- Sourdough loaf = 5d pantry
- Bagels = 5d pantry, 90d freezer
- English muffins = 7d pantry
- Flour tortillas = 7d pantry, 90d freezer
- Corn tortillas = 7d pantry
- Dinner rolls, Hawaiian rolls, potato rolls = 5d pantry
- Hamburger buns, hot dog buns = 5d pantry
- Pita bread = 5d pantry
- Naan bread = 5d pantry
- Croissants (plain, plain butter) = 2d pantry
- Donuts, glazed donuts = 2d pantry
- Muffins (bakery fresh) = 3d pantry
- Cinnamon rolls (bakery fresh) = 2d pantry
- Coffee cake = 3d pantry
- Pound cake, loaf cake = 4d pantry
- Banana bread = 4d pantry
- Scones = 2d pantry
- Biscotti = 14d pantry
- Breadsticks = 5d pantry
- Focaccia = 3d pantry

── PREPARED/DELI BAKERY ITEMS WITH PERISHABLE FILLINGS ──
CRITICAL: These contain perishable dairy or fresh fruit — their shelf life is SHORT:
- Cream puffs, eclairs, profiteroles = 2d fridge
- Cannoli (filled) = 2d fridge
- Tiramisu = 3d fridge
- Cheesecake (plain, NY style) = 5d fridge
- Cheesecake (with fresh fruit topping) = 3d fridge
- Chocolate dipped strawberries = 2d fridge
- Chocolate covered strawberries = 2d fridge
- Chocolate dipped fruit (any fresh fruit) = 2d fridge
- Fruit tart, fruit flan, fresh fruit pastry = 2d fridge
- Strawberry shortcake = 2d fridge
- Tres leches cake = 4d fridge
- Mousse cake, mirror glaze cake = 4d fridge
- Custard tart, egg tart = 3d fridge
- Boston cream pie = 3d fridge
- Layer cake with whipped cream frosting = 3d fridge
- Layer cake with buttercream frosting = 5d pantry or 7d fridge
- Cupcakes (buttercream) = 2d pantry
- Cupcakes (whipped cream) = 2d fridge
- Key lime pie = 4d fridge
- Lemon meringue pie = 2d fridge
- Pumpkin pie = 4d fridge
- Pecan pie = 4d pantry
- Apple pie, cherry pie (baked) = 2d pantry or 4d fridge
- Blueberry pie = 2d pantry

── FROZEN FOODS ──
- Frozen chicken breasts, thighs, wings = 270d freezer
- Frozen ground beef, ground turkey = 120d freezer
- Frozen beef patties, burgers = 120d freezer
- Frozen fish fillets, fish sticks = 180d freezer
- Frozen shrimp, frozen scallops = 180d freezer
- Frozen pizza (any brand) = 180d freezer
- Frozen vegetables (peas, corn, broccoli, etc.) = 365d freezer
- Frozen fruit (strawberries, blueberries, mango, etc.) = 365d freezer
- Ice cream, gelato, sherbet = 180d freezer
- Frozen yogurt = 120d freezer
- Popsicles, ice pops = 365d freezer
- Frozen meals (Lean Cuisine, Healthy Choice, Marie Callender's, Stouffer's) = 180d freezer
- Frozen burritos, frozen hot pockets = 180d freezer
- Frozen waffles, pancakes, French toast sticks = 180d freezer
- Frozen meatballs = 180d freezer
- Frozen edamame = 365d freezer
- Frozen dinner rolls = 180d freezer
- Frozen pie shells, frozen pastry dough = 180d freezer
- Frozen stir fry kits = 365d freezer

── CONDIMENTS & SAUCES ──
- Ketchup (opened) = 180d fridge
- Mustard (opened) = 180d fridge
- Mayonnaise (opened, refrigerated) = 60d fridge
- Miracle Whip (opened) = 60d fridge
- Ranch dressing (opened, refrigerated) = 60d fridge
- Italian dressing (opened) = 90d fridge
- Caesar dressing (opened, refrigerated) = 60d fridge
- Balsamic vinegar, red wine vinegar = 1095d pantry
- Soy sauce, tamari = 730d pantry
- Hot sauce (Frank's, Tabasco, Cholula) = 730d pantry
- BBQ sauce (opened) = 180d fridge
- Teriyaki sauce (opened) = 180d fridge
- Sriracha (opened) = 365d pantry
- Salsa (jarred, opened) = 30d fridge
- Salsa (fresh/refrigerated, opened) = 7d fridge
- Guacamole (fresh, refrigerated, opened) = 2d fridge
- Hummus (opened) = 7d fridge
- Pesto (refrigerated, opened) = 7d fridge
- Pesto (jarred, shelf stable) = 14d fridge after opening
- Tomato sauce, marinara (jarred, opened) = 7d fridge
- Alfredo sauce (jarred, opened) = 5d fridge
- Olive tapenade = 14d fridge
- Dijon mustard = 365d fridge
- Honey mustard = 365d fridge
- Worcestershire sauce = 365d pantry
- Fish sauce = 730d pantry
- Oyster sauce = 365d fridge
- Hoisin sauce = 365d fridge
- Pickle relish (opened) = 365d fridge
- Capers (opened) = 365d fridge
- Olives (opened jar) = 90d fridge
- Pickles (opened jar) = 90d fridge
- Jam, jelly, preserves (opened) = 180d fridge
- Maple syrup (opened) = 365d fridge
- Agave, molasses = 365d pantry

── PANTRY / DRY GOODS ──
- Dry pasta (spaghetti, penne, rotini, etc.) = 730d pantry
- Rice (white, jasmine, basmati) = 1825d pantry
- Brown rice = 180d pantry
- Quinoa, farro, barley, bulgur = 730d pantry
- Rolled oats, steel cut oats, instant oats = 730d pantry
- Canned beans (black, kidney, chickpeas, etc.) = 730d pantry
- Canned corn, canned green beans, canned peas = 730d pantry
- Canned tomatoes, tomato paste = 730d pantry
- Canned soup, canned broth, canned chili = 730d pantry
- Canned tuna, canned salmon = 1095d pantry
- Canned sardines, canned anchovies = 1095d pantry
- Canned coconut milk = 730d pantry
- Chips (potato, tortilla, kettle) = 60d pantry
- Pretzels = 60d pantry
- Popcorn (microwave, bagged) = 90d pantry
- Crackers (Ritz, Wheat Thins, Triscuit) = 90d pantry
- Cookies (Oreos, Chips Ahoy, etc.) = 90d pantry
- Granola bars, protein bars = 180d pantry
- Trail mix, mixed nuts = 90d pantry
- Cereal (any kind) = 180d pantry
- Pancake mix, waffle mix = 365d pantry
- Bread crumbs, panko = 180d pantry
- Peanut butter, almond butter, sunflower butter = 180d pantry
- Nutella, hazelnut spread = 180d pantry
- Olive oil, avocado oil = 365d pantry
- Vegetable oil, canola oil, coconut oil = 365d pantry
- Sesame oil = 180d pantry
- All-purpose flour, bread flour = 365d pantry
- Almond flour, coconut flour = 180d pantry
- White sugar, brown sugar = 3650d pantry
- Powdered sugar, confectioners sugar = 3650d pantry
- Salt (table, kosher, sea salt) = 3650d pantry
- Honey (pure) = 3650d pantry
- Maple syrup (unopened) = 1095d pantry
- Coffee (whole bean, sealed) = 180d pantry
- Coffee (ground, sealed) = 90d pantry
- Instant coffee = 730d pantry
- Tea bags, loose leaf tea = 730d pantry
- Baking soda = 180d pantry
- Baking powder = 180d pantry
- Cornstarch = 730d pantry
- Vanilla extract = 1825d pantry
- Cocoa powder = 730d pantry
- Chocolate chips = 730d pantry
- Chocolate bar (dark, milk, white) = 365d pantry
- Candy (hard candy, gummies, M&Ms, Skittles) = 365d pantry
- Dried pasta sauce mix = 730d pantry
- Ramen noodles, instant noodles = 365d pantry
- Spices and dried herbs (any) = 730d pantry
- Yeast (active dry, instant) = 365d pantry
- Gelatin, pudding mix = 730d pantry
- Protein powder (sealed tub) = 365d pantry
- Vitamins, supplements = 730d pantry

── BEVERAGES ──
- Orange juice (refrigerated carton, opened) = 7d fridge
- Apple juice, grape juice (opened) = 7d fridge
- Cold brew coffee (bottled, opened) = 7d fridge
- Lemonade (refrigerated, opened) = 7d fridge
- Sports drinks (Gatorade, Powerade, opened) = 5d fridge
- Kombucha (opened) = 7d fridge
- Coconut water (opened) = 5d fridge
- Energy drinks (unopened cans) = 270d pantry
- Soda (Coke, Pepsi, Sprite — unopened cans/bottles) = 270d pantry
- Sparkling water, LaCroix (unopened) = 365d pantry
- Bottled water = 730d pantry
- Beer (craft or domestic, unopened) = 180d pantry
- Wine (red or white, unopened bottle) = 730d pantry
- Wine (opened bottle) = 5d pantry
- Champagne, prosecco (opened) = 3d fridge
- Hard seltzer (White Claw, Truly, Nutrl, Bud Light Seltzer — unopened) = 365d pantry, fridge=null, freezer=null
- Hard cider (unopened) = 365d pantry
- Canned cocktails, wine coolers, hard lemonade = 365d pantry, fridge=null, freezer=null
- Malt beverages (Twisted Tea, Smirnoff Ice) = 365d pantry
- Liquor (vodka, whiskey, bourbon, rum, tequila, gin, brandy) = 3650d pantry, fridge=null, freezer=null
- Cream liqueurs (Baileys) = 365d pantry
- Vermouth (opened) = 60d fridge

── INTERNATIONAL / SPECIALTY ──
- Kimchi (opened) = 90d fridge
- Miso paste = 180d fridge
- Tahini (opened) = 90d pantry
- Coconut aminos = 365d pantry
- Rice vinegar = 1095d pantry
- Mirin = 365d pantry
- Rice wine = 365d pantry
- Gochujang, doenjang = 365d fridge
- Harissa = 30d fridge (opened)
- Tzatziki (opened) = 5d fridge
- Babaganoush = 5d fridge
- Naan (fresh, bakery) = 3d pantry
- Injera = 3d pantry
- Wonton wrappers, gyoza wrappers = 7d fridge
- Fresh udon, soba noodles = 5d fridge
- Tofu (firm, silken — opened) = 5d fridge
- Tempeh = 7d fridge
- Seitan = 7d fridge

── BABY & INFANT ──
- Baby formula (powder, sealed) = 365d pantry
- Baby food pouches (sealed) = 730d pantry
- Baby food (opened jar or pouch) = 2d fridge
- Breast milk (refrigerated) = 4d fridge

── HOUSEHOLD (non-food expiry) ──
- Cleaning products, laundry detergent, dish soap = 730d pantry
- Paper towels, toilet paper, napkins, tissues = 3650d pantry
- Personal care (shampoo, conditioner, body wash) = 730d pantry
- Toothpaste, mouthwash, deodorant = 730d pantry
- Cat litter, dog food/treats, pet supplies = 730d pantry
- Trash bags, plastic wrap, aluminum foil, Ziploc = 3650d pantry
- Batteries, light bulbs = 3650d pantry
- First aid, bandages, antiseptic = 730d pantry

════════════════════════════════════════
ABSOLUTE RULES — NEVER VIOLATE THESE
════════════════════════════════════════
1. COMPOSITE RULE: If an item contains fresh fruit, fresh cream, whipped cream, or custard as a component, its shelf life is governed by the most perishable ingredient. "Chocolate dipped strawberries" = 2d fridge because strawberries spoil in 2-3d. Chocolate coating does NOT extend strawberry shelf life.
2. NEVER assign pantry days to raw meat, raw poultry, raw seafood, deli meat, fresh dairy, or any item that requires refrigeration.
3. NEVER assign more than 7 days fridge to raw chicken, raw fish, raw ground meat, or raw shellfish.
4. NEVER exceed 365 days for any fresh, refrigerated, or perishable food.
5. Liquor (spirits), hard seltzer, hard cider, and canned cocktails: set fridge=null and freezer=null always.
6. Pantry-only non-perishables (cat litter, laundry detergent, dish soap): set fridge=null and freezer=null.
7. ALWAYS return all three storage values (fridge, freezer, pantry). Use null for any that are unsafe or inapplicable.
8. Each item in the input is ONE product. Never merge two products. Never split one product into two.
9. If the item name is ambiguous, default to the interpretation that results in the SHORTER, SAFER shelf life.
10. Fresh bakery items from the store deli (cakes, pies, pastries) have SHORT shelf lives. Do not apply pantry shelf life to cream-filled or fresh-fruit items.

Items:
{items_json}

Respond with ONLY a valid JSON array, no markdown."""

        for model_name in ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"):
            try:
                print(f"[GEMINI] Trying model: {model_name}", flush=True)
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
                    # Use Gemini's full_name if it returned one and it looks valid
                    gemini_full_name = (result.get("full_name") or "").strip()
                    # Remove consecutive duplicate words Gemini sometimes adds
                    # e.g. "Snyder's of Hanover Snap Pretzels Pretzels" -> remove second "Pretzels"
                    gemini_full_name = re.sub(
                        r'\b(\w+)\s+\1\b', r'\1', gemini_full_name, flags=re.IGNORECASE
                    ).strip()
                    # Only use Gemini's name if it is at least as long as what we already have
                    # This prevents Gemini from dropping words (e.g. "Cat Litter" -> "Cat")
                    existing_words = len(item["name"].split())
                    gemini_words = len(gemini_full_name.split()) if gemini_full_name else 0
                    if gemini_full_name and len(gemini_full_name) >= 3 and gemini_words >= existing_words:
                        item["name"] = gemini_full_name
                    final_name_for_classify = item["name"]
                    # Use Gemini's category answer — it's trained on all of this context
                    # Fall back to keyword classifier only if Gemini returned something invalid
                    gemini_category = (result.get("category") or "").strip()
                    our_category = gemini_category if gemini_category in ("Food", "Household") else _classify(final_name_for_classify)

                    # New: food_category (display subcategory) and photo_query from Gemini
                    valid_food_categories = {
                        "Produce", "Meat & Seafood", "Dairy & Eggs", "Bread & Bakery",
                        "Deli", "Beverages", "Snacks & Candy", "Cereal & Breakfast",
                        "Other", "Household"
                    }
                    gemini_food_category = (result.get("food_category") or "").strip()
                    our_food_category = gemini_food_category if gemini_food_category in valid_food_categories else None
                    gemini_photo_query = (result.get("photo_query") or "").strip()
                    # Gemini is the authority on shelf life — it has the full reference table
                    # and knows the specific product/brand. Our rules DB is only a safety fallback.
                    gemini_days_raw = result.get("expires_in_days")
                    gemini_storage_raw = result.get("storage", "fridge")
                    gemini_storage = gemini_storage_raw if gemini_storage_raw in valid_storages else "fridge"

                    # Sanity-check Gemini's answer against hard safety limits
                    name_l = final_name_for_classify.lower()
                    is_raw_meat = any(w in name_l for w in ["raw","chicken","beef","pork","turkey","fish","shrimp","salmon","tilapia","cod","halibut","ground meat","ground beef","ground turkey","ground pork","ground chicken","steak","lamb","scallop","crab","lobster"])
                    is_meat = any(w in name_l for w in ["chicken","beef","pork","turkey","fish","shrimp","salmon","steak","ground","bacon","deli","sausage","ham","lamb","veal","brisket"])
                    is_dairy = any(w in name_l for w in ["milk","cream","yogurt","butter","cheese","kefir","half-and-half"])
                    is_pantry_only = any(w in name_l for w in ["seltzer","litter","detergent","vodka","whiskey","rum","tequila","gin","liquor","cleaner","soap","shampoo","white claw","truly","nutrl","hard lemonade"])

                    if gemini_days_raw is not None:
                        shelf_days = int(gemini_days_raw)
                        # Safety cap: raw meat/fish must never exceed 7 days fridge
                        if is_raw_meat and gemini_storage == "fridge" and shelf_days > 7:
                            shelf_days = 4
                        # Global cap: nothing perishable exceeds 365 days
                        shelf_days = min(shelf_days, 365)
                        shelf_storage = gemini_storage
                        shelf_confidence = "high"
                    else:
                        # Gemini didn't return a value — fall back to our rules DB
                        rules = _rules_lookup(final_name_for_classify)
                        if rules:
                            shelf_days = rules["expires_in_days"]
                            shelf_storage = rules["storage"]
                            shelf_confidence = rules.get("confidence", "medium")
                        else:
                            shelf_days = 14
                            shelf_storage = gemini_storage
                            shelf_confidence = "low"

                    # Build shelf_life_by_storage from Gemini's result fields
                    # Gemini returns fridge/freezer/pantry directly in the result
                    g_fridge  = result.get("fridge")
                    g_freezer = result.get("freezer")
                    g_pantry  = result.get("pantry")

                    if any(v is not None for v in [g_fridge, g_freezer, g_pantry]):
                        # Gemini provided all three — use them directly
                        slbs = {
                            "fridge":  int(g_fridge)  if g_fridge  is not None else None,
                            "freezer": int(g_freezer) if g_freezer is not None else None,
                            "pantry":  int(g_pantry)  if g_pantry  is not None else None,
                        }
                    else:
                        # Gemini only gave expires_in_days — build slbs intelligently
                        rules = _rules_lookup(final_name_for_classify)
                        if rules:
                            slbs = rules.get("shelf_life_by_storage", {"fridge": shelf_days, "freezer": None, "pantry": None})
                        elif shelf_storage == "fridge":
                            if is_meat:
                                slbs = {"fridge": shelf_days, "freezer": min(shelf_days * 30, 365), "pantry": None}
                            elif is_dairy:
                                slbs = {"fridge": shelf_days, "freezer": None, "pantry": None}
                            else:
                                slbs = {"fridge": shelf_days, "freezer": min(shelf_days * 3, 365), "pantry": None}
                        elif shelf_storage == "freezer":
                            slbs = {"fridge": None, "freezer": shelf_days, "pantry": None}
                        else:  # pantry
                            if is_pantry_only:
                                slbs = {"fridge": None, "freezer": None, "pantry": shelf_days}
                            else:
                                slbs = {"fridge": None, "freezer": None, "pantry": shelf_days}
                    enrichment = {
                        "full_name": gemini_full_name or item["name"],
                        "expires_in_days": shelf_days,
                        "storage": shelf_storage,
                        "category": our_category,
                        "food_category": our_food_category,
                        "photo_query": gemini_photo_query or None,
                        "is_estimated": True,
                        "confidence": shelf_confidence,
                        "shelf_life_by_storage": slbs,
                    }
                    _ai_enrichment_cache[key] = enrichment
                    results[uncached_indices[j]] = enrichment.copy()

                print(f"[GEMINI] Success with {model_name} — results: {[(it['name'], r.get('expires_in_days'), r.get('storage')) for it, r in zip(uncached_items, gemini_results)]}", flush=True)
                gemini_succeeded = True
                break
            except Exception as exc:
                print(f"[GEMINI] {model_name} failed: {exc}", flush=True)
                continue

    if not gemini_succeeded:
        print(f"[GEMINI] All models failed — using fallback dictionary", flush=True)
        for j, i in enumerate(uncached_indices):
            key = items[i]["name"].lower().strip()
            enrichment = _fallback_for_item(items[i]["name"])
            # Always use our classifier for category
            enrichment["category"] = _classify(items[i]["name"])
            _ai_enrichment_cache[key] = enrichment
            results[i] = enrichment.copy()

    for i in range(len(results)):
        if results[i] is None:
            results[i] = _fallback_for_item(items[i]["name"])

    return results  # type: ignore

PACKSHOT_SERVICE_URL = (os.getenv("PACKSHOT_SERVICE_URL") or "").strip().rstrip("/")
PACKSHOT_SERVICE_KEY = (os.getenv("PACKSHOT_SERVICE_KEY") or "").strip()

KROGER_CLIENT_ID = (os.getenv("KROGER_CLIENT_ID") or "").strip()
KROGER_CLIENT_SECRET = (os.getenv("KROGER_CLIENT_SECRET") or "").strip()

# Edamam API keys
EDAMAM_FOOD_APP_ID  = (os.getenv("EDAMAM_FOOD_APP_ID")  or "").strip()
EDAMAM_FOOD_APP_KEY = (os.getenv("EDAMAM_FOOD_APP_KEY") or "").strip()
EDAMAM_RECIPE_APP_ID  = (os.getenv("EDAMAM_RECIPE_APP_ID")  or "").strip()
EDAMAM_RECIPE_APP_KEY = (os.getenv("EDAMAM_RECIPE_APP_KEY") or "").strip()
_kroger_token: Optional[str] = None
_kroger_token_expiry: float = 0.0

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

# Persistent URL cache — survives server restarts
_PERSISTENT_IMAGE_URL_CACHE: Dict[str, str] = {}
_PERSISTENT_CACHE_PATH = "/data/persistent_image_cache.json"


def _load_persistent_image_cache() -> None:
    """Load the persistent image URL cache from disk on startup."""
    global _PERSISTENT_IMAGE_URL_CACHE
    try:
        if os.path.exists(_PERSISTENT_CACHE_PATH):
            with open(_PERSISTENT_CACHE_PATH, "r") as f:
                _PERSISTENT_IMAGE_URL_CACHE = json.load(f)
            print(f"[PERSISTENT CACHE] Loaded {len(_PERSISTENT_IMAGE_URL_CACHE)} cached image URLs from disk.", flush=True)
        else:
            _PERSISTENT_IMAGE_URL_CACHE = {}
            print("[PERSISTENT CACHE] No cache file found, starting fresh.", flush=True)
    except Exception as e:
        _PERSISTENT_IMAGE_URL_CACHE = {}
        print(f"[PERSISTENT CACHE] Failed to load cache: {e}", flush=True)


def _save_persistent_image_cache() -> None:
    """Save the persistent image URL cache to disk."""
    try:
        os.makedirs(os.path.dirname(_PERSISTENT_CACHE_PATH), exist_ok=True)
        with open(_PERSISTENT_CACHE_PATH, "w") as f:
            json.dump(_PERSISTENT_IMAGE_URL_CACHE, f)
    except Exception as e:
        print(f"[PERSISTENT CACHE] Failed to save cache: {e}", flush=True)


def _get_persistent_image(name: str) -> Optional[str]:
    """Look up a product image URL from the persistent cache."""
    return _PERSISTENT_IMAGE_URL_CACHE.get(name.lower().strip())


def _set_persistent_image(name: str, url: str) -> None:
    """Save a product image URL to the persistent cache and write to disk."""
    _PERSISTENT_IMAGE_URL_CACHE[name.lower().strip()] = url
    _save_persistent_image_cache()

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
    "whole", "foods", "trader", "joe", "joes", "fresh", "market",
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
    ("fresh market", re.compile(r"\bthe\s+fresh\s+market\b|\bfresh\s+market\b", re.IGNORECASE)),
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
    "fresh market": re.compile(r"^\s*(?:the\s+)?fresh\s+market\s*$", re.IGNORECASE),
}

_STORE_TOKENS = {"publix", "walmart", "wal", "mart", "target", "costco", "kroger", "aldi", "whole", "foods", "trader", "joe", "joes", "wm", "fresh", "market"}
_GENERIC_HEADER_TOKENS = {"super", "markets", "market", "stores", "store", "wholesale", "pharmacy", "supercenter"}
_JUNK_EXACT_LINES = {
    # Single/double letter noise
    "t", "f", "tf", "t f", "ft", "tt", "ff",
    # Payment/financial
    "visa", "debit", "credit", "for", "cash", "tender",
    # Generic category labels that are NOT food items
    "grocery", "groceries", "produce", "deli", "bakery", "seafood",
    "meat", "dairy", "frozen", "pharmacy", "floral", "bulk",
    "general merchandise", "gm", "nonfood", "non food", "non-food",
    # Costco-specific section headers
    "bottom of basket", "top of basket",
    "instant savings", "coupon savings",
    "executive member", "executive membership",
    "executive rebate",
    "next renewal", "membership renewal",
    "warehouse", "costco wholesale",
    # Generic receipt labels
    "food product", "live food product", "food item",
    "sugar free food product", "live sugar free food product",
    "tote", "reusable bag", "bag charge", "bag fee",
    "wpv tote", "wpv",
    # ALDI section labels
    "special buy", "aldi finds", "weekly specials",
    # Kroger/Publix
    "digital coupon", "fuel points", "gas points",
    # Whole Foods / Trader Joe's
    "team member",
    # Price promotion lines (e.g. "2 for", "3 for", "2 for 5", "buy 2 get 1")
    "2 for", "3 for", "4 for", "5 for", "2 for 1", "buy 2", "buy 3",
    "mix match", "mix and match", "mix & match",
}

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
    r"\boak\s*grove\s*shoppes?\b",
    r"\boak\s*grove\b",
    r"\bshoppes?\b",
    # ── Costco section headers & meta lines ─────────────────────────────────
    r"\bbottom\s+of\s+basket\b",
    r"\btop\s+of\s+basket\b",
    r"\binstant\s+savings\b",
    r"\bcoupon\s+savings\b",
    r"\bexecutive\s+(?:member|membership|rebate)\b",
    r"\bnext\s+renewal\b",
    r"\bmembership\s+renewal\b",
    r"^\s*costco\s+wholesale\s*$",
    r"^\s*warehouse\s*$",
    r"\bcash\s*rebate\b",
    r"\bannual\s+member\b",
    r"\bbob\s+count\b",
    # ── Bag / tote charges (all stores) ──────────────────────────────────
    r"\breusable\s+bag\b",
    r"\bbag\s+(?:charge|fee)\b",
    r"\bcarry\s+out\s+bag\b",
    r"\bplastic\s+bag\b",
    r"\bwpv\s*tote\b",
    r"^\s*tote\s*$",
    r"^\s*wpv\s*$",
    # ── Generic category header labels (all stores) ──────────────────────
    r"^\s*produce\s*$",
    r"^\s*deli\s*$",
    r"^\s*bakery\s*$",
    r"^\s*seafood\s*$",
    r"^\s*(?:raw\s+)?meat\s*$",
    r"^\s*dairy\s*$",
    r"^\s*frozen\s*$",
    r"^\s*floral\s*$",
    r"^\s*bulk\s*$",
    r"^\s*(?:general\s+)?merchandise\s*$",
    r"^\s*non[-\s]?food\s*$",
    r"\bfood\s+product\b",
    r"\blive\s+(?:sugar\s+free\s+)?food\s+product\b",
    # ── Loyalty / rewards lines (all stores) ────────────────────────────
    r"\bdigital\s+coupon\b",
    r"\bfuel\s+points\b",
    r"\bgas\s+points\b",
    r"\bcash\s+rewards?\b",
    r"\breward\s+points\b",
    r"\bjust\s+for\s+u\b",
    r"\bclub\s+card\b",
    r"\bplus\s+card\b",
    r"\baldi\s+finds\b",
    r"\bspecial\s+buy\b",
    r"\bweekly\s+specials?\b",
    # ── OCR garbage patterns (universal) ──────────────────────────────
    r"^\s*[A-Za-z]{1,3}\s*$",
    r"^\s*[A-Za-z]{3,8}\s+\d{2,4}[A-Za-z]{1,4}\s*$",
    r"^\s*[BCDFGHJKLMNPQRSTVWXYZbcdfghjklmnpqrstvwxyz]{4,}\s*$",
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

    # Bacon / Great Value
    "gw": "great value",
    "bac": "bacon",
    "bcon": "bacon",

    # Chicken
    "chick": "chicken",

    # Buitoni tortellini
    "buit": "buitoni",
    "tortel": "tortellini",
    "tortl": "tortellini",
    "tortelini": "tortellini",

    # Cremo Barber Grade Italian Bergamot
    "bw": "barber grade",
    "ital": "italian",
    "berg": "bergamot",
    "brb": "barber",
    "brbr": "barber",

    # Ben's Original rice
    "bens": "ben's original",
    "fg": "flavored grain",
    "wld": "wild",
    "lrg": "large",

    # Fresh Step Lightweight
    "lght": "lightweight",
    "wght": "weight",
    "lwt": "lightweight",

    # AMC popcorn  
    "ktl": "kettle",
    "btr": "butter",

    # Granny Smith apples
    "smit": "smith",
    "grny": "granny",

    # Snyder's pretzels
    "snyd": "snyder's",
    "prtz": "pretzels",
    "snap": "snap pretzels",

    # Thomas bagels
    "thms": "thomas",

    # Rao's marinara
    "rao": "rao's homemade",

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
        "rf": "reduced fat",
        "ff": "fat free",
        "lf": "low fat",
        "ws": "whole",
        "nf": "nonfat",
        "choc": "chocolate",
        "van": "vanilla",
        "strbry": "strawberry",
        "strw": "strawberry",
        "blbry": "blueberry",
        "rsn": "raisin",
        "cin": "cinnamon",
        "cinn": "cinnamon",
        "xtr": "extra",
        "xtra": "extra",
        "orng": "orange",
        "jce": "juice",
        "mlk": "milk",
        "yog": "yogurt",
        "yg": "yogurt",
        "snk": "snack",
        "snks": "snacks",
        "brd": "bread",
        "wht": "wheat",
        "whl": "whole",
        "grn": "grain",
        "mlt": "multi",
        "sdwch": "sandwich",
        "chkn": "chicken",
        "bnls": "boneless",
        "brst": "breast",
        "thgh": "thigh",
        "drm": "drumstick",
        "tndrl": "tenderloin",
        "grnd": "ground",
        "trky": "turkey",
        "slcd": "sliced",
        "shrdd": "shredded",
        "grtd": "grated",
        "slc": "slice",
        "deli": "deli",
        "rst": "roast",
        "rstd": "roasted",
        "smkd": "smoked",
        "grl": "grilled",
        "bkd": "baked",
        "frd": "fried",
        "frzn": "frozen",
        "org": "organic",
        "nat": "natural",
        "prem": "premium",
        "sel": "select",
        "asst": "assorted",
        "vty": "variety",
        "pblx": "publix",
        "pub": "publix",
        "pbx": "publix",
        "pt": "pint",
    },
    "target": {
        # Good & Gather is Target's store brand
        "gg": "good and gather",
        "g&g": "good and gather",
        "mkt": "market pantry",
        "mp": "market pantry",
        "up&up": "up and up",
        "upup": "up and up",
        "smpl": "simply balanced",
        "org": "organic",
        "frsh": "fresh",
        "choc": "chocolate",
        "van": "vanilla",
        "strbry": "strawberry",
        "blbry": "blueberry",
        "btr": "butter",
        "bttr": "butter",
        "crm": "cream",
        "chz": "cheese",
        "chs": "cheese",
        "snk": "snack",
        "brd": "bread",
        "whl": "whole",
        "wht": "wheat",
        "mlt": "multi",
        "grn": "grain",
        "mlk": "milk",
        "yog": "yogurt",
        "grnd": "ground",
        "chkn": "chicken",
        "bnls": "boneless",
        "brst": "breast",
        "trky": "turkey",
        "frzn": "frozen",
        "slcd": "sliced",
        "shrdd": "shredded",
        "rst": "roast",
        "rstd": "roasted",
        "smkd": "smoked",
        "asst": "assorted",
        "vty": "variety",
        "sel": "select",
        "prem": "premium",
        "lg": "large",
        "sm": "small",
        "pk": "pack",
        "pck": "pack",
        "xtr": "extra",
        "xtra": "extra",
        "orig": "original",
        "nat": "natural",
        "lf": "low fat",
        "rf": "reduced fat",
        "ff": "fat free",
        "nf": "nonfat",
    },
    "walmart": {
        # Great Value is Walmart's primary store brand
        "gv": "great value",
        "gw": "great value",
        "grtvl": "great value",
        # Sam's Choice is the premium Walmart store brand
        "sc": "sam's choice",
        "samc": "sam's choice",
        # Equate is Walmart's health/personal care brand
        "eq": "equate",
        "eqt": "equate",
        # Parent's Choice is Walmart's baby brand
        "pc": "parent's choice",
        # Marketside is Walmart's fresh/deli brand
        "mktsd": "marketside",
        "mkts": "marketside",
        # Common Walmart receipt abbreviations
        "org": "organic",
        "nat": "natural",
        "frsh": "fresh",
        "frzn": "frozen",
        "orig": "original",
        "prem": "premium",
        "sel": "select",
        "asst": "assorted",
        "vty": "variety",
        "choc": "chocolate",
        "van": "vanilla",
        "strbry": "strawberry",
        "strwbry": "strawberry",
        "blbry": "blueberry",
        "rsn": "raisin",
        "cin": "cinnamon",
        "cinn": "cinnamon",
        "orng": "orange",
        "jce": "juice",
        "mlk": "milk",
        "yog": "yogurt",
        "btr": "butter",
        "bttr": "butter",
        "crm": "cream",
        "chs": "cheese",
        "chz": "cheese",
        "chkn": "chicken",
        "bnls": "boneless",
        "brst": "breast",
        "thgh": "thigh",
        "grnd": "ground",
        "trky": "turkey",
        "slcd": "sliced",
        "shrdd": "shredded",
        "rst": "roast",
        "rstd": "roasted",
        "smkd": "smoked",
        "grl": "grilled",
        "bkd": "baked",
        "snk": "snack",
        "brd": "bread",
        "whl": "whole",
        "wht": "wheat",
        "mlt": "multi",
        "grn": "grain",
        "sdwch": "sandwich",
        "lg": "large",
        "sm": "small",
        "pk": "pack",
        "pck": "pack",
        "xtr": "extra",
        "xtra": "extra",
        "lf": "low fat",
        "rf": "reduced fat",
        "ff": "fat free",
        "nf": "nonfat",
        "brn": "brown",
        "sug": "sugar",
        "bac": "bacon",
        "spcy": "spicy",
        "hny": "honey",
        "grl": "garlic",
        "onon": "onion",
        "tmt": "tomato",
        "ltc": "lettuce",
        "spin": "spinach",
        "broc": "broccoli",
        "caul": "cauliflower",
        "pot": "potato",
        "swpot": "sweet potato",
        "avdo": "avocado",
        "avo": "avocado",
    },
    "aldi": {
        # ALDI uses SimplyNature, Never Any!, Specially Selected, Earth Grown, Fit & Active
        "sn": "simply nature",
        "simntr": "simply nature",
        "simnat": "simply nature",
        "na": "never any",
        "nvr": "never",
        "ss": "specially selected",
        "spslct": "specially selected",
        "eg": "earth grown",
        "fa": "fit and active",
        "fitact": "fit and active",
        "bb": "bake house creations",
        "bhc": "bake house creations",
        "lm": "little journey",
        "pd": "park street deli",
        "sd": "sundae shoppe",
        "ch": "countryside creamery",
        "cc": "countryside creamery",
        # Common ALDI abbreviations
        "org": "organic",
        "nat": "natural",
        "frsh": "fresh",
        "frzn": "frozen",
        "orig": "original",
        "prem": "premium",
        "sel": "selected",
        "asst": "assorted",
        "choc": "chocolate",
        "van": "vanilla",
        "strbry": "strawberry",
        "blbry": "blueberry",
        "cin": "cinnamon",
        "cinn": "cinnamon",
        "mlk": "milk",
        "yog": "yogurt",
        "btr": "butter",
        "bttr": "butter",
        "crm": "cream",
        "chs": "cheese",
        "chz": "cheese",
        "chkn": "chicken",
        "bnls": "boneless",
        "brst": "breast",
        "grnd": "ground",
        "trky": "turkey",
        "slcd": "sliced",
        "shrdd": "shredded",
        "rst": "roast",
        "smkd": "smoked",
        "brd": "bread",
        "whl": "whole",
        "wht": "wheat",
        "grn": "grain",
        "lg": "large",
        "sm": "small",
        "pk": "pack",
        "lf": "low fat",
        "rf": "reduced fat",
    },
    "costco": {
        # Kirkland Signature is Costco's store brand — the most important one
        "ks": "kirkland signature",
        "ksl": "kirkland signature",
        "kslg": "kirkland signature",
        "kirk": "kirkland",
        "kirkl": "kirkland",
        "krkl": "kirkland",
        # Costco sells bulk — these are very common on their receipts
        "org": "organic",
        "org ": "organic",
        "nat": "natural",
        "frsh": "fresh",
        "frzn": "frozen",
        "orig": "original",
        "prem": "premium",
        "sel": "select",
        "asst": "assorted",
        "vty": "variety",
        "choc": "chocolate",
        "dk": "dark",
        "van": "vanilla",
        "strbry": "strawberry",
        "blbry": "blueberry",
        "rsn": "raisin",
        "cin": "cinnamon",
        "cinn": "cinnamon",
        "orng": "orange",
        "jce": "juice",
        "mlk": "milk",
        "yog": "yogurt",
        "grk": "greek",
        "btr": "butter",
        "bttr": "butter",
        "crm": "cream",
        "chs": "cheese",
        "chz": "cheese",
        "chkn": "chicken",
        "bnls": "boneless",
        "brst": "breast",
        "thgh": "thigh",
        "grnd": "ground",
        "trky": "turkey",
        "slcd": "sliced",
        "shrdd": "shredded",
        "grtd": "grated",
        "rst": "roast",
        "rstd": "roasted",
        "smkd": "smoked",
        "grl": "grilled",
        "bkd": "baked",
        "snk": "snack",
        "brd": "bread",
        "whl": "whole",
        "wht": "wheat",
        "mlt": "multi",
        "grn": "grain",
        "lg": "large",
        "sm": "small",
        "pk": "pack",
        "pck": "pack",
        "cnt": "count",
        "xtr": "extra",
        "xtra": "extra",
        "lf": "low fat",
        "rf": "reduced fat",
        "ff": "fat free",
        "nf": "nonfat",
        "brn": "brown",
        "sug": "sugar",
        "bac": "bacon",
        "spcy": "spicy",
        "hny": "honey",
        "pot": "potato",
        "avdo": "avocado",
        "avo": "avocado",
        "alnd": "almond",
        "alm": "almond",
        "csh": "cashew",
        "wlt": "walnut",
        "pst": "pistachio",
        "mxd": "mixed",
        "drfd": "dried",
        "grntd": "granulated",
        "unslt": "unsalted",
        "rstd": "roasted",
    },
    "kroger": {
        # Kroger's store brands: Simple Truth, Private Selection, Kroger Brand
        "st": "simple truth",
        "str": "simple truth",
        "simtruth": "simple truth",
        "ps": "private selection",
        "prsel": "private selection",
        "kr": "kroger",
        "krg": "kroger",
        # Common Kroger receipt abbreviations
        "org": "organic",
        "nat": "natural",
        "frsh": "fresh",
        "frzn": "frozen",
        "orig": "original",
        "prem": "premium",
        "sel": "select",
        "asst": "assorted",
        "vty": "variety",
        "choc": "chocolate",
        "van": "vanilla",
        "strbry": "strawberry",
        "blbry": "blueberry",
        "rsn": "raisin",
        "cin": "cinnamon",
        "cinn": "cinnamon",
        "orng": "orange",
        "jce": "juice",
        "mlk": "milk",
        "yog": "yogurt",
        "grk": "greek",
        "btr": "butter",
        "bttr": "butter",
        "crm": "cream",
        "chs": "cheese",
        "chz": "cheese",
        "chkn": "chicken",
        "bnls": "boneless",
        "brst": "breast",
        "thgh": "thigh",
        "grnd": "ground",
        "trky": "turkey",
        "slcd": "sliced",
        "shrdd": "shredded",
        "grtd": "grated",
        "rst": "roast",
        "rstd": "roasted",
        "smkd": "smoked",
        "grl": "grilled",
        "snk": "snack",
        "brd": "bread",
        "whl": "whole",
        "wht": "wheat",
        "mlt": "multi",
        "grn": "grain",
        "sdwch": "sandwich",
        "lg": "large",
        "sm": "small",
        "pk": "pack",
        "pck": "pack",
        "xtr": "extra",
        "lf": "low fat",
        "rf": "reduced fat",
        "ff": "fat free",
        "nf": "nonfat",
        "brn": "brown",
        "sug": "sugar",
        "bac": "bacon",
        "spcy": "spicy",
        "hny": "honey",
        "grl": "garlic",
        "pot": "potato",
        "swpot": "sweet potato",
        "avdo": "avocado",
        "avo": "avocado",
        "alnd": "almond",
        "csh": "cashew",
        "wlt": "walnut",
        "mxd": "mixed",
        "unslt": "unsalted",
        "slt": "salted",
    },
    "whole foods": {
        # 365 is Whole Foods' store brand (now 365 by Whole Foods Market)
        "365": "365 whole foods market",
        "wfm": "whole foods market",
        "org": "organic",
        "nat": "natural",
        "frsh": "fresh",
        "frzn": "frozen",
        "orig": "original",
        "prem": "premium",
        "sel": "select",
        "asst": "assorted",
        "choc": "chocolate",
        "dk": "dark",
        "van": "vanilla",
        "strbry": "strawberry",
        "blbry": "blueberry",
        "rsn": "raisin",
        "cin": "cinnamon",
        "orng": "orange",
        "jce": "juice",
        "mlk": "milk",
        "yog": "yogurt",
        "grk": "greek",
        "btr": "butter",
        "bttr": "butter",
        "crm": "cream",
        "chs": "cheese",
        "chz": "cheese",
        "chkn": "chicken",
        "bnls": "boneless",
        "brst": "breast",
        "grnd": "ground",
        "trky": "turkey",
        "slcd": "sliced",
        "shrdd": "shredded",
        "grtd": "grated",
        "rst": "roast",
        "rstd": "roasted",
        "smkd": "smoked",
        "snk": "snack",
        "brd": "bread",
        "whl": "whole",
        "wht": "wheat",
        "mlt": "multi",
        "grn": "grain",
        "lg": "large",
        "sm": "small",
        "pk": "pack",
        "xtr": "extra",
        "lf": "low fat",
        "rf": "reduced fat",
        "ff": "fat free",
        "nf": "nonfat",
        "brn": "brown",
        "sug": "sugar",
        "spcy": "spicy",
        "hny": "honey",
        "avdo": "avocado",
        "avo": "avocado",
        "alnd": "almond",
        "csh": "cashew",
        "wlt": "walnut",
        "mxd": "mixed",
        "gf": "gluten free",
        "glf": "gluten free",
        "vgn": "vegan",
        "veg": "vegetarian",
        "unslt": "unsalted",
        "slt": "salted",
        "ksh": "kosher",
        "hlal": "halal",
    },
    "trader joe's": {
        # TJ's uses very minimal abbreviations — their items are usually fully printed
        # but OCR can mangle them
        "tjs": "trader joe's",
        "tj": "trader joe's",
        "org": "organic",
        "nat": "natural",
        "frsh": "fresh",
        "frzn": "frozen",
        "orig": "original",
        "prem": "premium",
        "asst": "assorted",
        "choc": "chocolate",
        "dk": "dark",
        "mlk": "milk",
        "van": "vanilla",
        "strbry": "strawberry",
        "blbry": "blueberry",
        "cin": "cinnamon",
        "orng": "orange",
        "yog": "yogurt",
        "grk": "greek",
        "btr": "butter",
        "bttr": "butter",
        "crm": "cream",
        "chs": "cheese",
        "chz": "cheese",
        "chkn": "chicken",
        "bnls": "boneless",
        "brst": "breast",
        "grnd": "ground",
        "slcd": "sliced",
        "shrdd": "shredded",
        "rst": "roast",
        "smkd": "smoked",
        "brd": "bread",
        "whl": "whole",
        "wht": "wheat",
        "mlt": "multigrain",
        "grn": "grain",
        "lg": "large",
        "sm": "small",
        "pk": "pack",
        "gf": "gluten free",
        "vgn": "vegan",
        "veg": "vegetarian",
        "spcy": "spicy",
        "hny": "honey",
        "avdo": "avocado",
        "avo": "avocado",
        "alnd": "almond",
        "csh": "cashew",
        "mxd": "mixed",
        "unslt": "unsalted",
        "slt": "salted",
        "grl": "garlic",
        "lmn": "lemon",
        "lme": "lime",
        "mshr": "mushroom",
        "spnch": "spinach",
        "kl": "kale",
        "artch": "artichoke",
    },
    "fresh market": {
        # The Fresh Market uses full names more often but OCR still abbreviates
        "tfm": "the fresh market",
        "fm": "fresh market",
        "org": "organic",
        "nat": "natural",
        "frsh": "fresh",
        "frzn": "frozen",
        "orig": "original",
        "prem": "premium",
        "sel": "select",
        "asst": "assorted",
        "choc": "chocolate",
        "dk": "dark",
        "van": "vanilla",
        "strbry": "strawberry",
        "blbry": "blueberry",
        "mlk": "milk",
        "yog": "yogurt",
        "grk": "greek",
        "btr": "butter",
        "bttr": "butter",
        "crm": "cream",
        "chs": "cheese",
        "chkn": "chicken",
        "bnls": "boneless",
        "brst": "breast",
        "grnd": "ground",
        "slcd": "sliced",
        "shrdd": "shredded",
        "grtd": "grated",
        "rst": "roast",
        "rstd": "roasted",
        "smkd": "smoked",
        "brd": "bread",
        "whl": "whole",
        "wht": "wheat",
        "grn": "grain",
        "gf": "gluten free",
        "vgn": "vegan",
        "spcy": "spicy",
        "hny": "honey",
        "avdo": "avocado",
        "avo": "avocado",
        "alnd": "almond",
        "mxd": "mixed",
        "unslt": "unsalted",
        "slt": "salted",
        "ksh": "kosher",
    },
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
    "grnd": "ground",
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
    # Universal — no store hint needed
    "grk": "greek",
    "frzn": "frozen",
    "frsh": "fresh",
    "slcd": "sliced",
    "shrdd": "shredded",
    "grtd": "grated",
    "smkd": "smoked",
    "rstd": "roasted",
    "rst": "roast",
    "grl": "grilled",
    "bkd": "baked",
    "trky": "turkey",
    "thgh": "thigh",
    "spcy": "spicy",
    "hny": "honey",
    "choc": "chocolate",
    "van": "vanilla",
    "strbry": "strawberry",
    "blbry": "blueberry",
    "cin": "cinnamon",
    "cinn": "cinnamon",
    "orng": "orange",
    "jce": "juice",
    "mlk": "milk",
    "brn": "brown",
    "sug": "sugar",
    "bac": "bacon",
    "nat": "natural",
    "prem": "premium",
    "sel": "select",
    "asst": "assorted",
    "vty": "variety",
    "wht": "wheat",
    "whl": "whole",
    "mlt": "multi",
    "sdwch": "sandwich",
    "snk": "snack",
    "brd": "bread",
    "lf": "low fat",
    "rf": "reduced fat",
    "ff": "fat free",
    "nf": "nonfat",
    "unslt": "unsalted",
    "slt": "salted",
    "avdo": "avocado",
    "avo": "avocado",
    "alnd": "almond",
    "alm": "almond",
    "csh": "cashew",
    "wlt": "walnut",
    "pst": "pistachio",
    "mxd": "mixed",
    "gf": "gluten free",
    "vgn": "vegan",
    "ksh": "kosher",
    "hlal": "halal",
    "pot": "potato",
    "swpot": "sweet potato",
    "spin": "spinach",
    "broc": "broccoli",
    "caul": "cauliflower",
    "dk": "dark",
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
    "amc", "jack daniel's", "rao", "rao's", "panera", "publix", "king's hawaiian",
    "nutrl", "buitoni", "thomas", "thomas'", "snyder's", "snyder", "ben's", "great value",
    "sara lee", "wishbone", "cremo", "mojo", "buitoni", "granny smith",
    "fresh step", "tide", "boneless chicken",
}

# Hardcoded map is intentionally empty — all images now fetched dynamically
# from Unsplash → Kroger → Freepik in that order.
PRODUCT_IMAGE_MAP: Dict[str, str] = {}
FALLBACK_PRODUCT_IMAGE = "https://images.unsplash.com/photo-1542838132-92c53300491e?w=512&q=80"


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

    _IMAGE_CACHE.clear()
    _IMAGE_CONTENT_TYPE_CACHE.clear()
    _init_google_credentials_file()
    _load_learned_map()
    _load_pending_map()
    _load_persistent_image_cache()
    print("Startup complete. Image cache cleared.", flush=True)


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
    shelf_life_by_storage: Optional[Dict[str, Optional[int]]] = None


class InstacartLineItem(BaseModel):
    name: str
    quantity: float = 1.0
    unit: str = "each"
    upc: Optional[str] = None
    category: Optional[str] = None


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


def _image_url_for_item(base_url: str, name: str, category: Optional[str] = None, photo_query: Optional[str] = None) -> str:
    url = f"{base_url}/image?name={urllib.parse.quote((name or '').strip())}"
    if category:
        url += f"&category={urllib.parse.quote(category.strip())}"
    if photo_query:
        url += f"&photo_query={urllib.parse.quote(photo_query.strip())}"
    return url


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


# Words that, if a line starts with them, indicate it's a product — never an address.
_PRODUCT_LEAD_WORDS = {
    "FRSH", "SBR", "PBX", "TIDE", "DAWN", "GAIN", "BOUNTY", "CHARMIN", "GLAD",
    "SCOTT", "HEFTY", "FEBREZE", "LYSOL", "CLOROX", "PALMOLIVE", "AJAX",
    "PUBLIX", "GREAT", "SNYDER", "BUITONI", "KINGS", "KH", "AMC", "KROGER",
    "NABISCO", "OREO", "PEPSI", "COKE", "SPRITE", "GATORADE", "POWERADE",
    "TROPICANA", "DOLE", "CHOBANI", "YOPLAIT", "DANNON", "ACTIVIA",
    "LAYS", "RUFFLES", "DORITOS", "CHEETOS", "PRINGLES", "TOSTITOS",
    "CAMPBELL", "PROGRESSO", "HUNTS", "HEINZ", "KRAFT", "VELVEETA",
    "HIDDEN", "RANCH", "KENS", "WISHBONE", "NEWMANS",
    "BIRDS", "STOUFFERS", "LEAN", "HEALTHY", "WEIGHT",
    "SARA", "PEPPERIDGE", "ARNOLD", "NATURES", "OTIS",
    "BANQUET", "TYSON", "PERDUE", "FOSTER", "JIMMY", "HILLSHIRE",
    "OSCAR", "BALL", "LAND", "HORIZON", "ORGANIC",
    "FRESH", "ULTRA", "PREMIUM", "SELECT", "CHOICE",
}

def _looks_like_receipt_product_line(s: str) -> bool:
    """Return True if this line looks like a receipt product abbreviation, not an address.
    Criteria: all tokens are ALL-CAPS, avg token length <= 6, at least 2 tokens.
    Also True if the first token is a known product-lead word.
    """
    toks = s.split()
    if not toks:
        return False
    # Check for known product-lead word at start
    if toks[0].upper() in _PRODUCT_LEAD_WORDS:
        return True
    # All-caps short-token heuristic (receipt abbreviations)
    if len(toks) >= 2:
        upper_toks = [t for t in toks if re.match(r'^[A-Z0-9]+$', t)]
        if len(upper_toks) == len(toks):
            avg_len = sum(len(t) for t in toks) / len(toks)
            if avg_len <= 7:
                return True
    return False


def _is_header_or_address(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return True

    # Guard: if this looks like a receipt product line (all-caps abbreviations
    # or starts with a known brand/product word), never treat it as an address.
    if _looks_like_receipt_product_line(s):
        return False

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

    # Single word: too short or all consonants = noise
    if len(toks) == 1:
        word = toks[0]
        if word.isalpha() and len(word) <= 4:
            return True
        if word.isalpha() and len(word) >= 4 and not any(c in "aeiouAEIOU" for c in word):
            return True

    # Two words: both tiny with no vowels = OCR garbage
    if len(toks) == 2:
        both_short = all(len(t) <= 3 for t in toks)
        no_vowels  = all(not any(c in "aeiouAEIOU" for c in t) for t in toks if t.isalpha())
        if both_short and no_vowels:
            return True
        # Random letters + malformed number e.g. "Ahoxi 200lds"
        if toks[0].isalpha() and re.match(r"^\d{2,4}[A-Za-z]{1,4}$", toks[1]):
            return True

    # All tokens are 1-2 letters and none are known units = noise
    if len(toks) >= 2 and all(len(t) <= 2 and t.isalpha() for t in toks):
        if not any(t in {"oz", "lb", "ct", "ea", "gf", "og"} for t in toks):
            return True

    # Price promotion lines: "2 for $5", "3 for 10.00", "2 for 6", etc.
    raw = s.strip().lower()
    if re.match(r'^\d+\s+for\b', raw):
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


# Standalone single-word items that should never be merged with the next line
_STANDALONE_ITEMS = {
    "garlic", "carrots", "carrot", "apples", "apple", "bananas", "banana",
    "oranges", "orange", "grapes", "grape", "lemons", "lemon", "limes", "lime",
    "broccoli", "spinach", "lettuce", "onions", "onion", "potatoes", "potato",
    "tomatoes", "tomato", "strawberries", "blueberries", "milk", "eggs", "butter",
    "chicken", "beef", "pork", "salmon", "shrimp", "bread", "rice", "pasta",
    "celery", "mushrooms", "mushroom", "avocado", "avocados", "kale", "corn",
    "peaches", "peach", "plums", "plum", "cherries", "cherry", "mango", "mangos",
    "pears", "pear", "watermelon", "pineapple", "cauliflower", "asparagus",
    "zucchini", "cucumber", "peppers", "pepper", "radishes", "radish",
}

def _should_merge_with_next(head: str, tail: str, next_support: str) -> bool:
    if not head or not tail:
        return False

    head_key = dedupe_key(head)
    head_toks = head_key.split()
    if not head_toks:
        return False

    # Never merge if the head is already a complete standalone item name
    if len(head_toks) == 1 and head_toks[0] in _STANDALONE_ITEMS:
        return False

    # Never merge if the tail starts with the same brand/store word as the head
    # e.g. "PUBLIX PASTA" should not merge with "PUBLIX STICKS SALT"
    tail_key = dedupe_key(tail)
    tail_toks = tail_key.split()
    if head_toks and tail_toks and head_toks[0] == tail_toks[0] and len(tail_toks) >= 2:
        return False

    # Never merge if the tail starts with a known brand/product lead word
    # e.g. "PUBLIX PASTA" should not merge with "TIDE HE SPRING REN"
    if tail_toks and tail_toks[0].upper() in _PRODUCT_LEAD_WORDS:
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
    if re.search(r"\bsticks?\b", low) and re.search(r"\bsalted?\b|\bsalt\b", low) and "butter" not in low:
        low = re.sub(r"\bsticks?\b", "butter sticks", low)
        low = re.sub(r"\bsalted?\b|\bsalt\b", "salted", low)
        low = re.sub(r"\s+", " ", low).strip()

    # Fresh Step Lightweight Cat Litter
    if "fresh" in low and ("lght" in low or "lightweight" in low or "light weight" in low):
        return "fresh step lightweight cat litter"

    # Fresh Step Extreme Odor
    if "fresh" in low and ("xtrm" in low or "extreme" in low) and ("odor" in low or "step" in low):
        return "fresh step extreme odor control cat litter"

    # AMC Pop Kettle Butter Popcorn
    if ("amc" in low or low.startswith("amc ")) and ("ktl" in low or "kettle" in low or "pop" in low):
        return "amc kettle butter popcorn"

    # Buitoni 5 Cheese Tortellini
    if ("buit" in low or "buitoni" in low) and ("tortel" in low or "tortellini" in low or "chse" in low or "cheese" in low):
        return "buitoni 5 cheese tortellini"

    # Cremo Barber Grade Italian Bergamot
    if "cremo" in low and ("berg" in low or "bergamot" in low or "ital" in low or "bw" in low or "barber" in low):
        return "cremo barber grade italian bergamot"

    # Ben's Original Flavored Grain Large Wild Rice
    if ("ben" in low or "bens" in low) and ("wld" in low or "wild" in low or "fg" in low):
        return "ben's original large wild rice"

    # Great Value Brown Sugar Bacon
    if ("gw" in low or "great value" in low) and ("bac" in low or "bacon" in low) and ("brn" in low or "brown" in low or "sugar" in low):
        return "great value brown sugar bacon"

    # Boneless Chicken Breast
    if ("bnls" in low or "boneless" in low) and ("chick" in low or "chk" in low or "chicken" in low) and ("brst" in low or "breast" in low):
        return "boneless chicken breast"

    # Granny Smith Apples
    if "apple" in low and ("smit" in low or "smith" in low or "granny" in low or "grny" in low):
        return "granny smith apples"

    # Snyder's Snap Pretzels
    if ("snyder" in low or "snyd" in low) and ("snap" in low or "10ct" in low or "prtz" in low or "pretzels" in low):
        return "snyder's of hanover snap pretzels"  # Gemini must not add "Pretzels" again

    # Thomas' Plain Bagels
    if "thomas" in low and ("bagel" in low or "bgl" in low or "plain" in low or "pln" in low):
        return "thomas' plain bagels"

    # Rao's Marinara
    if ("rao" in low) and ("marinara" in low or "sauce" in low):
        return "rao's homemade marinara sauce"

    # Nutrl Vodka
    if "nutrl" in low or ("nutr" in low and "vodka" in low):
        return "nutrl vodka seltzer"

    # Publix Pasta (keep simple — it IS just publix pasta)
    # Wishbone Ranch Dressing
    if "wishbone" in low and "ranch" in low:
        return "wishbone ranch dressing"

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
    # Remove any consecutive duplicate words (e.g. "Snap Pretzels Pretzels" -> "Snap Pretzels")
    pretty = re.sub(r'\b(\w+)(\s+\1)+\b', r'\1', pretty, flags=re.IGNORECASE).strip()
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

    # Debug: log dropped lines so we can see what's being filtered
    _watch = {"frsh", "tide", "sbr", "kh", "pbx", "sharp", "savory", "buff", "marind", "spring", "xtrm", "renewal"}
    for _dl in _dropped_lines:
        _dl_low = (_dl.get("line") or "").lower()
        if any(w in _dl_low for w in _watch):
            print(f"[DROP] stage={_dl.get('stage')} line={_dl.get('line')!r} cleaned={_dl.get('cleaned')!r}", flush=True)
    for _c in candidates:
        _cl_low = (_c.cleaned_line or "").lower()
        if any(w in _cl_low for w in _watch):
            print(f"[CANDIDATE] {_c.cleaned_line!r}", flush=True)

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
            "image_url": _image_url_for_item(base_url, final_name, category),
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
                it["image_url"] = _image_url_for_item(base_url, enriched_final, it.get("category"))

        for it in items:
            it.pop("_raw_line", None)
            it.pop("_name_cleaned", None)
            it.pop("_expanded", None)

    if items:
        enriched = await enrich_items_with_ai(items)
        for it, info in zip(items, enriched):
            it["expires_in_days"] = info.get("expires_in_days")
            it["storage"] = info.get("storage")
            it["shelf_life_by_storage"] = info.get("shelf_life_by_storage")
            if info.get("category"):
                it["category"] = info["category"]
            # Pass through food_category (display subcategory) from Gemini
            if info.get("food_category"):
                it["food_category"] = info["food_category"]
            # Pass through photo_query from Gemini for smarter image lookup
            if info.get("photo_query"):
                it["photo_query"] = info["photo_query"]
            # Apply Gemini's full name if available
            gemini_name = (info.get("full_name") or "").strip()
            if gemini_name and len(gemini_name) >= 3:
                it["name"] = gemini_name
                # Include photo_query in image URL so the /image endpoint uses Gemini's exact query
                it["image_url"] = _image_url_for_item(base_url, gemini_name, it.get("category"), it.get("photo_query"))

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
            "image_url": _image_url_for_item(base_url, final_name, category),
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

    expanded_names = await asyncio.gather(*[_expand_receipt_name(i.name) for i in items])

    # Hardcoded UPC table for common household items — no external call needed
    _HOUSEHOLD_UPC_TABLE: Dict[str, str] = {
        # Paper Towels
        "bounty paper towels":                    "0030772056301",
        "bounty select-a-size paper towels":      "0030772058749",
        "bounty essentials paper towels":         "0030772058039",
        "paper towels":                           "0030772056301",
        "viva paper towels":                      "0054000000877",
        "brawny paper towels":                    "0057000001376",
        # Toilet Paper
        "charmin ultra soft toilet paper":        "0030772012282",
        "charmin ultra strong toilet paper":      "0030772012299",
        "charmin toilet paper":                   "0030772012282",
        "toilet paper":                           "0030772012282",
        "cottonelle toilet paper":                "0035000517012",
        "scott toilet paper":                     "0054000000860",
        "angel soft toilet paper":                "0057000001383",
        # Laundry
        "tide original laundry detergent":        "0037000259831",
        "tide pods laundry detergent":            "0037000780175",
        "tide laundry detergent":                 "0037000259831",
        "laundry detergent":                      "0037000259831",
        "gain original laundry detergent":        "0037000811473",
        "gain laundry detergent":                 "0037000811473",
        "arm and hammer laundry detergent":       "0033200020967",
        "persil laundry detergent":               "0046500014002",
        "all laundry detergent":                  "0072613440009",
        "dreft laundry detergent":                "0037000869504",
        "downy fabric softener":                  "0037000888192",
        "bounce dryer sheets":                    "0037000315902",
        "dryer sheets":                           "0037000315902",
        # Dish
        "dawn dish soap":                         "0037000951919",
        "dawn original dish soap":                "0037000951919",
        "dish soap":                              "0037000951919",
        "cascade dish pods":                      "0037000951230",
        "cascade platinum pods":                  "0037000914358",
        "cascade dishwasher detergent":           "0037000951230",
        "finish dishwasher pods":                 "0051700897706",
        # Cleaning
        "lysol disinfectant spray":               "0019200804602",
        "lysol wipes":                            "0019200804619",
        "clorox disinfecting wipes":              "0044600322354",
        "clorox bleach":                          "0044600301442",
        "mr clean multi surface cleaner":         "0037000901099",
        "windex glass cleaner":                   "0046500013319",
        "febreze air freshener":                  "0037000900849",
        "air freshener":                          "0037000900849",
        "swiffer sweeper refills":                "0037000807612",
        "swiffer wet jet pads":                   "0037000807841",
        # Trash Bags
        "glad trash bags":                        "0012587813552",
        "glad forceflex trash bags":              "0012587827740",
        "hefty ultra strong trash bags":          "0018600016511",
        "hefty trash bags":                       "0018600016511",
        "trash bags":                             "0012587813552",
        # Cat Litter
        "fresh step cat litter":                  "0044600313697",
        "fresh step extreme odor control cat litter": "0044600313697",
        "tidy cats cat litter":                   "0070230111013",
        "arm and hammer cat litter":              "0033200014591",
        "cat litter":                             "0044600313697",
        # Dog
        "pedigree dog food":                      "0023100113682",
        "purina dog food":                        "0017800129527",
        "milk bone dog treats":                   "0070330020206",
        # Toothpaste / Oral Care — brand-specific only, generic falls back to name search
        "colgate toothpaste":                     "0035000042606",
        "crest toothpaste":                       "0037000283249",
        "oral b toothbrush":                      "0300410096051",
        "listerine mouthwash":                    "0312547307013",
        # Deodorant — brand-specific only
        "old spice deodorant":                    "0037000918608",
        "secret deodorant":                       "0037000457282",
        "dove deodorant":                         "0011111093919",
        "degree deodorant":                       "0011111007413",
        # Shampoo / Body — brand-specific only
        "head and shoulders shampoo":             "0037000350729",
        "pantene shampoo":                        "0037000716365",
        "dove body wash":                         "0011111081046",
        "irish spring body wash":                 "0035000594327",
        "dove bar soap":                          "0011111093988",
        # Razors — brand-specific only
        "gillette razor":                         "0047400258484",
        "gillette mach3 razor":                   "0047400258484",
        "venus razor":                            "0047400258507",
        # Hand Soap / Sanitizer — brand-specific only
        "softsoap hand soap":                     "0037000200475",
        "purell hand sanitizer":                  "0045865200001",
        # Tissue / Napkins — brand-specific only
        "kleenex tissues":                        "0036000291452",
        "puffs tissues":                          "0037000100171",
        "bounty napkins":                         "0030772052051",
        # Plastic Wrap / Bags
        "ziploc bags":                            "0025700011904",
        "ziploc sandwich bags":                   "0025700011904",
        "glad ziploc bags":                       "0025700011904",
        "plastic wrap":                           "0025700011805",
        "saran wrap":                             "0025700011805",
        "aluminum foil":                          "0025700061276",
        "reynolds wrap aluminum foil":            "0025700061276",
        # Batteries
        "duracell batteries":                     "0041333038261",
        "energizer batteries":                    "0039800020321",
        "batteries":                              "0041333038261",
        # Light Bulbs
        "ge light bulbs":                         "0043168606851",
        "light bulbs":                            "0043168606851",
    }

    def _lookup_household_upc(name: str) -> Optional[str]:
        """Look up UPC from hardcoded table using fuzzy name matching."""
        name_lower = name.lower().strip()
        # Exact match first
        if name_lower in _HOUSEHOLD_UPC_TABLE:
            return _HOUSEHOLD_UPC_TABLE[name_lower]
        # Partial match — check if any key is contained in the name or vice versa
        # Sort by length descending so more specific keys win
        for key in sorted(_HOUSEHOLD_UPC_TABLE.keys(), key=len, reverse=True):
            if key in name_lower or name_lower in key:
                return _HOUSEHOLD_UPC_TABLE[key]
        return None

    # Resolve UPC for each item
    async def _resolve_upc(item: InstacartLineItem, expanded_name: str) -> Optional[str]:
        # If UPC was sent directly from the app, use it
        if item.upc and item.upc.strip():
            return item.upc.strip()
        # Skip for food items — Instacart already matches those well by name
        if (item.category or "").lower() == "food":
            return None
        # Look up from hardcoded table
        upc = _lookup_household_upc(expanded_name)
        if upc:
            print(f"[INSTACART UPC] '{expanded_name}' -> {upc} (hardcoded table)", flush=True)
        else:
            print(f"[INSTACART UPC] '{expanded_name}' -> no match in table", flush=True)
        return upc

    upcs = await asyncio.gather(*[_resolve_upc(items[idx], expanded_names[idx]) for idx in range(len(items))])

    def _build_line_item(idx: int) -> dict:
        entry: dict = {
            "name": expanded_names[idx],
            "quantity": items[idx].quantity,
            "unit": items[idx].unit,
            "line_item_measurements": [{"quantity": items[idx].quantity, "unit": items[idx].unit}]
        }
        if upcs[idx]:
            entry["upcs"] = [upcs[idx]]
        print(f"[INSTACART LINE ITEM] {entry}", flush=True)
        return entry

    payload = {
        "title": req.title,
        "link_type": "shopping_list",
        "line_items": [_build_line_item(idx) for idx in range(len(items))],
    }
    print(f"[INSTACART PAYLOAD] {payload}", flush=True)

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


async def _kroger_get_token() -> Optional[str]:
    global _kroger_token, _kroger_token_expiry
    if not KROGER_CLIENT_ID or not KROGER_CLIENT_SECRET:
        return None
    if _kroger_token and time.time() < _kroger_token_expiry:
        return _kroger_token
    try:
        import base64 as _b64
        creds = _b64.b64encode(f"{KROGER_CLIENT_ID}:{KROGER_CLIENT_SECRET}".encode()).decode()
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.post(
                "https://api.kroger.com/v1/connect/oauth2/token",
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Authorization": f"Basic {creds}",
                },
                data="grant_type=client_credentials&scope=product.compact",
            )
            if r.status_code == 200:
                data = r.json()
                _kroger_token = data.get("access_token")
                expires_in = int(data.get("expires_in", 1800))
                _kroger_token_expiry = time.time() + expires_in - 60
                print(f"[KROGER] got token, expires in {expires_in}s", flush=True)
                return _kroger_token
            else:
                print(f"[KROGER] token error status={r.status_code} body={r.text}", flush=True)
                return None
    except Exception as e:
        print(f"[KROGER] token exception: {e}", flush=True)
        return None


def _is_good_product_image(url: str) -> bool:
    """
    Returns True if the URL looks like a front-of-pack product shot.
    Rejects URLs that hint at nutrition labels, back-of-pack, or generic lifestyle shots.
    """
    if not url:
        return False
    url_lower = url.lower()
    # Reject known bad URL patterns
    bad_patterns = [
        "nutrition", "ingredient", "label",
        "istockphoto", "shutterstock", "gettyimages", "dreamstime",
        "alamy", "depositphotos",
        # Kroger store brand (mascot cartoon guy) — reject and fall through to Unsplash
        "/0001111",
    ]
    for pat in bad_patterns:
        if pat in url_lower:
            return False
    return True


async def _open_food_facts_image(name: str) -> Optional[str]:
    """Search Open Food Facts for a clean front-of-pack product image."""
    try:
        search_query = urllib.parse.quote(name.strip())
        off_url = (
            f"https://world.openfoodfacts.org/cgi/search.pl"
            f"?search_terms={search_query}&search_simple=1&action=process"
            f"&json=1&page_size=5&fields=product_name,brands,image_front_url,image_url"
        )
        async with httpx.AsyncClient(
            timeout=8.0,
            follow_redirects=True,
            headers={"User-Agent": "ShelfLife/1.0 (contact@shelflife.app)"},
        ) as oclient:
            oresp = await oclient.get(off_url)
            if oresp.status_code == 200:
                odata = oresp.json()
                products = odata.get("products", [])
                for product in products:
                    # Prefer image_front_url — that's the front of the package
                    candidate = product.get("image_front_url") or product.get("image_url")
                    if candidate and candidate.startswith("http") and _is_good_product_image(candidate):
                        # Verify reachable
                        try:
                            async with httpx.AsyncClient(timeout=5.0) as vc:
                                vr = await vc.head(candidate)
                                if vr.status_code == 200:
                                    print(f"[OFF] found image for '{name}': {candidate}", flush=True)
                                    return candidate
                        except Exception:
                            pass
                print(f"[OFF] no usable image found for '{name}'", flush=True)
            else:
                print(f"[OFF] error for '{name}' status={oresp.status_code}", flush=True)
    except Exception as e:
        print(f"[OFF IMAGE] error for '{name}': {e}", flush=True)
    return None


def _simplify_for_unsplash(name: str) -> str:
    """Strip brand names, sizes, weights, and noise so Unsplash gets a clean food keyword."""
    import re
    s = name.strip()

    # Remove common brand names
    brands = [
        "kraft", "heinz", "campbell\'s", "campbells", "general mills", "kellogg\'s", "kelloggs",
        "pepperidge farm", "nabisco", "frito-lay", "lay\'s", "lays", "doritos", "cheetos",
        "pringles", "oreo", "chips ahoy", "ritz", "triscuit", "wheat thins",
        "quaker", "nature valley", "kind", "clif", "luna",
        "coca-cola", "coke", "pepsi", "sprite", "fanta", "dr pepper", "mountain dew",
        "gatorade", "powerade", "red bull", "monster",
        "tropicana", "minute maid", "simply", "ocean spray", "dole",
        "chobani", "fage", "yoplait", "oikos", "siggi\'s", "siggis",
        "sargento", "tillamook", "cabot", "boar\'s head", "boars head",
        "oscar mayer", "hillshire farm", "jimmy dean", "tyson", "perdue",
        "birds eye", "green giant", "del monte", "bush\'s", "bushs",
        "hunts", "hunt\'s", "muir glen", "prego", "ragu", "newman\'s own", "newmans own",
        "annie\'s", "annies", "amy\'s", "amys", "cascadian farm",
        "stonyfield", "horizon", "organic valley",
        "land o lakes", "land o\'lakes", "breakstone\'s", "breakstones",
        "philadelphia", "velveeta", "laughing cow",
        "hellmann\'s", "hellmanns", "best foods", "duke\'s", "dukes",
        "french\'s", "frenchs", "gulden\'s", "guldens",
        "wishbone", "ken\'s", "kens", "hidden valley",
        "publix", "great value", "store brand", "generic", "signature select",
        "365", "simple truth", "open nature", "o organics",
        "progresso", "swanson", "lipton", "knorr", "idahoan",
        "uncle ben\'s", "uncle bens", "ben\'s original", "bens original",
        "rice-a-roni", "pasta roni", "hamburger helper", "betty crocker",
        "pillsbury", "jiffy", "bisquick",
        "jif", "skippy", "peter pan", "smucker\'s", "smuckers",
        "nutella", "biscoff",
        "domino", "c&h", "bob\'s red mill", "bobs red mill",
        "mccormick", "lawry\'s", "lawrys", "old bay", "tone\'s", "tones",
        "barilla", "ronzoni", "mueller\'s", "muellers", "de cecco",
        "bertolli", "colavita", "filippo berio",
        "dannon", "activia",
        "bolthouse", "naked juice", "odwalla",
        "ben & jerry\'s", "ben and jerrys", "haagen-dazs", "breyers", "dreyer\'s", "dreyers",
        "talenti", "arctic zero",
        "alexia", "ore-ida",
        "stouffer\'s", "stouffers", "lean cuisine", "healthy choice", "marie callender\'s",
        "digorno", "digiorno", "red baron", "tombstone",
        "kashi", "special k", "cheerios", "frosted flakes", "lucky charms",
        "cap\'n crunch", "capn crunch", "cocoa puffs", "froot loops", "fruit loops",
        "honey bunches", "life cereal",
        "thomas\'", "arnold", "pepperidge",
        "wonder", "nature\'s own", "natures own", "dave\'s killer", "daves killer",
        "mission", "old el paso", "taco bell", "ortega",
        "pam", "crisco", "wesson",
        "splenda", "truvia", "equal", "sweet\'n low",
        "crystal light", "mio", "true lemon",
        "emerald", "planters", "blue diamond", "wonderful",
        "sun-maid", "sunmaid", "craisins",
    ]

    s_lower = s.lower()
    for brand in brands:
        pattern = re.compile(re.escape(brand), re.IGNORECASE)
        s_lower = pattern.sub("", s_lower)

    # Remove size/weight/count patterns like 12oz, 2lb, 16 fl oz, 6ct, 32 count, etc.
    s_lower = re.sub(r'\b\d+(\.\d+)?\s*(oz|fl\.?\s*oz|lb|lbs|g|kg|ml|l|ct|count|pk|pack|piece|pcs|slices?|serving)s?\b', '', s_lower, flags=re.IGNORECASE)

    # Remove standalone numbers
    s_lower = re.sub(r'\b\d+\b', '', s_lower)

    # Remove filler words
    fillers = ["original", "classic", "premium", "value", "family size", "mega", "super",
               "new", "improved", "natural", "organic", "extra", "ultra",
               "select", "choice", "grade a", "grade b", "lite", "low fat",
               "fat free", "sugar free", "reduced", "less", "more", "plus",
               "&", "and", "with", "in", "the", "a", "an", "of"]
    for filler in fillers:
        s_lower = re.sub(r'\b' + re.escape(filler) + r'\b', '', s_lower, flags=re.IGNORECASE)

    # Clean up punctuation and extra spaces
    s_lower = re.sub(r'[^a-z\s]', ' ', s_lower)
    s_lower = re.sub(r'\s+', ' ', s_lower).strip()

    # If we stripped everything, fall back to last 1-2 meaningful words of original
    if not s_lower or len(s_lower) < 3:
        words = [w for w in name.lower().split() if len(w) > 3]
        s_lower = " ".join(words[-2:]) if words else name.lower()

    print(f"[UNSPLASH SIMPLIFY] '{name}' -> '{s_lower}'", flush=True)
    return s_lower


# ── Unsplash search-type overrides ────────────────────────────────────────────
# Maps a simplified name keyword to a precise Unsplash search query.
# This bypasses the generic suffix logic for items we know need a specific query.
_UNSPLASH_QUERY_OVERRIDES: dict = {

    # ── DAIRY & EGGS ─────────────────────────────────────────────────────────
    "half and half":           "half and half creamer carton bottle white background",
    "half-and-half":           "half and half creamer carton bottle white background",
    "heavy cream":             "heavy whipping cream carton white background",
    "whipping cream":          "heavy whipping cream carton white background",
    "heavy whipping cream":    "heavy whipping cream carton white background",
    "sour cream":              "sour cream tub container white background",
    "cream cheese":            "cream cheese block package white background",
    "whipped cream cheese":    "cream cheese tub container white background",
    "cottage cheese":          "cottage cheese container tub white background",
    "ricotta":                 "ricotta cheese container white background",
    "butter":                  "butter sticks wrapped package white background",
    "unsalted butter":         "butter sticks wrapped package white background",
    "salted butter":           "butter sticks wrapped package white background",
    "ghee":                    "ghee jar glass white background",
    "milk":                    "milk gallon jug white background",
    "whole milk":              "whole milk gallon jug white background",
    "2% milk":                 "milk gallon jug white background",
    "skim milk":               "skim milk carton jug white background",
    "1% milk":                 "milk carton white background",
    "chocolate milk":          "chocolate milk carton bottle white background",
    "almond milk":             "almond milk carton white background",
    "oat milk":                "oat milk carton white background",
    "soy milk":                "soy milk carton white background",
    "coconut milk":            "coconut milk can carton white background",
    "buttermilk":              "buttermilk carton bottle white background",
    "eggnog":                  "eggnog carton white background",
    "yogurt":                  "yogurt container cup white background",
    "greek yogurt":            "greek yogurt container cup white background",
    "vanilla yogurt":          "vanilla yogurt cup container white background",
    "strawberry yogurt":       "strawberry yogurt cup container white background",
    "blueberry yogurt":        "blueberry yogurt cup container white background",
    "drinkable yogurt":        "yogurt drink bottle white background",
    "kefir":                   "kefir bottle white background",
    "cheese":                  "block cheese package white background",
    "cheddar":                 "cheddar cheese block package white background",
    "cheddar cheese":          "cheddar cheese block package white background",
    "sharp cheddar":           "sharp cheddar cheese block package white background",
    "mild cheddar":            "mild cheddar cheese block package white background",
    "shredded cheddar":        "shredded cheddar cheese bag package white background",
    "shredded cheese":         "shredded cheese bag package white background",
    "shredded mozzarella":     "shredded mozzarella cheese bag package white background",
    "shredded mexican blend":  "shredded mexican blend cheese bag white background",
    "parmesan":                "parmesan cheese wedge block white background",
    "parmesan cheese":         "parmesan cheese wedge package white background",
    "parmesan wedge":          "parmesan cheese wedge block white background",
    "grated parmesan":         "grated parmesan cheese shaker white background",
    "mozzarella":              "fresh mozzarella cheese ball package white background",
    "fresh mozzarella":        "fresh mozzarella cheese ball package white background",
    "string cheese":           "string cheese sticks package white background",
    "swiss cheese":            "swiss cheese slices package white background",
    "provolone":               "provolone cheese slices package white background",
    "pepper jack":             "pepper jack cheese block package white background",
    "colby jack":              "colby jack cheese block package white background",
    "muenster":                "muenster cheese slices package white background",
    "brie":                    "brie cheese wheel package white background",
    "feta":                    "feta cheese block crumbled package white background",
    "feta cheese":             "feta cheese block crumbled package white background",
    "blue cheese":             "blue cheese crumbled package white background",
    "gouda":                   "gouda cheese wheel package white background",
    "american cheese":         "american cheese slices package white background",
    "velveeta":                "velveeta cheese block package white background",
    "eggs":                    "eggs carton closed white background",
    "large eggs":              "eggs carton closed white background",
    "egg whites":              "liquid egg whites carton white background",
    "hard boiled eggs":        "hard boiled eggs package white background",

    # ── MEAT & POULTRY (RAW) ─────────────────────────────────────────────────
    "chicken breast":          "raw chicken breast package tray white background",
    "boneless chicken breast": "raw chicken breast package tray white background",
    "boneless skinless chicken breast": "raw chicken breast package tray white background",
    "chicken thighs":          "raw chicken thighs package tray white background",
    "chicken wings":           "raw chicken wings package tray white background",
    "chicken drumsticks":      "raw chicken drumsticks package tray white background",
    "whole chicken":           "whole raw chicken package white background",
    "ground chicken":          "raw ground chicken package tray white background",
    "ground beef":             "raw ground beef package tray white background",
    "ground chuck":            "raw ground beef package tray white background",
    "ground turkey":           "raw ground turkey package tray white background",
    "ground pork":             "raw ground pork package tray white background",
    "steak":                   "raw beef steak package white background",
    "ribeye":                  "raw ribeye steak package white background",
    "sirloin":                 "raw sirloin steak package white background",
    "ny strip":                "raw new york strip steak package white background",
    "filet mignon":            "raw filet mignon steak package white background",
    "chuck roast":             "raw chuck roast beef package white background",
    "beef roast":              "raw beef roast package white background",
    "brisket":                 "raw beef brisket package white background",
    "pork chops":              "raw pork chops package tray white background",
    "pork tenderloin":         "raw pork tenderloin package white background",
    "pork ribs":               "raw pork ribs package white background",
    "baby back ribs":          "raw baby back ribs package white background",
    "lamb chops":              "raw lamb chops package white background",
    "turkey":                  "whole turkey raw package white background",
    "ground beef 80/20":       "raw ground beef package tray white background",
    "ground beef 85/15":       "raw ground beef package tray white background",
    "ground beef 90/10":       "raw ground beef package tray white background",

    # ── MEAT (PROCESSED / DELI) ──────────────────────────────────────────────
    "bacon":                   "bacon strips package sealed white background",
    "brown sugar bacon":       "bacon strips package sealed white background",
    "turkey bacon":            "turkey bacon package sealed white background",
    "canadian bacon":          "canadian bacon package white background",
    "pancetta":                "pancetta package white background",
    "prosciutto":              "prosciutto package sliced white background",
    "salami":                  "salami package sliced white background",
    "pepperoni":               "pepperoni package sliced white background",
    "ham":                     "deli ham package sliced white background",
    "deli ham":                "deli ham package sliced white background",
    "deli turkey":             "deli turkey package sliced white background",
    "deli chicken":            "deli chicken package sliced white background",
    "deli roast beef":         "deli roast beef package sliced white background",
    "bologna":                 "bologna package sliced white background",
    "hot dogs":                "hot dogs package sealed white background",
    "sausage":                 "sausage links package sealed white background",
    "italian sausage":         "italian sausage links package white background",
    "bratwurst":               "bratwurst sausage package white background",
    "kielbasa":                "kielbasa sausage package white background",
    "andouille":               "andouille sausage package white background",
    "smoked sausage":          "smoked sausage package white background",
    "rotisserie chicken":      "rotisserie chicken whole cooked package white background",
    "deli rotisserie chicken": "rotisserie chicken whole cooked package white background",

    # ── SEAFOOD ──────────────────────────────────────────────────────────────
    "salmon":                  "raw salmon fillet package tray white background",
    "salmon fillet":           "raw salmon fillet package tray white background",
    "tilapia":                 "raw tilapia fillet package tray white background",
    "cod":                     "raw cod fillet package tray white background",
    "halibut":                 "raw halibut fillet package tray white background",
    "mahi mahi":               "raw mahi mahi fillet package tray white background",
    "tuna steak":              "raw tuna steak package tray white background",
    "shrimp":                  "raw shrimp package bag white background",
    "frozen shrimp":           "frozen shrimp bag package white background",
    "scallops":                "raw scallops package tray white background",
    "lobster tail":            "lobster tail raw package white background",
    "crab legs":               "crab legs package white background",
    "imitation crab":          "imitation crab package white background",
    "smoked salmon":           "smoked salmon package sealed white background",
    "lox":                     "smoked salmon lox package white background",
    "tuna":                    "canned tuna can white background",
    "canned tuna":             "canned tuna can white background",
    "canned salmon":           "canned salmon can white background",
    "sardines":                "canned sardines tin white background",
    "anchovies":               "canned anchovies tin white background",

    # ── PRODUCE — VEGETABLES ─────────────────────────────────────────────────
    "broccoli":                "fresh broccoli head white background",
    "broccoli florets":        "fresh broccoli florets bag white background",
    "cauliflower":             "fresh cauliflower head white background",
    "brussels sprouts":        "fresh brussels sprouts bag white background",
    "green beans":             "fresh green beans bag white background",
    "snap peas":               "fresh snap peas bag white background",
    "asparagus":               "fresh asparagus bunch white background",
    "spinach":                 "fresh baby spinach bag container white background",
    "baby spinach":            "fresh baby spinach bag container white background",
    "kale":                    "fresh kale bunch white background",
    "romaine":                 "romaine lettuce hearts package white background",
    "romaine lettuce":         "romaine lettuce hearts package white background",
    "iceberg lettuce":         "iceberg lettuce head white background",
    "lettuce":                 "fresh lettuce head white background",
    "spring mix":              "spring mix salad bag white background",
    "salad kit":               "salad kit bag package white background",
    "caesar salad kit":        "caesar salad kit bag package white background",
    "mixed greens":            "mixed greens bag package white background",
    "arugula":                 "fresh arugula bag white background",
    "tomatoes":                "fresh tomatoes white background",
    "tomato":                  "fresh tomato white background",
    "grape tomatoes":          "fresh grape tomatoes container white background",
    "cherry tomatoes":         "fresh cherry tomatoes container white background",
    "roma tomatoes":           "fresh roma tomatoes white background",
    "bell pepper":             "fresh bell pepper white background",
    "red bell pepper":         "fresh red bell pepper white background",
    "green bell pepper":       "fresh green bell pepper white background",
    "yellow bell pepper":      "fresh yellow bell pepper white background",
    "orange bell pepper":      "fresh orange bell pepper white background",
    "jalapeño":                "fresh jalapeno pepper white background",
    "jalapeno":                "fresh jalapeno pepper white background",
    "poblano":                 "fresh poblano pepper white background",
    "onion":                   "fresh yellow onion white background",
    "yellow onion":            "fresh yellow onion white background",
    "red onion":               "fresh red onion white background",
    "white onion":             "fresh white onion white background",
    "green onions":            "fresh green onions scallions white background",
    "scallions":               "fresh scallions green onions white background",
    "garlic":                  "fresh garlic bulb white background",
    "minced garlic":           "minced garlic jar white background",
    "avocado":                 "fresh avocado whole white background",
    "cucumber":                "fresh cucumber white background",
    "zucchini":                "fresh zucchini white background",
    "celery":                  "fresh celery stalks bunch white background",
    "carrots":                 "fresh carrots bunch bag white background",
    "baby carrots":            "fresh baby carrots bag white background",
    "potatoes":                "fresh potatoes bag white background",
    "russet potatoes":         "fresh russet potatoes bag white background",
    "red potatoes":            "fresh red potatoes bag white background",
    "yukon gold potatoes":     "fresh yukon gold potatoes bag white background",
    "sweet potato":            "fresh sweet potato white background",
    "sweet potatoes":          "fresh sweet potatoes bag white background",
    "mushrooms":               "fresh mushrooms package white background",
    "white mushrooms":         "fresh white mushrooms package white background",
    "baby bella mushrooms":    "fresh baby bella mushrooms package white background",
    "portobello mushrooms":    "fresh portobello mushrooms package white background",
    "corn":                    "fresh corn on the cob white background",
    "corn on the cob":         "fresh corn cob white background",
    "edamame":                 "fresh edamame bag package white background",
    "butternut squash":        "fresh butternut squash whole white background",
    "acorn squash":            "fresh acorn squash whole white background",
    "cabbage":                 "fresh cabbage head white background",
    "bok choy":                "fresh bok choy white background",
    "fennel":                  "fresh fennel bulb white background",
    "leeks":                   "fresh leeks white background",
    "radishes":                "fresh radishes bunch white background",
    "artichoke":               "fresh artichoke white background",
    "broccolini":              "fresh broccolini bunch white background",

    # ── PRODUCE — FRUIT ──────────────────────────────────────────────────────
    "apples":                  "fresh apples bag white background",
    "apple":                   "fresh apple white background",
    "granny smith apples":     "granny smith apples bag white background",
    "honeycrisp apples":       "honeycrisp apples bag white background",
    "gala apples":             "gala apples bag white background",
    "bananas":                 "fresh bananas bunch white background",
    "banana":                  "fresh banana white background",
    "oranges":                 "fresh navel oranges bag white background",
    "navel oranges":           "fresh navel oranges bag white background",
    "mandarins":               "fresh mandarin oranges bag white background",
    "clementines":             "fresh clementines bag white background",
    "grapefruit":              "fresh grapefruit white background",
    "lemons":                  "fresh lemons bag white background",
    "limes":                   "fresh limes bag white background",
    "strawberries":            "fresh strawberries container white background",
    "blueberries":             "fresh blueberries container white background",
    "raspberries":             "fresh raspberries container white background",
    "blackberries":            "fresh blackberries container white background",
    "grapes":                  "fresh grapes bag white background",
    "red grapes":              "fresh red grapes bag white background",
    "green grapes":            "fresh green grapes bag white background",
    "cotton candy grapes":     "fresh cotton candy grapes bag white background",
    "peaches":                 "fresh peaches white background",
    "nectarines":              "fresh nectarines white background",
    "plums":                   "fresh plums white background",
    "mango":                   "fresh mango white background",
    "pineapple":               "fresh pineapple whole white background",
    "watermelon":              "fresh whole watermelon white background",
    "cantaloupe":              "fresh cantaloupe melon whole white background",
    "honeydew":                "fresh honeydew melon whole white background",
    "kiwi":                    "fresh kiwi fruit white background",
    "cherries":                "fresh cherries bag white background",
    "pomegranate":             "fresh pomegranate whole white background",
    "pears":                   "fresh pears bag white background",
    "papaya":                  "fresh papaya white background",

    # ── FRESH HERBS ──────────────────────────────────────────────────────────
    "basil":                   "fresh basil bunch white background",
    "cilantro":                "fresh cilantro bunch white background",
    "parsley":                 "fresh parsley bunch white background",
    "rosemary":                "fresh rosemary bunch white background",
    "thyme":                   "fresh thyme bunch white background",
    "mint":                    "fresh mint leaves bunch white background",
    "dill":                    "fresh dill bunch white background",
    "chives":                  "fresh chives bunch white background",

    # ── BEVERAGES ────────────────────────────────────────────────────────────
    "water":                   "bottled water plastic bottle white background",
    "spring water":            "spring water plastic bottle white background",
    "sparkling water":         "sparkling water can bottle white background",
    "lacroix":                 "lacroix sparkling water can white background",
    "bottled water":           "bottled water plastic bottle white background",
    "distilled water":         "distilled water gallon jug white background",
    "purified water":          "purified water gallon jug white background",
    "schnucks water":          "bottled water plastic bottle white background",
    "dasani":                  "dasani water bottle white background",
    "smartwater":              "smartwater bottle white background",
    "fiji water":              "fiji water bottle white background",
    "evian":                   "evian water bottle white background",
    "orange juice":            "orange juice carton bottle white background",
    "apple juice":             "apple juice carton bottle white background",
    "grape juice":             "grape juice bottle white background",
    "cranberry juice":         "cranberry juice bottle white background",
    "lemonade":                "lemonade carton bottle white background",
    "tropical punch":          "juice drink bottle white background",
    "fruit punch":             "fruit punch bottle white background",
    "coffee":                  "coffee bag sealed product white background",
    "ground coffee":           "ground coffee bag sealed white background",
    "coffee beans":            "whole bean coffee bag sealed white background",
    "instant coffee":          "instant coffee jar white background",
    "cold brew":               "cold brew coffee bottle white background",
    "tea":                     "tea bags box package white background",
    "green tea":               "green tea bags box white background",
    "black tea":               "black tea bags box white background",
    "herbal tea":              "herbal tea bags box white background",
    "soda":                    "soda can bottle white background",
    "coca cola":               "coca cola can soda white background",
    "pepsi":                   "pepsi can soda white background",
    "sprite":                  "sprite can soda white background",
    "dr pepper":               "dr pepper can soda white background",
    "mountain dew":            "mountain dew can soda white background",
    "ginger ale":              "ginger ale can bottle white background",
    "root beer":               "root beer can bottle white background",
    "sports drink":            "sports drink bottle gatorade white background",
    "gatorade":                "gatorade sports drink bottle white background",
    "powerade":                "powerade sports drink bottle white background",
    "energy drink":            "energy drink can white background",
    "red bull":                "red bull energy drink can white background",
    "monster energy":          "monster energy drink can white background",
    "kombucha":                "kombucha bottle white background",
    "coconut water":           "coconut water carton bottle white background",
    "aloe water":              "aloe water bottle white background",
    "beer":                    "beer bottle can white background",
    "wine":                    "wine bottle white background",
    "red wine":                "red wine bottle white background",
    "white wine":              "white wine bottle white background",
    "champagne":               "champagne bottle white background",
    "prosecco":                "prosecco bottle white background",
    "hard seltzer":            "hard seltzer can white background",
    "white claw":              "white claw hard seltzer can white background",
    "truly hard seltzer":      "truly hard seltzer can white background",
    "vodka seltzer":           "vodka seltzer can white background",
    "nutrl vodka seltzer":     "nutrl vodka seltzer can white background",
    "liquor":                  "liquor bottle spirits white background",
    "vodka":                   "vodka bottle white background",
    "whiskey":                 "whiskey bottle white background",
    "bourbon":                 "bourbon whiskey bottle white background",
    "tequila":                 "tequila bottle white background",
    "rum":                     "rum bottle white background",
    "gin":                     "gin bottle white background",
    "jack daniels":            "jack daniels whiskey bottle white background",
    "protein shake":           "protein shake drink bottle white background",

    # ── BREAD & BAKERY ───────────────────────────────────────────────────────
    "bread":                   "bread loaf sealed bag white background",
    "white bread":             "white bread loaf sealed bag white background",
    "wheat bread":             "wheat bread loaf sealed bag white background",
    "whole wheat bread":       "whole wheat bread loaf bag white background",
    "multigrain bread":        "multigrain bread loaf bag white background",
    "sourdough":               "sourdough bread loaf white background",
    "sourdough bread":         "sourdough bread loaf white background",
    "bagels":                  "bagels package bag white background",
    "plain bagels":            "plain bagels package bag white background",
    "everything bagels":       "everything bagels package bag white background",
    "english muffins":         "english muffins package white background",
    "flour tortillas":         "flour tortillas package bag white background",
    "corn tortillas":          "corn tortillas package bag white background",
    "pita bread":              "pita bread package bag white background",
    "naan":                    "naan bread package white background",
    "dinner rolls":            "dinner rolls package bag white background",
    "hawaiian rolls":          "king's hawaiian rolls package white background",
    "pretzel buns":            "pretzel buns package bag white background",
    "hamburger buns":          "hamburger buns package bag white background",
    "hot dog buns":            "hot dog buns package bag white background",
    "croissants":              "croissants package bakery white background",
    "donuts":                  "donuts glazed package box white background",
    "muffins":                 "muffins package bakery white background",
    "cinnamon rolls":          "cinnamon rolls package bakery white background",
    "coffee cake":             "coffee cake package bakery white background",
    "pound cake":              "pound cake package bakery white background",
    "banana bread":            "banana bread loaf package white background",
    "focaccia":                "focaccia bread white background",
    "breadsticks":             "breadsticks package bag white background",
    "biscuits":                "biscuits can package white background",
    "crescent rolls":          "crescent rolls can package white background",

    # ── PASTA & GRAINS ───────────────────────────────────────────────────────
    # CRITICAL: all pasta must say "dry pasta box bag" to avoid dish photos
    "pasta":                   "dry pasta box bag white background",
    "spaghetti":               "dry spaghetti pasta box bag white background",
    "linguine":                "dry linguine pasta box bag white background",
    "fettuccine":              "dry fettuccine pasta box bag white background",
    "angel hair":              "dry angel hair pasta box bag white background",
    "penne":                   "dry penne pasta box bag white background",
    "rigatoni":                "dry rigatoni pasta box bag white background",
    "ziti":                    "dry ziti pasta box bag white background",
    "rotini":                  "dry rotini pasta box bag white background",
    "farfalle":                "dry farfalle bow tie pasta box bag white background",
    "bow tie pasta":           "dry farfalle bow tie pasta box bag white background",
    "fusilli":                 "dry fusilli pasta box bag white background",
    "shells":                  "dry shell pasta box bag white background",
    "shell pasta":             "dry shell pasta box bag white background",
    "medium shells":           "dry medium shell pasta box bag white background",
    "large shells":            "dry large shell pasta box bag white background",
    "small shells":            "dry small shell pasta box bag white background",
    "stuffed shells":          "dry jumbo shell pasta box bag white background",
    "jumbo shells":            "dry jumbo shell pasta box bag white background",
    "conchiglie":              "dry shell pasta box bag white background",
    "cavatappi":               "dry cavatappi pasta box bag white background",
    "orecchiette":             "dry orecchiette pasta box bag white background",
    "gemelli":                 "dry gemelli pasta box bag white background",
    "ditalini":                "dry ditalini pasta box bag white background",
    "orzo":                    "dry orzo pasta box bag white background",
    "elbow macaroni":          "dry elbow macaroni pasta box bag white background",
    "macaroni":                "dry macaroni pasta box bag white background",
    "lasagna":                 "dry lasagna noodles box white background",
    "lasagna noodles":         "dry lasagna noodles box white background",
    "egg noodles":             "dry egg noodles bag package white background",
    "ramen noodles":           "ramen noodle package cup white background",
    "rice noodles":            "dry rice noodles bag package white background",
    "udon noodles":            "dry udon noodles package white background",
    "soba noodles":            "dry soba noodles package white background",
    "rice":                    "white rice bag package white background",
    "white rice":              "white rice bag package white background",
    "jasmine rice":            "jasmine rice bag package white background",
    "basmati rice":            "basmati rice bag package white background",
    "brown rice":              "brown rice bag package white background",
    "wild rice":               "wild rice bag package white background",
    "quinoa":                  "quinoa bag package white background",
    "farro":                   "farro grain bag package white background",
    "barley":                  "barley grain bag package white background",
    "couscous":                "couscous box bag package white background",
    "polenta":                 "polenta tube roll package white background",
    "grits":                   "grits package box white background",

    # ── CEREAL & BREAKFAST ───────────────────────────────────────────────────
    "cereal":                  "cereal box package white background",
    "cheerios":                "cheerios cereal box white background",
    "frosted flakes":          "frosted flakes cereal box white background",
    "lucky charms":            "lucky charms cereal box white background",
    "special k":               "special k cereal box white background",
    "granola":                 "granola bag package white background",
    "oats":                    "rolled oats container canister white background",
    "rolled oats":             "rolled oats container canister white background",
    "instant oatmeal":         "instant oatmeal packets box white background",
    "overnight oats":          "oats container package white background",
    "pancake mix":             "pancake mix box package white background",
    "waffle mix":              "waffle mix box package white background",
    "maple syrup":             "maple syrup bottle jug white background",
    "protein bar":             "protein bar wrapper package white background",
    "granola bar":             "granola bar wrapper package white background",
    "kind bar":                "kind bar wrapper package white background",
    "clif bar":                "clif bar wrapper package white background",

    # ── FROZEN FOODS ─────────────────────────────────────────────────────────
    "frozen pizza":            "frozen pizza box package white background",
    "frozen vegetables":       "frozen vegetables bag package white background",
    "frozen fruit":            "frozen fruit bag package white background",
    "frozen strawberries":     "frozen strawberries bag package white background",
    "frozen blueberries":      "frozen blueberries bag package white background",
    "frozen mango":            "frozen mango bag package white background",
    "frozen broccoli":         "frozen broccoli bag package white background",
    "frozen corn":             "frozen corn bag package white background",
    "frozen peas":             "frozen peas bag package white background",
    "frozen edamame":          "frozen edamame bag package white background",
    "frozen chicken":          "frozen chicken bag package white background",
    "frozen chicken breasts":  "frozen chicken breasts bag package white background",
    "frozen shrimp":           "frozen shrimp bag package white background",
    "frozen fish":             "frozen fish fillets bag package white background",
    "fish sticks":             "fish sticks box package white background",
    "frozen waffles":          "frozen waffles box package white background",
    "frozen pancakes":         "frozen pancakes box package white background",
    "frozen meatballs":        "frozen meatballs bag package white background",
    "ice cream":               "ice cream tub container white background",
    "gelato":                  "gelato container tub white background",
    "frozen yogurt":           "frozen yogurt container tub white background",
    "popsicles":               "popsicles box package white background",
    "frozen burrito":          "frozen burrito package white background",
    "hot pocket":              "hot pockets box package white background",
    "lean cuisine":            "lean cuisine frozen meal box white background",
    "stouffers":               "stouffers frozen meal box white background",
    "healthy choice":          "healthy choice frozen meal box white background",

    # ── PANTRY & DRY GOODS ───────────────────────────────────────────────────
    "flour":                   "all purpose flour bag package white background",
    "all-purpose flour":       "all purpose flour bag package white background",
    "all purpose flour":       "all purpose flour bag package white background",
    "bread flour":             "bread flour bag package white background",
    "almond flour":            "almond flour bag package white background",
    "sugar":                   "white sugar bag package white background",
    "white sugar":             "white sugar bag package white background",
    "brown sugar":             "brown sugar bag package white background",
    "powdered sugar":          "powdered sugar bag package white background",
    "confectioners sugar":     "powdered sugar bag package white background",
    "salt":                    "salt container shaker white background",
    "kosher salt":             "kosher salt box container white background",
    "sea salt":                "sea salt container white background",
    "black pepper":            "black pepper grinder container white background",
    "olive oil":               "olive oil bottle white background",
    "extra virgin olive oil":  "extra virgin olive oil bottle white background",
    "vegetable oil":           "vegetable oil bottle white background",
    "canola oil":              "canola oil bottle white background",
    "avocado oil":             "avocado oil bottle white background",
    "coconut oil":             "coconut oil jar white background",
    "sesame oil":              "sesame oil bottle white background",
    "cooking spray":           "cooking spray can white background",
    "pam cooking spray":       "pam cooking spray can white background",
    "peanut butter":           "peanut butter jar white background",
    "almond butter":           "almond butter jar white background",
    "sunflower butter":        "sunflower butter jar white background",
    "nutella":                 "nutella jar white background",
    "jelly":                   "grape jelly jar white background",
    "jam":                     "jam jar white background",
    "strawberry jam":          "strawberry jam jar white background",
    "honey":                   "honey bear bottle jar white background",
    "agave":                   "agave nectar bottle white background",
    "molasses":                "molasses bottle jar white background",
    "vanilla extract":         "vanilla extract bottle white background",
    "baking soda":                  "baking soda box orange white background",
    "arm and hammer baking soda":   "arm hammer baking soda orange box white background",
    "arm & hammer baking soda":     "arm hammer baking soda orange box white background",
    "arm hammer baking soda":       "arm hammer baking soda orange box white background",
    "arm hammer":                   "arm hammer baking soda orange box white background",
    "baking powder":           "baking powder can container white background",
    "yeast":                   "active dry yeast packet jar white background",
    "cornstarch":              "cornstarch box package white background",
    "cocoa powder":            "cocoa powder container white background",
    "chocolate chips":         "chocolate chips bag package white background",
    "dark chocolate":          "dark chocolate bar white background",
    "milk chocolate":          "milk chocolate bar white background",
    "chocolate bar":           "chocolate bar wrapped package white background",
    "spices":                  "spice jar container white background",
    "cinnamon":                "cinnamon spice jar white background",
    "garlic powder":           "garlic powder spice jar white background",
    "onion powder":            "onion powder spice jar white background",
    "paprika":                 "paprika spice jar white background",
    "cumin":                   "cumin spice jar white background",
    "chili powder":            "chili powder spice jar white background",
    "italian seasoning":       "italian seasoning spice jar white background",
    "oregano":                 "oregano spice jar white background",
    "bread crumbs":            "bread crumbs canister package white background",
    "panko":                   "panko bread crumbs package white background",
    "gelatin":                 "gelatin box package white background",
    "pudding mix":             "pudding mix box package white background",
    "jello":                   "jello box package white background",
    "protein powder":          "protein powder tub container white background",
    "vitamins":                "vitamin supplement bottle white background",
    "multivitamin":            "multivitamin bottle white background",

    # ── CANNED & JARRED GOODS ────────────────────────────────────────────────
    "canned tomatoes":         "canned tomatoes can white background",
    "diced tomatoes":          "diced tomatoes can white background",
    "crushed tomatoes":        "crushed tomatoes can white background",
    "tomato paste":            "tomato paste can white background",
    "tomato sauce":            "tomato sauce jar can white background",
    "marinara sauce":          "marinara sauce jar white background",
    "alfredo sauce":           "alfredo sauce jar white background",
    "pasta sauce":             "pasta sauce jar white background",
    "salsa":                   "salsa jar white background",
    "pesto":                   "pesto jar white background",
    "chicken broth":           "chicken broth carton white background",
    "beef broth":              "beef broth carton white background",
    "vegetable broth":         "vegetable broth carton white background",
    "bone broth":              "bone broth carton white background",
    "soup":                    "soup can white background",
    "tomato soup":             "tomato soup can white background",
    "chicken noodle soup":     "chicken noodle soup can white background",
    "vegetable soup":          "vegetable soup can white background",
    "clam chowder":            "clam chowder can white background",
    "beans":                   "beans can white background",
    "black beans":             "black beans can white background",
    "kidney beans":            "kidney beans can white background",
    "chickpeas":               "chickpeas garbanzo beans can white background",
    "garbanzo beans":          "chickpeas garbanzo beans can white background",
    "pinto beans":             "pinto beans can white background",
    "white beans":             "white beans cannellini can white background",
    "refried beans":           "refried beans can white background",
    "corn":                    "canned corn can white background",
    "canned corn":             "canned corn can white background",
    "green beans can":         "canned green beans can white background",
    "peas can":                "canned peas can white background",
    "coconut milk can":        "coconut milk can white background",
    "tuna":                    "canned tuna can white background",
    "salmon can":              "canned salmon can white background",
    "sardines":                "sardines tin can white background",
    "olives":                  "olives jar white background",
    "pickles":                 "pickles jar white background",
    "dill pickles":            "dill pickles jar white background",
    "capers":                  "capers jar white background",
    "artichoke hearts":        "artichoke hearts jar can white background",
    "roasted red peppers":     "roasted red peppers jar white background",
    "sun dried tomatoes":      "sun dried tomatoes jar white background",

    # ── CONDIMENTS & SAUCES ──────────────────────────────────────────────────
    "ketchup":                 "ketchup bottle white background",
    "mustard":                 "mustard bottle yellow white background",
    "dijon mustard":           "dijon mustard jar white background",
    "honey mustard":           "honey mustard bottle jar white background",
    "mayonnaise":              "mayonnaise jar white background",
    "miracle whip":            "miracle whip jar white background",
    "ranch dressing":          "ranch dressing bottle white background",
    "caesar dressing":         "caesar dressing bottle white background",
    "italian dressing":        "italian dressing bottle white background",
    "balsamic vinegar":        "balsamic vinegar bottle white background",
    "apple cider vinegar":     "apple cider vinegar bottle white background",
    "white vinegar":           "white vinegar bottle white background",
    "soy sauce":               "soy sauce bottle white background",
    "teriyaki sauce":          "teriyaki sauce bottle white background",
    "hot sauce":               "hot sauce bottle white background",
    "franks hot sauce":        "franks red hot sauce bottle white background",
    "tabasco":                 "tabasco hot sauce bottle white background",
    "sriracha":                "sriracha hot sauce bottle white background",
    "bbq sauce":               "bbq sauce bottle white background",
    "worcestershire sauce":    "worcestershire sauce bottle white background",
    "fish sauce":              "fish sauce bottle white background",
    "oyster sauce":            "oyster sauce bottle white background",
    "hoisin sauce":            "hoisin sauce bottle white background",
    "hummus":                  "hummus container tub white background",
    "guacamole":               "guacamole container tub white background",
    "queso dip":               "queso dip jar container white background",
    "buffalo sauce":           "buffalo wing sauce bottle white background",
    "sweet baby rays":         "sweet baby rays bbq sauce bottle white background",
    "pickle relish":           "pickle relish jar white background",
    "tahini":                  "tahini jar white background",
    "miso paste":              "miso paste package tub white background",
    "gochujang":               "gochujang paste tub white background",
    "kimchi":                  "kimchi jar package white background",

    # ── SNACKS ───────────────────────────────────────────────────────────────
    "chips":                   "potato chips bag package white background",
    "potato chips":            "potato chips bag package white background",
    "tortilla chips":          "tortilla chips bag package white background",
    "doritos":                 "doritos chips bag package white background",
    "lays chips":              "lays potato chips bag package white background",
    "pringles":                "pringles chips can white background",
    "cheetos":                 "cheetos bag package white background",
    "fritos":                  "fritos corn chips bag package white background",
    "popcorn":                 "popcorn bag package white background",
    "microwave popcorn":       "microwave popcorn bag package white background",
    "pretzels":                "pretzels bag package white background",
    "snap pretzels":           "snap pretzels bag package white background",
    "crackers":                "crackers box package white background",
    "ritz crackers":           "ritz crackers box package white background",
    "wheat thins":             "wheat thins crackers box package white background",
    "triscuits":               "triscuits crackers box package white background",
    "graham crackers":         "graham crackers box package white background",
    "rice cakes":              "rice cakes bag package white background",
    "trail mix":               "trail mix bag package white background",
    "mixed nuts":              "mixed nuts can jar white background",
    "nuts":                    "mixed nuts bag container white background",
    "almonds":                 "almonds bag package white background",
    "cashews":                 "cashews bag package white background",
    "walnuts":                 "walnuts bag package white background",
    "pecans":                  "pecans bag package white background",
    "pistachios":              "pistachios bag package white background",
    "wonderful pistachios":    "pistachios bag package white background",
    "peanuts":                 "peanuts bag package white background",
    "sunflower seeds":         "sunflower seeds bag package white background",
    "pumpkin seeds":           "pumpkin seeds bag package white background",
    "dried fruit":             "dried fruit bag package white background",
    "raisins":                 "raisins box bag package white background",
    "dried cranberries":       "dried cranberries bag package white background",
    "craisins":                "craisins dried cranberries bag package white background",
    "fruit snacks":            "fruit snacks pouch package white background",
    "gummies":                 "gummy candy bag package white background",
    "candy":                   "candy bag package white background",
    "chocolate covered":       "chocolate candy package white background",
    "m&ms":                    "m&ms candy bag package white background",
    "oreos":                   "oreos cookie package white background",
    "cookies":                 "cookies bag box package white background",
    "chips ahoy":              "chips ahoy cookies package white background",
    "protein bar":             "protein bar wrapper package white background",
    "rxbar":                   "rxbar protein bar wrapper white background",

    # ── DELI / PREPARED ──────────────────────────────────────────────────────
    "mac and cheese":          "macaroni and cheese box package white background",
    "macaroni and cheese":     "macaroni and cheese box package white background",
    "macaroni & cheese":       "macaroni and cheese box package white background",
    "kraft mac and cheese":    "kraft macaroni and cheese box package white background",
    "chocolate dipped strawberries": "chocolate dipped strawberries package container white background",
    "dipped strawberries":     "chocolate dipped strawberries package white background",
    "cheesecake":              "cheesecake package container white background",
    "rotisserie chicken":      "rotisserie chicken package store white background",
    "deli chicken":            "rotisserie chicken package store white background",
    "pulled pork":             "pulled pork vacuum sealed package white background",
    "deli meatloaf":           "meatloaf sliced package white background",

    # ── HOUSEHOLD ────────────────────────────────────────────────────────────
    "paper towels":            "paper towels roll package white background",
    "bounty":                  "bounty paper towels roll package white background",
    "toilet paper":            "toilet paper rolls package white background",
    "charmin":                 "charmin toilet paper package white background",
    "tissue":                  "facial tissue box white background",
    "kleenex":                 "kleenex tissue box white background",
    "napkins":                 "paper napkins package white background",
    "trash bags":              "trash bags box package white background",
    "ziploc bags":             "ziploc bags box package white background",
    "plastic wrap":            "plastic wrap box white background",
    "aluminum foil":           "aluminum foil box white background",
    "parchment paper":         "parchment paper box white background",
    "dish soap":               "dish soap bottle white background",
    "dawn dish soap":          "dawn dish soap bottle white background",
    "laundry detergent":       "laundry detergent bottle white background",
    "tide":                    "tide laundry detergent bottle white background",
    "gain":                    "gain laundry detergent bottle white background",
    "fabric softener":         "fabric softener bottle white background",
    "dryer sheets":            "dryer sheets box white background",
    "dishwasher pods":         "dishwasher pods tub white background",
    "cascade":                 "cascade dishwasher pods tub white background",
    "cleaning spray":          "cleaning spray bottle white background",
    "lysol":                   "lysol spray bottle white background",
    "clorox":                  "clorox bleach bottle white background",
    "windex":                  "windex glass cleaner spray bottle white background",
    "febreze":                 "febreze air freshener spray bottle white background",
    "swiffer":                 "swiffer sweeper product white background",
    "cat litter":              "cat litter box package white background",
    "fresh step":              "fresh step cat litter box package white background",
    "tidy cats":               "tidy cats cat litter container white background",
    "arm hammer litter":       "arm hammer cat litter box package white background",
    "dog food":                "dog food bag package white background",
    "cat food":                "cat food can package white background",
    "pet treats":              "pet treats bag package white background",
    "shampoo":                 "shampoo bottle white background",
    "conditioner":             "conditioner bottle white background",
    "body wash":               "body wash bottle white background",
    "soap":                    "bar soap package white background",
    "hand soap":               "hand soap pump bottle white background",
    "toothpaste":              "toothpaste tube white background",
    "toothbrush":              "toothbrush package white background",
    "deodorant":               "deodorant stick white background",
    "sunscreen":               "sunscreen bottle tube white background",
    "lotion":                  "lotion bottle white background",
    "bandages":                "bandages box package white background",
    "ibuprofen":               "ibuprofen bottle white background",
    "acetaminophen":           "acetaminophen bottle white background",
    "antacid":                 "antacid tablets bottle white background",
    "batteries":               "batteries package white background",
    "duracell":                "duracell batteries package white background",
    "duracell batteries":      "duracell aa batteries package white background",
    "energizer":               "energizer batteries package white background",

    # ── Personal care / hygiene (Costco + all stores) ─────────────────────
    "shave gel":               "shave gel can white background",
    "shaving gel":             "shaving gel can white background",
    "shaving cream":           "shaving cream can white background",
    "gillette shave gel":      "gillette shave gel can white background",
    "edge shave gel":          "edge shave gel can white background",
    "dove":                    "dove body wash bottle white background",
    "dove body wash":          "dove body wash bottle white background",
    "dove soap":               "dove soap bar package white background",
    "dove invisible":          "dove invisible deodorant stick white background",
    "dove invs":               "dove invisible deodorant stick white background",
    "degree deodorant":        "degree deodorant stick white background",
    "degree ultra":            "degree deodorant stick white background",
    "old spice":               "old spice deodorant stick white background",
    "cerave":                  "cerave moisturizing cream tub white background",
    "cerave cream":            "cerave moisturizing cream tub white background",
    "cerave lotion":           "cerave moisturizing lotion bottle white background",
    "cerave cleanser":         "cerave hydrating cleanser bottle white background",
    "neutrogena":              "neutrogena face wash bottle white background",
    "olay":                    "olay moisturizer bottle white background",
    "crest":                   "crest toothpaste tube white background",
    "crest toothpaste":        "crest toothpaste tube white background",
    "crest mouthwash":         "crest mouthwash bottle white background",
    "crest mwash":             "crest mouthwash bottle white background",
    "listerine":               "listerine mouthwash bottle white background",
    "mouthwash":               "mouthwash bottle white background",
    "colgate":                 "colgate toothpaste tube white background",
    "oral b":                  "oral b toothbrush package white background",
    "floss":                   "dental floss package white background",
    "razor":                   "razor package white background",
    "gillette razor":          "gillette razor package white background",
    "venus razor":             "venus razor package white background",
    "shea conditioner":        "conditioner bottle white background",
    "shea condtnr":            "conditioner bottle white background",
    "head and shoulders":      "head and shoulders shampoo bottle white background",
    "pantene":                 "pantene shampoo bottle white background",
    "tresemme":                "tresemme shampoo bottle white background",

    # ── Baby / kids products ───────────────────────────────────────────────
    "diapers":                 "diapers package white background",
    "huggies":                 "huggies diapers package white background",
    "huggies pull ups":        "huggies pull ups training pants package white background",
    "pull ups":                "pull ups training pants package white background",
    "pull-ups":                "pull ups training pants package white background",
    "pampers":                 "pampers diapers package white background",
    "luvs":                    "luvs diapers package white background",
    "ks diaper":               "kirkland diapers package white background",
    "kirkland diapers":        "kirkland diapers package white background",
    "baby wipes":              "baby wipes package white background",
    "wipes":                   "baby wipes package white background",
    "baby formula":            "baby formula can white background",
    "formula":                 "baby formula can white background",

    # ── Laundry / fabric care ──────────────────────────────────────────────
    "downy":                   "downy fabric softener bottle white background",
    "downy fabric softener":   "downy fabric softener bottle white background",
    "downy ultimt":            "downy unstopables beads white background",
    "downy unstopables":       "downy unstopables scent beads white background",
    "bounce":                  "bounce dryer sheets box white background",
    "arm and hammer detergent": "arm hammer laundry detergent bottle white background",
    "arm hammer detergent":    "arm hammer laundry detergent bottle white background",
    "all detergent":           "all laundry detergent bottle white background",
    "persil":                  "persil laundry detergent bottle white background",

    # ── Dental / oral care ────────────────────────────────────────────────
    "dental chews":            "dog dental chews bag package white background",
    "ks dental ch":            "dog dental chews bag package white background",
    "milk bone":               "milk bone dog treats box white background",
    "greenies":                "greenies dental dog treats package white background",

    # ── Vitamins / supplements ────────────────────────────────────────────
    "vitamin c":               "vitamin c supplement bottle white background",
    "vitamin d":               "vitamin d supplement bottle white background",
    "multivitamin":            "multivitamin bottle white background",
    "fish oil":                "fish oil supplement bottle white background",
    "melatonin":               "melatonin supplement bottle white background",
    "zinc":                    "zinc supplement bottle white background",
    "dura c":                  "vitamin c supplement bottle white background",
    "dura c 14pk":             "vitamin c supplement bottle white background",
}


def _is_household_item(name: str) -> bool:
    """Returns True if the item name looks like a household/cleaning product."""
    n = name.lower()
    signals = [
        "paper towel", "toilet paper", "bath tissue", "tissue", "napkin",
        "trash bag", "garbage bag", "ziploc", "plastic wrap", "aluminum foil",
        "parchment", "wax paper", "plastic bag", "storage bag",
        "dish soap", "dishwasher", "laundry", "detergent", "fabric softener",
        "dryer sheet", "bleach", "cleaner", "disinfect", "windex", "lysol",
        "clorox", "swiffer", "mop", "broom", "sponge", "scrub",
        "air freshener", "febreze", "candle", "batteries", "light bulb",
        "hand soap", "body wash", "shampoo", "conditioner", "toothpaste",
        "toothbrush", "deodorant", "razor", "mouthwash", "floss",
        "cotton ball", "q-tip", "bandage", "band-aid", "lotion", "sunscreen",
        "toilet", "shower", "soap", "wash", "clean", "wipe",
    ]
    return any(s in n for s in signals)


def _is_food_photo(photo: dict) -> bool:
    """
    Check Unsplash photo tags/description/alt to confirm it's a usable food or product photo.
    Much stricter than before — rejects shelf photos, store interiors, labels/text-heavy
    images, people photos, and anything that is clearly not the product itself.
    """
    # ── Build combined text from all Unsplash metadata ────────────────────────
    desc = (photo.get("description") or "").lower()
    alt  = (photo.get("alt_description") or "").lower()
    tags = " ".join((t.get("title") or "").lower() for t in (photo.get("tags") or []))
    combined = f"{desc} {alt} {tags}".strip()

    # ── HARD REJECT — these phrases disqualify a photo immediately ────────────
    # Grocery/store interior shots
    hard_reject_phrases = [
        "supermarket", "grocery store", "store aisle", "store shelf", "retail shelf",
        "shopping aisle", "grocery aisle", "superstore", "hypermarket", "warehouse store",
        "shopping cart", "shopping basket", "checkout", "convenience store",
        "refrigerator aisle", "freezer aisle", "produce aisle", "deli counter",
        "pantry shelf", "pantry", "kitchen shelf", "food shelf",
        # Nutrition label / text-heavy product shots we don't want
        "nutrition label", "nutrition facts", "ingredient list", "ingredients list",
        "nutrition information",
        # People / lifestyle shots
        "person eating", "woman eating", "man eating", "child eating",
        "woman cooking", "man cooking", "chef cooking",
        "family dinner", "dinner table", "restaurant table", "restaurant meal",
        # Nature / outdoor
        "farm field", "crop field", "agriculture field", "harvest field",
    ]
    # Single-word hard rejects that reliably mean shelf/store photos
    hard_reject_words = {
        "aisle", "shelves", "supermarket", "hypermarket", "pantry",
        "portrait", "fashion", "clothing", "architecture",
    }

    for phrase in hard_reject_phrases:
        if phrase in combined:
            print(f"[PHOTO CHECK] hard rejected (phrase: '{phrase}'): {combined[:80]}", flush=True)
            return False

    combined_words = set(combined.split())
    for word in hard_reject_words:
        if word in combined_words:
            print(f"[PHOTO CHECK] hard rejected (word: '{word}'): {combined[:80]}", flush=True)
            return False

    # ── ACCEPT SIGNALS — photo must have at least one of these ────────────────
    food_keywords = {
        # General food/product
        "food", "ingredient", "product", "package", "packaging",
        "container", "bottle", "jar", "can", "carton", "bag", "box", "tub",
        # Cooking / eating context
        "meal", "dish", "recipe", "cook", "kitchen", "eat",
        # Produce
        "fruit", "vegetable", "produce", "fresh", "raw",
        # Protein
        "meat", "chicken", "beef", "pork", "fish", "seafood", "shrimp", "egg",
        # Dairy
        "dairy", "cheese", "milk", "cream", "butter", "yogurt",
        # Beverages
        "drink", "beverage", "juice", "water", "coffee", "tea",
        # Bread / grains
        "bread", "pasta", "rice", "cereal", "oat", "grain",
        # Snacks / sweets
        "snack", "chip", "cracker", "cookie", "cake", "dessert", "chocolate",
        "candy", "ice cream", "frozen",
        # Condiments / pantry
        "sauce", "condiment", "spice", "herb", "oil", "vinegar", "honey",
        "ketchup", "mustard", "salsa", "dressing",
        # Common produce names
        "apple", "banana", "orange", "berry", "strawberry", "tomato", "potato",
        "onion", "garlic", "pepper", "carrot", "broccoli", "spinach", "lettuce",
        "lemon", "lime", "avocado", "mushroom", "corn", "bean", "pea",
        "grape", "peach", "mango", "pineapple", "watermelon", "zucchini",
        # Other food words
        "soup", "salad", "pizza", "burger", "taco", "sushi", "noodle",
        "nutrition", "organic", "healthy",
    }

    if any(kw in combined for kw in food_keywords):
        return True

    # ── No tags at all — accept reluctantly (better than nothing) ─────────────
    if not combined.strip():
        return True

    # ── Has tags but none are food-related — reject ───────────────────────────
    print(f"[PHOTO CHECK] rejected (no food keywords): {combined[:80]}", flush=True)
    return False


async def _spoonacular_image(name: str) -> Optional[str]:
    """
    Fetch a clean, generic, non-branded ingredient image from Spoonacular.
    Always strips brand names before searching so "Doritos" -> "chips",
    "Jif Peanut Butter" -> "peanut butter", etc.
    Returns the image URL string or None.
    """
    if not SPOONACULAR_API:
        return None
    try:
        # Strip brand so we always search the generic ingredient name
        simplified = _simplify_for_unsplash(name)
        if not simplified or len(simplified) < 2:
            simplified = name.strip()

        search_url = "https://api.spoonacular.com/food/ingredients/search"
        params = {
            "query": simplified,
            "number": 1,
            "apiKey": SPOONACULAR_API,
        }
        async with httpx.AsyncClient(timeout=6.0, follow_redirects=True) as client:
            resp = await client.get(search_url, params=params)
            if resp.status_code != 200:
                print(f"[SPOONACULAR] HTTP {resp.status_code} for '{simplified}'", flush=True)
                return None
            data = resp.json()
            results = data.get("results", [])
            if not results:
                print(f"[SPOONACULAR] no results for '{simplified}'", flush=True)
                return None
            image_name = results[0].get("image", "")
            if not image_name:
                return None
            img_url = f"https://spoonacular.com/cdn/ingredients_250x250/{image_name}"
            print(f"[SPOONACULAR] ✅ '{name}' -> '{simplified}' -> {img_url}", flush=True)
            return img_url
    except Exception as e:
        print(f"[SPOONACULAR IMAGE] error for '{name}': {e}", flush=True)
    return None


async def _unsplash_image(name: str, is_household: bool = False, photo_query: Optional[str] = None) -> Optional[str]:
    """
    Search Unsplash for a product photo.
    ── Smart query logic ───────────────────────────────────────────────────────
    1. Check _UNSPLASH_QUERY_OVERRIDES for a hand-crafted exact query
    2. Otherwise, build a smart query from the simplified name + type-based suffix
    3. Fall back through progressively broader queries before giving up
    """
    try:
        access_key = os.getenv("UNSPLASH_ACCESS_KEY", "").strip()
        if not access_key:
            return None

        # Simplify the name so Unsplash gets a clean keyword
        simplified = _simplify_for_unsplash(name)
        name_lower = name.lower()
        simplified_lower = simplified.lower()

        # ── STEP 1: Check override dictionary (exact query for known items) ───────
        # Check full simplified name first, then each word combo
        override_query = None
        # Try exact simplified match
        if simplified_lower in _UNSPLASH_QUERY_OVERRIDES:
            override_query = _UNSPLASH_QUERY_OVERRIDES[simplified_lower]
        else:
            # Try original name (lowercased) match
            if name_lower in _UNSPLASH_QUERY_OVERRIDES:
                override_query = _UNSPLASH_QUERY_OVERRIDES[name_lower]
            else:
                # Try to match any override key that's fully contained in the simplified name
                # Sort by key length descending so longer/more-specific keys win first
                for key, val in sorted(_UNSPLASH_QUERY_OVERRIDES.items(), key=lambda x: len(x[0]), reverse=True):
                    if key in simplified_lower or key in name_lower:
                        override_query = val
                        print(f"[UNSPLASH OVERRIDE] '{name}' matched key '{key}'", flush=True)
                        break


        # Gemini returned an exact query for this item — ALWAYS use it, even if override dict matched.
        # Gemini knows the exact item from the receipt; the override dict is a best guess.
        if photo_query:
            override_query = photo_query
            print(f"[UNSPLASH GEMINI QUERY] '{name}' using Gemini photo_query: '{photo_query}'", flush=True)

        # Detect item type for suffix selection
        packaged_signals = [
            "fillet", "fillets", "nugget", "nuggets", "strip", "strips", "tender", "tenders",
            "patty", "patties", "breaded", "battered", "frozen", "canned", "jarred",
            "sliced", "shredded", "diced", "chopped", "seasoned",
            "roasted", "baked", "smoked", "cured", "deli", "luncheon",
            "sauce", "dressing", "marinade", "seasoning", "mix", "blend",
            "cereal", "cracker", "chip", "cookie", "cake", "roll",
            "pasta", "noodle", "rice cake", "popcorn", "pretzel",
            "yogurt", "cream cheese", "cottage cheese", "sour cream",
            "juice", "drink", "soda", "water", "milk", "creamer",
        ]
        is_packaged = any(sig in name_lower for sig in packaged_signals)

        if is_household:
            suffix = "retail product packaging grocery store clean white background"
            fallback_query = "household product retail packaging white background"
        elif is_packaged:
            suffix = "grocery store product packaging isolated white background"
            fallback_query = "packaged grocery product isolated white background"
        else:
            suffix = "fresh grocery store product isolated white background"
            fallback_query = "fresh grocery food product isolated white background"

        # ── Inner search helper ─────────────────────────────────────────────────
        async def _search(query: str, append_suffix: bool = True, require_food_check: bool = True) -> Optional[str]:
            full_query = f"{query} {suffix}".strip() if append_suffix else query
            encoded = urllib.parse.quote(full_query)
            url = (
                f"https://api.unsplash.com/search/photos"
                f"?query={encoded}&per_page=15&orientation=squarish&content_filter=high"
            )
            async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as uc:
                resp = await uc.get(url, headers={"Authorization": f"Client-ID {access_key}"})
                if resp.status_code != 200:
                    print(f"[UNSPLASH] HTTP {resp.status_code} for '{full_query}'", flush=True)
                    return None
                results = resp.json().get("results", [])
                for photo in results:
                    img_url = (
                        (photo.get("urls") or {}).get("regular")
                        or (photo.get("urls") or {}).get("small")
                    )
                    if not img_url or not img_url.startswith("http"):
                        continue
                    if require_food_check and not _is_food_photo(photo):
                        print(f"[UNSPLASH] rejected photo for '{full_query}'", flush=True)
                        continue
                    print(f"[UNSPLASH] ✅ found for '{full_query}': {img_url[:60]}", flush=True)
                    return img_url
                print(f"[UNSPLASH] no valid photo for '{full_query}'", flush=True)
            return None

        # ── STEP 3: Run searches in priority order ──────────────────────────────

        # Household items must skip the food photo check entirely —
        # paper towels, cat litter, dish soap have no food keywords in their Unsplash tags.
        food_check = not is_household

        # Priority 1: Override query (hand-crafted, no suffix added — it's already complete)
        if override_query:
            img_url = await _search(override_query, append_suffix=False, require_food_check=food_check)
            if img_url:
                return img_url

        # Priority 2: Simplified name + suffix
        img_url = await _search(simplified, require_food_check=food_check)
        if img_url:
            return img_url

        # Priority 3: Last 2 words of simplified name + suffix
        words = simplified.split()
        if len(words) > 2:
            short = " ".join(words[-2:])
            img_url = await _search(short, require_food_check=food_check)
            if img_url:
                return img_url

        # Priority 4: First meaningful word + suffix
        if words:
            img_url = await _search(words[0], require_food_check=food_check)
            if img_url:
                return img_url

        # Priority 5: Generic fallback, skip food check always
        img_url = await _search(fallback_query, append_suffix=False, require_food_check=False)
        return img_url

    except Exception as e:
        print(f"[UNSPLASH IMAGE] error for '{name}': {e}", flush=True)
    return None


async def _expand_receipt_name(name: str) -> str:
    """Use Gemini to expand receipt abbreviations into real product names."""
    try:
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            return name

        # Only expand if the name looks abbreviated
        words = name.strip().split()
        avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
        looks_abbreviated = (
            len(name) < 35 or
            avg_word_len < 4.5 or
            any(len(w) <= 2 for w in words if w.isalpha())
        )
        if not looks_abbreviated:
            return name

        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = (
            f"This is a grocery receipt item name that may be abbreviated or truncated: \"{name}\"\n"
            f"Expand it into the full real product name as it would appear on store shelves.\n"
            f"Rules:\n"
            f"- Return ONLY the expanded product name, nothing else\n"
            f"- Keep brand names if present\n"
            f"- If it's already a clear full name, return it as-is\n"
            f"- Do not add size, weight, or price info\n"
            f"- Examples: 'Ck Sl Cookie' -> 'Chocolate Slice Cookie', "
            f"'Ny Xsh Cheddar' -> 'New York Extra Sharp Cheddar', "
            f"'Dipped Straw' -> 'Chocolate Dipped Strawberries', "
            f"'Hvy Whp Crm' -> 'Heavy Whipping Cream', "
            f"'Buitoni Pes Bas Sauce' -> 'Buitoni Pesto Basil Sauce', "
            f"'Corn Yellow' -> 'Yellow Corn', 'Straw Bry' -> 'Strawberries', "
            f"'Grn Bns' -> 'Green Beans'\n"
            f"Expanded name:"
        )
        # 3 second timeout — if Gemini is slow, just use the original name
        response = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, model.generate_content, prompt),
            timeout=3.0
        )
        expanded = response.text.strip().strip('"').strip("'")
        if expanded and len(expanded) > 2:
            print(f"[EXPAND NAME] '{name}' -> '{expanded}'", flush=True)
            return expanded
        return name
    except asyncio.TimeoutError:
        print(f"[EXPAND NAME] timeout for '{name}', using original", flush=True)
        return name
    except Exception as e:
        print(f"[EXPAND NAME] error for '{name}': {e}", flush=True)
        return name


async def _google_product_image(name: str) -> Optional[str]:
    """Search Google Custom Search for clean product images from grocery/retail sites."""
    try:
        api_key = os.getenv("GOOGLE_IMAGE_API_KEY", "").strip()
        cx = os.getenv("GOOGLE_IMAGE_CX", "").strip()
        if not api_key or not cx:
            print("[GOOGLE IMAGE] missing GOOGLE_IMAGE_API_KEY or GOOGLE_IMAGE_CX", flush=True)
            return None

        async def _search(query: str) -> Optional[str]:
            # Search just the product name — the search engine is already
            # restricted to retail sites via the CX configuration
            params = {
                "key": api_key,
                "cx": cx,
                "q": query,
                "searchType": "image",
                "num": 5,
                "imgType": "photo",
                "imgSize": "medium",
                "safe": "active",
            }
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as gc:
                resp = await gc.get(
                    "https://www.googleapis.com/customsearch/v1",
                    params=params
                )
                if resp.status_code == 200:
                    data = resp.json()
                    items = data.get("items", [])
                    for item in items:
                        img_url = item.get("link", "")
                        if img_url and img_url.startswith("http") and _is_good_product_image(img_url):
                            print(f"[GOOGLE IMAGE] found for '{query}': {img_url}", flush=True)
                            return img_url
                    print(f"[GOOGLE IMAGE] no usable result for '{query}'", flush=True)
                elif resp.status_code == 429:
                    print("[GOOGLE IMAGE] quota exceeded", flush=True)
                else:
                    print(f"[GOOGLE IMAGE] error status={resp.status_code} for '{query}'", flush=True)
            return None

        # Try full name first (keeps brand name — best chance of exact product match)
        img_url = await _search(name)

        # Fall back to simplified name (strips brand/size) if full name got nothing
        if not img_url:
            simplified = _simplify_for_unsplash(name)
            if simplified.lower() != name.lower():
                img_url = await _search(simplified)

        # Last resort — just the last 2 meaningful words
        if not img_url:
            words = (simplified if 'simplified' in dir() else name).split()
            if len(words) > 2:
                short = " ".join(words[-2:])
                img_url = await _search(short)

        return img_url

    except Exception as e:
        print(f"[GOOGLE IMAGE] error for '{name}': {e}", flush=True)
    return None


async def _edamam_food_image(name: str) -> Optional[str]:
    """Search Edamam Food Database API for a food photo."""
    if not EDAMAM_FOOD_APP_ID or not EDAMAM_FOOD_APP_KEY:
        return None
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            params = {
                "app_id":  EDAMAM_FOOD_APP_ID,
                "app_key": EDAMAM_FOOD_APP_KEY,
                "ingr":    name,
            }
            resp = await client.get("https://api.edamam.com/api/food-database/v2/parser", params=params)
            if resp.status_code != 200:
                print(f"[EDAMAM FOOD] HTTP {resp.status_code} for '{name}'", flush=True)
                return None
            data = resp.json()
            hints = data.get("hints", [])
            for hint in hints[:5]:
                food = hint.get("food", {})
                img = food.get("image", "")
                if img and img.startswith("http"):
                    print(f"[EDAMAM FOOD] ✅ found image for '{name}': {img[:60]}", flush=True)
                    return img
            # Also check parsed item
            for parsed in data.get("parsed", []):
                food = parsed.get("food", {})
                img = food.get("image", "")
                if img and img.startswith("http"):
                    print(f"[EDAMAM FOOD] ✅ parsed image for '{name}': {img[:60]}", flush=True)
                    return img
            print(f"[EDAMAM FOOD] no image found for '{name}'", flush=True)
            return None
    except Exception as e:
        print(f"[EDAMAM FOOD] error for '{name}': {e}", flush=True)
        return None


async def _edamam_recipe_image(name: str) -> Optional[str]:
    """Search Edamam Recipe API for a recipe photo.
    Only returns an image if the recipe title is a reasonable match for the search term.
    """
    if not EDAMAM_RECIPE_APP_ID or not EDAMAM_RECIPE_APP_KEY:
        return None
    try:
        # Build a set of meaningful words from the search name for relevance checking
        stop = {"a", "an", "the", "and", "or", "of", "with", "in", "on", "for", "to", "de", "la"}
        name_tokens = {w.lower() for w in re.split(r"\W+", name) if w and w.lower() not in stop and len(w) > 2}

        async with httpx.AsyncClient(timeout=8.0) as client:
            params = {
                "app_id":  EDAMAM_RECIPE_APP_ID,
                "app_key": EDAMAM_RECIPE_APP_KEY,
                "type":    "public",
                "q":       name,
            }
            resp = await client.get(
                "https://api.edamam.com/api/recipes/v2",
                params=params,
                headers={"Accept": "application/json"}
            )
            if resp.status_code != 200:
                print(f"[EDAMAM RECIPE] HTTP {resp.status_code} for '{name}'", flush=True)
                return None
            data = resp.json()
            hits = data.get("hits", [])
            for hit in hits[:8]:
                recipe = hit.get("recipe", {})
                img = recipe.get("image", "")
                recipe_title = recipe.get("label", "")
                if not img or not img.startswith("http"):
                    continue
                # Check relevance — at least one meaningful word from the search
                # must appear in the recipe title
                title_lower = recipe_title.lower()
                if name_tokens and not any(tok in title_lower for tok in name_tokens):
                    print(f"[EDAMAM RECIPE] skipping irrelevant result '{recipe_title}' for '{name}'", flush=True)
                    continue
                print(f"[EDAMAM RECIPE] ✅ found image for '{name}' via '{recipe_title}': {img[:60]}", flush=True)
                return img
            print(f"[EDAMAM RECIPE] no relevant image found for '{name}'", flush=True)
            return None
    except Exception as e:
        print(f"[EDAMAM RECIPE] error for '{name}': {e}", flush=True)
        return None


async def _freepik_image(name: str) -> Optional[str]:
    """Search Freepik stock photos by name."""
    try:
        api_key = os.getenv("FREEPIK_API_KEY", "").strip()
        if not api_key:
            return None
        search_query = urllib.parse.quote(name.strip())
        url = f"https://api.freepik.com/v1/resources?term={search_query}&filters[content_type][photo]=1&limit=5&order=relevance"
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as fc:
            resp = await fc.get(url, headers={"x-freepik-api-key": api_key, "Accept-Language": "en-US"})
            if resp.status_code == 200:
                data = resp.json()
                resources = data.get("data", [])
                for resource in resources:
                    img_url = (resource.get("image") or {}).get("source", {}).get("url")
                    if img_url and img_url.startswith("http") and _is_good_product_image(img_url):
                        print(f"[FREEPIK] found image for '{name}': {img_url}", flush=True)
                        return img_url
                print(f"[FREEPIK] no usable image found for '{name}'", flush=True)
            else:
                print(f"[FREEPIK] error for '{name}' status={resp.status_code}", flush=True)
    except Exception as e:
        print(f"[FREEPIK IMAGE] error for '{name}': {e}", flush=True)
    return None


async def _kroger_image(name: str) -> Optional[str]:
    """
    Search Kroger API for a product image.
    ── Smart matching rules ───────────────────────────────────────────────────────
    1. Extract "core" tokens from the item name (the noun, not adjectives/brands)
    2. ALL core tokens must be present in the Kroger product description
    3. Reject /0001111 store brand images and images without an http URL
    4. If strict match fails, try a simplified 1-2 word search as a fallback
    """
    token = await _kroger_get_token()
    if not token:
        return None

    # Words to completely ignore during matching
    _stop_words = {
        "a", "an", "the", "and", "or", "of", "with", "in", "on", "for", "to",
        "oz", "lb", "ct", "pk", "pack", "bottle", "bottles", "bag", "bags",
        "box", "can", "jar", "gallon", "fl", "g", "kg", "ml", "liter",
        "count", "organic", "fresh", "large", "small", "medium",
        # Adjective/quality words that don't identify the product
        "grade", "premium", "select", "choice", "natural", "original",
        "classic", "regular", "extra", "super", "ultra", "value",
        "new", "improved", "lite", "light", "reduced", "low", "fat", "free",
        "sugar", "sodium", "calorie",
    }

    def _tokenize(text: str) -> set:
        return {
            w for w in re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
            if w not in _stop_words and len(w) >= 3
        }

    # Core search tokens — what the product fundamentally IS
    search_tokens = _tokenize(name)

    # ── Identify "must-match" core nouns from the item name ───────────────────
    # These are the defining words (e.g. for "sour cream" both "sour" and "cream"
    # must be in the Kroger product, not just any shared word like "spring").
    # For names with 1-2 meaningful words, ALL must match.
    # For names with 3+ meaningful words, at least 2 must match.
    core_tokens = search_tokens  # default: all tokens must match
    min_overlap_needed = len(search_tokens)  # require full overlap by default
    if len(search_tokens) >= 3:
        min_overlap_needed = max(2, len(search_tokens) - 1)  # allow 1 miss for longer names

    async def _search_kroger(search_name: str, tokens: set, min_overlap: int) -> Optional[str]:
        """Perform one Kroger search and return the best matching image URL."""
        try:
            search_term = urllib.parse.quote(search_name)
            async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
                r = await client.get(
                    f"https://api.kroger.com/v1/products?filter.term={search_term}&filter.limit=8",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Accept": "application/json",
                    },
                )
                if r.status_code != 200:
                    print(f"[KROGER] HTTP {r.status_code} for '{search_name}'", flush=True)
                    return None

                products = r.json().get("data", [])
                if not products:
                    print(f"[KROGER] no products for '{search_name}'", flush=True)
                    return None

                for product in products:
                    description = (product.get("description") or "").lower().strip()
                    brand       = (product.get("brand")       or "").lower().strip()
                    full_text   = f"{brand} {description}".strip()

                    if not description:
                        continue

                    product_tokens = _tokenize(full_text)

                    # Count how many of our search tokens appear in this product
                    overlap = tokens & product_tokens
                    overlap_count = len(overlap)

                    if overlap_count < min_overlap:
                        print(
                            f"[KROGER] rejected '{description}' — overlap={overlap_count}/{min_overlap} "
                            f"(need {min_overlap}) for '{search_name}'",
                            flush=True
                        )
                        continue

                    # Good match — find the best image
                    images = product.get("images", [])

                    # Prefer "front" perspective images, then any other
                    def image_priority(img_obj):
                        perspective = (img_obj.get("perspective") or "").lower()
                        return 0 if perspective == "front" else 1

                    for img in sorted(images, key=image_priority):
                        sizes = img.get("sizes", []) or []
                        # Prefer larger sizes: xlarge > large > medium > small > thumbnail
                        size_order = {"xlarge": 0, "large": 1, "medium": 2, "small": 3, "thumbnail": 4}
                        for sz in sorted(sizes, key=lambda s: size_order.get((s.get("size") or "").lower(), 99)):
                            url = (sz.get("url") or "").strip()
                            if not url or not url.startswith("http"):
                                continue
                            # Reject store brand /0001111 mascot images
                            if "/0001111" in url:
                                print(f"[KROGER] rejected store brand URL for '{search_name}': {url[:60]}", flush=True)
                                continue
                            print(
                                f"[KROGER] ✅ matched '{description}' (overlap={overlap_count}) for '{search_name}': {url[:60]}",
                                flush=True
                            )
                            return url

                print(f"[KROGER] no valid match for '{search_name}'", flush=True)
                return None

        except Exception as e:
            print(f"[KROGER] exception for '{search_name}': {e}", flush=True)
            return None

    # ── Attempt 1: Full name, require all core tokens to match ──────────────────
    result = await _search_kroger(name, core_tokens, min_overlap_needed)
    if result:
        return result

    # ── Attempt 2: Simplified 1-2 word fallback ───────────────────────────────
    # Use the simplified Unsplash name as a fallback search term
    simplified = _simplify_for_unsplash(name)
    if simplified and simplified.lower() != name.lower():
        simplified_tokens = _tokenize(simplified)
        if simplified_tokens:
            # Require all simplified tokens to match (they're already stripped down)
            result = await _search_kroger(simplified, simplified_tokens, len(simplified_tokens))
            if result:
                return result

    print(f"[KROGER] ❌ no image found for '{name}'", flush=True)
    return None


async def _gemini_icon(name: str, photo_query: Optional[str] = None) -> Optional[bytes]:
    """
    Generate a colorful flat illustration icon for a grocery item.
    Uses Gemini REST API directly (httpx) — no SDK image generation needed.
    Returns raw PNG bytes, or None on failure.
    """
    try:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            return None

        # ── Step 1: Build subject description ────────────────────────────────
        if photo_query and len(photo_query.strip()) > 3:
            subject = photo_query.strip()
        else:
            text_model = genai.GenerativeModel("gemini-2.5-flash")
            describe_prompt = (
                f'A grocery item named "{name}" needs a product illustration for a mobile app.\n'
                f'Describe in 3-6 words exactly what physical object to draw.\n'
                f'Rules:\n'
                f'- Be specific about the form: "paper towel roll", "cat litter jug", "raw chicken breast", "pasta shells box"\n'
                f'- Never say "plate of" or "bowl of" or "dish of" — always the raw product or its package\n'
                f'- For produce: "fresh [item]" e.g. "fresh strawberries", "bunch of bananas"\n'
                f'- For packaged goods: include container type e.g. "sour cream container", "orange juice carton"\n'
                f'Return ONLY the 3-6 word description, nothing else.'
            )
            resp = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, text_model.generate_content, describe_prompt),
                timeout=4.0
            )
            subject = resp.text.strip().strip('"').strip("'")
            if not subject:
                subject = name

        illustration_prompt = (
            f"Colorful flat vector illustration of {subject}, "
            f"clean white background, bold vibrant colors, simple friendly shapes, "
            f"modern grocery app icon style, no text, no labels, no shadows, centered"
        )
        print(f"[GEMINI ICON] generating for '{name}': {illustration_prompt}", flush=True)

        # ── Step 2: Call Gemini image generation via REST API ────────────────
        # Uses httpx directly — works with any SDK version installed.
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-3.1-flash-image:generateContent?key={api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": illustration_prompt}]}],
            "generationConfig": {"responseModalities": ["IMAGE"]}
        }
        async with httpx.AsyncClient(timeout=25.0) as hc:
            r = await hc.post(url, json=payload)
            if r.status_code != 200:
                print(f"[GEMINI ICON] HTTP {r.status_code} for '{name}': {r.text[:200]}", flush=True)
                return None
            data = r.json()

        # Extract base64 image from response
        for candidate in data.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    png_bytes = base64.b64decode(inline["data"])
                    print(f"[GEMINI ICON] ✅ generated {len(png_bytes)} bytes for '{name}'", flush=True)
                    return png_bytes

        print(f"[GEMINI ICON] no image in response for '{name}': {str(data)[:200]}", flush=True)
        return None

    except asyncio.TimeoutError:
        print(f"[GEMINI ICON] timeout for '{name}'", flush=True)
        return None
    except Exception as e:
        print(f"[GEMINI ICON] error for '{name}': {e}", flush=True)
        return None


@app.get("/image")
async def get_product_image(name: str = Query(...), upc: Optional[str] = Query(None), product_id: Optional[str] = Query(None), category: Optional[str] = Query(None), photo_query: Optional[str] = Query(None)):
    ck = _cache_key(name, upc, product_id)

    if ck in _IMAGE_CACHE and ck in _IMAGE_CONTENT_TYPE_CACHE:
        return Response(content=_IMAGE_CACHE[ck], media_type=_IMAGE_CONTENT_TYPE_CACHE[ck])

    # ── STEP 1: PERSISTENT CACHE ─────────────────────────────────────────────
    # Generated icons are stored as data:image/png;base64,... strings.
    # Once an icon is generated it lasts forever — instant on all future scans.
    cached_val = _get_persistent_image(name)
    if cached_val:
        if cached_val.startswith("data:image/"):
            # It's a stored base64 icon — decode and return directly
            try:
                header, b64data = cached_val.split(",", 1)
                ctype = header.split(":")[1].split(";")[0]
                img_bytes = base64.b64decode(b64data)
                _IMAGE_CACHE[ck] = img_bytes
                _IMAGE_CONTENT_TYPE_CACHE[ck] = ctype
                _trim_caches_if_needed()
                print(f"[ICON CACHE] hit for '{name}'", flush=True)
                return Response(content=img_bytes, media_type=ctype)
            except Exception as e:
                print(f"[ICON CACHE] decode error for '{name}': {e}", flush=True)
        else:
            # Legacy URL entry — ignore it, regenerate with icon pipeline
            print(f"[ICON CACHE] legacy URL for '{name}', regenerating as icon", flush=True)

    # ── STEP 2: GENERATE ICON WITH GEMINI ────────────────────────────────────
    # Gemini generates a colorful flat illustration for the exact item.
    # On first scan this takes ~5 seconds. After that it's cached forever.
    icon_bytes = await _gemini_icon(name, photo_query=photo_query)
    if icon_bytes:
        # Store as base64 data URL in persistent cache — lasts forever
        b64 = base64.b64encode(icon_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{b64}"
        _set_persistent_image(name, data_url)
        _IMAGE_CACHE[ck] = icon_bytes
        _IMAGE_CONTENT_TYPE_CACHE[ck] = "image/png"
        _trim_caches_if_needed()
        print(f"[GEMINI ICON] cached new icon for '{name}'", flush=True)
        return Response(content=icon_bytes, media_type="image/png")

    # ── STEP 3: FALLBACK ──────────────────────────────────────────────────────
    # Gemini failed (timeout, API error). Return a tiny transparent placeholder.
    # This will NOT be cached so next scan tries Gemini again.
    print(f"[IMAGE] Gemini icon failed for '{name}', returning placeholder", flush=True)
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


@app.post("/admin/clear-image-cache")
async def admin_clear_image_cache(key: Optional[str] = Query(None)):
    """Clears both the in-memory image cache and the persistent disk cache."""
    _require_admin(key)
    _IMAGE_CACHE.clear()
    _IMAGE_CONTENT_TYPE_CACHE.clear()
    global _PERSISTENT_IMAGE_URL_CACHE
    _PERSISTENT_IMAGE_URL_CACHE = {}
    try:
        if os.path.exists(_PERSISTENT_CACHE_PATH):
            os.remove(_PERSISTENT_CACHE_PATH)
    except Exception as e:
        print(f"[CACHE CLEAR] failed to delete cache file: {e}", flush=True)
    print("[CACHE CLEAR] All image caches cleared by admin.", flush=True)
    return {"status": "ok", "message": "All image caches cleared. Next requests will fetch fresh photos."}


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




# ---------------------------------------------------------------------------
# /suggest-meals  — Gemini-powered meal suggestions based on pantry items
# ---------------------------------------------------------------------------
class SuggestMealsRequest(BaseModel):
    ingredients: List[str]          # food item names from the user's pantry
    slot: str                       # "breakfast", "lunch", or "dinner"
    count: int = 8                  # how many suggestions to return

class MealSuggestion(BaseModel):
    title: str
    description: str
    missing: List[str]              # ingredients needed but not in pantry
    photo_url: Optional[str] = None

@app.post("/suggest-meals")
@app.post("/suggest-meals/")
async def suggest_meals(req: SuggestMealsRequest) -> List[MealSuggestion]:
    slot = req.slot.lower().strip()
    if slot not in ("breakfast", "lunch", "dinner"):
        slot = "dinner"

    ingredients_text = ", ".join(req.ingredients) if req.ingredients else "various pantry staples"

    prompt = f"""You are a helpful meal planning assistant.

The user has these ingredients available: {ingredients_text}

Suggest exactly {req.count} {slot} meal ideas. Prioritize meals the user can make with what they already have.
If a meal needs 1-3 extra ingredients, that is okay — list them clearly.
Do NOT suggest meals that need more than 3 ingredients the user doesn't have.

Return ONLY a valid JSON array with exactly {req.count} objects. Each object must have:
- "title": short meal name (e.g. "Scrambled Eggs with Toast")
- "description": one sentence description (e.g. "A simple and filling breakfast ready in 10 minutes")
- "missing": array of ingredient names the user needs to buy (empty array if they have everything)

Example format:
[
  {{"title": "Chicken Stir Fry", "description": "Quick and tasty stir fry with vegetables over rice.", "missing": []}},
  {{"title": "Pasta Primavera", "description": "Light pasta with fresh vegetables in olive oil.", "missing": ["pasta", "zucchini"]}}
]

Only return the JSON array. No markdown, no explanation, no code blocks."""

    suggestions: List[MealSuggestion] = []

    if not GEMINI_API_KEY:
        return suggestions

    try:
        for model_name in ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"):
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                raw = response.text.strip()
                # Strip markdown code fences if present
                raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.IGNORECASE)
                raw = re.sub(r"\n?```$", "", raw, flags=re.IGNORECASE)
                raw = raw.strip()
                parsed = json.loads(raw)
                if not isinstance(parsed, list):
                    continue

                # Build suggestions — Edamam recipe image first, then Unsplash with food hint
                async def fetch_photo(title: str) -> Optional[str]:
                    photo = await _edamam_recipe_image(title)
                    if not photo:
                        # Append "food dish" so Unsplash returns a plated meal, not random
                        photo = await _unsplash_image(title + " food dish")
                    return photo

                photo_tasks = [fetch_photo(item.get("title", "")) for item in parsed]
                photos = await asyncio.gather(*photo_tasks)

                for item, photo in zip(parsed, photos):
                    title = (item.get("title") or "").strip()
                    description = (item.get("description") or "").strip()
                    missing = [m.strip() for m in (item.get("missing") or []) if m.strip()]
                    if title:
                        suggestions.append(MealSuggestion(
                            title=title,
                            description=description,
                            missing=missing,
                            photo_url=photo
                        ))

                print(f"[SUGGEST-MEALS] {model_name} returned {len(suggestions)} {slot} suggestions", flush=True)
                break

            except Exception as exc:
                print(f"[SUGGEST-MEALS] {model_name} failed: {exc}", flush=True)
                continue

    except Exception as exc:
        print(f"[SUGGEST-MEALS] Fatal error: {exc}", flush=True)

    return suggestions



# ---------------------------------------------------------------------------
# /recipe-detail  — Gemini generates full recipe for a given meal name
# ---------------------------------------------------------------------------
class RecipeDetailRequest(BaseModel):
    title: str          # meal name e.g. "Chicken Stir Fry"
    description: str = ""

class RecipeDetailResponse(BaseModel):
    title: str
    description: str
    prep_time: str
    cook_time: str
    servings: str
    ingredients: List[str]
    instructions: List[str]
    photo_url: Optional[str] = None

@app.post("/recipe-detail")
@app.post("/recipe-detail/")
async def recipe_detail(req: RecipeDetailRequest) -> RecipeDetailResponse:
    prompt = f"""You are a professional chef writing a clear, easy-to-follow recipe.

Write a complete recipe for: {req.title}

Return ONLY a valid JSON object with these exact fields:
- "title": the meal name
- "description": one appealing sentence about the dish
- "prep_time": preparation time as a string (e.g. "10 minutes")
- "cook_time": cooking time as a string (e.g. "20 minutes")
- "servings": number of servings as a string (e.g. "4 servings")
- "ingredients": array of strings, each one ingredient with amount (e.g. "2 chicken breasts", "1 cup rice", "3 cloves garlic")
- "instructions": array of strings, each one clear numbered step (do NOT include the number, just the instruction text)

Example format:
{{
  "title": "Chicken Stir Fry",
  "description": "A quick and flavorful stir fry with tender chicken and crisp vegetables.",
  "prep_time": "10 minutes",
  "cook_time": "15 minutes",
  "servings": "4 servings",
  "ingredients": ["2 chicken breasts, sliced thin", "1 cup broccoli florets", "2 tbsp soy sauce"],
  "instructions": ["Heat oil in a large wok over high heat.", "Add chicken and cook until golden, about 5 minutes.", "Add vegetables and stir fry for 3 more minutes."]
}}

Only return the JSON object. No markdown, no explanation, no code blocks."""

    fallback = RecipeDetailResponse(
        title=req.title,
        description=req.description,
        prep_time="",
        cook_time="",
        servings="",
        ingredients=[],
        instructions=[]
    )

    if not GEMINI_API_KEY:
        return fallback

    try:
        for model_name in ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"):
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                raw = response.text.strip()
                raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.IGNORECASE)
                raw = re.sub(r"\n?```$", "", raw, flags=re.IGNORECASE)
                raw = raw.strip()
                parsed = json.loads(raw)

                # Fetch photo from Edamam first, fall back to Unsplash with food hint
                photo_url = None
                try:
                    photo_url = await _edamam_recipe_image(req.title)
                    if not photo_url:
                        photo_url = await _unsplash_image(req.title + " food dish")
                except Exception:
                    pass

                print(f"[RECIPE-DETAIL] {model_name} generated recipe for: {req.title}", flush=True)

                return RecipeDetailResponse(
                    title=parsed.get("title", req.title),
                    description=parsed.get("description", req.description),
                    prep_time=parsed.get("prep_time", ""),
                    cook_time=parsed.get("cook_time", ""),
                    servings=parsed.get("servings", ""),
                    ingredients=[str(i) for i in parsed.get("ingredients", [])],
                    instructions=[str(s) for s in parsed.get("instructions", [])],
                    photo_url=photo_url
                )

            except Exception as exc:
                print(f"[RECIPE-DETAIL] {model_name} failed: {exc}", flush=True)
                continue

    except Exception as exc:
        print(f"[RECIPE-DETAIL] Fatal error: {exc}", flush=True)

    return fallback

# =========================
# /parse-barcode
# Accepts a product name from a UPC lookup and runs it through
# the same Gemini AI enrichment + image fetch as a receipt scan.
# Returns a single-item array in the same ParsedItem format.
# =========================

class BarcodeRequest(BaseModel):
    product_name: str


@app.post("/parse-barcode")
@app.post("/parse-barcode/")
async def parse_barcode(req: BarcodeRequest, request: Request):
    product_name = req.product_name.strip()
    if not product_name:
        raise HTTPException(status_code=422, detail="product_name is required")

    print(f"[BARCODE] Looking up: {product_name!r}", flush=True)

    # Step 1: Use Gemini to get shelf life + category — identical prompt to receipt flow
    item_dict = {"name": product_name, "category": "Food"}
    enriched_list = await enrich_items_with_ai([item_dict])
    enriched = enriched_list[0] if enriched_list else {}

    full_name    = (enriched.get("full_name") or product_name).strip()
    category     = (enriched.get("category") or "Food").strip()
    expires_days = enriched.get("expires_in_days")
    storage      = enriched.get("storage") or "pantry"
    fridge_days  = enriched.get("fridge")
    freezer_days = enriched.get("freezer")
    pantry_days  = enriched.get("pantry")

    # Build shelf_life_by_storage exactly like the receipt flow
    shelf_life: Dict[str, int] = {}
    if isinstance(fridge_days, int):  shelf_life["fridge"]  = fridge_days
    if isinstance(freezer_days, int): shelf_life["freezer"] = freezer_days
    if isinstance(pantry_days, int):  shelf_life["pantry"]  = pantry_days

    # Step 2: Fetch product image — same pipeline as receipt: Unsplash → Freepik
    base_url = _public_base_url(request)
    image_url = _image_url_for_item(base_url, full_name, category)

    # Step 3: Build response in the same ParsedItem format the iOS app expects
    result = {
        "name":                  full_name,
        "original_name":         product_name,
        "quantity":              1,
        "category":              category,
        "expires_in_days":       expires_days,
        "storage":               storage,
        "shelf_life_by_storage": shelf_life if shelf_life else None,
        "image_url":             image_url,
    }

    print(f"[BARCODE] Result: {full_name!r} cat={category} expires={expires_days}d storage={storage}", flush=True)

    # Return as a single-item array so the iOS app handles it
    # identically to a receipt response
    return [result]

# =========================
# /parse-label
# Accepts a pill bottle label image. OCR reads whatever text is
# visible on the bottle front, then Gemini identifies the exact
# product and looks up the FULL known drug label information for
# that specific product — directions, warnings, interactions, etc.
# OCR identifies the product. Gemini supplies the complete info.
# =========================

class LabelSection(BaseModel):
    title: str
    content: str

class LabelParseResponse(BaseModel):
    product_name: str
    sections: list[LabelSection]
    raw_ocr: str
    disclaimer: str

@app.post("/parse-label")
@app.post("/parse-label/")
async def parse_label(file: UploadFile = File(...)):
    # ── Step 1: Read image bytes ──────────────────────────────────────────────
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Empty image file.")

    print(f"[LABEL] Received image: {file.filename} ({len(image_bytes)} bytes)", flush=True)

    # ── Step 2: OCR via Google Cloud Vision to identify the product ───────────
    try:
        vision_client = vision.ImageAnnotatorClient()
        vision_image = vision.Image(content=image_bytes)
        ocr_response = vision_client.text_detection(image=vision_image)
        texts = ocr_response.text_annotations
        if not texts:
            raise HTTPException(status_code=422, detail="No text found on label. Try better lighting or a closer photo.")
        raw_ocr = texts[0].description.strip()
        print(f"[LABEL] OCR extracted {len(raw_ocr)} chars: {raw_ocr[:200]!r}", flush=True)
    except Exception as exc:
        print(f"[LABEL] OCR failed: {exc}", flush=True)
        raise HTTPException(status_code=500, detail=f"OCR failed: {exc}")

    # ── Step 3: Gemini identifies the product and looks up full label info ────
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured.")

    prompt = f"""You are a medication label assistant. A user has scanned the front of a pill bottle.

The OCR text read from the bottle is:
\"\"\"
{raw_ocr}
\"\"\"

Your job:
1. Identify the exact product name and brand from the OCR text (e.g. "Excedrin Migraine Relief", "Tylenol Extra Strength", "Advil Ibuprofen 200mg").
2. Using your knowledge of that specific FDA-approved product, return the COMPLETE official label information for it — including all sections that appear on the full drug label, even if they were not visible on the front of the bottle due to peel labels or limited visible text.
3. Only return factual, publicly known drug label information for this exact product. Do not invent anything.
4. If you cannot confidently identify the product from the OCR text, return an empty sections array.

Return this JSON format:
{{
  "product_name": "Full product name and brand",
  "sections": [
    {{"title": "Active Ingredients", "content": "full content"}},
    {{"title": "Uses", "content": "full content"}},
    {{"title": "Directions", "content": "full content"}},
    {{"title": "Warnings", "content": "full content"}},
    {{"title": "Do Not Use", "content": "full content"}},
    {{"title": "Ask a Doctor", "content": "full content"}},
    {{"title": "Ask a Doctor or Pharmacist", "content": "full content"}},
    {{"title": "Stop Use and Ask a Doctor", "content": "full content"}},
    {{"title": "Inactive Ingredients", "content": "full content"}},
    {{"title": "Storage", "content": "full content"}},
    {{"title": "Other Information", "content": "full content"}}
  ]
}}

Only include sections that exist for this product. Return ONLY valid JSON. No markdown, no explanation, no extra text."""

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        gemini_response = model.generate_content(prompt)
        raw_json = gemini_response.text.strip()

        # Strip markdown code fences if present
        if raw_json.startswith("```"):
            raw_json = re.sub(r"^```[a-z]*\n?", "", raw_json)
            raw_json = re.sub(r"\n?```$", "", raw_json)
            raw_json = raw_json.strip()

        parsed = json.loads(raw_json)
        print(f"[LABEL] Gemini returned {len(parsed.get('sections', []))} sections for: {parsed.get('product_name', '')!r}", flush=True)

    except Exception as exc:
        print(f"[LABEL] Gemini lookup failed: {exc}", flush=True)
        raise HTTPException(status_code=500, detail=f"Label lookup failed: {exc}")

    sections = parsed.get("sections", [])
    if not sections:
        raise HTTPException(status_code=422, detail="Could not identify product from label. Try a clearer photo of the product name.")

    return {
        "product_name": parsed.get("product_name", ""),
        "sections": sections,
        "raw_ocr": raw_ocr,
        "disclaimer": "Information is sourced from the FDA-approved label for this product. This is not medical advice."
    }
    
@app.middleware("http")
async def _log_requests(request: Request, call_next):
    try:
        resp = await call_next(request)
        return resp
    finally:
        print(f"{request.method} {request.url.path}")
