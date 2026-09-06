"""
============================================================================
IMAGESHOP AI CATEGORIZATION PIPELINE  v5
============================================================================

KEY IMPROVEMENTS OVER v4 (see PIPELINE_POSTMORTEM.md for full details):

  ACCURACY
  --------
  1. Subcategory validation: case-normalize → fuzzy match (difflib) → fallback
     (Eliminates silent wrong assignments from ISSUE-01)
  2. Low-confidence classifications get AIStatus='needs_review' instead of being
     silently inserted as wrong data (from ISSUE-04)

  PERFORMANCE  (estimated total: 5–6h → ~1.5–2h for 5k images)
  -----------
  3. LLaVA batch inference: LLAVA_BATCH_SIZE images per forward pass (~2.5x Phase 1)
  4. Mistral batch inference: MISTRAL_BATCH_SIZE captions per 2 calls instead of 2N
     calls (~4–8x Phase 2)
  5. Disk-cached raw image bytes: Phase 2 reads from local disk, not S3 (~50% less
     S3 bandwidth, eliminates double-download)
  6. JSONL append-only caption cache: O(1) write per caption vs O(n²) full-file
     rewrite (from ISSUE-06)
  7. Single reused aioboto3 S3 client: one TLS handshake instead of one per request
     (from ISSUE-09)

  PRODUCTION SAFETY
  -----------------
  8. Startup config validation: fails immediately with clear message if .env is
     incomplete — before loading any models (from ISSUE-13)
  9. Thread-safe DB connections: threading.local gives each thread its own
     connection — no race conditions (from ISSUE-11)
 10. Failed images excluded from "done" dedup and retried on next run (up to 3x)
     (from ISSUE-12)
 11. Fixed S3 public URL to include AWS region (was broken for non-us-east-1)
     (from ISSUE-14)

  HARDWARE SAFETY
  ---------------
 12. Optional GPU temperature monitoring via pynvml: auto-pauses when GPU reaches
     GPU_MAX_TEMP_C°C (from ISSUE-10)
     Install: pip install nvidia-ml-py3

  CODE QUALITY
  ------------
 13. asyncio.get_running_loop() replaces deprecated get_event_loop() (ISSUE-16)
 14. Prompt strings built without indentation — clean left-aligned text to Mistral
     (from ISSUE-03)
 15. All dead commented-out v3/v4 code removed (from ISSUE-17)

VRAM BUDGET (RTX 22 GB):
  Phase 1: LLaVA float16 ~14 GB + batch=4 overhead ~2 GB = ~16 GB  [OK]
  Phase 2: Mistral float16 ~8 GB + batch=8 overhead ~2 GB = ~10 GB  [OK]

PREREQUISITES:
  pip install torch transformers pillow aioboto3 pyodbc tqdm aiofiles \
              python-dotenv accelerate sentencepiece
  pip install nvidia-ml-py3   # optional, enables GPU temperature monitoring
============================================================================
"""

# ============================================================================
#  STANDARD LIBRARY
# ============================================================================
import os
import io
import re
import json
import asyncio
import logging
import sys
import threading
import hashlib
import difflib
import shutil
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Windows UTF-8 fix — must happen BEFORE any logger is created
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ============================================================================
#  THIRD-PARTY IMPORTS
# ============================================================================
import torch
from PIL import Image
from transformers import (
    LlavaForConditionalGeneration,
    LlavaProcessor,
    LlamaTokenizer,
    CLIPImageProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
)
import aioboto3
import pyodbc
from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Optional GPU temperature monitoring
# Install with: pip install nvidia-ml-py3
# If not installed, all GPU monitoring calls are silently skipped.
# --------------------------------------------------------------------------
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False

load_dotenv()


# ============================================================================
#  ANSI COLOUR HELPERS
# ============================================================================
class C:
    _tty    = sys.stdout.isatty()
    RESET   = "\033[0m"  if _tty else ""
    BOLD    = "\033[1m"  if _tty else ""
    GREEN   = "\033[92m" if _tty else ""
    YELLOW  = "\033[93m" if _tty else ""
    CYAN    = "\033[96m" if _tty else ""
    RED     = "\033[91m" if _tty else ""
    GRAY    = "\033[90m" if _tty else ""
    BLUE    = "\033[94m" if _tty else ""
    MAGENTA = "\033[95m" if _tty else ""


# ============================================================================
#  CONFIGURATION
# ============================================================================
@dataclass
class Config:

    # ── AWS ──────────────────────────────────────────────────────────────────
    AWS_ACCESS_KEY_ID:     str = field(default_factory=lambda: os.getenv("AWS_ACCESS_KEY_ID", ""))
    AWS_SECRET_ACCESS_KEY: str = field(default_factory=lambda: os.getenv("AWS_SECRET_ACCESS_KEY", ""))
    AWS_REGION:            str = field(default_factory=lambda: os.getenv("AWS_REGION", "us-east-1"))

    # ── S3 buckets ───────────────────────────────────────────────────────────
    S3_PRIVATE_BUCKET: str = field(default_factory=lambda: os.getenv("S3_PRIVATE_BUCKET", ""))
    S3_PUBLIC_BUCKET:  str = field(default_factory=lambda: os.getenv("S3_PUBLIC_BUCKET", ""))

    # ── S3 prefixes ──────────────────────────────────────────────────────────
    S3_PRIVATE_PREFIX:   str = field(default_factory=lambda: os.getenv("S3_PRIVATE_PREFIX", "originals/"))
    S3_THUMBNAIL_PREFIX: str = field(default_factory=lambda: os.getenv("S3_THUMBNAIL_PREFIX", "thumbnails/"))
    S3_WATERMARK_PREFIX: str = field(default_factory=lambda: os.getenv("S3_WATERMARK_PREFIX", "watermarks/"))

    # ── SQL Server ───────────────────────────────────────────────────────────
    SQL_SERVER:   str = field(default_factory=lambda: os.getenv("SQL_SERVER", ""))
    SQL_DATABASE: str = field(default_factory=lambda: os.getenv("SQL_DATABASE", ""))
    SQL_USERNAME: str = field(default_factory=lambda: os.getenv("SQL_USERNAME", ""))
    SQL_PASSWORD: str = field(default_factory=lambda: os.getenv("SQL_PASSWORD", ""))
    SQL_DRIVER:   str = field(default_factory=lambda: os.getenv("SQL_DRIVER", "ODBC Driver 18 for SQL Server"))

    # ── Parallelism ──────────────────────────────────────────────────────────
    DOWNLOAD_CONCURRENCY: int = field(default_factory=lambda: int(os.getenv("DOWNLOAD_CONCURRENCY", "20")))
    UPLOAD_CONCURRENCY:   int = field(default_factory=lambda: int(os.getenv("UPLOAD_CONCURRENCY", "20")))
    COMPRESS_THREADS:     int = field(default_factory=lambda: int(os.getenv("COMPRESS_THREADS", "8")))
    # [FIX-06] 1000 is the ListObjectsV2 maximum. 500 doubled the number of
    # listing calls for no benefit.
    S3_PAGE_SIZE:         int = field(default_factory=lambda: int(os.getenv("S3_PAGE_SIZE", "1000")))

    # ── AI batch sizes ───────────────────────────────────────────────────────
    # Safe defaults for 22 GB VRAM.
    # VRAM usage: LLaVA base ~14 GB + ~0.5 GB per image in batch.
    # VRAM usage: Mistral base ~8 GB + ~0.3 GB per prompt in batch.
    LLAVA_BATCH_SIZE:   int = field(default_factory=lambda: int(os.getenv("LLAVA_BATCH_SIZE", "8")))
    MISTRAL_BATCH_SIZE: int = field(default_factory=lambda: int(os.getenv("MISTRAL_BATCH_SIZE", "12")))

    # ── [FIX-08] RE-CLASSIFY MODE ──────────────────────────────────────────
    # Set RECLASSIFY_MODE=true in .env to re-run ONLY the classification step
    # over images that are already in the database, using the caption already
    # stored in the AICaption column.
    #
    # In this mode the pipeline does not touch S3 at all: no listing, no
    # download, no upload, no compression, and LLaVA is never loaded. It reads
    # captions, runs Mistral, and UPDATEs CategoryId / SubCategoryId. Nothing
    # is inserted and nothing is deleted.
    #
    # Leave it false (or absent) and the script behaves exactly as before.
    RECLASSIFY_MODE:  bool = field(default_factory=lambda:
        os.getenv("RECLASSIFY_MODE", "false").strip().lower() in ("1", "true", "yes", "on"))

    # Optional comma-separated TaxonomyKeys to restrict the pass, e.g.
    #   RECLASSIFY_CATEGORIES=Wallpaper,Backgrounds,Nature,Drawings
    # Images already sitting in a specific category are usually already right,
    # so limiting the pass to the broad buckets is cheaper and lower risk.
    # Leave empty to re-classify every completed image.
    RECLASSIFY_CATEGORIES: str = field(default_factory=lambda:
        os.getenv("RECLASSIFY_CATEGORIES", "").strip())

    # Cap the pass for testing. Empty or 0 means no cap.
    RECLASSIFY_LIMIT: Optional[int] = field(default_factory=lambda:
        int(os.getenv("RECLASSIFY_LIMIT")) if os.getenv("RECLASSIFY_LIMIT") else None)

    # Dry run: classify and log what WOULD change, but write nothing.
    # Run once with this true before the real pass.
    RECLASSIFY_DRY_RUN: bool = field(default_factory=lambda:
        os.getenv("RECLASSIFY_DRY_RUN", "false").strip().lower() in ("1", "true", "yes", "on"))

    RECLASSIFY_CHECKPOINT: str = field(default_factory=lambda:
        os.getenv("RECLASSIFY_CHECKPOINT", "reclassify_checkpoint.json"))

    # [FIX-02] Caption length budget. v5 hardcoded 180, which is far more than the
    # four-field caption ever needs. In batched generation the whole batch runs
    # until the LONGEST sequence finishes, so an over-large budget wastes time on
    # every image in the batch. 110 covers the structured caption comfortably.
    LLAVA_MAX_NEW_TOKENS: int = field(default_factory=lambda: int(os.getenv("LLAVA_MAX_NEW_TOKENS", "110")))

    # ── Image compression ────────────────────────────────────────────────────
    THUMBNAIL_MAX_PX:  int = field(default_factory=lambda: int(os.getenv("THUMBNAIL_MAX_PX", "400")))
    THUMBNAIL_QUALITY: int = field(default_factory=lambda: int(os.getenv("THUMBNAIL_QUALITY", "75")))
    WATERMARK_MAX_PX:  int = field(default_factory=lambda: int(os.getenv("WATERMARK_MAX_PX", "1200")))
    WATERMARK_QUALITY: int = field(default_factory=lambda: int(os.getenv("WATERMARK_QUALITY", "82")))

    # ── Processing cap ───────────────────────────────────────────────────────
    MAX_IMAGES: Optional[int] = field(
        default_factory=lambda: int(os.getenv("MAX_IMAGES", "0")) or None
    )

    # ── Files ────────────────────────────────────────────────────────────────
    CHECKPOINT_FILE:     str = field(default_factory=lambda: os.getenv("CHECKPOINT_FILE", "pipeline_checkpoint.json"))
    CHECKPOINT_INTERVAL: int = field(default_factory=lambda: int(os.getenv("CHECKPOINT_INTERVAL", "100")))
    # JSONL format (one JSON object per line — append-only, O(1) writes)
    CAPTIONS_CACHE_FILE: str = field(default_factory=lambda: os.getenv("CAPTIONS_CACHE_FILE", "captions_cache.jsonl"))
    # Raw image bytes cached to disk so Phase 2 does not re-download from S3
    IMAGE_CACHE_DIR:     str = field(default_factory=lambda: os.getenv("IMAGE_CACHE_DIR", ".image_cache"))
    LOG_FILE:            str = field(default_factory=lambda: os.getenv("LOG_FILE", "pipeline.log"))
    LOG_LEVEL:           int = logging.INFO

    # ── Mistral generation ───────────────────────────────────────────────────
    MISTRAL_TEMPERATURE:    float = field(default_factory=lambda: float(os.getenv("MISTRAL_TEMPERATURE", "0.1")))
    # [FIX-01] v5 defined this but never used it — both classification calls were
    # hardcoded to max_new_tokens=20. Long subcategory names such as
    # {"subcategory": "Positive Covid Test Picture"} need ~22 tokens, so they were
    # truncated mid-JSON, failed to parse, and silently fell back to the first
    # subcategory in the list. 40 leaves headroom for every name in the taxonomy.
    MISTRAL_MAX_NEW_TOKENS: int   = field(default_factory=lambda: int(os.getenv("MISTRAL_MAX_NEW_TOKENS", "40")))

    # ── GPU thermal protection ───────────────────────────────────────────────
    GPU_MAX_TEMP_C:  int = field(default_factory=lambda: int(os.getenv("GPU_MAX_TEMP_C", "80")))
    GPU_COOLDOWN_S:  int = field(default_factory=lambda: int(os.getenv("GPU_COOLDOWN_S", "60")))
    # Check GPU temperature every N images (0 = disable)
    GPU_CHECK_EVERY: int = field(default_factory=lambda: int(os.getenv("GPU_CHECK_EVERY", "100")))

    def validate(self):
        """
        Fail fast before loading models if required .env values are missing.
        Called once at startup — saves wasting an hour of model loading on a misconfigured run.
        """
        required = {
            "AWS_ACCESS_KEY_ID":     self.AWS_ACCESS_KEY_ID,
            "AWS_SECRET_ACCESS_KEY": self.AWS_SECRET_ACCESS_KEY,
            "S3_PRIVATE_BUCKET":     self.S3_PRIVATE_BUCKET,
            "S3_PUBLIC_BUCKET":      self.S3_PUBLIC_BUCKET,
            "SQL_SERVER":            self.SQL_SERVER,
            "SQL_DATABASE":          self.SQL_DATABASE,
            "SQL_USERNAME":          self.SQL_USERNAME,
            "SQL_PASSWORD":          self.SQL_PASSWORD,
        }
        missing = [k for k, v in required.items() if not v.strip()]
        if missing:
            raise ValueError(
                f"\n[ERROR] Missing required .env values: {', '.join(missing)}\n"
                f"Copy .env.example to .env and fill in all values before running.\n"
            )


# ============================================================================
#  TAXONOMY  (do not modify — values are foreign keys in the database)
# ============================================================================
# ==========================================================================
#  TAXONOMY v2  --  paste this over the existing TAXONOMY block in p6.py
#  Generated from taxonomy_v2.json. Must stay in sync with the database;
#  run 05_seed_taxonomy.sql after any change here.
#  25 categories, 615 subcategory slots
# ==========================================================================

TAXONOMY: Dict[str, List[str]] = {
    "Backgrounds": [
        "Background", "White Background", "Black Background", "Christmas Background",
        "Cool Background", "Cute Background", "Pink Background", "Aesthetic Backgrounds",
        "Blue Background", "Fall Background", "Red Background", "Green Background",
        "Halloween Background", "Purple Background", "Beach Background",
        "Space Background", "Anime Background", "Flower Background", "Galaxy Background",
        "Gold Background", "Yellow Background", "Grey Background", "Heart Background",
        "Light Blue Background", "Rainbow Background", "Winter Background",
        "Marble Background", "Office Background", "Spring Background",
        "Birthday Background", "Stockphoto-Graf",
    ],
    "Nature": [
        "Nature Background", "Sunset", "Flowers", "Sunrise", "Autumn", "Water", "Beach",
        "Ocean", "Desert", "Star", "Moon", "Spring", "Exercise", "Summer", "Forest",
        "Camping", "Rain", "Hiking", "Clouds", "Lake", "Winter", "Mountains", "Underwater",
        "Wind", "Recycling", "Moss", "Dusk", "Nature Wallpaper", "People In Nature",
        "Gardening", "Wildlife", "Fall Wallpaper", "Red Moon", "Fall Leaves", "Landscapes",
        "Ocean Background", "Travel Mania",
    ],
    "Business": [
        "Stock Market", "Recession", "Money", "Business Casual", "Office",
        "Cryptocurrency", "Teleworking", "Business", "Digital Marketing",
        "Customer Service", "Work From Home", "Marketing", "Bankruptcy", "Globalization",
        "Economics", "Fintech", "Biotechnology", "Business Plan", "Business Card",
        "Small Business", "Infographics", "Conference Room", "Business Card Design",
        "Happy Work Anniversary", "Conference Call", "Business Man", "Business Woman",
        "Hybrid Work", "Business Travel", "Teamwork",
    ],
    "People": [
        "People", "Team Work", "Children", "Families", "Women", "Group Photo", "Friends",
        "Baby", "Party", "City Streets", "Sporting Event", "Concerts", "Doctor", "Boy",
        "Men", "Business People", "Faces", "Kids", "Wedding", "Happy People",
        "Crowd Of People", "People Icons", "People Walking", "People Eating",
        "Mom And Son", "Dad And Son", "Siblings", "Senior Citizens", "Authentic People",
        "Diverse People", "Drawings Of People", "Silhouettes Of People",
        "Large Group Of People",
    ],
    "Medical": [
        "Hospital", "Nurse", "Positive Pregnancy Test", "Heart", "Doctor", "Eye",
        "Eyeball", "Bacteria", "Medical", "Science", "Mri", "Brain", "Medicine", "Blood",
        "X Ray", "Dental", "Skin Disease", "Cell", "Magnetic Resonance Imaging",
        "Dental Implant", "Medical Background", "Coughing", "Cancer", "Heart Anatomy",
    ],
    "Food": [
        "Food", "Food Truck", "Healthy Food", "Diet", "Mexican Food", "Food Icons",
        "Food Clipart", "Food Vectors", "Seafood", "Food Bank", "Food Pantry", "Fast Food",
        "Italian Food", "Food Delivery", "Food Waste", "Food Background",
        "Food Truck Mockup", "Food Safety", "Chinese Food", "Canned Food", "Dog Food",
        "Junk Food", "Indian Food", "Food Manufacturing",
    ],
    "Technology": [
        "Technology Background", "Laptop", "Cryptocurrency", "Computer", "Podcast",
        "Innovation", "Artificial Intelligence", "Cybersecurity", "Fintech",
        "Information Technology", "Biotechnology", "Robotics", "Smartphone",
        "Blockchain Technology", "Generative Ai", "Nanotechnology", "Science Fiction",
        "Old Computer", "Medical Technology", "Computer Clipart", "Computer Drawing",
        "Cloud Technology", "Healthcare Technology", "Financial Technology",
        "Construction Technology", "Tech Company", "Computer Wallpaper",
        "Technical Difficulties Screen", "Computer Cartoon", "Technology Clipart",
    ],
    "Travel": [
        "Passport", "Beach", "Cruise", "Luggage", "Airport", "Travel Insurance",
        "Spring Break", "Travel Agent", "Family Travel", "Beach Background",
        "Luxury Travel", "Beach Sunset", "Flower Pictures", "Summer Vacation", "Palm Tree",
        "World Travel", "Business Travel", "Adventure Travel", "Travel Background",
        "Holiday Travel", "Space Travel", "Roadtrip", "Vacation Mode", "Out Of Office",
    ],
    "Winter": [
        "Winter Background", "Winter Wallpaper", "Winter Banner", "Winter Border",
        "Winter Holiday", "Winter Scene", "Winter Wonderland", "Snowy Trees",
        "Winter Landscapes", "Winter Solstice", "Winter Home", "Winter Storm",
        "Winter Road", "Winter Trees", "Winter Sky", "Winter Hat", "Family Winter",
        "Winter Coat", "Winter Wedding", "Winter Sports", "Winter Flowers", "Winter Break",
        "Winter Vacation", "Winter Texture", "Winter Icon", "Winter Logo", "Winter Lights",
        "Dog Winter", "Winter Village", "Winter Jacket Mockup",
    ],
    "Clipart": [
        "Heart Clipart", "Flower Clipart", "Book Clipart", "Star Clipart", "Sun Clipart",
        "Basketball Clipart", "Dog Clipart", "Fall Clipart", "Football Clipart",
        "Apple Clipart", "Christmas Clipart", "Halloween Clipart", "Butterfly Clipart",
        "Cat Clipart", "Car Clipart", "Fire Clipart", "Fish Clipart", "Money Clipart",
        "School Clipart", "Snowflake Clipart", "Tree Clipart", "Happy Birthday Clipart",
        "Baseball Clipart", "Bee Clipart", "Dinosaur Clipart", "House Clipart",
        "Airplane Clipart", "Camera Clipart", "Cow Clipart", "Crown Clipart",
        "Pizza Clipart", "Arrow Clipart",
    ],
    "Healthcare": [
        "Ambulance", "Doctors Office Background", "Dermatologist", "Mental Health",
        "Nutrition", "Pediatrician", "Emergency Room", "Surgery", "Wellness", "Nurse",
        "Preventative Medicine", "Medical Insurance", "Sports Medicine", "Telemedicine",
        "Doctors Office", "Hospital Room Background", "First Aid", "Operation",
        "Positive Covid Test Picture", "Medical Equipment", "Dentist Office",
        "Alternative Medicine", "Medical Wallpaper", "Hospital Room", "Medical Logo",
        "Medical Technology", "Healthy Tonsils", "Medical Images", "Hospital Sign",
        "Mental Wellness",
    ],
    "Family": [
        "Home", "Welcome To The Family", "Family Vacation", "Family Dinner", "Pregnancy",
        "Baby", "Parenting", "Wedding", "Marriage", "Children", "Happy Family",
        "Family Travel", "Family Quotes", "Engagement", "Family Therapy", "Siblings",
        "Family Fun", "Family Planning", "Grandparents", "Family Clipart",
        "Family And Friends", "Family Drawing", "Family Cartoon", "Family Outside",
    ],
    "Wallpaper": [
        "Cool Wallpapers", "Cute Wallpapers", "Aesthetic Wallpaper", "Black Wallpaper",
        "Pink Wallpaper", "Anime Wallpaper", "4K Wallpaper", "Desktop Wallpaper",
        "Blue Wallpaper", "Background Wallpaper", "Purple Wallpaper", "Fall Wallpaper",
        "Red Wallpaper", "Space Wallpaper", "Stitch Wallpaper", "Green Wallpaper",
        "Black And White Wallpaper", "Flower Wallpaper", "Galaxy Wallpaper",
        "Heart Wallpaper", "Beach Wallpaper", "Beautiful Wallpaper", "Butterfly Wallpaper",
        "3D Wallpaper", "Nature Wallpaper", "Football Wallpaper", "Laptop Wallpaper",
        "Sunset Wallpaper", "Winter Wallpaper", "Wolf Wallpaper", "Christmas Wallpaper",
        "Halloween Wallpaper", "Computer Wallpaper", "Cat Wallpaper", "Dog Wallpaper",
    ],
    "Drawings": [
        "Butterfly Drawing", "Cool Drawing", "Flower Drawing", "Rose Drawing",
        "Skull Drawing", "Dragon Drawing", "Eye Drawing", "Heart Drawing",
        "Mushroom Drawing", "Tree Drawing", "Christmas Drawing", "Frog Drawing",
        "Horse Drawing", "Bunny Drawing", "Car Drawing", "Cow Drawing", "Fish Drawing",
        "Hand Drawing", "Snake Drawing", "Sunflower Drawing", "Wolf Drawing", "Ai Drawing",
        "Anime Drawing", "Bee Drawing", "Bird Drawing", "Elephant Drawing", "Fire Drawing",
        "Fox Drawing", "Girl Drawing", "Lion Drawing", "Moon Drawing", "Pumpkin Drawing",
        "Shark Drawing", "Skeleton Drawing", "Soccer Ball Drawing", "Football Drawing",
    ],
    "Mockups": [
        "Hoodie Mockup", "Wall Art Mockup", "Menu Mockup", "Tshirt Mockup", "Book Mockup",
        "Poster Mockup", "Beanie Mockup", "Business Card Mockup", "Website Mockup",
        "Magazine Mockup", "Billboard Mockup", "Tote Bag Mockup", "Hat Mockup",
        "Laptop Mockup", "Logo Mockup", "Phone Mockup", "Sticker Mockup", "Box Mockup",
        "Sweatpants Mockup", "Clothing Mockup", "Varsity Jacket Mockup", "Shorts Mockup",
        "Brochure Mockup", "Trucker Hat Mockup", "Computer Mockup", "App Mockup",
        "Banner Mockup", "Flyer Mockup", "Postcard Mockup", "Product Mockup",
    ],
    "Valentines Day": [
        "Happy Valentines Day", "Valentines Day Background", "Valentines Day Banner",
        "Valentines Day Border", "Valentines Day Vector", "Valentines Day Clipart",
        "Valentines Day Dinner", "Valentines Day Celebration", "Valentines Day Party",
        "Valentines Day Flowers", "Valentines Day Card", "Valentines Day Heart",
        "Valentines Day Chocolate", "Valentines Day Text", "Valentines Day Couple",
        "Valentines Day Food", "Valentines Day Presents", "Valentines Day Gifts",
        "Valentines Day Flyer", "Valentines Day Poster", "Valentines Day Icon",
        "Valentines Day Cookies", "Valentines Day Sale", "Valentines Day Date",
        "Dog Valentine", "Cat Valentine", "Will You Be My Valentine", "Valentine Heart",
        "Valentines SVG", "Valentines PNG",
    ],
    "Abstract": [
        "Abstract Art", "Abstract Design", "Abstract Painting", "Abstract Drawing",
        "Abstract Sketches", "Abstract Architecture", "Abstract Shapes",
        "Abstract Patterns", "Abstract Lines", "Geometric Abstract",
    ],
    "Animal": [
        "Dog", "Cat", "Horse", "Bird", "Wolf", "Lion", "Tiger", "Bear", "Fox", "Deer",
        "Elephant", "Rabbit", "Butterfly", "Fish", "Marine Animals", "Insects", "Reptiles",
        "Amphibians", "Farm Animals", "Wild Animals", "Animal Portraits",
        "Animal Silhouettes", "Animal Patterns",
    ],
    "Icons": [
        "3D Icons", "Flat Icons", "Outline Icons", "Filled Icons", "Minimal Icons",
        "Navigation Icons", "Ecommerce Icons", "Education Icons", "Healthcare Icons",
        "Finance Icons", "Travel Icons", "Security Icons", "Weather Icons", "Arrow Icons",
        "App Icons", "Settings Icons", "Interface Symbols",
    ],
    "Education": [
        "Classroom", "Students Studying", "Online Learning", "School", "Teachers",
        "Students", "Graduation", "Study Group", "Digital Classroom", "Training Workshop",
        "Tutoring", "Academic Research", "Library", "College Students",
        "University Campus", "Remote Learning",
    ],
    "Logos": [
        "Monogram Logo", "Wordmark Logo", "Lettermark Logo", "Mascot Logo", "Badge Logo",
        "Emblem Logo", "Abstract Logo", "Geometric Logo", "Vintage Logo", "Luxury Logo",
        "Startup Branding", "Visual Identity", "Negative Space Logo", "Hand Drawn Logo",
    ],
    "Plants & Flowers": [
        "Tropical Plants", "Houseplants", "Indoor Plants", "Potted Plants",
        "Flowering Plants", "Leaves And Foliage", "Plant Textures", "Plant Silhouettes",
        "Seasonal Flowers", "Spring Flowers", "Dried Flowers", "Floral Arrangements",
        "Plant Close-Ups", "Natural Greenery", "Plant Illustrations", "Flower Close-Ups",
    ],
    "Sports": [
        "Running", "Cycling", "Swimming", "Cross Training", "Cardio Training",
        "Personal Training", "Group Fitness", "Home Workouts", "Endurance Sports",
        "Recovery And Stretching", "Fitness Technology", "Sports Equipment",
        "Competitive Sports", "Fitness Lifestyle", "Outdoor Training",
        "Sports Action Shots",
    ],
    "Signs & Symbols": [
        "Astrological Signs", "Street Signs", "Warning Signs", "Traffic Signs",
        "Medical Symbols", "Tech Symbols", "Weather Symbols", "Currency Symbols",
        "Religious Symbols", "Peace Symbols", "LGBTQ+ Symbols", "Gender Symbols",
    ],
    "Holidays": [
        "American Holidays", "Family Holidays", "International Holidays",
        "Religious Holidays", "Celebrations", "Fireworks", "Presents", "Holiday Feast",
        "Spring Holidays", "Summer Holidays", "Fall Holidays",
    ],
}

# ==========================================================================
#  CATEGORY_DESCRIPTIONS  --  used to build the Step 1 classification prompt.
#  These carry the disambiguation rules. Editing them changes how the model
#  decides between overlapping categories, so change them deliberately.
# ==========================================================================

CATEGORY_DESCRIPTIONS: Dict[str, str] = {
    "Backgrounds":
        "A plain background surface: texture, gradient, solid colour field or "
        "seamless repeating pattern with NO composed artwork and no "
        "recognisable subject. If the image has a deliberate artistic "
        "composition, use Abstract instead.",
    "Nature":
        "An outdoor natural scene where the landscape or environment is the "
        "subject: mountains, forests, oceans, skies, weather. If a single "
        "plant or flower is the subject, use Plants & Flowers. If an animal "
        "is the subject, use Animal.",
    "Business":
        "Corporate, office, finance and professional working life.",
    "People":
        "One or more people are the main subject: portraits, lifestyle, "
        "groups, emotion.",
    "Medical":
        "Clinical medicine: hospitals, procedures, medical equipment, doctors "
        "at work.",
    "Food":
        "Food, drink, ingredients, cooking and dining.",
    "Technology":
        "Computing, devices, software, AI, networks and digital technology.",
    "Travel":
        "Destinations, landmarks, tourism, transport and journeys.",
    "Winter":
        "Snow, ice, cold-weather scenes and winter seasonal imagery.",
    "Clipart":
        "Simple flat vector-style illustrations and cut-out graphic elements "
        "intended for reuse in documents and designs.",
    "Healthcare":
        "Wellness, care, fitness for health, therapy and patient wellbeing "
        "outside clinical procedures.",
    "Family":
        "Family relationships and family life across generations.",
    "Wallpaper":
        "A finished image intended to be used as a desktop or phone "
        "wallpaper, with a recognisable scene or composition designed to fill "
        "a screen. If the image has NO subject and is a flat surface, use "
        "Backgrounds. If the subject is an animal, plant, person or place, "
        "use that category instead.",
    "Drawings":
        "Hand-drawn or hand-painted artwork with visible drawing media: "
        "pencil, ink, charcoal, watercolour, or digital brushwork imitating "
        "them.",
    "Mockups":
        "A blank or placeholder product, device screen, packaging or print "
        "item presented for a design to be placed onto.",
    "Valentines Day":
        "Romantic and Valentines Day themed imagery: hearts, roses, couples, "
        "love motifs.",
    "Abstract":
        "Non-representational artwork with a deliberate composition: shapes, "
        "lines, geometry, or painterly forms arranged as the subject itself. "
        "NOT a plain texture or gradient (that is Backgrounds).",
    "Animal":
        "A living animal is the main subject of the image. Prefer the "
        "specific species when one of the listed species applies.",
    "Icons":
        "Sets or single pieces of user-interface iconography: small symbolic "
        "graphics designed for apps, websites and software. NOT real-world "
        "signage (that is Signs & Symbols).",
    "Education":
        "Learning, teaching and academic settings, including students, "
        "teachers, classrooms and study activity.",
    "Logos":
        "Logo and brand identity designs. Product photography and printed "
        "brand collateral belong in Mockups, not here.",
    "Plants & Flowers":
        "A plant or flower is the main subject. Wide landscapes, forests and "
        "outdoor scenery belong in Nature, not here.",
    "Sports":
        "Sport, fitness and physical training activity, equipment or "
        "environments.",
    "Signs & Symbols":
        "Real-world signage and cultural, religious or conceptual symbols. "
        "Application and website iconography belongs in Icons, not here.",
    "Holidays":
        "Holiday celebrations, festive occasions and gift-giving. Winter and "
        "Valentines Day have their own categories and take priority when they "
        "clearly apply.",
}

# ==========================================================================
#  PRIORITY_RULES  --  appended verbatim to the Step 1 prompt.
#  These exist because several categories in this taxonomy answer different
#  questions about the same image (what is in it, how it was made, what it
#  is used for). Without an explicit ordering the model defaults to the
#  use-case answer, which is why Wallpaper and Backgrounds absorbed 60% of
#  the first 60,000 images.
# ==========================================================================

PRIORITY_RULES = """
DECISION ORDER - apply these in sequence and stop at the first match:

1. If a living animal is the main subject, choose Animal.
2. If a plant or flower is the main subject, choose Plants & Flowers.
3. If one or more people are the main subject, choose the best of
   People, Family, Business, Medical, Healthcare, Education or Sports.
4. If the image shows a recognisable object, place, food or scene,
   choose the category for that subject.
5. Only if the image has NO recognisable subject:
     - deliberate artistic composition of shapes, lines or forms -> Abstract
     - flat texture, gradient, solid colour or seamless pattern -> Backgrounds
6. Choose Wallpaper ONLY when the image is a finished screen-filling scene
   and no category from steps 1 to 4 applies.

Wallpaper and Backgrounds are last resorts, never first choices.
"""


# Pre-computed lookup: lowercase category name → canonical name (for case-insensitive matching)
_CAT_LOWER: Dict[str, str] = {k.lower(): k for k in TAXONOMY}

# Pre-computed lookup: (category, lowercase subcategory) → canonical subcategory name
_SUB_LOWER: Dict[Tuple[str, str], str] = {
    (cat, sub.lower()): sub
    for cat, subs in TAXONOMY.items()
    for sub in subs
}


# ============================================================================
#  GPU TEMPERATURE MONITORING  (requires nvidia-ml-py3)
# ============================================================================
def gpu_temp() -> Optional[int]:
    """Return current GPU 0 temperature in Celsius, or None if unavailable."""
    if not _NVML_OK:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    except Exception:
        return None


def maybe_cooldown(logger: logging.Logger, max_temp: int, cooldown_s: int):
    """
    If GPU temperature is at or above max_temp, block for cooldown_s seconds.
    This prevents sustained thermal stress on workstation hardware.
    """
    temp = gpu_temp()
    if temp is None:
        return
    if temp >= max_temp:
        logger.warning(
            f"{C.YELLOW}[THERMAL] GPU at {temp}°C (limit {max_temp}°C) — "
            f"pausing {cooldown_s}s to cool down...{C.RESET}"
        )
        time.sleep(cooldown_s)
        after = gpu_temp()
        if after:
            logger.info(f"{C.GREEN}[THERMAL] GPU now at {after}°C — resuming.{C.RESET}")


# ============================================================================
#  LOGGING
# ============================================================================
def setup_logging(config: Config) -> logging.Logger:
    logger = logging.getLogger("ImageShopPipeline")
    logger.setLevel(config.LOG_LEVEL)
    logger.handlers = []

    fmt_file    = logging.Formatter("%(asctime)s | %(levelname)-8s | %(funcName)s | %(message)s")
    fmt_console = logging.Formatter("%(asctime)s | %(message)s")

    fh = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt_file)

    utf8_stdout = open(
        sys.stdout.fileno(), mode="w",
        encoding="utf-8", errors="replace",
        buffering=1, closefd=False,
    )
    ch = logging.StreamHandler(stream=utf8_stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt_console)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ============================================================================
#  PROGRESS TRACKER
# ============================================================================
class ProgressTracker:
    def __init__(self, logger: logging.Logger):
        self.logger      = logger
        self.total       = 0
        self.index       = 0
        self.success     = 0
        self.failed      = 0
        self.skipped     = 0
        self.needs_review = 0

    def set_total(self, n: int):
        self.total = n
        self.logger.info(f"{C.BOLD}{C.CYAN}[INFO] Total images to process: {n}{C.RESET}")

    def log_image_start(self, key: str):
        self.index += 1
        filename = key.split("/")[-1]
        idx_str  = f"{self.index}/{self.total}" if self.total else str(self.index)
        self.logger.info(
            f"\n{C.BOLD}{'-'*62}{C.RESET}\n"
            f"{C.BOLD}{C.BLUE}>>> Processing image {idx_str}{C.RESET}  "
            f"{C.GRAY}{filename}{C.RESET}"
        )

    def log_caption(self, caption: str):
        display = caption if len(caption) <= 160 else caption[:157] + "..."
        self.logger.info(f"  {C.CYAN}[Caption]   : {C.RESET}{display}")

    def log_classification(self, main_cat: str, sub_cat: str, confident: bool):
        flag = "" if confident else f" {C.YELLOW}[needs_review]{C.RESET}"
        self.logger.info(f"  {C.GREEN}[Category]  : {C.RESET}{C.BOLD}{main_cat}{C.RESET}{flag}")
        self.logger.info(f"  {C.GREEN}[Sub-cat]   : {C.RESET}{sub_cat}")

    def log_success(self, needs_review: bool = False):
        self.success += 1
        if needs_review:
            self.needs_review += 1
            status = f"{C.YELLOW}[NEEDS_REVIEW]{C.RESET}"
        else:
            status = f"{C.GREEN}[OK]{C.RESET}"
        self.logger.info(
            f"  {status} Saved to DB  "
            f"{C.GRAY}[done={self.success} failed={self.failed} "
            f"review={self.needs_review} skip={self.skipped}]{C.RESET}"
        )

    def log_failure(self, reason: str):
        self.failed += 1
        self.logger.error(
            f"  {C.RED}[FAIL]{C.RESET}: {reason}  "
            f"{C.GRAY}[done={self.success} failed={self.failed}]{C.RESET}"
        )

    def log_skip(self):
        self.skipped += 1
        if self.skipped % 50 == 1:
            self.logger.info(
                f"  {C.GRAY}[SKIP] Skipped {self.skipped} already-processed images...{C.RESET}"
            )

    def log_batch_stats(self, elapsed_s: float):
        rate      = self.success / max(elapsed_s, 1)
        remaining = max(self.total - self.index, 0)
        eta_s     = remaining / max(rate, 0.001)
        eta_str   = f"{int(eta_s//60)}m {int(eta_s%60)}s" if self.total else "?"
        bar_width = 30
        filled    = int(bar_width * self.index / max(self.total, 1))
        bar       = "#" * filled + "." * (bar_width - filled)
        self.logger.info(
            f"\n{C.BOLD}{C.MAGENTA}  +-- Progress -------------------------------------------+{C.RESET}\n"
            f"{C.MAGENTA}  |{C.RESET}  [{bar}] {self.index}/{self.total}\n"
            f"{C.MAGENTA}  |{C.RESET}  OK:{self.success}  FAIL:{self.failed}  "
            f"REVIEW:{self.needs_review}  SKIP:{self.skipped}\n"
            f"{C.MAGENTA}  |{C.RESET}  Speed:{rate:.2f}/s   ETA:{eta_str}\n"
            f"{C.MAGENTA}  +-------------------------------------------------------+{C.RESET}"
        )

    def print_summary(self, elapsed_s: float):
        self.logger.info(
            f"\n{C.BOLD}{'='*62}\n"
            f"{'  PIPELINE COMPLETE':^62}\n"
            f"{'='*62}{C.RESET}\n"
            f"  Total listed    : {self.total}\n"
            f"  Skipped (dup)   : {self.skipped}\n"
            f"  Processed OK    : {self.success}\n"
            f"  Needs review    : {self.needs_review}\n"
            f"  Failed          : {self.failed}\n"
            f"  Duration        : {elapsed_s:.1f} s\n"
            f"  Speed           : {self.success / max(elapsed_s, 1):.2f} img/s\n"
            f"{C.BOLD}{'='*62}{C.RESET}"
        )


# ============================================================================
#  CHECKPOINT MANAGER
#  Tracks completed and failed keys for resume and retry.
# ============================================================================
class CheckpointManager:
    MAX_RETRIES = 3  # give up after this many consecutive failures for one key

    def __init__(self, config: Config, logger: logging.Logger):
        self.path   = config.CHECKPOINT_FILE
        self.logger = logger
        self._completed:  set            = set()
        self._failed:     Dict[str, int] = {}   # key → retry count
        self._last_token: Optional[str]  = None
        self._count:      int            = 0

    def load(self):
        if not os.path.exists(self.path):
            self.logger.info("No checkpoint found — starting fresh.")
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            self._completed  = set(data.get("completed_keys", []))
            self._failed     = data.get("failed_keys", {})
            self._last_token = data.get("last_continuation_token")
            self._count      = data.get("processed_count", 0)
            self.logger.info(
                f"Checkpoint loaded: {len(self._completed)} done, "
                f"{len(self._failed)} failed-with-retries, "
                f"token={'yes' if self._last_token else 'no'}"
            )
        except Exception as e:
            self.logger.warning(f"Checkpoint load failed: {e} — fresh start.")

    def save(self, continuation_token: Optional[str] = None):
        data = {
            "completed_keys":          list(self._completed),
            "failed_keys":             self._failed,
            "last_continuation_token": continuation_token or self._last_token,
            "processed_count":         self._count,
            "timestamp":               datetime.now(timezone.utc).isoformat(),
        }
        with open(self.path, "w") as f:
            json.dump(data, f)

    def mark_done(self, key: str):
        self._completed.add(key)
        self._failed.pop(key, None)   # clear retry counter on success
        self._count += 1

    def mark_failed_key(self, key: str):
        self._failed[key] = self._failed.get(key, 0) + 1

    def is_done(self, key: str) -> bool:
        # Exhausted retries counts as "done" (permanently skipped)
        return key in self._completed or self._failed.get(key, 0) >= self.MAX_RETRIES

    def reset_token(self):
        self._last_token = None

    def clear(self):
        if os.path.exists(self.path):
            os.remove(self.path)
            self.logger.info("Checkpoint cleared.")

    @property
    def last_token(self) -> Optional[str]:
        return self._last_token

    @property
    def count(self) -> int:
        return self._count


# ============================================================================
#  CAPTIONS CACHE  (JSONL — append-only, O(1) per write)
#
#  v4 wrote the entire JSON file on every caption save → O(n²) total I/O.
#  v5 appends one JSON line per caption → O(1) per write, O(n) total.
# ============================================================================
class CaptionsCache:
    def __init__(self, config: Config, logger: logging.Logger):
        self.path   = config.CAPTIONS_CACHE_FILE
        self.logger = logger
        self._data: Dict[str, Dict[str, str]] = {}

    def load(self):
        if not os.path.exists(self.path):
            return
        loaded = 0
        corrupt = 0
        try:
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        # last-write-wins handles any duplicate keys from interrupted runs
                        self._data[record["key"]] = {
                            "caption": record["caption"],
                            "tags":    record["tags"],
                        }
                        loaded += 1
                    except (json.JSONDecodeError, KeyError):
                        corrupt += 1
            msg = f"Captions cache loaded: {loaded} captions"
            if corrupt:
                msg += f" ({corrupt} corrupt lines skipped)"
            self.logger.info(msg)
        except Exception as e:
            self.logger.warning(f"Captions cache load failed: {e} — starting fresh.")

    def save_caption(self, key: str, caption: str, tags: str):
        self._data[key] = {"caption": caption, "tags": tags}
        # Append one line — never rewrites the whole file
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps({"key": key, "caption": caption, "tags": tags},
                           ensure_ascii=False) + "\n"
            )

    def has(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str) -> Optional[Dict[str, str]]:
        return self._data.get(key)

    def all_keys(self) -> List[str]:
        return list(self._data.keys())

    def delete(self):
        if os.path.exists(self.path):
            os.remove(self.path)
            self.logger.info("Captions cache deleted after successful run.")


# ============================================================================
#  IMAGE BYTES CACHE  (disk cache — eliminates Phase 2 S3 re-downloads)
#
#  Phase 1 saves raw image bytes to disk.
#  Phase 2 reads from disk instead of re-downloading from S3.
#  Saves ~50% of S3 bandwidth and eliminates one full round of S3 GET requests.
# ============================================================================
class ImageBytesCache:
    def __init__(self, config: Config, logger: logging.Logger):
        self.dir    = Path(config.IMAGE_CACHE_DIR)
        self.logger = logger
        self.dir.mkdir(exist_ok=True)

    def _path(self, key: str) -> Path:
        # MD5 hash of the S3 key avoids issues with slashes and long path names
        h = hashlib.md5(key.encode()).hexdigest()
        return self.dir / h

    def save(self, key: str, data: bytes):
        self._path(key).write_bytes(data)

    def get(self, key: str) -> Optional[bytes]:
        p = self._path(key)
        return p.read_bytes() if p.exists() else None

    def has(self, key: str) -> bool:
        return self._path(key).exists()

    def delete_key(self, key: str):
        # Called after Phase 2 successfully processes each image — frees disk space
        # immediately instead of holding all raw bytes until the full run completes.
        try:
            self._path(key).unlink(missing_ok=True)
        except Exception:
            pass

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)
        self.logger.info(f"Image bytes cache removed ({self.dir})")


# ============================================================================
#  DATABASE MANAGER  (thread-safe via threading.local)
#
#  v4 used a shared list as a connection pool — not thread-safe under concurrent
#  access from ThreadPoolExecutor workers.
#  v5 uses threading.local so each OS thread gets its own dedicated connection.
# ============================================================================
class DatabaseManager:

    def __init__(self, config: Config, logger: logging.Logger):
        self.config    = config
        self.logger    = logger
        self._conn_str = self._build_conn_str()
        self._local    = threading.local()          # one connection per thread
        self._cat_cache:    Dict[str, int]             = {}
        self._subcat_cache: Dict[Tuple[str, str], int] = {}

    def _build_conn_str(self) -> str:
        return (
            f"DRIVER={{{self.config.SQL_DRIVER}}};"
            f"SERVER={self.config.SQL_SERVER},1433;"
            f"DATABASE={self.config.SQL_DATABASE};"
            f"UID={self.config.SQL_USERNAME};"
            f"PWD={self.config.SQL_PASSWORD};"
            "Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
        )

    def _get_conn(self) -> pyodbc.Connection:
        """Return the connection for the calling thread, creating it if needed."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = pyodbc.connect(self._conn_str)
            self._local.conn = conn
        return conn

    def _reset_conn(self):
        """Force reconnect on the next _get_conn call (used after connection failure)."""
        conn = getattr(self._local, "conn", None)
        if conn:
            try:
                conn.close()
            except Exception:
                pass
        self._local.conn = None

    # ── [FIX-08] Re-classify support ───────────────────────────────────────
    # Both methods below are only ever called when RECLASSIFY_MODE is on.

    def fetch_captions_for_reclassify(
        self,
        categories: Optional[List[str]],
        after_id:   int,
        batch:      int,
    ) -> List[tuple]:
        """
        Pull the next slice of already-processed images that have a caption.

        Paging is by Id rather than by OFFSET. That matters here: this pass
        UPDATEs the same rows it is reading, so an OFFSET-based page would
        shift under us. Ordering by Id and asking for Id > last_seen is stable
        no matter what the updates do, and it doubles as the resume token.

        Returns [(Id, AICaption, CurrentCategoryKey), ...]
        """
        conn   = self._get_conn()
        cursor = conn.cursor()

        sql = """
            SELECT TOP (?) i.Id, i.AICaption, c.TaxonomyKey
            FROM   AiImagesImages i
            LEFT JOIN AIImagesCategories c ON c.Id = i.CategoryId
            WHERE  i.Id > ?
              AND  i.AICaption IS NOT NULL
              AND  LEN(i.AICaption) > 20
              AND  i.AIStatus IN ('completed', 'needs_review')
        """
        params: List = [batch, after_id]

        if categories:
            placeholders = ",".join("?" for _ in categories)
            sql += f" AND c.TaxonomyKey IN ({placeholders})"
            params.extend(categories)

        sql += " ORDER BY i.Id"
        cursor.execute(sql, params)
        return [(r.Id, r.AICaption, r.TaxonomyKey) for r in cursor.fetchall()]

    def count_reclassify_candidates(self, categories: Optional[List[str]]) -> int:
        conn   = self._get_conn()
        cursor = conn.cursor()
        sql = """
            SELECT COUNT(*)
            FROM   AiImagesImages i
            LEFT JOIN AIImagesCategories c ON c.Id = i.CategoryId
            WHERE  i.AICaption IS NOT NULL
              AND  LEN(i.AICaption) > 20
              AND  i.AIStatus IN ('completed', 'needs_review')
        """
        params: List = []
        if categories:
            placeholders = ",".join("?" for _ in categories)
            sql += f" AND c.TaxonomyKey IN ({placeholders})"
            params.extend(categories)
        cursor.execute(sql, params)
        return cursor.fetchone()[0]

    def update_classification(
        self,
        image_id:        int,
        category_id:     int,
        subcategory_id:  Optional[int],
        status:          str,
    ) -> bool:
        """
        UPDATE the classification columns on one existing row.

        Deliberately narrow: it touches CategoryId, SubCategoryId, AIStatus and
        UpdatedAt and nothing else. FileName, FilePath, ThumbnailUrl,
        OriginalFilePath, AICaption, AITags, Price and every other column are
        left exactly as they are, so the website keeps working while this runs.
        """
        try:
            conn   = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE AiImagesImages
                SET    CategoryId    = ?,
                       SubCategoryId = ?,
                       AIStatus      = ?,
                       UpdatedAt     = GETUTCDATE()
                WHERE  Id = ?
                """,
                (category_id, subcategory_id, status, image_id),
            )
            conn.commit()
            return True
        except Exception as e:
            self.logger.error(f"[FAIL] DB update (Id={image_id}): {e}")
            self._reset_conn()
            return False

    def load_category_mappings(self):
        conn   = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT Id, TaxonomyKey FROM AIImagesCategories WHERE TaxonomyKey IS NOT NULL"
        )
        for row in cursor.fetchall():
            self._cat_cache[row.TaxonomyKey] = row.Id

        cursor.execute("""
            SELECT sc.Id, sc.TaxonomyKey, c.TaxonomyKey AS CatKey
            FROM AIImagesSubCategories sc
            JOIN AIImagesCategories c ON sc.CategoryId = c.Id
            WHERE sc.TaxonomyKey IS NOT NULL
        """)
        for row in cursor.fetchall():
            self._subcat_cache[(row.CatKey, row.TaxonomyKey)] = row.Id

        self.logger.info(
            f"[OK] Category cache: {len(self._cat_cache)} cats, "
            f"{len(self._subcat_cache)} subcats"
        )

    def get_processed_original_paths(self) -> set:
        """
        Return OriginalFilePath values already in DB with a non-failed status.
        Excludes AIStatus='failed' so those images are retried on the next run.
        """
        conn   = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT OriginalFilePath FROM AiImagesImages "
            "WHERE OriginalFilePath IS NOT NULL AND AIStatus != 'failed'"
        )
        paths = {row[0] for row in cursor.fetchall()}
        self.logger.info(f"[OK] DB duplicate check: {len(paths)} existing non-failed paths")
        return paths

    def insert_image(
        self, *,
        filename:       str,
        original_path:  str,
        thumbnail_url:  Optional[str],
        watermark_url:  Optional[str],
        main_category:  str,
        sub_category:   str,
        caption:        str,
        tags:           str,
        status:         str = "completed",   # "completed" or "needs_review"
    ) -> bool:
        category_id    = self._cat_cache.get(main_category)
        subcategory_id = self._subcat_cache.get((main_category, sub_category))

        if category_id is None:
            self.logger.warning(f"[WARN] Unknown category '{main_category}' — skipped.")
            return False
        if subcategory_id is None:
            self.logger.warning(
                f"[WARN] Unknown subcategory '{main_category}/{sub_category}' "
                f"— inserting with NULL subcategory"
            )

        # Retry once on transient connection failures
        for attempt in range(2):
            try:
                conn   = self._get_conn()
                cursor = conn.cursor()

                # If a previous failed record exists for this path, remove it first
                cursor.execute(
                    "DELETE FROM AiImagesImages "
                    "WHERE OriginalFilePath = ? AND AIStatus = 'failed'",
                    (original_path,)
                )

                cursor.execute("""
                    INSERT INTO AiImagesImages (
                        FileName, FilePath, Title, Description,
                        OriginalFilePath, MainDescription,
                        CategoryId, SubCategoryId,
                        AICaption, AITags,
                        ThumbnailUrl,
                        Price, ViewCount, DownloadCount, IsActive,
                        UploadDate, CreatedAt, UpdatedAt,
                        AIProcessedAt, AIStatus
                    ) VALUES (
                        ?,?,?,?,?,?,?,?,?,?,?,
                        1.00,0,0,1,
                        GETUTCDATE(),GETUTCDATE(),GETUTCDATE(),
                        GETUTCDATE(),?
                    )
                """, (
                    filename,
                    watermark_url or original_path,
                    filename,
                    caption,
                    original_path,
                    caption[:500] or "",
                    category_id,
                    subcategory_id,
                    caption,
                    tags,
                    thumbnail_url,
                    status,
                ))
                conn.commit()
                return True

            except pyodbc.OperationalError:
                if attempt == 0:
                    self.logger.warning("[WARN] DB connection lost — reconnecting...")
                    self._reset_conn()
                    continue
                self.logger.error(f"[FAIL] DB connection failed after retry for {filename}")
                return False
            except Exception as e:
                self.logger.error(f"[FAIL] INSERT failed for {filename}: {e}")
                try:
                    self._get_conn().rollback()
                except Exception:
                    pass
                return False

        return False

    def mark_failed(self, original_path: str, error: str):
        for attempt in range(2):
            try:
                conn   = self._get_conn()
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO AiImagesImages (
                        FileName, FilePath, OriginalFilePath, Description,
                        Price, ViewCount, DownloadCount, IsActive,
                        UploadDate, CreatedAt, UpdatedAt, AIStatus, AIErrorMessage
                    ) VALUES (?,?,?,?,1.00,0,0,0,GETUTCDATE(),GETUTCDATE(),GETUTCDATE(),'failed',?)
                """, (
                    os.path.basename(original_path),
                    original_path,
                    original_path,
                    "Processing failed",
                    error[:1000],
                ))
                conn.commit()
                return
            except pyodbc.OperationalError:
                if attempt == 0:
                    self._reset_conn()
                    continue
                self.logger.error(f"[FAIL] mark_failed reconnect failed: {original_path}")
                return
            except Exception as e:
                self.logger.error(f"[FAIL] mark_failed error: {e}")
                return

    def close_current_thread(self):
        conn = getattr(self._local, "conn", None)
        if conn:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None


# ============================================================================
#  S3 MANAGER  (single reused aioboto3 client)
#
#  v4 created a new aioboto3.Session and client for every download/upload
#  → one TLS handshake per request → massive overhead at scale.
#  v5 opens ONE client at pipeline start and reuses it throughout.
# ============================================================================
class S3Manager:

    def __init__(self, config: Config, logger: logging.Logger):
        self.config     = config
        self.logger     = logger
        self._session   = aioboto3.Session(
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            region_name=config.AWS_REGION,
        )
        self._client    = None
        self._client_cm = None

    async def open(self):
        """Open the S3 client. Call once before any download/upload operations."""
        self._client_cm = self._session.client("s3")
        self._client    = await self._client_cm.__aenter__()
        self.logger.info(f"{C.GREEN}[OK] S3 client opened (connection pool active){C.RESET}")

    async def close(self):
        """Close the S3 client. Call once after all operations are done."""
        if self._client_cm:
            await self._client_cm.__aexit__(None, None, None)
            self._client    = None
            self._client_cm = None

    async def list_keys(self, continuation_token: Optional[str] = None):
        """Async generator yielding (s3_key, next_continuation_token) for all images."""
        params: Dict = {
            "Bucket":  self.config.S3_PRIVATE_BUCKET,
            "Prefix":  self.config.S3_PRIVATE_PREFIX,
            "MaxKeys": self.config.S3_PAGE_SIZE,
        }
        if continuation_token:
            params["ContinuationToken"] = continuation_token

        while True:
            resp = await self._client.list_objects_v2(**params)
            for obj in resp.get("Contents", []):
                key = obj["Key"]
                if key.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
                    yield key, resp.get("NextContinuationToken")
            if not resp.get("IsTruncated"):
                break
            params["ContinuationToken"] = resp["NextContinuationToken"]

    async def download(self, key: str, semaphore: asyncio.Semaphore) -> Optional[bytes]:
        """Download one object with exponential-backoff retry (3 attempts)."""
        async with semaphore:
            for attempt in range(1, 4):
                try:
                    resp = await self._client.get_object(
                        Bucket=self.config.S3_PRIVATE_BUCKET, Key=key
                    )
                    return await resp["Body"].read()
                except Exception as e:
                    if attempt == 3:
                        self.logger.error(f"[FAIL] Download failed ({key}): {e}")
                        return None
                    await asyncio.sleep(2 ** attempt)

    async def upload(self, data: bytes, key: str, semaphore: asyncio.Semaphore) -> Optional[str]:
        """Upload bytes to the public bucket. Returns region-correct public URL."""
        async with semaphore:
            for attempt in range(1, 4):
                try:
                    await self._client.put_object(
                        Bucket=self.config.S3_PUBLIC_BUCKET,
                        Key=key,
                        Body=data,
                        ContentType="image/jpeg",
                        CacheControl="max-age=31536000",
                    )
                    # Use region-specific URL (fixes broken URLs for non-us-east-1 buckets)
                    return (
                        f"https://{self.config.S3_PUBLIC_BUCKET}"
                        f".s3.{self.config.AWS_REGION}.amazonaws.com/{key}"
                    )
                except Exception as e:
                    if attempt == 3:
                        self.logger.error(f"[FAIL] Upload failed ({key}): {e}")
                        return None
                    await asyncio.sleep(2 ** attempt)

    def thumbnail_key(self, filename: str) -> str:
        return f"{self.config.S3_THUMBNAIL_PREFIX}{filename}"

    def watermark_key(self, filename: str) -> str:
        return f"{self.config.S3_WATERMARK_PREFIX}{filename}"

    def original_url(self, key: str) -> str:
        return (
            f"https://{self.config.S3_PRIVATE_BUCKET}"
            f".s3.{self.config.AWS_REGION}.amazonaws.com/{key}"
        )


# ============================================================================
#  IMAGE PROCESSOR
# ============================================================================
class ImageProcessor:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config   = config
        self.logger   = logger
        self.executor = ThreadPoolExecutor(max_workers=config.COMPRESS_THREADS)

    def _resize_to_jpeg(self, img: Image.Image, max_px: int, quality: int) -> bytes:
        if img.mode != "RGB":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "RGBA":
                bg.paste(img, mask=img.split()[3])
            else:
                bg.paste(img)
            img = bg
        if max(img.width, img.height) > max_px:
            img = img.copy()
            img.thumbnail((max_px, max_px), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()

    def process(self, raw_bytes: bytes) -> Tuple[bytes, bytes]:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            img.load()
            return (
                self._resize_to_jpeg(img, self.config.THUMBNAIL_MAX_PX, self.config.THUMBNAIL_QUALITY),
                self._resize_to_jpeg(img, self.config.WATERMARK_MAX_PX,  self.config.WATERMARK_QUALITY),
            )

    async def process_async(self, raw_bytes: bytes) -> Tuple[bytes, bytes]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self.process, raw_bytes)

    def shutdown(self):
        self.executor.shutdown(wait=True)


# ============================================================================
#  AI ENGINE  v5
#
#  PHASE 1 — LLaVA-1.5-7B  (image → caption)
#    Batch size configurable via LLAVA_BATCH_SIZE (default 4).
#    Falls back to sequential on any batch-level error.
#
#  PHASE 2 — Mistral-7B-Instruct-v0.3  (caption → category + subcategory)
#    Two-step design preserved (less confusion than one 481-option prompt).
#    Batch size configurable via MISTRAL_BATCH_SIZE (default 8).
#    Per batch: 2 Mistral calls (vs 2N calls in v4 → up to 8x reduction).
#
#  SUBCATEGORY VALIDATION (three-tier):
#    1. Exact match
#    2. Case-insensitive match   ← catches most "failures" in v4
#    3. difflib fuzzy match (cutoff=0.75)  ← catches typos/minor differences
#    4. Fallback to first subcategory + mark as needs_review
# ============================================================================
class AIEngine:

    CAPTION_MODEL_ID    = "llava-hf/llava-1.5-7b-hf"
    CLASSIFIER_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

    _STOP_WORDS = frozenset({
        "about", "after", "also", "because", "been", "being", "between",
        "could", "doing", "during", "eight", "every", "first", "five",
        "found", "four", "from", "have", "image", "into", "just", "know",
        "like", "make", "more", "much", "nine", "only", "other", "over",
        "same", "seven", "should", "since", "six", "some", "still", "such",
        "than", "that", "their", "them", "then", "there", "these", "they",
        "this", "those", "three", "through", "under", "until", "very",
        "when", "where", "which", "while", "will", "with", "would", "your",
        "primary", "subject", "setting", "background", "visual", "style",
        "colors", "mood", "dominant", "overall", "atmosphere",
    })

    def __init__(self, config: Config, logger: logging.Logger):
        self.config  = config
        self.logger  = logger
        self.device  = "cuda" if torch.cuda.is_available() else "cpu"
        self._llava_model:       Optional[LlavaForConditionalGeneration] = None
        self._llava_processor:   Optional[LlavaProcessor]                = None
        self._mistral_model:     Optional[AutoModelForCausalLM]          = None
        self._mistral_tokenizer: Optional[AutoTokenizer]                 = None

        self.logger.info(f"AI Engine initialised — device: {self.device}")
        if self.device == "cuda":
            self.logger.info(f"   GPU : {torch.cuda.get_device_name(0)}")
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info(f"   VRAM: {total:.1f} GB total")

    # =========================================================================
    #  PHASE 1 — LLaVA
    # =========================================================================

    def load_caption_model(self):
        self.logger.info(
            f"{C.CYAN}[Phase 1] Loading LLaVA-1.5-7B caption model...{C.RESET}\n"
            f"  {C.GRAY}First run downloads ~14 GB — cached after that{C.RESET}"
        )
        # Use slow/pure-Python LlamaTokenizer to avoid the Windows Rust tokenizer bug
        tokenizer = LlamaTokenizer.from_pretrained(
            self.CAPTION_MODEL_ID, use_fast=False, legacy=True,
        )
        # Left-padding required for correct batch inference on decoder-only models
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token    = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        image_processor = CLIPImageProcessor.from_pretrained(self.CAPTION_MODEL_ID)
        self._llava_processor = LlavaProcessor(
            tokenizer=tokenizer, image_processor=image_processor,
        )
        self._llava_model = LlavaForConditionalGeneration.from_pretrained(
            self.CAPTION_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self._llava_model.eval()

        if self.device == "cuda":
            used = torch.cuda.memory_allocated() / 1e9
            self.logger.info(f"  {C.GREEN}[OK] LLaVA loaded ({used:.1f} GB VRAM){C.RESET}")

    def unload_caption_model(self):
        self.logger.info(f"{C.YELLOW}[Phase 1→2] Unloading LLaVA to free VRAM...{C.RESET}")
        del self._llava_model
        del self._llava_processor
        self._llava_model     = None
        self._llava_processor = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
            free = (
                torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated()
            ) / 1e9
            self.logger.info(f"  LLaVA unloaded — {free:.1f} GB VRAM free")

    _CAPTION_PROMPT = (
        "USER: <image>\n"
        "Describe this image using exactly these four fields:\n"
        "Primary subject: [what is the main object, scene, or focus of the image]\n"
        "Setting/background: [what environment, backdrop, or surface is visible]\n"
        "Visual style: [is this a photo, digital art, illustration, drawing, pattern, texture, or clipart]\n"
        "Colors and mood: [dominant colors and overall atmosphere]\n"
        "ASSISTANT:"
    )

    def _raw_to_caption_tags(self, raw: str) -> Tuple[str, str]:
        tags = ", ".join(sorted({
            w.strip(".,!?;:\"'[]()").lower()
            for w in raw.split()
            if len(w) > 4
            and w.strip(".,!?;:\"'[]()").isalpha()
            and w.strip(".,!?;:\"'[]()").lower() not in self._STOP_WORDS
        }))
        return raw.strip(), tags

    def generate_captions_batch(self, images: List[Image.Image]) -> List[Tuple[str, str]]:
        """
        Run LLaVA on a batch of images simultaneously.
        Falls back to sequential processing on any batch-level error.
        Returns list of (caption, tags) for each image.
        """
        try:
            return self._batch_caption(images)
        except Exception as e:
            self.logger.warning(
                f"{C.YELLOW}[WARN] LLaVA batch failed ({e}) — falling back to sequential{C.RESET}"
            )
            return [self._single_caption(img) for img in images]

    def _batch_caption(self, images: List[Image.Image]) -> List[Tuple[str, str]]:
        prompts = [self._CAPTION_PROMPT] * len(images)
        inputs  = self._llava_processor(
            text=prompts, images=images, return_tensors="pt", padding=True,
        )
        first_device = next(self._llava_model.parameters()).device
        inputs = {k: v.to(first_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._llava_model.generate(
                **inputs,
                max_new_tokens=self.config.LLAVA_MAX_NEW_TOKENS,   # [FIX-02]
                do_sample=False,
                pad_token_id=self._llava_processor.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        results   = []
        for i in range(len(images)):
            raw = self._llava_processor.tokenizer.decode(
                output_ids[i, input_len:], skip_special_tokens=True
            )
            results.append(self._raw_to_caption_tags(raw))
        return results

    def _single_caption(self, img: Image.Image) -> Tuple[str, str]:
        inputs = self._llava_processor(
            text=self._CAPTION_PROMPT, images=img, return_tensors="pt",
        )
        first_device = next(self._llava_model.parameters()).device
        inputs = {k: v.to(first_device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self._llava_model.generate(
                **inputs,
                max_new_tokens=self.config.LLAVA_MAX_NEW_TOKENS,   # [FIX-02]
                do_sample=False,
                pad_token_id=self._llava_processor.tokenizer.eos_token_id,
            )
        raw = self._llava_processor.tokenizer.decode(
            output_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return self._raw_to_caption_tags(raw)

    # =========================================================================
    #  PHASE 2 — Mistral
    # =========================================================================

    def load_classifier_model(self):
        self.logger.info(
            f"{C.CYAN}[Phase 2] Loading Mistral-7B-Instruct-v0.3 classifier...{C.RESET}\n"
            f"  {C.GRAY}First run downloads ~14 GB — cached after that{C.RESET}"
        )
        self._mistral_tokenizer = AutoTokenizer.from_pretrained(
            self.CLASSIFIER_MODEL_ID, use_fast=True,
        )
        # Left-padding for correct decoder-only batch generation
        self._mistral_tokenizer.padding_side = "left"
        if self._mistral_tokenizer.pad_token is None:
            self._mistral_tokenizer.pad_token    = self._mistral_tokenizer.eos_token
            self._mistral_tokenizer.pad_token_id = self._mistral_tokenizer.eos_token_id

        self._mistral_model = AutoModelForCausalLM.from_pretrained(
            self.CLASSIFIER_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self._mistral_model.eval()
        if self.device == "cuda":
            used = torch.cuda.memory_allocated() / 1e9
            self.logger.info(f"  {C.GREEN}[OK] Mistral loaded ({used:.1f} GB VRAM){C.RESET}")

    def unload_classifier_model(self):
        self.logger.info(f"{C.YELLOW}[Cleanup] Unloading Mistral...{C.RESET}")
        del self._mistral_model
        del self._mistral_tokenizer
        self._mistral_model     = None
        self._mistral_tokenizer = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
        self.logger.info("  Mistral unloaded.")

    # Built once at import time. The category list is now 25 entries with a
    # scope line each, so rebuilding this string per image would be wasteful.
    _CATEGORY_BLOCK = "\n".join(
        f"- {c}: {CATEGORY_DESCRIPTIONS.get(c, '')}" for c in TAXONOMY
    )

    def _build_category_prompt(self, caption: str) -> str:
        """
        [FIX-07] Step 1 prompt, rebuilt for taxonomy v2.

        The previous version listed bare category names plus eight ad-hoc
        decision rules. With 16 categories that mostly worked; with 25 it does
        not, because several categories answer different questions about the
        same picture. A wolf in snow is truthfully an Animal, a Winter image
        and a Wallpaper all at once, and nothing in the old prompt said which
        question to answer first. The model settled on the use-case answer
        every time, which is how Wallpaper and Backgrounds ended up holding
        roughly 60% of the first 60,000 images.

        Two changes fix that:
          1. Every category now carries a one-line scope description, so the
             model can see what each one is actually for.
          2. PRIORITY_RULES gives an explicit order to work through, ending
             with Wallpaper and Backgrounds as last resorts rather than
             first choices.

        The prompt is longer than before, which adds a little prefill time per
        batch. That cost is worth paying: prefill is compute-bound and cheap on
        this GPU, whereas a wrong category also drags the subcategory with it,
        because Step 2 then picks from the wrong list.
        """
        return "\n".join([
            "[INST]",
            "You are an image classification assistant for a stock image website.",
            "",
            "Given the image description below, choose the single best category.",
            "",
            "=== AVAILABLE CATEGORIES ===",
            self._CATEGORY_BLOCK,
            "",
            "=== DECISION RULES ===",
            PRIORITY_RULES.strip(),
            "",
            "=== IMAGE DESCRIPTION ===",
            caption,
            "",
            "=== YOUR TASK ===",
            "Return ONLY a valid JSON object. No explanation. No extra text.",
            'Category name must match EXACTLY one entry from the list above.',
            '{"category": "<category name>"}',
            "[/INST]",
        ])

    def _build_subcategory_prompt(self, caption: str, category: str) -> str:
        sub_list = "\n".join(f"- {s}" for s in TAXONOMY[category])
        return "\n".join([
            "[INST]",
            "You are an image classification assistant.",
            "",
            f"The image has already been assigned to category: {category}",
            "",
            f"=== VALID SUBCATEGORIES FOR {category.upper()} ===",
            sub_list,
            "",
            "=== IMAGE DESCRIPTION ===",
            caption,
            "",
            "=== YOUR TASK ===",
            "Return ONLY a valid JSON object. No explanation. No extra text.",
            "The subcategory MUST be copied EXACTLY, word for word, from the list above.",
            '{"subcategory": "<subcategory name>"}',
            "[/INST]",
        ])

    def _parse_category(self, raw: str) -> Tuple[str, bool]:
        """
        Parse and validate a category from Mistral's raw output.
        Returns (category, is_confident).

        [FIX-03] v5 returned only the string. Every parse failure silently became
        "Backgrounds" with AIStatus='completed', so category-level mistakes were
        invisible in the database. A wrong category also forces the subcategory to
        be picked from the wrong list, so one bad parse produced two bad fields.
        Now an unparseable or unknown category is flagged, and the caller marks the
        row needs_review.
        """
        cat = None
        try:
            cat = json.loads(raw).get("category", "").strip()
        except json.JSONDecodeError:
            m = re.search(r'"category"\s*:\s*"([^"]+)"', raw)
            if m:
                cat = m.group(1).strip()

        if not cat:
            self.logger.warning(
                f"[WARN] No category parsed from Mistral output "
                f"(raw={raw[:80]!r}) — defaulting to Backgrounds, needs_review"
            )
            return "Backgrounds", False

        # 1. Exact match
        if cat in TAXONOMY:
            return cat, True

        # 2. Case-insensitive match (catches "backgrounds" → "Backgrounds")
        if cat.lower() in _CAT_LOWER:
            return _CAT_LOWER[cat.lower()], True

        # 3. Fuzzy match
        matches = difflib.get_close_matches(cat, list(TAXONOMY.keys()), n=1, cutoff=0.6)
        if matches:
            self.logger.debug(f"Category fuzzy match: '{cat}' → '{matches[0]}'")
            return matches[0], True

        self.logger.warning(
            f"[WARN] Unknown category '{cat}' — defaulting to Backgrounds, needs_review"
        )
        return "Backgrounds", False

    def _parse_subcategory(self, raw: str, category: str) -> Tuple[str, bool]:
        """
        Parse and validate a subcategory from Mistral's raw output.
        Returns (subcategory, is_confident).
        is_confident=False triggers needs_review AIStatus in the database.
        """
        valid_subs = TAXONOMY[category]
        sub = None
        try:
            sub = json.loads(raw).get("subcategory", "").strip()
        except json.JSONDecodeError:
            m = re.search(r'"subcategory"\s*:\s*"([^"]+)"', raw)
            if m:
                sub = m.group(1).strip()

        if sub:
            # 1. Exact match
            if sub in valid_subs:
                return sub, True

            # 2. Case-insensitive match
            key = (category, sub.lower())
            if key in _SUB_LOWER:
                return _SUB_LOWER[key], True

            # 3. Fuzzy match (cutoff 0.75 — strict enough to avoid bad matches)
            matches = difflib.get_close_matches(sub, valid_subs, n=1, cutoff=0.75)
            if matches:
                self.logger.debug(
                    f"Subcategory fuzzy: '{sub}' → '{matches[0]}' (cat={category})"
                )
                return matches[0], True

            self.logger.warning(
                f"[WARN] '{sub}' not in {category} subcategories — "
                f"using first as fallback, marking needs_review"
            )

        # No usable output — use first subcategory and flag for review
        return valid_subs[0], False

    def classify_captions_batch(
        self, captions: List[str]
    ) -> List[Tuple[str, str, bool]]:
        """
        Classify a batch of captions in two steps.

        Step 1: One Mistral forward pass for all N captions → N categories.
        Step 2: One Mistral forward pass for all N captions → N subcategories.
        Total: 2 forward passes for N captions (vs 2N passes in v4).

        Returns list of (category, subcategory, is_confident).
        Falls back to individual sequential calls on any batch-level error.
        """
        try:
            return self._batch_classify(captions)
        except Exception as e:
            self.logger.warning(
                f"{C.YELLOW}[WARN] Mistral batch failed ({e}) — falling back to sequential{C.RESET}"
            )
            results = []
            for caption in captions:
                try:
                    cat, sub, conf = self._single_classify(caption)
                    results.append((cat, sub, conf))
                except Exception as e2:
                    self.logger.error(f"[FAIL] Sequential classify failed: {e2}")
                    results.append(("Backgrounds", TAXONOMY["Backgrounds"][0], False))
            return results

    def _run_mistral_batch(self, prompts: List[str], max_new_tokens: int) -> List[str]:
        """Tokenize, run, and decode a batch of prompts on Mistral."""
        inputs = self._mistral_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self._mistral_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._mistral_tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        return [
            self._mistral_tokenizer.decode(
                output_ids[i, input_len:], skip_special_tokens=True
            ).strip()
            for i in range(len(prompts))
        ]

    def _batch_classify(self, captions: List[str]) -> List[Tuple[str, str, bool]]:
        # [FIX-01] token budget now comes from config instead of a hardcoded 20
        n_tok = self.config.MISTRAL_MAX_NEW_TOKENS

        # ── Step 1: categories ──────────────────────────────────────────────
        cat_prompts = [self._build_category_prompt(c) for c in captions]
        cat_raws    = self._run_mistral_batch(cat_prompts, max_new_tokens=n_tok)
        cat_results = [self._parse_category(r) for r in cat_raws]
        categories  = [c for c, _ in cat_results]

        # ── Step 2: subcategories ────────────────────────────────────────────
        sub_prompts = [
            self._build_subcategory_prompt(c, cat)
            for c, cat in zip(captions, categories)
        ]
        sub_raws = self._run_mistral_batch(sub_prompts, max_new_tokens=n_tok)

        results = []
        for r, (cat, cat_conf) in zip(sub_raws, cat_results):
            sub, sub_conf = self._parse_subcategory(r, cat)
            # [FIX-03] a row is only "confident" when BOTH levels parsed cleanly
            results.append((cat, sub, cat_conf and sub_conf))
        return results

    def _single_classify(self, caption: str) -> Tuple[str, str, bool]:
        n_tok = self.config.MISTRAL_MAX_NEW_TOKENS
        # Step 1
        cat_raw = self._run_mistral_batch(
            [self._build_category_prompt(caption)], n_tok
        )[0]
        category, cat_conf = self._parse_category(cat_raw)
        # Step 2
        sub_raw = self._run_mistral_batch(
            [self._build_subcategory_prompt(caption, category)], n_tok
        )[0]
        sub, sub_conf = self._parse_subcategory(sub_raw, category)
        return category, sub, cat_conf and sub_conf


# ============================================================================
#  PIPELINE ORCHESTRATOR
# ============================================================================
class Pipeline:

    DOWNLOAD_BATCH_SIZE = 50

    def __init__(self, config: Config):
        self.config    = config
        self.logger    = setup_logging(config)
        self.s3        = S3Manager(config, self.logger)
        self.img       = ImageProcessor(config, self.logger)
        self.ai        = AIEngine(config, self.logger)
        self.db        = DatabaseManager(config, self.logger)
        self.ckpt      = CheckpointManager(config, self.logger)
        self.captions  = CaptionsCache(config, self.logger)
        self.img_cache = ImageBytesCache(config, self.logger)
        self.progress  = ProgressTracker(self.logger)
        self._dl_sem:  Optional[asyncio.Semaphore] = None
        self._ul_sem:  Optional[asyncio.Semaphore] = None
        # [FIX-05] tracks how many images had been processed at the last
        # checkpoint write, so saves cannot fall out of alignment (see below)
        self._last_ckpt_save: int = 0

    # ── Key collection ────────────────────────────────────────────────────────

    async def _collect_all_keys(self, db_done_paths: set) -> Tuple[List[str], Optional[str]]:
        self.logger.info(f"{C.CYAN}Listing S3 keys...{C.RESET}")
        to_process:   List[str]     = []
        last_token:   Optional[str] = None
        total_listed: int           = 0

        # [FIX-06] Always list from the beginning of the bucket.
        #
        # v5 resumed from a saved S3 ContinuationToken. That token points at the
        # NEXT page, not at the position where processing actually stopped. With
        # MAX_IMAGES the loop breaks in the MIDDLE of a page, so the rest of that
        # page was never listed again on the following run — those images were
        # silently lost forever. This was a data-loss bug, not a duplicate bug,
        # which is why it was never noticed.
        #
        # Listing the whole bucket costs one API call per 1000 keys (a few seconds
        # for 20k objects). The checkpoint's completed-key set plus the database
        # dedup handle the skipping, and both are exact. Trading a few seconds of
        # listing for correctness is the right call here.
        start_token = None

        async for key, token in self.s3.list_keys(continuation_token=start_token):
            last_token    = token
            total_listed += 1
            original_url  = self.s3.original_url(key)
            if self.ckpt.is_done(key) or original_url in db_done_paths:
                self.progress.log_skip()
                continue
            to_process.append(key)
            if self.config.MAX_IMAGES and len(to_process) >= self.config.MAX_IMAGES:
                break

        # Stale continuation token: restart from beginning, DB dedup skips already-done
        if total_listed == 0 and start_token:
            self.logger.warning(
                f"{C.YELLOW}[WARN] S3 returned 0 results with saved token — "
                f"token expired, restarting from bucket beginning.{C.RESET}"
            )
            self.ckpt.reset_token()
            to_process.clear()
            last_token   = None
            total_listed = 0
            self.progress.skipped = 0
            async for key, token in self.s3.list_keys(continuation_token=None):
                last_token    = token
                total_listed += 1
                original_url  = self.s3.original_url(key)
                if self.ckpt.is_done(key) or original_url in db_done_paths:
                    self.progress.log_skip()
                    continue
                to_process.append(key)
                if self.config.MAX_IMAGES and len(to_process) >= self.config.MAX_IMAGES:
                    break

        self.logger.info(
            f"{C.CYAN}Listed: {total_listed} total | "
            f"{self.progress.skipped} already done | "
            f"{len(to_process)} to process{C.RESET}"
        )
        self.progress.set_total(len(to_process))
        return to_process, last_token

    # ── Phase 1: Captioning ───────────────────────────────────────────────────

    async def run_phase1_captioning(self, keys: List[str]):
        self.logger.info(
            f"\n{C.BOLD}{C.CYAN}{'='*62}\n"
            f"  PHASE 1  —  CAPTIONING  (LLaVA-1.5-7B, batch={self.config.LLAVA_BATCH_SIZE})\n"
            f"{'='*62}{C.RESET}"
        )
        self.ai.load_caption_model()

        total      = len(keys)
        captioned  = 0
        skipped    = 0
        loop       = asyncio.get_running_loop()
        img_checks = 0

        for batch_start in range(0, total, self.DOWNLOAD_BATCH_SIZE):
            batch         = keys[batch_start: batch_start + self.DOWNLOAD_BATCH_SIZE]
            needs_caption = [k for k in batch if not self.captions.has(k)]
            skipped      += len(batch) - len(needs_caption)

            if not needs_caption:
                continue

            # Download concurrently
            raw_list = await asyncio.gather(*[
                self.s3.download(k, self._dl_sem) for k in needs_caption
            ])

            # Save raw bytes to disk cache so Phase 2 does not re-download
            valid: List[Tuple[str, bytes]] = []
            for key, raw in zip(needs_caption, raw_list):
                if raw is None:
                    continue
                self.img_cache.save(key, raw)   # O(1) disk write
                valid.append((key, raw))

            if not valid:
                continue

            # GPU temperature check (every GPU_CHECK_EVERY images)
            img_checks += len(valid)
            if self.config.GPU_CHECK_EVERY > 0 and img_checks >= self.config.GPU_CHECK_EVERY:
                maybe_cooldown(self.logger, self.config.GPU_MAX_TEMP_C, self.config.GPU_COOLDOWN_S)
                img_checks = 0

            # Caption in sub-batches on GPU (run in thread pool to keep event loop free)
            llava_bs = self.config.LLAVA_BATCH_SIZE

            def caption_all(items: List[Tuple[str, bytes]]) -> List[Tuple[str, str, str, Optional[str]]]:
                results = []
                for i in range(0, len(items), llava_bs):
                    sub   = items[i: i + llava_bs]
                    imgs  = []
                    valid_sub = []
                    for k, raw_bytes in sub:
                        try:
                            imgs.append(Image.open(io.BytesIO(raw_bytes)).convert("RGB"))
                            valid_sub.append(k)
                        except Exception as e:
                            results.append((k, "", "", str(e)))
                    if not imgs:
                        continue
                    try:
                        caps_tags = self.ai.generate_captions_batch(imgs)
                        for k, (cap, tags) in zip(valid_sub, caps_tags):
                            results.append((k, cap, tags, None))
                    except Exception as e:
                        # Last-resort: sequential fallback
                        for k, img in zip(valid_sub, imgs):
                            try:
                                cap, tags = self.ai._single_caption(img)
                                results.append((k, cap, tags, None))
                            except Exception as e2:
                                results.append((k, "", "", str(e2)))
                    finally:
                        for img in imgs:
                            try:
                                img.close()
                            except Exception:
                                pass
                return results

            batch_results = await loop.run_in_executor(None, caption_all, valid)

            for key, cap, tags, err in batch_results:
                if err:
                    # [FIX-04] v5 logged this and moved on without counting it.
                    # Phase 2 then silently dropped the key (no caption in cache),
                    # so the image vanished from every counter. The run reported
                    # "Failed: 0" while images were missing, and that false zero
                    # triggered the full cleanup that wiped the checkpoint and
                    # caches. Now it is counted and recorded as a retryable failure.
                    self.logger.error(f"[FAIL] Caption ({key.split('/')[-1]}): {err}")
                    self.db.mark_failed(self.s3.original_url(key), f"Caption: {err}")
                    self.ckpt.mark_failed_key(key)
                    self.progress.log_failure(f"Caption: {err}")
                    self.img_cache.delete_key(key)
                    continue
                self.captions.save_caption(key, cap, tags)
                captioned += 1
                self.logger.info(
                    f"  {C.CYAN}[Captioned]{C.RESET} "
                    f"{key.split('/')[-1]}  "
                    f"{C.GRAY}{cap[:80]}...{C.RESET}"
                )

            done_so_far = batch_start + len(batch)
            if done_so_far % 500 == 0:
                self.logger.info(
                    f"{C.MAGENTA}  Phase 1 progress: "
                    f"{done_so_far}/{total}  captioned={captioned}  cache_hits={skipped}{C.RESET}"
                )

        self.logger.info(
            f"\n{C.GREEN}[OK] Phase 1 complete — "
            f"{captioned} new captions, {skipped} from cache.{C.RESET}"
        )
        self.ai.unload_caption_model()

    # ── Phase 2: Classification + Upload + DB insert ──────────────────────────

    async def run_phase2_classification(self, keys: List[str], last_token: Optional[str]):
        self.logger.info(
            f"\n{C.BOLD}{C.CYAN}{'='*62}\n"
            f"  PHASE 2  —  CLASSIFY + UPLOAD + DB  "
            f"(Mistral-7B, batch={self.config.MISTRAL_BATCH_SIZE})\n"
            f"{'='*62}{C.RESET}"
        )
        self.ai.load_classifier_model()

        loop       = asyncio.get_running_loop()
        mistral_bs = self.config.MISTRAL_BATCH_SIZE
        img_checks = 0

        for batch_start in range(0, len(keys), self.DOWNLOAD_BATCH_SIZE):
            batch       = keys[batch_start: batch_start + self.DOWNLOAD_BATCH_SIZE]
            to_classify = [
                k for k in batch
                if self.captions.has(k) and not self.ckpt.is_done(k)
            ]
            if not to_classify:
                continue

            # Get raw bytes from disk cache; fall back to S3 download on cache miss
            async def get_raw(key: str) -> Optional[bytes]:
                raw = await loop.run_in_executor(None, self.img_cache.get, key)
                if raw is None:
                    self.logger.warning(
                        f"{C.YELLOW}[WARN] Cache miss for {key.split('/')[-1]} "
                        f"— downloading from S3{C.RESET}"
                    )
                    raw = await self.s3.download(key, self._dl_sem)
                return raw

            raw_list = await asyncio.gather(*[get_raw(k) for k in to_classify])
            valid    = [(k, r) for k, r in zip(to_classify, raw_list) if r is not None]

            if not valid:
                continue

            # GPU temperature check
            img_checks += len(valid)
            if self.config.GPU_CHECK_EVERY > 0 and img_checks >= self.config.GPU_CHECK_EVERY:
                maybe_cooldown(self.logger, self.config.GPU_MAX_TEMP_C, self.config.GPU_COOLDOWN_S)
                img_checks = 0

            # Classify in Mistral sub-batches (GPU work — run in thread pool)
            def classify_all(items):
                results = []
                for i in range(0, len(items), mistral_bs):
                    sub     = items[i: i + mistral_bs]
                    captions_list = []
                    tags_list     = []
                    for key, _ in sub:
                        cached = self.captions.get(key)
                        captions_list.append(cached["caption"] if cached else "")
                        tags_list.append(cached["tags"] if cached else "")
                    try:
                        classifications = self.ai.classify_captions_batch(captions_list)
                        for (key, raw), (cat, sub_cat, conf), tags in zip(
                            sub, classifications, tags_list
                        ):
                            results.append((key, raw, cat, sub_cat, conf, tags, None))
                    except Exception as e:
                        for key, raw in sub:
                            results.append((key, raw, None, None, False, "", str(e)))
                return results

            batch_results = await loop.run_in_executor(None, classify_all, valid)

            # Upload + DB insert for each classified image
            upload_tasks = []
            for key, raw, category, subcategory, confident, tags, err in batch_results:
                if err:
                    self.logger.error(f"[FAIL] Classification ({key.split('/')[-1]}): {err}")
                    self.db.mark_failed(self.s3.original_url(key), err)
                    self.ckpt.mark_failed_key(key)
                    self.progress.log_failure(f"Classification: {err}")
                    continue

                cached  = self.captions.get(key)
                caption = cached["caption"] if cached else ""
                tags    = cached["tags"]    if cached else tags

                self.progress.log_image_start(key)
                self.progress.log_caption(caption)
                self.progress.log_classification(category, subcategory, confident)

                upload_tasks.append(
                    self._upload_and_store(
                        key=key,
                        raw_bytes=raw,
                        category=category,
                        subcategory=subcategory,
                        confident=confident,
                        caption=caption,
                        tags=tags,
                    )
                )

            await asyncio.gather(*upload_tasks)

            # [FIX-05] Periodic checkpoint save.
            # v5 used `done % CHECKPOINT_INTERVAL == 0`. With batch=50 and
            # interval=100 that only lined up while nothing failed. A single
            # failure shifted the sequence to 49, 99, 149... which never hits a
            # multiple of 100, so the checkpoint would not be written even once
            # for the entire run. Comparing against the last saved count instead
            # of using modulo cannot be knocked out of alignment.
            done = self.progress.success + self.progress.failed
            if done - self._last_ckpt_save >= self.config.CHECKPOINT_INTERVAL:
                self.ckpt.save(continuation_token=last_token)
                self._last_ckpt_save = done
                self.logger.info(
                    f"{C.GRAY}  [CKPT] Saved at {done} processed{C.RESET}"
                )

        self.ai.unload_classifier_model()
        self.logger.info(f"{C.GREEN}[OK] Phase 2 complete.{C.RESET}")

    async def _upload_and_store(
        self, *,
        key:         str,
        raw_bytes:   bytes,
        category:    str,
        subcategory: str,
        confident:   bool,
        caption:     str,
        tags:        str,
    ):
        filename = key.split("/")[-1]

        # Compress
        try:
            thumb_bytes, wmark_bytes = await self.img.process_async(raw_bytes)
        except Exception as e:
            self.logger.error(f"[FAIL] Compression ({filename}): {e}")
            self.db.mark_failed(self.s3.original_url(key), str(e))
            self.ckpt.mark_failed_key(key)
            self.progress.log_failure(f"Compression: {e}")
            # Free disk space even on failure — raw bytes no longer needed
            self.img_cache.delete_key(key)
            return

        # Upload thumbnail + watermark concurrently
        thumb_url, wmark_url = await asyncio.gather(
            self.s3.upload(thumb_bytes, self.s3.thumbnail_key(filename), self._ul_sem),
            self.s3.upload(wmark_bytes, self.s3.watermark_key(filename), self._ul_sem),
        )

        # DB insert (needs_review if Mistral's output was uncertain)
        status = "completed" if confident else "needs_review"
        ok = self.db.insert_image(
            filename=filename,
            original_path=self.s3.original_url(key),
            thumbnail_url=thumb_url,
            watermark_url=wmark_url,
            main_category=category,
            sub_category=subcategory,
            caption=caption,
            tags=tags,
            status=status,
        )

        if ok:
            self.ckpt.mark_done(key)
            self.progress.log_success(needs_review=not confident)
        else:
            self.db.mark_failed(self.s3.original_url(key), "DB insert returned False")
            self.ckpt.mark_failed_key(key)
            self.progress.log_failure("DB insert returned False")

        # Free disk space immediately after processing — raw bytes no longer needed.
        # This keeps .image_cache/ size near zero throughout Phase 2 instead of
        # holding all images on disk until the full run completes.
        self.img_cache.delete_key(key)

    # ── Main entry point ─────────────────────────────────────────────────────

    # ══════════════════════════════════════════════════════════════════════
    #  [FIX-08] RE-CLASSIFY PASS
    #
    #  Repairs the categories of images that are already in the database,
    #  using the caption that was stored alongside each one when it was first
    #  processed. This exists because the taxonomy changed: images classified
    #  under the old 16-category prompt sit in Wallpaper and Backgrounds even
    #  when a more specific category now fits them.
    #
    #  Why this is cheap: captioning is the expensive step and it is already
    #  done. AICaption lives in the same row as the image, keyed by the same
    #  Id, so there is no matching problem and nothing to look up in S3. The
    #  pass loads Mistral only, reads captions, and writes two columns back.
    #
    #  What it never does: no S3 call of any kind, no LLaVA, no INSERT, no
    #  DELETE, no thumbnail or watermark regeneration.
    # ══════════════════════════════════════════════════════════════════════

    def _load_reclassify_checkpoint(self) -> int:
        path = Path(self.config.RECLASSIFY_CHECKPOINT)
        if not path.exists():
            return 0
        try:
            last = int(json.loads(path.read_text(encoding="utf-8")).get("last_id", 0))
            self.logger.info(f"{C.CYAN}[RESUME] Continuing after image Id {last}{C.RESET}")
            return last
        except Exception as e:
            self.logger.warning(f"[WARN] Could not read reclassify checkpoint: {e}")
            return 0

    def _save_reclassify_checkpoint(self, last_id: int, done: int, changed: int):
        try:
            Path(self.config.RECLASSIFY_CHECKPOINT).write_text(
                json.dumps({
                    "last_id":    last_id,
                    "processed":  done,
                    "changed":    changed,
                    "saved_at":   datetime.now(timezone.utc).isoformat(),
                }, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            self.logger.warning(f"[WARN] Could not save reclassify checkpoint: {e}")

    async def run_reclassify(self):
        start  = datetime.now(timezone.utc)
        dry    = self.config.RECLASSIFY_DRY_RUN
        cats   = [c.strip() for c in self.config.RECLASSIFY_CATEGORIES.split(",") if c.strip()] or None

        self.logger.info(
            f"\n{C.BOLD}{C.GREEN}{'='*64}\n"
            f"{'  RE-CLASSIFY PASS  —  Mistral only, no S3, UPDATE only':^64}\n"
            f"{'='*64}{C.RESET}"
        )
        if dry:
            self.logger.info(f"{C.YELLOW}  DRY RUN — nothing will be written to the database{C.RESET}")
        self.logger.info(f"  Scope       : {', '.join(cats) if cats else 'all completed images'}")
        self.logger.info(f"  Limit       : {self.config.RECLASSIFY_LIMIT or 'none'}")
        self.logger.info(f"  Batch size  : {self.config.MISTRAL_BATCH_SIZE}")

        self.db.load_category_mappings()

        total = self.db.count_reclassify_candidates(cats)
        limit = self.config.RECLASSIFY_LIMIT
        target = min(total, limit) if limit else total
        self.logger.info(f"  Candidates  : {total:,}  (will process {target:,})\n")
        if target == 0:
            self.logger.info("Nothing to do.")
            return

        self.ai.load_classifier_model()

        last_id  = self._load_reclassify_checkpoint()
        done     = 0
        changed  = 0
        unchanged= 0
        review   = 0
        failed   = 0
        moves: Dict[str, int] = {}
        bs = self.config.MISTRAL_BATCH_SIZE
        loop = asyncio.get_running_loop()

        try:
            while done < target:
                take = min(bs, target - done)
                rows = self.db.fetch_captions_for_reclassify(cats, last_id, take)
                if not rows:
                    self.logger.info("No more rows returned — reached the end.")
                    break

                ids      = [r[0] for r in rows]
                captions = [r[1] for r in rows]
                old_cats = [r[2] for r in rows]

                results = await loop.run_in_executor(
                    None, self.ai.classify_captions_batch, captions
                )

                for img_id, old_cat, (new_cat, new_sub, confident) in zip(ids, old_cats, results):
                    cat_id = self.db._cat_cache.get(new_cat)
                    if cat_id is None:
                        self.logger.error(
                            f"[FAIL] Id={img_id}: category '{new_cat}' is not in the database. "
                            f"Run 05_seed_taxonomy.sql first."
                        )
                        failed += 1
                        continue
                    sub_id = self.db._subcat_cache.get((new_cat, new_sub))
                    status = "completed" if confident else "needs_review"
                    if not confident:
                        review += 1

                    if old_cat != new_cat:
                        changed += 1
                        key = f"{old_cat} -> {new_cat}"
                        moves[key] = moves.get(key, 0) + 1
                        self.logger.debug(f"  Id={img_id}: {key} / {new_sub}")
                    else:
                        unchanged += 1

                    if not dry:
                        if not self.db.update_classification(img_id, cat_id, sub_id, status):
                            failed += 1

                last_id = ids[-1]
                done   += len(rows)

                if done % 500 < bs:
                    pct = done * 100 // max(target, 1)
                    el  = (datetime.now(timezone.utc) - start).total_seconds()
                    rate = done / el if el else 0
                    eta  = (target - done) / rate if rate else 0
                    self.logger.info(
                        f"{C.CYAN}  [{pct:3d}%] {done:,}/{target:,}  "
                        f"moved={changed:,}  same={unchanged:,}  review={review:,}  "
                        f"{rate:.2f} img/s  ETA {eta/3600:.1f}h{C.RESET}"
                    )
                if not dry:
                    self._save_reclassify_checkpoint(last_id, done, changed)

        except KeyboardInterrupt:
            self.logger.warning(f"\n{C.YELLOW}Interrupted — checkpoint saved, rerun to continue{C.RESET}")
            if not dry:
                self._save_reclassify_checkpoint(last_id, done, changed)
        finally:
            self.ai.unload_classifier_model()
            self.db.close_current_thread()

        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        self.logger.info(
            f"\n{C.BOLD}{C.GREEN}{'='*64}\n"
            f"{'  RE-CLASSIFY COMPLETE':^64}\n"
            f"{'='*64}{C.RESET}"
        )
        self.logger.info(f"  Processed     : {done:,}")
        self.logger.info(f"  Moved         : {changed:,}")
        self.logger.info(f"  Unchanged     : {unchanged:,}")
        self.logger.info(f"  Needs review  : {review:,}")
        self.logger.info(f"  Failed        : {failed:,}")
        self.logger.info(f"  Duration      : {elapsed/3600:.2f} h")
        self.logger.info(f"  Speed         : {done/elapsed:.2f} img/s" if elapsed else "")
        if dry:
            self.logger.info(f"\n{C.YELLOW}  DRY RUN — no rows were written{C.RESET}")

        if moves:
            self.logger.info(f"\n{C.BOLD}  Top category moves:{C.RESET}")
            for k, v in sorted(moves.items(), key=lambda x: -x[1])[:25]:
                self.logger.info(f"    {v:6,}  {k}")

        if done >= target and not dry:
            try:
                Path(self.config.RECLASSIFY_CHECKPOINT).unlink(missing_ok=True)
                self.logger.info("\n  Checkpoint cleared (pass finished).")
            except Exception:
                pass

    async def run(self):
        start = datetime.now(timezone.utc)
        self.logger.info(
            f"\n{C.BOLD}{C.GREEN}{'='*64}\n"
            f"{'  IMAGESHOP PIPELINE  v5  —  LLaVA + Mistral-7B':^64}\n"
            f"{'='*64}{C.RESET}"
        )

        self._dl_sem = asyncio.Semaphore(self.config.DOWNLOAD_CONCURRENCY)
        self._ul_sem = asyncio.Semaphore(self.config.UPLOAD_CONCURRENCY)

        # Open a single reused S3 client for the entire run
        await self.s3.open()

        try:
            # 1. Checkpoint + caption cache
            self.logger.info(f"\n{C.BOLD}[Step 1/4] Loading checkpoint and caption cache...{C.RESET}")
            self.ckpt.load()
            self.captions.load()

            # 2. DB dedup (exclude failed records so they get retried)
            self.logger.info(f"\n{C.BOLD}[Step 2/4] Querying DB for processed images...{C.RESET}")
            db_done_paths = self.db.get_processed_original_paths()

            # 3. Category ID mappings
            self.logger.info(f"\n{C.BOLD}[Step 3/4] Loading category mappings from DB...{C.RESET}")
            self.db.load_category_mappings()

            # 4. S3 key list
            self.logger.info(f"\n{C.BOLD}[Step 4/4] Listing S3 keys...{C.RESET}")
            to_process, last_token = await self._collect_all_keys(db_done_paths)

            if not to_process:
                self.logger.info(
                    f"{C.GREEN}[OK] Nothing to do — all images already processed!{C.RESET}"
                )
                return

            # Phase 1: caption all images
            try:
                await self.run_phase1_captioning(to_process)
            except KeyboardInterrupt:
                self.logger.warning(
                    f"\n{C.YELLOW}[WARN] Interrupted during Phase 1 — "
                    f"captions saved, image cache preserved for Phase 2 resume.{C.RESET}"
                )
                raise

            # Phase 2: classify + upload + insert
            try:
                await self.run_phase2_classification(to_process, last_token)
            except KeyboardInterrupt:
                self.logger.warning(
                    f"\n{C.YELLOW}[WARN] Interrupted during Phase 2 — saving checkpoint...{C.RESET}"
                )
                self.ckpt.save(continuation_token=last_token)
                raise

        finally:
            await self.s3.close()
            self.img.shutdown()
            self.db.close_current_thread()

        # Cleanup on full success
        if self.progress.failed == 0:
            self.ckpt.clear()
            self.captions.delete()
            self.img_cache.cleanup()
        else:
            self.ckpt.save(continuation_token=last_token)
            self.logger.info(
                f"{C.YELLOW}[WARN] {self.progress.failed} image(s) failed — "
                f"checkpoint and cache kept for next run resume.{C.RESET}"
            )

        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        self.progress.print_summary(elapsed)


# ============================================================================
#  ENTRY POINT
# ============================================================================
async def main():
    config = Config()
    config.validate()           # Fail fast before loading any models
    pipeline = Pipeline(config)

    # [FIX-08] One flag decides which of the two passes runs. Everything else
    # about the script is unchanged, so flipping RECLASSIFY_MODE back to false
    # returns it to normal processing with no other edit.
    if config.RECLASSIFY_MODE:
        await pipeline.run_reclassify()
    else:
        await pipeline.run()


if __name__ == "__main__":
    sep = "=" * 66
    print("")
    print(sep)
    print("  IMAGESHOP AI PIPELINE  v5  —  LLaVA-1.5-7B + Mistral-7B-Instruct")
    print(sep)
    print("")
    print("  PHASE 1  Caption : LLaVA-1.5-7B      (~14 GB VRAM, batch=LLAVA_BATCH_SIZE)")
    print("  PHASE 2  Classify: Mistral-7B-Instruct (~8 GB VRAM, batch=MISTRAL_BATCH_SIZE)")
    print("")
    print("  KEY IMPROVEMENTS OVER v4:")
    print("    - Batch GPU inference:  ~2.5x Phase 1, ~4x Phase 2")
    print("    - Disk image cache:     eliminates double S3 download")
    print("    - JSONL caption cache:  O(1) writes (was O(n²))")
    print("    - Fuzzy subcategory matching: far fewer silent misfiled images")
    print("    - Thread-safe DB, GPU temp monitoring, startup validation")
    print("")
    print("  RESUME SUPPORT:")
    print("    Phase 1 resumes from captions_cache.jsonl")
    print("    Phase 2 resumes from pipeline_checkpoint.json + .image_cache/")
    print("    Failed images are retried on next run (up to 3 attempts)")
    print("")
    print("  BEFORE RUNNING:")
    print("    1. Ensure .env is filled in (see .env.example)")
    print("    2. pip install torch transformers pillow aioboto3 pyodbc")
    print("                   python-dotenv accelerate sentencepiece")
    print("    3. pip install nvidia-ml-py3   # optional — GPU temp monitoring")
    print("")
    print("  FIRST RUN downloads ~28 GB model weights (cached after):")
    print("    LLaVA-1.5-7B   ~14 GB")
    print("    Mistral-7B     ~14 GB")
    print(sep)
    print("")
    asyncio.run(main())
