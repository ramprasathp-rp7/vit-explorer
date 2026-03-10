# backend/config.py
# ─────────────────────────────────────────────────────────────────────────────
# Edit these values to match your experiment setup.
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent          # repo root
IMAGES_DIR = ROOT / "assets" / "images"
MODELS_DIR = ROOT / "assets" / "models"
COMPACTED_MODELS_DIR = ROOT / "assets" / "compacted_models"

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Baseline model (for benchmarking) ─────────────────────────────────────────
BASELINE_MODEL_ID    = "baseline_vit_imagenette_v1_best.pth"
BASELINE_PARAM_COUNT = 85_806_346

# ── Dataset folder name ────────────────────────────────────────────────────────
# The name of the dataset folder inside assets/images/.
# e.g. if your images live at assets/images/imagenette2-320/val/n01440764/...
# set this to "imagenette2-320". Set to None to use assets/images/ directly.
DATASET_FOLDER = "imagenette2-320"

# Sub-folders inside the dataset to scan for images (train, val, or both).
# Images from all listed splits are merged together per class.
DATASET_SPLITS = ["val", "train"]

# ── Model architecture ─────────────────────────────────────────────────────────
MODEL_HF_NAME = "google/vit-base-patch16-224"
NUM_LABELS = 10

# ── Class names (Imagenette order) ─────────────────────────────────────────────
# Used for the model prediction labels (ordered by label index 0-9).
# Set to None to fall back to "Class N" labels.
CLASS_NAMES = [
    "tench",
    "English springer",
    "cassette player",
    "chain saw",
    "church",
    "French horn",
    "garbage truck",
    "gas pump",
    "golf ball",
    "parachute",
]

# ── Folder ID → display name mapping ──────────────────────────────────────────
# Maps the WordNet folder IDs (e.g. "n01440764") to human-readable class names.
# These are shown in the class browser UI.
# Add or edit entries here if you use a different dataset.
CLASS_FOLDER_NAMES: dict[str, str] = {
    "n01440764": "Tench",
    "n02102040": "English Springer",
    "n02979186": "Cassette Player",
    "n03000684": "Chain Saw",
    "n03028079": "Church",
    "n03394916": "French Horn",
    "n03417042": "Garbage Truck",
    "n03425413": "Gas Pump",
    "n03445777": "Golf Ball",
    "n03888257": "Parachute",
}

# ── Optional: override display names for specific model files ──────────────────
# Key = filename stem (without .pth), Value = display name shown in the UI.
# If a model file isn't listed here, its display name is auto-derived from
# the filename (underscores → spaces, title-cased).
MODEL_DISPLAY_NAMES: dict[str, str] = {
    # "baseline_vit_imagenette_v1_best": "Baseline ViT",
    # "beta_0_00_model": "DeepLIFT",
    # "beta_1_00_model": "LRP",
    # "method_magnitude_weighted_model": "Hybrid – Magnitude Weighted",
}

# ── Rollout hyper-parameters ───────────────────────────────────────────────────
DEFAULT_DISCARD_RATIO = 0.05
DEFAULT_HEAD_FUSION   = "mean"   # "mean" | "max"
