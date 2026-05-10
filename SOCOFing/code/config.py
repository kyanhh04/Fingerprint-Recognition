# ============================================================================
# Configuration - Fingerprint Recognition Pipeline
# ============================================================================

import os

# === Image Settings ===
IMAGE_SIZE = 512
IMAGE_FORMAT = "BMP"

# === Enhancement Settings ===
CLAHE_ENABLED = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

GABOR_ENABLED = True
GABOR_WAVELENGTH = 5.0
GABOR_SIGMA = 3.0

# === Feature Extraction ===
MINUTIAE_ENABLED = True
LBP_ENABLED = False  # Set to True for extra features, False for speed
LBP_P = 8
LBP_R = 1

# === Orientation Histogram ===
ORIENTATION_BINS = 16
GRID_SIZE = (4, 4)  # 4x4 grid for orientation calculation

# === Search Settings ===
TOP_K = 5
ROTATION_RANGE = 15  # ±15 degrees
ROTATION_STEP = 5    # 5 degree increments

# === Database ===
DB_PATH = "fingerprints.db"
DB_CHECKPOINT_SIZE = 10  # Save checkpoint every N images

# === Data Paths (Updated - images now in ../Real/) ===
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "Real")
OUTPUT_FOLDER = "output"
RESULTS_CSV = os.path.join(OUTPUT_FOLDER, "results.csv")
EVALUATION_CSV = os.path.join(OUTPUT_FOLDER, "evaluation.csv")

# === Segmentation ===
SEGMENTATION_METHOD = "otsu"  # otsu, adaptive
MORPHOLOGY_KERNEL_SIZE = 5
MORPHOLOGY_ITERATIONS = 2

# === Thresholds ===
BINARIZATION_THRESHOLD = 127
MIN_COMPONENT_SIZE = 50

# === Logging ===
LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(OUTPUT_FOLDER, "pipeline.log")

# Create output folder if not exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
