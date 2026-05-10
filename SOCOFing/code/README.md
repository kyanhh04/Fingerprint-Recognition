# ============================================================================
# README - Fingerprint Recognition Pipeline with Auto-Approve
# ============================================================================

## Overview

Complete end-to-end fingerprint recognition system with:
- **Minutiae Detection** (primary feature)
- **Orientation Histogram** (secondary feature)
- **LBP** (optional auxiliary feature)
- **CLAHE & Gabor Enhancement** (configurable)
- **Rotation Re-ranking** (±15°)
- **AUTO-APPROVE System** (NEW - automatic match verification)
- **SQLite Database** (persistent storage)
- **Streamlit UI** (interactive demo)

---

## Project Structure

```
SOCOFing/Real/
├── config.py                 # Global configuration
├── preprocess.py            # Image preprocessing pipeline
├── extract_features.py       # Feature extraction (minutiae, orientation, LBP)
├── build_database.py        # Build SQLite database from BMP folder
├── search.py                # Search logic
├── evaluate.py              # Evaluation metrics & benchmarking
├── app.py                   # Streamlit web UI
├── main.py                  # CLI orchestrator
└── README.md                # This file

data_processed_bmp/           # Input BMP images (24-bit RGB)
output/
├── fingerprints.db          # SQLite database
├── results.csv              # Search results
├── evaluation.csv           # Evaluation metrics
└── pipeline.log             # Execution log

```

---

## Installation

### 1. Requirements
- Python 3.8+
- pip (Python package manager)

### 2. Install Dependencies

```bash
pip install opencv-python numpy pillow scikit-image scikit-learn scipy pandas streamlit
```

### 3. Create Output Directory

```bash
mkdir output
```

---

## Configuration

Edit `config.py` to customize:

```python
# Image Settings
IMAGE_SIZE = 512
CLAHE_ENABLED = True           # CLAHE enhancement
GABOR_ENABLED = True           # Gabor filter
MINUTIAE_ENABLED = True        # Minutiae detection
LBP_ENABLED = False            # Set True for extra features

# Search
TOP_K = 5
ROTATION_RANGE = 15            # ±15 degrees
ROTATION_STEP = 5

# AUTO-APPROVE (NEW)
AUTO_APPROVE_ENABLED = True
AUTO_APPROVE_SCORE_THRESHOLD = 0.75      # If similarity >= 75%
AUTO_APPROVE_GAP_THRESHOLD = 0.15        # If (top1 - top2) >= 15%
AUTO_APPROVE_CONFIDENCE_THRESHOLD = 0.80 # Final confidence threshold

# Database
DB_PATH = "fingerprints.db"
DATA_FOLDER = "data_processed_bmp"
```

---

## Usage

### Step 1: Build Database from BMP Images

```bash
python build_database.py
```

**What it does:**
- Scans `data_processed_bmp/` for BMP files
- Preprocesses each image (grayscale, resize, CLAHE, Gabor)
- Extracts features (minutiae, orientation, LBP)
- Stores in SQLite database
- **Output:** `fingerprints.db`

**Expected output:**
```
INFO - Found 100 BMP files
INFO - Processed: finger001.bmp (ID=1)
...
INFO - Database building completed: 100 processed, 0 failed
```

---

### Step 2: Search for Matches with Auto-Approve

#### CLI Search

```bash
# Basic search
python search.py --query path/to/query.bmp

# With options
python search.py --query path/to/query.bmp --top-k 10 --no-rotation

# Save results with auto-approve decisions
python search.py --query path/to/query.bmp --top-k 5
```

**Output example:**
```
================================================================================
Query: path/to/query.bmp
================================================================================
Rank  Filename                   Similarity  AutoApprove  Confidence
--------------------------------------------------------------
1     finger001.bmp              0.8523      ✓ YES        0.850
2     finger012.bmp              0.7234      ✗ NO         0.723
3     finger045.bmp              0.6891      ✗ NO         0.689
...

Summary: 1/5 auto-approved

Top-1 Match:
  Filename: finger001.bmp
  Similarity: 0.8523
  Auto-Approved: True
  Reason: Score=0.852, Gap=0.129, Confidence=0.850
================================================================================
```

#### Programmatic Search

```python
from search import search_fingerprint

# Search
results = search_fingerprint(
    query_image_path="path/to/query.bmp",
    top_k=5,
    use_rotation=True
)

# Results structure
for match in results['results']:
    print(f"Rank {match['rank']}: {match['filename']}")
    print(f"  Similarity: {match['similarity']:.4f}")
    print(f"  Auto-Approved: {match['auto_approve']}")
    print(f"  Confidence: {match['confidence']:.3f}")
    print(f"  Reason: {match['reason']}")
```

---

### Step 3: Evaluate on Test Set

```bash
# Evaluate on test images
python evaluate.py --test-folder path/to/test_images/ --top-k 5

# With custom output
python evaluate.py --test-folder path/to/test_images/ --output results.csv
```

**Metrics computed:**
- **Top-1 Accuracy:** % of queries where top-1 match is correct
- **Precision@5:** % of top-5 results that are correct
- **Auto-Approve Accuracy:** % of auto-approved results that are correct

**Output CSV columns:**
```
query, ground_truth_id, top_1_match, top_1_accuracy, precision_at_5, 
top_1_auto_approved, auto_approved_accuracy, reason
```

**Evaluation report:**
```
================================================================================
EVALUATION REPORT
================================================================================

Aggregate Metrics:
  Total Queries: 50
  Top-1 Accuracy: 0.9200
  Precision@5 (mean): 0.9500 (±0.0400)

Auto-Approve Analysis:
  Total Auto-Approved: 45
  Queries with Top-1 Auto-Approved: 42
  Auto-Approve Accuracy (mean): 0.9778 (±0.0150)

================================================================================
```

---

### Step 4: Interactive Web UI (Streamlit)

```bash
streamlit run app.py
```

**Browser opens at:** `http://localhost:8501`

**Features:**
- Upload query fingerprint
- View top-K matches with auto-approve status
- Real-time configuration adjustment
- Visualization of search results
- Debug mode for advanced users

---

## Auto-Approve Logic (NEW)

The auto-approve system makes 3-tier decisions:

### Decision Criteria

```python
score_pass = similarity >= 0.75  # Score threshold
gap_pass = (top1 - top2) >= 0.15  # Gap threshold

if score_pass AND gap_pass:
    confidence = (score + gap) / 2
    auto_approve = confidence >= 0.80
else:
    auto_approve = False
    confidence = score  # Use score as confidence
```

### Database Storage

Auto-approve decisions are stored in `auto_approve_decisions` table:

```sql
CREATE TABLE auto_approve_decisions (
    decision_id INTEGER PRIMARY KEY,
    query_image_id INTEGER,
    match_image_id INTEGER,
    rank INTEGER,
    similarity_score REAL,
    gap_to_next REAL,
    auto_approve BOOLEAN,
    confidence REAL,
    reason TEXT,
    created_at TIMESTAMP
);
```

### Example Decision Reasons

- ✓ `Score=0.852, Gap=0.129, Confidence=0.850` → **AUTO-APPROVED**
- ✗ `Score PASS (0.82), but Gap FAIL (0.08)` → NOT APPROVED (gap too small)
- ✗ `Both Score and Gap FAIL (Score=0.65, Gap=0.05)` → NOT APPROVED

---

## Feature Engineering

### 1. Minutiae Detection
- Detects bifurcations (ridge splits) and endings
- Fixed-size vector: 256 position entries + 5 metadata
- Sensitive to ridge structure quality

### 2. Orientation Histogram
- 16-bin histogram per 4×4 grid cell (64 values total)
- Captures ridge orientation patterns
- Invariant to intensity variations

### 3. LBP (Optional)
- Local Binary Pattern on enhanced image
- 256-bin histogram
- Set `LBP_ENABLED = True` to use

### 4. Enhancement Pipeline
```
Input BMP (24-bit RGB)
    ↓
Grayscale + Resize (512×512)
    ↓
CLAHE (Contrast-Limited AHE)
    ↓
Gabor Filter (orient enhancement)
    ↓
Binarize + Skeletonize
    ↓
Feature Extraction
```

---

## Database Schema

### `images` Table
```
image_id     (PRIMARY KEY)
filename     (UNIQUE)
filepath
image_size
feature_size
created_at   (TIMESTAMP)
```

### `features` Table
```
feature_id   (PRIMARY KEY)
image_id     (FOREIGN KEY → images)
minutiae_count
bifurcations
endings
feature_vector   (BLOB - concatenated features)
orientation_hist (BLOB)
lbp_hist         (BLOB, optional)
created_at   (TIMESTAMP)
```

### `auto_approve_decisions` Table (NEW)
```
decision_id
query_image_id
match_image_id
rank
similarity_score
gap_to_next
auto_approve     (BOOLEAN)
confidence       (REAL 0-1)
reason           (TEXT - explanation)
created_at       (TIMESTAMP)
```

---

## Performance Tuning

### Speed vs Accuracy

#### Fast Mode (set in `config.py`)
```python
CLAHE_ENABLED = True      # Keep ON (minimal cost)
GABOR_ENABLED = False     # Turn OFF (expensive)
LBP_ENABLED = False       # Keep OFF
ROTATION_RANGE = 0        # Skip rotation
```

#### High Accuracy Mode
```python
CLAHE_ENABLED = True      # ON
GABOR_ENABLED = True      # ON
LBP_ENABLED = True        # ON
ROTATION_RANGE = 15       # Full range
```

### Expected Timings
- **Preprocess:** ~100ms per image
- **Feature extraction:** ~50ms per image
- **Database build (100 images):** ~20 seconds
- **Single search (no rotation):** ~200ms
- **Single search (with rotation ±15°):** ~2 seconds

---

## Troubleshooting

### Database Already Exists
Delete `fingerprints.db` before rebuilding:
```bash
rm fingerprints.db  # Linux/Mac
del fingerprints.db  # Windows
```

### Out of Memory on Large Database
Reduce `DB_CHECKPOINT_SIZE` in `config.py`

### Low Auto-Approve Rate
Adjust thresholds in `config.py`:
```python
AUTO_APPROVE_SCORE_THRESHOLD = 0.65      # Lower score threshold
AUTO_APPROVE_GAP_THRESHOLD = 0.10        # Lower gap threshold
AUTO_APPROVE_CONFIDENCE_THRESHOLD = 0.70 # Lower confidence
```

### Slow Search
- Set `GABOR_ENABLED = False`
- Set `ROTATION_RANGE = 0`
- Reduce `TOP_K`

---

## Key Features Summary

| Feature | Status | Config |
|---------|--------|--------|
| Minutiae Detection | ✓ Primary | `MINUTIAE_ENABLED` |
| Orientation Histogram | ✓ Secondary | Always ON |
| LBP Features | ○ Optional | `LBP_ENABLED` |
| CLAHE Enhancement | ✓ Recommended | `CLAHE_ENABLED` |
| Gabor Filter | ○ Optional | `GABOR_ENABLED` |
| Rotation Re-ranking | ○ Optional | `ROTATION_RANGE` |
| **Auto-Approve** | **✓ NEW** | `AUTO_APPROVE_ENABLED` |
| SQLite Storage | ✓ Default | `DB_PATH` |
| Streamlit UI | ○ Optional | `app.py` |

---

## References

- OpenCV: https://docs.opencv.org
- Scikit-Image: https://scikit-image.org
- Streamlit: https://streamlit.io

---

## License

For evaluation and research purposes.

---

## Contact

For issues or questions, check log file: `output/pipeline.log`

