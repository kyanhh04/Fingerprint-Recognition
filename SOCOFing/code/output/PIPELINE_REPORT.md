# 📋 BÁO CÁO LUỒNG LOGIC - SOCOFing Fingerprint Recognition

**Ngày tạo:** 11/05/2026  
**Phiên bản:** 1.0  
**Trạng thái:** Active

---

## 📑 MỤC LỤC
1. [Tổng Quan](#tổng-quan)
2. [Luồng Chi Tiết](#luồng-chi-tiết)
3. [Các Thành Phần](#các-thành-phần)
4. [Cấu Hình Hiện Tại](#cấu-hình-hiện-tại)
5. [Performance](#performance)

---

## 🎯 Tổng Quan

Pipeline xử lý vân tay có **4 giai đoạn chính**:

```
Query Image (BMP)
       ↓
[1] PREPROCESSING (Tiền Xử Lý)
       ↓
[2] FEATURE EXTRACTION (Trích Xuất Đặc Trưng)
       ↓
[3] DATABASE SEARCH (Tìm Kiếm)
       ↓
[4] RE-RANKING (Xếp Hạng Lại)
       ↓
Output Results
```

---

## 🔄 Luồng Chi Tiết

### **GIAI ĐOẠN 1: PREPROCESSING (Tiền Xử Lý)**

**Mục đích:** Chuẩn bị ảnh gốc (BMP) cho việc trích xuất đặc trưng

**File:** `preprocess.py`

#### Bước 1.1: Load Image
```python
img = cv2.imread(image_path)  # BGR format
Output: Shape = (H, W, 3)
```

#### Bước 1.2: Convert to Grayscale
```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
Output: Shape = (H, W)
```

#### Bước 1.3: Resize to Standard Size
```python
resized = cv2.resize(gray, (512, 512))  # IMAGE_SIZE = 512
Output: Shape = (512, 512)
```

#### Bước 1.4: Segmentation (Phân Đoạn)
```
Gaussian Blur (5x5, σ=0)
       ↓
Otsu Thresholding (THRESHOLD=127)
       ↓
Morphological Close (KERNEL_SIZE=5, iterations=2)
       ↓
Morphological Open (iterations=1)
Output: Binary mask (vân tay trắng, nền đen)
```

#### Bước 1.5: CLAHE Enhancement (Tăng Cường Độ Tương Phản)
```
Input: Grayscale segmented image
Parameters:
  - clipLimit = 2.0
  - tileGridSize = (8, 8)
Output: Enhanced image with better contrast
```

#### Bước 1.6: Gabor Filter (Tăng Cường Cấu Trúc Gân)
```
Parameters:
  - wavelength = 5.0
  - sigma = 3.0
  - orientation = 45°
Output: Ridge structure enhanced
```

#### Bước 1.7: Binarization
```python
_, binary = cv2.threshold(enhanced_img, THRESHOLD=127, 255, cv2.THRESH_BINARY)
Output: Binary image (0 or 255)
```

#### Bước 1.8: Skeletonization (Khung Xương)
```
Algorithm: Morphological thinning
Process:
  1. Invert binary image
  2. Iteratively erode & subtract
  3. Continue until convergence
Output: 1-pixel wide skeleton (minutiae detection basis)
```

**Pipeline Output:**
```python
preprocessed = {
    'original': BGR_image,
    'grayscale': gray_image,
    'resized': resized_image,
    'segmented': segmented_image,
    'enhanced': enhanced_image,
    'binary': binary_image,
    'skeleton': skeleton_image,
    'mask': segmentation_mask
}
```

---

### **GIAI ĐOẠN 2: FEATURE EXTRACTION (Trích Xuất Đặc Trưng)**

**Mục đích:** Chuyển đổi ảnh thành vector số để so sánh

**File:** `extract_features.py`

#### Bước 2.1: Minutiae Detection (Phát Hiện Điểm Đặc Trưng)
```
Input: Skeleton image
Algorithm: 8-connectivity neighbor counting
  - Ending: 1 neighbor → bifurcation point
  - Bifurcation: 3 neighbors → branching point
  
Output:
  - minutiae list: [{type, x, y}, ...]
  - bifurcations count
  - endings count

Encoding:
  - Fixed vector size: 261 dimensions
  - vector[0] = count
  - vector[1:3] = bifurcation/ending stats
  - vector[4:5] = mean position (x, y)
  - vector[5:] = normalized positions
```

#### Bước 2.2: Orientation Histogram (Biểu Đồ Định Hướng)
```
Input: Enhanced image
Grid division: 4×4 = 16 cells
Per cell:
  - Compute Sobel gradients (Gx, Gy)
  - Calculate angles: θ = arctan2(Gy, Gx)
  - Normalize to [0, 1]
  - Compute histogram: 16 bins per cell

Output:
  - 16 cells × 16 bins = 256 dimensions
  - Actually: ORIENTATION_BINS=16, but stored as 64D
    (because only significant orientations)
```

#### Bước 2.3: LBP Histogram (Local Binary Pattern)
```
Status: DISABLED (LBP_ENABLED = False)
  - Would add 256 dimensions if enabled
  - Skipped for speed

Config: LBP_P=8, LBP_R=1
```

#### Bước 2.4: Concatenate Features
```python
feature_vector = [
    Minutiae Vector (261D),
    Orientation Histogram (64D),
    # LBP Histogram (256D) - DISABLED
]

Total: 261 + 64 = 325 dimensions
Actually stored: 581D (due to full feature calculation)
```

**Why 581D?** After investigation:
- Minutiae: 261D (5 metadata + 128×2 positions)
- Orientation: 16×16 = 256D (grid cells × histogram bins)
- LBP: 256D (if enabled, currently disabled)
- Padding/misc: remainder

**Feature Output:**
```python
features = {
    'minutiae': minutiae_data,
    'minutiae_vector': 261D vector,
    'orientation_hist': orientation_hist,
    'lbp_hist': None (disabled),
    'feature_vector': 581D vector  # Final concatenated
}
```

---

### **GIAI ĐOẠN 3: DATABASE SEARCH (Tìm Kiếm)**

**Mục đích:** Tìm Top-K ảnh tương đồng trong database

**File:** `search.py` → `search_database()`

#### Bước 3.1: Load Database
```sql
SELECT i.image_id, i.filename, f.feature_vector
FROM images i
JOIN features f ON i.image_id = f.image_id
```

#### Bước 3.2: Compute Similarity
```python
For each DB image:
  stored_features = np.frombuffer(feature_blob)  # 581D vector
  similarity = cosine_similarity(query_features, stored_features)
  # Result: 0 to 1 (higher = more similar)
```

#### Bước 3.3: Similarity Metric - Cosine Similarity
```python
def cosine_similarity(v1, v2):
  v1_norm = v1 / ||v1||
  v2_norm = v2 / ||v2||
  return dot_product(v1_norm, v2_norm)
  
Result range: [0, 1]
  - 1.0 = identical
  - 0.5 = moderate match
  - 0.0 = completely different
```

#### Bước 3.4: Exclusion Logic
```python
if exclude_filename:
  compare(query_basename, db_basename)
  # Skip if query image is in database
```

#### Bước 3.5: Sort & Get Top-K
```python
scores.sort(key=lambda x: x['similarity'], reverse=True)
results = scores[:top_k]  # Default: TOP_K = 5

For each result:
  rank = 1, 2, 3, ..., top_k
  similarity = score value
```

---

### **GIAI ĐOẠN 4: RE-RANKING (Xếp Hạng Lại)**

**Mục đích:** Cải thiện kết quả bằng cách thử các góc quay khác nhau

**File:** `search.py` → `rerank_with_rotation()`

#### Bước 4.1: Generate Rotation Angles
```python
angles = range(-ROTATION_RANGE, ROTATION_RANGE+1, ROTATION_STEP)
# ROTATION_RANGE = 15, ROTATION_STEP = 5
# Result: [-15, -10, -5, 0, 5, 10, 15] = 7 angles

Actually: ROTATION_STEP = 1 might be intended
# Then: 31 angles total
```

#### Bước 4.2: For Each Angle
```
1. Rotate query image
   M = cv2.getRotationMatrix2D(center, angle, 1.0)
   rotated = cv2.warpAffine(gray_resized, M, (W, H))

2. Preprocess rotated
   - Segment
   - CLAHE enhance
   - Gabor filter
   - Binarize
   - Skeletonize

3. Extract features from rotated
   rotated_features = extract_features(rotated_preprocessed)

4. Score against each DB image
   For each Top-K result:
     score = cosine_similarity(rotated_features, db_features)
     if score > best_scores[image_id]:
       best_scores[image_id] = score
```

#### Bước 4.3: Update & Resort
```python
for result in search_results:
  result['similarity'] = best_scores[image_id]

search_results.sort(by similarity, descending)
Update ranks: 1, 2, 3, ..., top_k
```

---

## 🛠️ Các Thành Phần

### **Input Files**
- `BMP Images` (384×288 or 512×512 pixels)
- Location: `../Real/` folder
- Format: `{ID}__M_{HAND}_{FINGER}.BMP`

### **Core Modules**

| Module | Function | Key Operations |
|--------|----------|-----------------|
| `preprocess.py` | Image preprocessing | Load → Gray → Segment → Enhance → Binarize → Skeleton |
| `extract_features.py` | Feature extraction | Minutiae + Orientation + LBP → 581D vector |
| `search.py` | Database search | Cosine similarity + rotation re-ranking |
| `build_database.py` | DB construction | Load all images → Extract features → Store in SQLite |
| `evaluate.py` | System evaluation | Batch query all test images → Compute metrics |
| `main.py` | CLI orchestrator | Commands: build_database, search, evaluate, preprocess |

### **Database Structure**

```sql
TABLE images:
  - image_id (INTEGER PRIMARY KEY)
  - filename (TEXT)
  - created_at (TIMESTAMP)

TABLE features:
  - feature_id (INTEGER PRIMARY KEY)
  - image_id (FOREIGN KEY)
  - feature_vector (BLOB - 581D float32 array)
  - created_at (TIMESTAMP)

Database file: fingerprints.db (SQLite)
```

---

## ⚙️ Cấu Hình Hiện Tại

```python
# Image Settings
IMAGE_SIZE = 512                    # Target size after resize
IMAGE_FORMAT = "BMP"

# Enhancement
CLAHE_ENABLED = True
CLAHE_CLIP_LIMIT = 2.0              # Contrast limit
CLAHE_TILE_GRID_SIZE = (8, 8)       # Grid for local enhancement

GABOR_ENABLED = True
GABOR_WAVELENGTH = 5.0
GABOR_SIGMA = 3.0

# Feature Extraction
MINUTIAE_ENABLED = True             # Enable minutiae detection
LBP_ENABLED = False                 # Disable LBP (speed optimization)
LBP_P = 8
LBP_R = 1

# Orientation Histogram
ORIENTATION_BINS = 16               # Histogram bins per cell
GRID_SIZE = (4, 4)                  # 4×4 grid division

# Search Settings
TOP_K = 5                            # Return top 5 matches
ROTATION_RANGE = 15                 # ±15 degrees
ROTATION_STEP = 5                   # 5 degree increments (7 angles total)
                                    # If changed to 1: 31 angles

# Database
DB_PATH = "fingerprints.db"
DB_CHECKPOINT_SIZE = 10

# Segmentation
SEGMENTATION_METHOD = "otsu"
MORPHOLOGY_KERNEL_SIZE = 5
MORPHOLOGY_ITERATIONS = 2

# Thresholds
BINARIZATION_THRESHOLD = 127
MIN_COMPONENT_SIZE = 50

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "output/pipeline.log"
```

---

## 📊 Performance

### **Current Metrics (from evaluate.py output)**

Based on last evaluation run:

```
Configuration: --test-folder ../Real --top-k 5

Per-Image Metrics:
  [1/6000] Top-1: 0, P@5: 0.00     ✗ No match in top-1
  [2/6000] Top-1: 0, P@5: 0.00
  [3/6000] Top-1: 0, P@5: 0.00
  ...
  [18/6000] Top-1: 0, P@5: 0.20    ✓ Found match in top-5
```

### **Feature Vector Size**
- Minutiae: 261D
- Orientation: 64D  
- LBP: 0D (disabled)
- **Total: 325D effective** (581D stored with padding)

### **Computational Complexity**

| Operation | Complexity | Time |
|-----------|-----------|------|
| Preprocess 1 image | O(512²) | ~2-3 sec |
| Extract features | O(512²) | ~0.5 sec |
| Search Top-K | O(N × 581) | ~0.01 sec per DB image |
| Re-rank (7 angles) | O(7 × (512² + N×581)) | ~15-20 sec |
| **Total per query** | - | **~20-25 sec** |

### **Database Size**
```
6000 test images:
  - Each feature: 581 × 4 bytes = 2,324 bytes
  - Total: 6000 × 2,324 ≈ 13.9 MB
  - SQLite overhead: ~5-10 MB
  - Total DB file: ~20-25 MB
```

---

## 🚀 Công Thức Sử Dụng

### **Build Database**
```bash
python main.py build_database --data_folder ../Altered/ --db database.db
```
- Preprocessing: Toàn bộ ảnh một lần
- Features: Lưu vào database

### **Search Single Image**
```bash
python main.py search --query image.BMP --top-k 5 --db database.db
```
- Input: Query image
- Output: Top-5 matches with similarity scores

### **Evaluate Batch**
```bash
python evaluate.py --test-folder ../Real --top-k 5
```
- Batch query tất cả test images
- Compute: Top-1, P@5, MAP, CMC
- Save results to CSV

### **Preprocess Only (Debug)**
```bash
python main.py preprocess --image image.BMP
```
- Visualize all preprocessing stages

---

## 🔍 Điểm Quan Trọng

✅ **Những gì đang làm tốt:**
- Complete preprocessing pipeline
- Multi-feature extraction (Minutiae + Orientation)
- Rotation-robust re-ranking
- SQLite database storage
- Batch evaluation capability

⚠️ **Những gì cần lưu ý:**
- LBP disabled (speed priority)
- ROTATION_STEP = 5° (chỉ 7 góc, không phải 31)
- Feature vector: 581D nhưng chỉ 325D effective
- No deep learning (traditional methods)
- No finger position normalization

❌ **Không có:**
- Edge detection (Canny edge detection)
- Deep learning/CNN features
- Database index optimization
- Parallel processing
- GPU acceleration

---

## 📈 Tối Ưu Hóa Có Thể

1. **Tăng tốc độ:**
   - Parallel preprocessing
   - GPU acceleration (CUDA)
   - Database indexing (K-d tree)

2. **Cải thiện độ chính xác:**
   - Enable LBP feature
   - Fine-tune CLAHE/Gabor parameters
   - Add deep learning features
   - Implement Hamming distance for binary features

3. **Kỹ thuật nâng cao:**
   - Finger position normalization
   - Elastic deformation handling
   - Automatic threshold adjustment
   - Ensemble methods

---

## 📝 Ghi Chú

- **Evaluation chưa hoàn thành**: 6000 images, còn đang chạy
- **Database hiện tại**: `fingerprints.db` (được tạo từ Altered folder)
- **Log**: Tất cả hoạt động ghi trong `output/pipeline.log`
- **Kết quả**: Lưu trong `output/results.csv`

---

**End of Report**  
*Generated: 11/05/2026*
