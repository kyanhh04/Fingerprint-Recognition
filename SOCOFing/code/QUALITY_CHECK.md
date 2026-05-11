# Quality Check & Similarity Threshold

## Vấn đề
Hệ thống trước đây trả về kết quả với độ tương đồng cao ngay cả khi ảnh đầu vào **không phải là vân tay** (ví dụ: ảnh phong cảnh, chữ viết, ảnh màu, v.v.).

## Giải pháp

### 1. **Quality Check (Kiểm tra chất lượng ảnh) - Nâng cao**
Trước khi tìm kiếm, hệ thống kiểm tra xem ảnh có phải là vân tay hợp lệ không với **7 tiêu chí**:

#### Các tiêu chí kiểm tra:
1. **Ridge Density**: Mật độ vân (0.05 - 0.20) - Vân tay có mật độ vân đặc trưng
2. **Contrast**: Độ tương phản (20 - 60) - Ảnh tự nhiên thường có contrast rất cao (>60)
3. **Edge Density**: Mật độ cạnh (0.05 - 0.25) - Cảnh phức tạp có edge density cao (>0.30)
4. **Entropy**: Độ phức tạp (4.5 - 6.5) - Ảnh tự nhiên có entropy rất cao (>7.0)
5. **Binary Ratio**: Tỷ lệ đen/trắng (0.3 - 0.7) - Vân tay có tỷ lệ cân bằng
6. **Color Variance**: Độ lệch màu (<10) - Vân tay là ảnh xám, ảnh màu có variance cao
7. **Texture Uniformity (LBP)**: Độ đồng nhất texture (<3.0) - Cảnh tự nhiên có texture rất đa dạng

#### Điểm chất lượng (Quality Score):
- `>= 0.6`: Ảnh hợp lệ ✅ (ngưỡng tăng từ 0.5 lên 0.6)
- `< 0.6`: Ảnh không hợp lệ ❌

### 2. **Similarity Threshold (Ngưỡng độ tương đồng)**
Chỉ trả về kết quả có độ tương đồng >= ngưỡng (mặc định: **0.4**, tăng từ 0.3)

#### Điều chỉnh ngưỡng:
- **Cao (0.5 - 0.8)**: Chỉ trả về kết quả rất giống
- **Trung bình (0.3 - 0.5)**: Cân bằng ⭐ (khuyến nghị)
- **Thấp (0.1 - 0.3)**: Trả về nhiều kết quả hơn

## Cấu hình

### File `config.py`:
```python
SIMILARITY_THRESHOLD = 0.4  # Ngưỡng độ tương đồng tối thiểu (tăng từ 0.3)
MIN_QUALITY_SCORE = 0.6     # Điểm chất lượng tối thiểu (tăng từ 0.5)
```

### Giao diện Streamlit:
- Slider "Similarity Threshold" trong sidebar
- Hiển thị Quality Score khi tìm kiếm
- Cảnh báo khi ảnh không hợp lệ

## Kết quả

### Ảnh vân tay hợp lệ:
```
✅ Found 5 matches!
Quality Score: 0.85 / 1.0
```

### Ảnh không phải vân tay (ảnh phong cảnh):
```
⚠️ Image Quality Issue: Very high contrast (85.34 - likely not fingerprint); 
Very high edge density (0.35 - likely complex scene); 
Very high entropy (7.45 - too complex); 
Strong color content (45.23 - likely natural image); 
Very varied texture (3.25 - likely natural scene)

Quality Score: 0.14 / 1.0
💡 Please upload a clear fingerprint image.
```

### Không có kết quả trên ngưỡng:
```
⚠️ No matches found with similarity >= 0.3
Best match similarity: 0.1234 (threshold: 0.30)
```

## Test

### Test quality check với ảnh vân tay:
```bash
python test_quality.py ../Real/100__M_Left_index_finger.BMP
```

### Test với ảnh phong cảnh:
```bash
python test_landscape.py
```

### Test với ảnh tùy chỉnh:
```bash
python test_landscape.py /path/to/your/image.jpg
```

### Kết quả mong đợi:
```
### TEST 1: Synthetic Landscape Image ###
RESULTS:
  Valid: ❌ NO
  Score: 0.14 / 1.0
  Reason: Very high contrast; Very high edge density; Strong color content

### TEST 3: Real Fingerprint (for comparison) ###
RESULTS:
  Valid: ✅ YES
  Score: 0.86 / 1.0
  Reason: Valid fingerprint
```

## API Changes

### `search_fingerprint()` return value:
```python
{
    'query': 'path/to/image.bmp',
    'results': [...],
    'total_matches': 5,
    'quality_check': {
        'is_valid': True,
        'score': 0.85,
        'reason': 'Valid fingerprint'
    },
    'threshold': 0.3,
    'warning': None,  # hoặc thông báo cảnh báo
    'best_similarity': 0.95  # chỉ có khi không có kết quả
}
```

## Lợi ích

1. ✅ **Ngăn chặn false positives**: Không trả về kết quả khi ảnh không phải vân tay
2. ✅ **Cải thiện độ chính xác**: Chỉ trả về kết quả có độ tương đồng cao
3. ✅ **Trải nghiệm người dùng tốt hơn**: Cảnh báo rõ ràng khi ảnh không hợp lệ
4. ✅ **Linh hoạt**: Có thể điều chỉnh ngưỡng theo nhu cầu
