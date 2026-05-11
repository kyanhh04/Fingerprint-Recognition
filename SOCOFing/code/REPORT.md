# Báo Cáo Dự Án: Hệ Thống Truy Vấn Ảnh Vân Tay

## 1. Mục tiêu tổng quát
Dự án hướng tới xây dựng một hệ thống truy vấn ảnh vân tay hoàn chỉnh, từ tiền xử lý ảnh đến trả về kết quả Top‑5, tối ưu cho truy vấn nhanh trên cơ sở dữ liệu SQLite chứa ~6000 mẫu.

## 1.2 Mục tiêu cụ thể
- Xây dựng CSDL SQLite: thiết kế bảng lưu metadata và vector đặc trưng, có khả năng lưu trữ hàng nghìn vector float32 và metadata liên quan.
- Trích rút đặc trưng: Lưu ý hiện trạng mã nguồn hiện tại trích vector 517‑D (261D minutiae + 256D orientation). Dự án có thể giảm chiều xuống 25‑D cho báo cáo/so sánh bằng phương pháp PCA nếu cần.
- Thiết kế cơ chế tìm kiếm trọng số: kết hợp khoảng cách Euclidean trên vector với trọng số ưu tiên cho đặc trưng hình thái học (morphological features).
- Xây dựng giao diện demo: triển khai Streamlit app cho phép upload ảnh, điều chỉnh ngưỡng và xem Top‑5 kết quả.
- Đánh giá hiệu năng: dùng Precision@5 và Top‑1 Accuracy để định lượng trên bộ dữ liệu thực nghiệm.

## 1.3 Đối tượng và phạm vi
- Đối tượng: ảnh vân tay kỹ thuật số thu bằng cảm biến quang học, được chuẩn hoá theo kích thước và định dạng.
- Quy mô dữ liệu: hệ thống vận hành trên ~6000 mẫu ảnh.
- Phạm vi: tập trung vào trích xuất đặc trưng mức thấp và truy vấn tương đồng; không thực hiện xác thực pháp lý, chỉ hỗ trợ truy vấn nhanh trong cơ sở dữ liệu.

## 1.4 Phương pháp thực hiện (Quy trình)
Quy trình được tổ chức theo các pha xử lý logic:

1. Tiền xử lý
   - Chuẩn hoá kích thước ảnh về 512×512 pixels; chuyển sang grayscale.
   - Loại nhiễu cơ bản (Gaussian blur, CLAHE nếu cần).

2. ROI Extraction (Foreground Mask)
   - Sinh mặt nạ foreground để loại bỏ nền trắng và giữ vùng có rãnh.

3. Nhị phân hóa và skeletonization
   - Binarize → (tùy chọn) skeletonize để làm nổi bật đường rãnh phục vụ trích minutiae.

4. Trích rút đặc trưng (517‑D)
    - Mô tả chung: mã hiện tại trích vector 517 chiều, được tạo bằng cách ghép hai nhóm đặc trưng chính:
       - **Minutiae vector (261D):** 5 giá trị metadata đầu (số minutiae, số bifurcation, số ending, mean_x, mean_y) + tọa độ x,y cho tối đa 128 minutiae (256 giá trị). Nếu số minutiae thực tế < 128, phần còn lại được đặt 0.
       - **Orientation histogram (256D):** chia ảnh thành lưới 4×4 (config.GRID_SIZE) và tính histogram hướng trong mỗi ô với `ORIENTATION_BINS=16` → 4×4×16 = 256.
    - Kết hợp: `feature_vector = concat(minutiae_vector, orientation_hist)` → tổng 517 chiều.
    - Nếu cần giảm chiều cho báo cáo: có thể áp dụng PCA (ví dụ 517→25) bằng cách fit PCA trên toàn bộ ma trận features trong DB và lưu vectors 25‑D (script PCA được cung cấp trong `code/tools`).

5. Lưu trữ SQLite
   - Bảng `images(image_id INTEGER PRIMARY KEY, filename TEXT, metadata JSON)`.
   - Bảng `features(image_id INTEGER, feature_vector BLOB)` — lưu float32.tobytes().

6. Truy vấn và xếp hạng
   - Từ ảnh truy vấn: preprocess → extract 25D vector.
   - Chuẩn hoá (L2‑normalize) → tính khoảng cách Euclidean.
   - Chuyển sang similarity: similarity = 1 − (d / d_max). Với vector chuẩn hoá, dùng d_max=2 ⇒ similarity = 1 − d/2.
   - Áp ngưỡng `SIMILARITY_THRESHOLD` và trả Top‑K (mặc định K=5).
   - (Tùy chọn) Re‑rank bằng visual checks (SSIM/ORB) hoặc kết hợp trọng số.

## Đánh giá hiệu năng
- Metrics: Precision@5, Top‑1 Accuracy.
- Cách đo: chạy trên tập kiểm thử (ví dụ 10% tổng dữ liệu), sinh `evaluation.csv` chứa: query, ground_truth_id, top1_filename, top1_score, precision_at_5.
- Ngưỡng khởi điểm: `SIMILARITY_THRESHOLD = 0.4` (tinh chỉnh sau khi thử nghiệm).

## Kiến trúc và công nghệ
- Ngôn ngữ: Python
- Thư viện: OpenCV, NumPy, scikit‑image, sqlite3, Streamlit
- Cấu trúc kho mã (chính):
  - `preprocess.py`: load, resize, mask, enhance, binarize, skeletonize, quality check
  - `extract_features.py`: hàm trích 25D
  - `search.py`: tìm kiếm trong SQLite + lọc/top‑k
  - `app.py`: Streamlit demo
  - `evaluate.py`: script đánh giá (Precision@5, Top‑1)

## Kế hoạch triển khai (gợi ý)
- Tuần 1–2: Thiết kế DB, triển khai tiền xử lý và hàm trích 25D.
- Tuần 3: Triển khai search + demo Streamlit.
- Tuần 4: Chạy đánh giá, tinh chỉnh ngưỡng, xuất báo cáo.

## Ghi chú về giảm chiều (PCA)
- Mục tiêu: giữ lại phần lớn phương sai của vector 517‑D trong không gian 25‑D để phù hợp với yêu cầu báo cáo.
- Quy trình ngắn:
   1. Đọc tất cả feature_vector từ bảng `features` trong `fingerprints.db`.
   2. Stack vào ma trận (N x 517), chạy `sklearn.decomposition.PCA(n_components=25)`.
   3. Lưu ma trận components và mean, đồng thời lưu vectors 25‑D vào bảng `features_pca(image_id, vector25 BLOB)` hoặc `pca_features.csv`.
   4. Thay thế hoặc song song sử dụng vector 25‑D cho các thí nghiệm hoặc báo cáo.

--

_Đã cập nhật báo cáo để phản ánh trạng thái hiện tại (vector 517‑D) và thêm phương án giảm chiều (PCA)._

## Kết luận & Khuyến nghị
- Hệ thống CBIR hiện nay phù hợp để truy vấn tương đồng dựa trên đặc trưng rãnh; để nâng độ tin cậy trực quan, nên bổ sung lớp kiểm tra visual (SSIM/ORB) hoặc xây dựng embedding học sâu nếu cần tính chính xác cao.

---

_File này tạo tự động theo yêu cầu. Muốn mình xuất thêm `README.md` cập nhật hoặc PDF báo cáo không?_
