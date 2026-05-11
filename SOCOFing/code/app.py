# ============================================================================
# Streamlit Web App - Interactive Fingerprint Search Demo
# ============================================================================

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import logging
import os

import config
import preprocess
import extract_features
from search import search_fingerprint, print_search_results

logger = logging.getLogger(__name__)

# ============================================================================
# 1. Streamlit Configuration
# ============================================================================
st.set_page_config(page_title="Hệ thống nhận dạng vân tay", layout="wide")

# ============================================================================
# 2. Initialize Session State
# ============================================================================
if 'search_results' not in st.session_state:
    st.session_state.search_results = None

if 'query_image' not in st.session_state:
    st.session_state.query_image = None

if 'query_filename' not in st.session_state:
    st.session_state.query_filename = None

# ============================================================================
# 3. Header
# ============================================================================
st.title("Hệ thống nhận dạng vân tay")
st.markdown("---")

# ============================================================================
# 4. Sidebar Configuration
# ============================================================================
with st.sidebar:
    st.header("Cấu hình")
    
    top_k = st.slider("Số kết quả hiển thị", 1, 20, config.TOP_K)
    
    st.markdown("---")
    st.subheader("Ngưỡng tìm kiếm")
    
    similarity_threshold = st.slider(
        "Độ tương đồng tối thiểu", 
        0.0, 1.0, 
        config.SIMILARITY_THRESHOLD,
        0.05,
        help="Chỉ hiển thị kết quả có độ giống >= ngưỡng này. Càng cao càng chính xác."
    )
    
    # Update config with user's threshold
    config.SIMILARITY_THRESHOLD = similarity_threshold
    
    st.markdown("---")
    

# ============================================================================
# 5. Main UI - Two Columns
# ============================================================================
col1, col2 = st.columns([1, 2])

# ============================================================================
# 6. Column 1: Query Image Upload
# ============================================================================
with col1:
    st.subheader("Ảnh truy vấn")
    
    uploaded_file = st.file_uploader(
        "Tải lên ảnh vân tay (BMP, JPG, PNG)",
        type=["bmp", "BMP", "jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open("temp_query.bmp", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.query_image = "temp_query.bmp"
        st.session_state.query_filename = uploaded_file.name  # Store original filename
        
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh đã tải lên")
        
        # Image info
        st.info(f"Kích thước: {image.size}")

# ============================================================================
# 7. Column 2: Search & Results
# ============================================================================
with col2:
    st.subheader("Kết quả tìm kiếm")
    
    if st.session_state.query_image is not None:
        # Search button
        if st.button("Tìm kiếm", use_container_width=True):
            with st.spinner("Đang tìm kiếm..."):
                try:
                    # Run search (pass original filename to exclude from results)
                    search_result = search_fingerprint(
                        st.session_state.query_image,
                        top_k=top_k,
                        db_path=config.DB_PATH,
                        exclude_filename=st.session_state.query_filename
                    )
                    
                    st.session_state.search_results = search_result
                    
                    # Check quality
                    quality_check = search_result.get('quality_check', {})
                    
                    if not quality_check.get('is_valid', True):
                        st.error("Vui lòng tải lên ảnh vân tay rõ ràng.")
                        if st.checkbox("Hiển thị chi tiết kỹ thuật"):
                            st.info(f"Điểm chất lượng: {quality_check.get('score', 0.0):.2f} / 1.0")
                            st.caption(f"Lý do: {quality_check.get('reason', 'Không rõ')}")
                    elif 'warning' in search_result:
                        st.warning(f"{search_result['warning']}")
                        if 'best_similarity' in search_result:
                            st.info(f"Độ tương đồng cao nhất: {search_result['best_similarity']:.4f} (ngưỡng: {search_result.get('threshold', 0.4):.2f})")
                    else:
                        st.toast(f"Tìm thấy {search_result['total_matches']} kết quả!")
                        if st.session_state.get('show_quality_info', False):
                            st.info(f"Điểm chất lượng: {quality_check.get('score', 0.0):.2f} / 1.0")
                
                except Exception as e:
                    st.error(f"Tìm kiếm thất bại: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Display results
    if st.session_state.search_results is not None:
        results = st.session_state.search_results['results']
        
        # Results table
        st.markdown("#### Kết quả khớp nhất")
        
        for idx, result in enumerate(results[:top_k]):
            with st.container(border=True):
                col_rank, col_img, col_info = st.columns([1, 2, 3])
                
                with col_rank:
                    st.metric("Hạng", f"#{result['rank']}")
                
                with col_img:
                    # Display matched fingerprint image
                    try:
                        matched_image_path = os.path.join(config.DATA_FOLDER, result['filename'])
                        if os.path.exists(matched_image_path):
                            matched_img = Image.open(matched_image_path)
                            st.image(matched_img, caption=f"Kết quả #{result['rank']}", use_container_width=True)
                        else:
                            st.warning("Không tìm thấy ảnh")
                    except Exception as e:
                        st.error(f"Lỗi tải ảnh: {e}")
                
                with col_info:
                    st.write(f"**File:** {result['filename']}")
                    st.write(f"**Độ tương đồng:** {result['similarity']:.4f}")
                    
                    # Progress bar for similarity score
                    st.progress(min(result['similarity'], 1.0))

# ============================================================================
# 8. Bottom Section - Statistics
# ============================================================================
st.markdown("---")

if st.session_state.search_results is not None:
    results = st.session_state.search_results['results']
    st.metric("Tổng số kết quả", len(results))

# ============================================================================
# 10. Footer
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size: 0.8em; color: gray;'>
Hệ thống nhận dạng vân tay | 
Đặc trưng: 517D (Minutiae 261D + Orientation 256D) | 
Metric: Euclidean Distance (normalized) |
Database: SQLite
</div>
""", unsafe_allow_html=True)

# ============================================================================
# Running: streamlit run app.py
# ============================================================================
