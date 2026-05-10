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
st.set_page_config(page_title="Fingerprint Recognition", layout="wide")

# ============================================================================
# 2. Initialize Session State
# ============================================================================
if 'search_results' not in st.session_state:
    st.session_state.search_results = None

if 'query_image' not in st.session_state:
    st.session_state.query_image = None

# ============================================================================
# 3. Header
# ============================================================================
st.title("Fingerprint Recognition System")
st.markdown("---")

# ============================================================================
# 4. Sidebar Configuration
# ============================================================================
with st.sidebar:
    st.header("Configuration")
    
    top_k = st.slider("Top-K Results", 1, 20, config.TOP_K)
    use_rotation = st.checkbox("Enable Rotation Re-ranking", True)
    
    st.markdown("---")
    
    if st.button("ℹInfo"):
        st.info("""
        **Pipeline Features:**
        - ✓ Minutiae Detection (bifurcations & endings)
        - ✓ Orientation Histogram (16-bin, 4x4 grid)
        - ✓ CLAHE Enhancement (automatic)
        - ✓ Gabor Filter (optional)
        - ✓ Rotation Re-ranking (±15°)
        """)

# ============================================================================
# 5. Main UI - Two Columns
# ============================================================================
col1, col2 = st.columns([1, 2])

# ============================================================================
# 6. Column 1: Query Image Upload
# ============================================================================
with col1:
    st.subheader("Query Image")
    
    uploaded_file = st.file_uploader(
        "Upload BMP fingerprint image",
        type=["bmp", "BMP", "jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open("temp_query.bmp", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.query_image = "temp_query.bmp"
        
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")
        
        # Image info
        st.info(f"Image size: {image.size}")

# ============================================================================
# 7. Column 2: Search & Results
# ============================================================================
with col2:
    st.subheader("Search Results")
    
    if st.session_state.query_image is not None:
        # Search button
        if st.button("Search", use_container_width=True):
            with st.spinner("Searching..."):
                try:
                    # Run search
                    search_result = search_fingerprint(
                        st.session_state.query_image,
                        top_k=top_k,
                        use_rotation=use_rotation,
                        db_path=config.DB_PATH
                    )
                    
                    st.session_state.search_results = search_result
                    st.success(f"Found {search_result['total_matches']} matches!")
                
                except Exception as e:
                    st.error(f"Search failed: {e}")
    
    # Display results
    if st.session_state.search_results is not None:
        results = st.session_state.search_results['results']
        
        # Results table
        st.markdown("#### Top Matches")
        
        for idx, result in enumerate(results[:top_k]):
            with st.container(border=True):
                col_rank, col_info, col_status = st.columns([1, 3, 1])
                
                with col_rank:
                    st.metric("Rank", f"#{result['rank']}")
                
                with col_info:
                    st.write(f"**File:** {result['filename']}")
                    st.write(f"**Similarity:** {result['similarity']:.4f}")

# ============================================================================
# 8. Bottom Section - Statistics
# ============================================================================
st.markdown("---")

if st.session_state.search_results is not None:
    results = st.session_state.search_results['results']
    st.metric("Total Matches", len(results))

# ============================================================================
# 9. Debug Section
# ============================================================================
if st.checkbox("Debug Mode"):
    st.subheader("Debug Information")
    
    # Configuration
    with st.expander("Current Configuration"):
        st.write(f"Top-K: {top_k}")
        st.write(f"Rotation Re-ranking: {use_rotation}")
    
    # Database stats
    with st.expander("Database Statistics"):
        try:
            import sqlite3
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM images')
            total_images = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM features')
            total_features = cursor.fetchone()[0]
            
            conn.close()
            
            st.write(f"Total Images: {total_images}")
            st.write(f"Total Features: {total_features}")
        except Exception as e:
            st.error(f"Error reading database: {e}")

# ============================================================================
# 10. Footer
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size: 0.8em; color: gray;'>
Fingerprint Recognition System | 
Features: Minutiae + Orientation | 
Database: SQLite
</div>
""", unsafe_allow_html=True)

# ============================================================================
# Running: streamlit run app.py
# ============================================================================
