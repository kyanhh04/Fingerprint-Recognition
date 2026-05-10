# ============================================================================
# Preprocessing - Image Loading, Segmentation, Enhancement
# ============================================================================

import cv2
import numpy as np
from PIL import Image
import logging

import config

logger = logging.getLogger(__name__)

# ============================================================================
# 1. Load BMP Image
# ============================================================================
def load_bmp(image_path):
    """Load BMP image using OpenCV (BGR format)"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        logger.debug(f"Loaded image: {image_path}, shape={img.shape}")
        return img
    except Exception as e:
        logger.error(f"Error loading {image_path}: {e}")
        return None

# ============================================================================
# 2. Convert to Grayscale
# ============================================================================
def to_grayscale(img):
    """Convert BGR image to grayscale"""
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

# ============================================================================
# 3. Resize to Standard Size
# ============================================================================
def resize_image(img, size=config.IMAGE_SIZE):
    """Resize image to standard size"""
    if img is None:
        return None
    resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return resized

# ============================================================================
# 4. Segmentation (Fingerprint Region Detection)
# ============================================================================
def segment_fingerprint(img):
    """Segment fingerprint region using Otsu thresholding"""
    if img is None:
        return None, None
    
    # Apply Gaussian blur first
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Otsu thresholding
    _, binary = cv2.threshold(blurred, config.BINARIZATION_THRESHOLD, 255, 
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                       (config.MORPHOLOGY_KERNEL_SIZE, 
                                        config.MORPHOLOGY_KERNEL_SIZE))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, 
                             iterations=config.MORPHOLOGY_ITERATIONS)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, 
                             iterations=1)
    
    return binary, None

# ============================================================================
# 5. CLAHE Enhancement (Contrast Limited Adaptive Histogram Equalization)
# ============================================================================
def apply_clahe(img):
    """Apply CLAHE to enhance contrast"""
    if img is None or not config.CLAHE_ENABLED:
        return img
    
    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT, 
                           tileGridSize=config.CLAHE_TILE_GRID_SIZE)
    enhanced = clahe.apply(img)
    return enhanced

# ============================================================================
# 6. Gabor Filter Enhancement
# ============================================================================
def apply_gabor_filter(img, mask=None):
    """Apply Gabor filter to enhance ridge structure"""
    if img is None or not config.GABOR_ENABLED:
        return img
    
    # Create Gabor kernel
    kernel_size = int(2 * np.ceil(3 * config.GABOR_SIGMA) + 1)
    kernel = cv2.getGaborKernel((kernel_size, kernel_size), 
                                config.GABOR_SIGMA, 
                                np.pi/4,  # 45 degree orientation
                                config.GABOR_WAVELENGTH, 
                                0.5, 0)
    
    # Apply filter
    filtered = cv2.filter2D(img, -1, kernel)
    
    # Normalize
    filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return filtered

# ============================================================================
# 7. Binarization
# ============================================================================
def binarize_image(img):
    """Binarize image"""
    if img is None:
        return None
    
    _, binary = cv2.threshold(img, config.BINARIZATION_THRESHOLD, 255, 
                              cv2.THRESH_BINARY)
    return binary

# ============================================================================
# 8. Skeletonization
# ============================================================================
def skeletonize_image(binary_img):
    """Skeletonize binary image to 1-pixel wide skeleton"""
    if binary_img is None:
        return None
    
    # Invert: white ridge, black background
    inverted = cv2.bitwise_not(binary_img)
    
    # Morphological skeleton
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skeleton = np.zeros_like(inverted)
    
    while True:
        eroded = cv2.morphologyEx(inverted, cv2.MORPH_ERODE, kernel)
        dilated = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp = cv2.subtract(inverted, dilated)
        skeleton = cv2.bitwise_or(skeleton, temp)
        inverted = eroded.copy()
        
        if cv2.countNonZero(inverted) == 0:
            break
    
    return skeleton

# ============================================================================
# 9. Full Preprocessing Pipeline
# ============================================================================
def preprocess_image(image_path, apply_enhancements=True):
    """
    Full preprocessing pipeline:
    1. Load BMP
    2. Convert to grayscale
    3. Resize to standard size
    4. Segment fingerprint region
    5. Apply CLAHE & Gabor enhancement
    6. Binarize & Skeletonize
    
    Returns: dict with preprocessed images
    """
    result = {
        'original': None,
        'grayscale': None,
        'resized': None,
        'segmented': None,
        'enhanced': None,
        'binary': None,
        'skeleton': None,
        'mask': None
    }
    
    # 1. Load
    img_bgr = load_bmp(image_path)
    if img_bgr is None:
        return None
    result['original'] = img_bgr
    
    # 2. Grayscale
    gray = to_grayscale(img_bgr)
    result['grayscale'] = gray
    
    # 3. Resize
    resized = resize_image(gray)
    result['resized'] = resized
    
    # 4. Segment
    mask, _ = segment_fingerprint(resized)
    result['segmented'] = resized.copy()
    result['mask'] = mask
    
    # 5. Enhance
    enhanced = resized.copy()
    if apply_enhancements:
        enhanced = apply_clahe(enhanced)
        enhanced = apply_gabor_filter(enhanced, mask)
    result['enhanced'] = enhanced
    
    # 6. Binarize
    binary = binarize_image(enhanced)
    result['binary'] = binary
    
    # 7. Skeletonize
    skeleton = skeletonize_image(binary)
    result['skeleton'] = skeleton
    
    logger.info(f"Preprocessed: {image_path}")
    return result

# ============================================================================
# 10. Utility: Save Preprocessed Images
# ============================================================================
def save_preprocessed_images(preprocessed, output_dir, image_id):
    """Save all preprocessed stages for visualization"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    stages = ['original', 'grayscale', 'resized', 'enhanced', 'binary', 'skeleton']
    for stage in stages:
        if preprocessed[stage] is not None:
            path = os.path.join(output_dir, f"{image_id}_{stage}.png")
            cv2.imwrite(path, preprocessed[stage])
            logger.debug(f"Saved: {path}")
