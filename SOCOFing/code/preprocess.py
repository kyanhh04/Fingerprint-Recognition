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
# 10. Quality Check - Verify if image is a valid fingerprint
# ============================================================================
def check_fingerprint_quality(preprocessed):
    """
    Check if the preprocessed image is likely a valid fingerprint.
    Returns: (is_valid, quality_score, reason)
    
    Checks:
    1. Ridge density (fingerprints have characteristic ridge patterns)
    2. Minutiae count (fingerprints should have sufficient minutiae)
    3. Contrast (fingerprints should have good contrast)
    4. Edge density (fingerprints have specific edge patterns)
    5. Texture uniformity (fingerprints have consistent texture)
    """
    if preprocessed is None or preprocessed['skeleton'] is None:
        return False, 0.0, "Preprocessing failed"
    
    skeleton = preprocessed['skeleton']
    enhanced = preprocessed['enhanced']
    binary = preprocessed['binary']
    
    quality_checks = []
    reasons = []
    
    # 1. Check ridge density (skeleton should have reasonable white pixels)
    # Fingerprints typically have 5-20% ridge density
    ridge_density = np.count_nonzero(skeleton) / skeleton.size
    if ridge_density < 0.02:  # Too few ridges
        quality_checks.append(0.0)
        reasons.append(f"Very low ridge density ({ridge_density:.4f})")
    elif ridge_density > 0.25:  # Too many ridges (noise or not fingerprint)
        quality_checks.append(0.0)
        reasons.append(f"Very high ridge density ({ridge_density:.4f})")
    elif ridge_density < 0.05 or ridge_density > 0.20:
        quality_checks.append(0.3)
        reasons.append(f"Unusual ridge density ({ridge_density:.4f})")
    else:
        quality_checks.append(1.0)
    
    # 2. Check contrast (fingerprints should have moderate contrast)
    # Natural images often have very high contrast
    contrast = enhanced.std()
    if contrast < 15:  # Too low contrast
        quality_checks.append(0.0)
        reasons.append(f"Very low contrast ({contrast:.2f})")
    elif contrast > 60:  # Too high contrast (likely natural image)
        quality_checks.append(0.0)
        reasons.append(f"Very high contrast ({contrast:.2f} - likely not fingerprint)")
    elif contrast < 20 or contrast > 50:
        quality_checks.append(0.4)
        reasons.append(f"Unusual contrast ({contrast:.2f})")
    else:
        quality_checks.append(1.0)
    
    # 3. Check edge density using Canny
    # Fingerprints have moderate edge density
    edges = cv2.Canny(enhanced, 50, 150)
    edge_density = np.count_nonzero(edges) / edges.size
    if edge_density < 0.03:  # Too few edges
        quality_checks.append(0.0)
        reasons.append(f"Very low edge density ({edge_density:.4f})")
    elif edge_density > 0.30:  # Too many edges (complex scene)
        quality_checks.append(0.0)
        reasons.append(f"Very high edge density ({edge_density:.4f} - likely complex scene)")
    elif edge_density < 0.05 or edge_density > 0.25:
        quality_checks.append(0.3)
        reasons.append(f"Unusual edge density ({edge_density:.4f})")
    else:
        quality_checks.append(1.0)
    
    # 4. Check histogram entropy
    # Fingerprints have moderate entropy (not too uniform, not too complex)
    hist = cv2.calcHist([enhanced], [0], None, [256], [0, 256])
    hist_normalized = hist / hist.sum()
    entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
    if entropy < 3.5:  # Too uniform
        quality_checks.append(0.0)
        reasons.append(f"Very low entropy ({entropy:.2f} - too uniform)")
    elif entropy > 7.0:  # Too complex (natural images)
        quality_checks.append(0.0)
        reasons.append(f"Very high entropy ({entropy:.2f} - too complex)")
    elif entropy < 4.5 or entropy > 6.5:
        quality_checks.append(0.4)
        reasons.append(f"Unusual entropy ({entropy:.2f})")
    else:
        quality_checks.append(1.0)
    
    # 5. Check binary image characteristics
    # Fingerprints should have balanced black/white ratio
    white_ratio = np.count_nonzero(binary) / binary.size
    if white_ratio < 0.2 or white_ratio > 0.8:
        quality_checks.append(0.0)
        reasons.append(f"Unbalanced binary ratio ({white_ratio:.2f})")
    elif white_ratio < 0.3 or white_ratio > 0.7:
        quality_checks.append(0.5)
        reasons.append(f"Unusual binary ratio ({white_ratio:.2f})")
    else:
        quality_checks.append(1.0)
    
    # 6. Check for color information (fingerprints are grayscale)
    # If original image has strong color, it's likely not a fingerprint
    if preprocessed['original'] is not None:
        original = preprocessed['original']
        if len(original.shape) == 3:
            b, g, r = cv2.split(original)
            color_variance = np.mean([
                np.std(b - g),
                np.std(g - r),
                np.std(r - b)
            ])
            if color_variance > 15:  # Strong color content
                quality_checks.append(0.0)
                reasons.append(f"Strong color content ({color_variance:.2f} - likely natural image)")
            elif color_variance > 10:
                quality_checks.append(0.5)
                reasons.append(f"Some color content ({color_variance:.2f})")
            else:
                quality_checks.append(1.0)
    
    # 7. Check texture uniformity using Local Binary Pattern
    # Fingerprints have consistent directional patterns
    from skimage.feature import local_binary_pattern
    lbp = local_binary_pattern(enhanced, 8, 1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
    lbp_hist = lbp_hist.astype(float) / lbp_hist.sum()
    lbp_entropy = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-10))
    
    if lbp_entropy > 3.0:  # Too varied texture (natural scene)
        quality_checks.append(0.0)
        reasons.append(f"Very varied texture ({lbp_entropy:.2f} - likely natural scene)")
    elif lbp_entropy < 1.5:  # Too uniform
        quality_checks.append(0.3)
        reasons.append(f"Very uniform texture ({lbp_entropy:.2f})")
    else:
        quality_checks.append(1.0)
    
    # Calculate overall quality score
    quality_score = np.mean(quality_checks)
    
    # Stricter validation: require at least 60% of checks to pass
    is_valid = quality_score >= 0.6
    
    if not is_valid:
        reason = "; ".join(reasons) if reasons else "Multiple quality issues detected"
    else:
        reason = "Valid fingerprint"
    
    logger.info(f"Quality check: score={quality_score:.2f}, valid={is_valid}")
    logger.debug(f"Details: ridge_density={ridge_density:.4f}, contrast={contrast:.2f}, "
                f"edge_density={edge_density:.4f}, entropy={entropy:.2f}, "
                f"white_ratio={white_ratio:.2f}")
    
    return is_valid, quality_score, reason


# ============================================================================
# 11. Utility: Save Preprocessed Images
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
