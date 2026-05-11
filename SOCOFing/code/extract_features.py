# ============================================================================
# Feature Extraction - Minutiae, Orientation Histogram, LBP
# ============================================================================

import cv2
import numpy as np
import logging
from scipy import ndimage
from skimage.feature import local_binary_pattern

import config

logger = logging.getLogger(__name__)

# ============================================================================
# 1. Minutiae Detection (Bifurcations & Endings)
# ============================================================================
def detect_minutiae(skeleton):
    """
    Detect minutiae (bifurcations and endings) from skeleton image.
    Uses pixel neighbor count to identify minutiae types.
    """
    if skeleton is None:
        return {'minutiae': [], 'bifurcations': 0, 'endings': 0}
    
    minutiae = []
    bifurcations = 0
    endings = 0
    
    # Normalize skeleton to 0/1
    skeleton_bin = (skeleton > 127).astype(np.uint8)
    
    h, w = skeleton_bin.shape
    
    # Check each skeleton pixel
    for i in range(1, h-1):
        for j in range(1, w-1):
            if skeleton_bin[i, j] == 1:  # If pixel is part of skeleton
                # Count neighbors (8-connectivity)
                neighbors = skeleton_bin[i-1:i+2, j-1:j+2]
                neighbor_count = np.sum(neighbors) - 1  # Exclude center pixel
                
                # Ending: 1 neighbor
                if neighbor_count == 1:
                    minutiae.append({'type': 'ending', 'x': j, 'y': i})
                    endings += 1
                # Bifurcation: 3 neighbors
                elif neighbor_count == 3:
                    minutiae.append({'type': 'bifurcation', 'x': j, 'y': i})
                    bifurcations += 1
    
    logger.debug(f"Detected minutiae: {len(minutiae)} (endings={endings}, bifurcations={bifurcations})")
    
    return {
        'minutiae': minutiae,
        'bifurcations': bifurcations,
        'endings': endings,
        'count': len(minutiae)
    }

# ============================================================================
# 2. Minutiae Vector Representation
# ============================================================================
def encode_minutiae(minutiae_data, max_minutiae=128):
    """
    Convert minutiae to fixed-size vector.
    Vector: [count, bifurc_ratio, ending_ratio, position_mean_x, position_mean_y, ...]
    """
    minutiae = minutiae_data['minutiae']
    count = len(minutiae)
    
    # Initialize vector
    vector = np.zeros(max_minutiae * 2 + 5)  # 2 features per minutiae + metadata
    
    # Metadata
    vector[0] = count
    vector[1] = minutiae_data['bifurcations']
    vector[2] = minutiae_data['endings']
    
    if count > 0:
        positions = np.array([[m['x'], m['y']] for m in minutiae])
        vector[3] = np.mean(positions[:, 0])  # mean x
        vector[4] = np.mean(positions[:, 1])  # mean y
        
        # Encode minutiae (position normalized)
        for idx, m in enumerate(minutiae[:max_minutiae]):
            vector[5 + idx*2] = m['x']
            vector[5 + idx*2 + 1] = m['y']
    
    return vector

# ============================================================================
# 3. Orientation Histogram
# ============================================================================
def compute_orientation_histogram(enhanced_img, mask=None, n_bins=config.ORIENTATION_BINS):
    """
    Compute orientation histogram of ridge structure.
    Divide image into 4x4 grid, compute dominant orientation per cell.
    """
    if enhanced_img is None:
        return np.zeros(config.GRID_SIZE[0] * config.GRID_SIZE[1] * n_bins)
    
    h, w = enhanced_img.shape
    grid_h, grid_w = config.GRID_SIZE
    cell_h, cell_w = h // grid_h, w // grid_w
    
    orientation_hist = []
    
    # Compute Sobel gradients
    sobelx = cv2.Sobel(enhanced_img, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(enhanced_img, cv2.CV_32F, 0, 1, ksize=3)
    
    # Compute angles
    angles = np.arctan2(sobely, sobelx)
    angles = (angles + np.pi) / (2 * np.pi)  # Normalize to [0, 1]
    
    # Process each grid cell
    for gi in range(grid_h):
        for gj in range(grid_w):
            y_start = gi * cell_h
            y_end = (gi + 1) * cell_h
            x_start = gj * cell_w
            x_end = (gj + 1) * cell_w
            
            cell_angles = angles[y_start:y_end, x_start:x_end]
            
            # Histogram of orientations in this cell
            hist, _ = np.histogram(cell_angles.flatten(), bins=n_bins, range=(0, 1))
            hist = hist / (hist.sum() + 1e-6)  # Normalize
            
            orientation_hist.extend(hist)
    
    return np.array(orientation_hist)

# ============================================================================
# 4. LBP (Local Binary Pattern) - Optional Feature
# ============================================================================
def compute_lbp_histogram(img, P=config.LBP_P, R=config.LBP_R, n_bins=256):
    """
    Compute LBP (Local Binary Pattern) histogram.
    Optional: for high-precision scenarios.
    """
    if img is None or not config.LBP_ENABLED:
        return np.zeros(n_bins)
    
    lbp = local_binary_pattern(img, P, R, method='uniform')
    hist, _ = np.histogram(lbp.flatten(), bins=n_bins, range=(0, n_bins))
    hist = hist / (hist.sum() + 1e-6)
    
    return hist

# ============================================================================
# 5. Feature Extraction - Main Function
# ============================================================================
def extract_features(preprocessed_dict):
    """
    Extract all features from preprocessed fingerprint.
    
    Returns:
        features dict with:
        - minutiae_vector
        - orientation_hist
        - lbp_hist (if enabled)
        - feature_vector (concatenated)
    """
    features = {
        'minutiae': None,
        'minutiae_vector': None,
        'orientation_hist': None,
        'lbp_hist': None,
        'feature_vector': None
    }
    
    # 1. Minutiae Detection
    if config.MINUTIAE_ENABLED:
        minutiae_data = detect_minutiae(preprocessed_dict['skeleton'])
        features['minutiae'] = minutiae_data
        features['minutiae_vector'] = encode_minutiae(minutiae_data)
    
    # 2. Orientation Histogram
    orientation_hist = compute_orientation_histogram(
        preprocessed_dict['enhanced'],
        mask=preprocessed_dict['mask'],
        n_bins=config.ORIENTATION_BINS
    )
    features['orientation_hist'] = orientation_hist
    
    # 3. LBP Histogram (optional)
    if config.LBP_ENABLED:
        lbp_hist = compute_lbp_histogram(preprocessed_dict['enhanced'])
        features['lbp_hist'] = lbp_hist
    
    # 4. Concatenate all features
    feature_parts = []
    if features['minutiae_vector'] is not None:
        feature_parts.append(features['minutiae_vector'])
    feature_parts.append(orientation_hist)
    if features['lbp_hist'] is not None:
        feature_parts.append(features['lbp_hist'])
    
    features['feature_vector'] = np.concatenate(feature_parts)
    
    logger.debug(f"Extracted features: vector_size={len(features['feature_vector'])}")
    
    return features

# ============================================================================
# 6. Feature Normalization
# ============================================================================
def normalize_features(feature_vector):
    """Normalize feature vector to unit length"""
    norm = np.linalg.norm(feature_vector)
    if norm > 0:
        return feature_vector / norm
    return feature_vector

# ============================================================================
# 7. Distance Metrics
# ============================================================================
def euclidean_distance(vec1, vec2):
    """Euclidean distance between two normalized feature vectors"""
    # Normalize vectors first to unit length
    v1_norm = normalize_features(vec1)
    v2_norm = normalize_features(vec2)
    # Compute distance on normalized vectors (range: [0, 2])
    return np.linalg.norm(v1_norm - v2_norm)

def cosine_similarity(vec1, vec2):
    """Cosine similarity (0 to 1, higher is more similar)"""
    v1_norm = normalize_features(vec1)
    v2_norm = normalize_features(vec2)
    similarity = np.dot(v1_norm, v2_norm)
    return max(0, min(1, similarity))  # Clamp to [0, 1]

def compute_similarity_score(vec1, vec2, metric='cosine'):
    """
    Compute similarity score.
    metric: 'cosine' or 'euclidean' (normalized)
    Both metrics now work well with high-dimensional vectors (517D):
    - Cosine: measures angle between vectors
    - Euclidean: measures distance between normalized vectors (range 0-2)
    """
    if metric == 'cosine':
        return cosine_similarity(vec1, vec2)
    else:  # euclidean (normalized)
        # Distance on normalized vectors is in range [0, 2]
        # Convert to similarity: 1 - (distance / 2)
        dist = euclidean_distance(vec1, vec2)
        return 1.0 - (dist / 2.0)  # Maps [0, 2] to [1, 0]
