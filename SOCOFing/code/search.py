# ============================================================================
# Search - Query, Retrieve Top-K, Re-rank with Rotation
# ============================================================================

import os
import sqlite3
import numpy as np
import logging
import argparse
from datetime import datetime

import config
import preprocess
import extract_features
from extract_features import compute_similarity_score

logger = logging.getLogger(__name__)

# ============================================================================
# 1. Query Database for Matches
# ============================================================================
def search_database(query_features, db_path=config.DB_PATH, top_k=config.TOP_K, metric='cosine', exclude_filename=None):
    """
    Search database for matches.
    Returns: list of (image_id, filename, similarity_score, rank)
    
    Args:
        query_features: Feature vector of query image
        db_path: Path to database
        top_k: Number of top results to return
        metric: Similarity metric to use
        exclude_filename: Filename to exclude from results (e.g., query image itself)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Fetch all images with their features
    cursor.execute('''
        SELECT i.image_id, i.filename, f.feature_vector
        FROM images i
        JOIN features f ON i.image_id = f.image_id
    ''')
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        logger.warning("Database is empty")
        return []
    
    # Compute similarities
    scores = []
    excluded_count = 0
    
    for image_id, filename, feature_blob in rows:
        # Skip if this is the query image itself
        # Compare both basename and full filename to handle different cases
        if exclude_filename:
            exclude_base = os.path.basename(exclude_filename)
            filename_base = os.path.basename(filename)
            
            if exclude_base == filename_base or exclude_base == filename:
                logger.info(f"✓ Excluding query image: exclude='{exclude_filename}' vs db='{filename}'")
                excluded_count += 1
                continue
            
        stored_features = np.frombuffer(feature_blob, dtype=np.float32)
        
        try:
            similarity = compute_similarity_score(query_features, stored_features, metric=metric)
            scores.append({
                'image_id': image_id,
                'filename': filename,
                'similarity': similarity
            })
        except Exception as e:
            logger.warning(f"Error computing similarity for {filename}: {e}")
            continue
    
    if excluded_count > 0:
        logger.info(f"Excluded {excluded_count} image(s) from search results")
    
    # Sort by similarity (descending)
    scores.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Return top-k
    results = scores[:top_k]
    for rank, result in enumerate(results, 1):
        result['rank'] = rank
    
    logger.debug(f"Found {len(results)} matches")
    return results

# ============================================================================
# 2. Re-rank with Rotation
# ============================================================================
def rerank_with_rotation(query_image_path, query_features, search_results, 
                        db_path=config.DB_PATH):
    """
    Re-rank results by trying rotated versions of query image.
    Keep the best match per database image.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    rotation_angles = range(-config.ROTATION_RANGE, config.ROTATION_RANGE + 1, config.ROTATION_STEP)
    best_scores = {}  # image_id -> best_score
    
    # Try each rotation
    for angle in rotation_angles:
        # Rotate query image
        gray = preprocess.to_grayscale(preprocess.load_bmp(query_image_path))
        gray_resized = preprocess.resize_image(gray)
        
        h, w = gray_resized.shape
        center = (w // 2, h // 2)
        
        import cv2
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(gray_resized, M, (w, h))
        
        # Preprocess rotated
        mask, _ = preprocess.segment_fingerprint(rotated)
        enhanced = preprocess.apply_clahe(rotated)
        enhanced = preprocess.apply_gabor_filter(enhanced, mask)
        binary = preprocess.binarize_image(enhanced)
        skeleton = preprocess.skeletonize_image(binary)
        
        # Extract features from rotated
        rotated_preprocessed = {
            'original': None,
            'grayscale': rotated,
            'resized': rotated,
            'enhanced': enhanced,
            'binary': binary,
            'skeleton': skeleton,
            'mask': mask
        }
        rotated_features = extract_features.extract_features(rotated_preprocessed)
        
        # Score against each database image
        for result in search_results:
            image_id = result['image_id']
            
            cursor.execute('SELECT feature_vector FROM features WHERE image_id = ?', (image_id,))
            row = cursor.fetchone()
            if row is None:
                continue
            
            stored_features = np.frombuffer(row[0], dtype=np.float32)
            score = compute_similarity_score(rotated_features['feature_vector'], stored_features)
            
            if image_id not in best_scores or score > best_scores[image_id]:
                best_scores[image_id] = score
    
    conn.close()
    
    # Update search results with best scores
    for result in search_results:
        if result['image_id'] in best_scores:
            result['similarity'] = best_scores[result['image_id']]
    
    # Re-sort
    search_results.sort(key=lambda x: x['similarity'], reverse=True)
    for rank, result in enumerate(search_results, 1):
        result['rank'] = rank
    
    logger.info(f"Re-ranked with rotation (±{config.ROTATION_RANGE}°)")
    return search_results

# ============================================================================
# 3. Full Search Pipeline
# ============================================================================
def search_fingerprint(query_image_path, top_k=config.TOP_K, use_rotation=True, 
                      db_path=config.DB_PATH, exclude_filename=None):
    """
    Full search pipeline:
    1. Preprocess query image
    2. Check fingerprint quality
    3. Extract features
    4. Search database
    5. Filter by similarity threshold
    6. Re-rank with rotation (optional)
    
    Args:
        query_image_path: Path to query image
        top_k: Number of top results to return
        use_rotation: Whether to use rotation re-ranking
        db_path: Path to database
        exclude_filename: Original filename to exclude from results (e.g., when querying an image from the database)
    
    Returns: dict with results and quality info
    """
    logger.info(f"Searching for: {query_image_path}")
    
    # 1. Preprocess query
    preprocessed = preprocess.preprocess_image(query_image_path, apply_enhancements=True)
    if preprocessed is None:
        logger.error("Failed to preprocess query image")
        return {
            'query': query_image_path,
            'results': [],
            'total_matches': 0,
            'error': 'Preprocessing failed',
            'quality_check': {'is_valid': False, 'score': 0.0, 'reason': 'Preprocessing failed'}
        }
    
    # 2. Check fingerprint quality
    is_valid, quality_score, quality_reason = preprocess.check_fingerprint_quality(preprocessed)
    
    quality_info = {
        'is_valid': is_valid,
        'score': quality_score,
        'reason': quality_reason
    }
    
    if not is_valid:
        logger.warning(f"Quality check failed: {quality_reason}")
        return {
            'query': query_image_path,
            'results': [],
            'total_matches': 0,
            'warning': f'Image quality check failed: {quality_reason}',
            'quality_check': quality_info
        }
    
    # 3. Extract features
    query_features = extract_features.extract_features(preprocessed)
    
    # 4. Search database (exclude query image from results)
    results = search_database(
        query_features['feature_vector'], 
        db_path=db_path, 
        top_k=top_k * 2,  # Get more results for filtering
        exclude_filename=exclude_filename
    )
    
    if not results:
        logger.warning("No matches found")
        return {
            'query': query_image_path,
            'results': [],
            'total_matches': 0,
            'quality_check': quality_info
        }
    
    # 5. Filter by similarity threshold
    filtered_results = [r for r in results if r['similarity'] >= config.SIMILARITY_THRESHOLD]
    
    if not filtered_results:
        logger.warning(f"No matches above similarity threshold ({config.SIMILARITY_THRESHOLD})")
        return {
            'query': query_image_path,
            'results': [],
            'total_matches': 0,
            'warning': f'No matches found with similarity >= {config.SIMILARITY_THRESHOLD}',
            'quality_check': quality_info,
            'best_similarity': results[0]['similarity'] if results else 0.0
        }
    
    # 6. Re-rank with rotation
    if use_rotation:
        filtered_results = rerank_with_rotation(query_image_path, query_features, filtered_results, db_path)
        # Re-filter after rotation
        filtered_results = [r for r in filtered_results if r['similarity'] >= config.SIMILARITY_THRESHOLD]
    
    # Return top-k
    final_results = filtered_results[:top_k]
    
    logger.info(f"Found {len(final_results)} matches above threshold")
    
    return {
        'query': query_image_path,
        'results': final_results,
        'total_matches': len(final_results),
        'quality_check': quality_info,
        'threshold': config.SIMILARITY_THRESHOLD
    }

# ============================================================================
# 4. Print Results
# ============================================================================
def print_search_results(search_result):
    """Pretty print search results"""
    if search_result is None:
        print("No results")
        return
    
    print(f"\n{'='*80}")
    print(f"Query: {search_result['query']}")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Filename':<30} {'Similarity':<12}")
    print(f"{'-'*80}")
    
    for result in search_result['results']:
        filename = result['filename'][:28]
        print(f"{result['rank']:<6} {filename:<30} {result['similarity']:<12.4f}")
    
    print(f"{'-'*80}")
    
    # Top-1 details
    if search_result['results']:
        top1 = search_result['results'][0]
        print(f"\nTop-1 Match:")
        print(f"  Filename: {top1['filename']}")
        print(f"  Similarity: {top1['similarity']:.4f}")
    
    print(f"{'='*80}\n")

# ============================================================================
# 7. Main - CLI Search
# ============================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler()
        ]
    )
    
    parser = argparse.ArgumentParser(description='Fingerprint Search')
    parser.add_argument('--query', type=str, required=True, help='Query image path')
    parser.add_argument('--top-k', type=int, default=config.TOP_K, help='Return top-k matches')
    parser.add_argument('--no-rotation', action='store_true', help='Disable rotation re-ranking')
    parser.add_argument('--db', type=str, default=config.DB_PATH, help='Database path')
    
    args = parser.parse_args()
    
    # Search
    results = search_fingerprint(
        args.query,
        top_k=args.top_k,
        use_rotation=not args.no_rotation,
        db_path=args.db
    )
    
    # Print results
    print_search_results(results)
