# ============================================================================
# Batch Evaluation - Run 100 random queries and compute metrics
# ============================================================================

import os
import sqlite3
import numpy as np
import logging
import csv
import random
from pathlib import Path

import config
from search import search_fingerprint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_ground_truth_id(filename):
    """Extract ground truth ID from filename (first number before underscore)"""
    basename = Path(filename).stem
    parts = basename.split('_')
    if len(parts) > 0:
        try:
            return int(parts[0])
        except:
            return basename
    return basename

def get_random_queries(db_path, num_queries=100):
    """Get random query image paths from database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('SELECT filename FROM images ORDER BY RANDOM() LIMIT ?', (num_queries,))
    rows = cursor.fetchall()
    conn.close()
    
    # Resolve paths using current DATA_FOLDER
    queries = []
    for (filename,) in rows:
        filepath = os.path.join(config.DATA_FOLDER, filename)
        queries.append((filepath, filename))
    
    return queries

def evaluate_single_query(filepath, filename):
    """Evaluate a single query"""
    ground_truth_id = get_ground_truth_id(filename)
    
    try:
        search_result = search_fingerprint(
            filepath,
            top_k=config.TOP_K,
            db_path=config.DB_PATH,
            exclude_filename=filename  # Exclude query itself from results
        )
        
        if not search_result or 'results' not in search_result or not search_result['results']:
            return {
                'query_filename': filename,
                'ground_truth_id': ground_truth_id,
                'top_1_match': None,
                'top_1_score': 0.0,
                'top_1_accuracy': 0,
                'precision_at_5': 0.0,
                'visual_check_passed': None,
                'error': 'No results'
            }
        
        results = search_result['results']
        
        # Top-1 evaluation
        top_1_result = results[0]
        top_1_filename = top_1_result['filename']
        top_1_id = get_ground_truth_id(top_1_filename)
        top_1_match = 1 if top_1_id == ground_truth_id else 0
        top_1_score = top_1_result.get('similarity', 0.0)
        visual_check_passed = top_1_result.get('visual_check', {}).get('passed', None)
        
        # Precision@5
        top_5_matches = sum(1 for r in results[:5] 
                           if get_ground_truth_id(r['filename']) == ground_truth_id)
        precision_at_5 = top_5_matches / min(5, len(results))
        
        return {
            'query_filename': filename,
            'ground_truth_id': ground_truth_id,
            'top_1_match': top_1_filename,
            'top_1_score': top_1_score,
            'top_1_accuracy': top_1_match,
            'precision_at_5': precision_at_5,
            'visual_check_passed': visual_check_passed,
            'error': None
        }
    
    except Exception as e:
        logger.error(f"Error evaluating {filename}: {e}")
        return {
            'query_filename': filename,
            'ground_truth_id': ground_truth_id,
            'error': str(e),
            'top_1_accuracy': 0,
            'precision_at_5': 0.0
        }

def run_batch_evaluation(num_queries=100):
    """Run batch evaluation on random queries"""
    logger.info(f"Starting batch evaluation with {num_queries} random queries...")
    
    # Get random queries
    queries = get_random_queries(config.DB_PATH, num_queries)
    logger.info(f"Fetched {len(queries)} random query images from database")
    
    all_results = []
    
    for idx, (filepath, filename) in enumerate(queries):
        logger.info(f"[{idx+1}/{len(queries)}] Evaluating {filename}...")
        metrics = evaluate_single_query(filepath, filename)
        all_results.append(metrics)
        
        if metrics['error'] is None:
            logger.info(f"  Top-1 Accuracy: {metrics['top_1_accuracy']}, "
                       f"Precision@5: {metrics['precision_at_5']:.3f}, "
                       f"Top-1 Score: {metrics['top_1_score']:.4f}, "
                       f"Visual Check: {metrics['visual_check_passed']}")
    
    # Save to CSV
    output_path = os.path.join(config.OUTPUT_FOLDER, 'evaluation.csv')
    os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        fieldnames = ['query_filename', 'ground_truth_id', 'top_1_match', 'top_1_score', 
                      'top_1_accuracy', 'precision_at_5', 'visual_check_passed', 'error']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    
    logger.info(f"Results saved to {output_path}")
    
    # Compute aggregated metrics
    valid_results = [r for r in all_results if r.get('error') is None]
    
    if valid_results:
        top_1_accuracies = [r['top_1_accuracy'] for r in valid_results]
        precision_at_5_values = [r['precision_at_5'] for r in valid_results]
        visual_checks = [r['visual_check_passed'] for r in valid_results if r['visual_check_passed'] is not None]
        
        top_1_accuracy = np.mean(top_1_accuracies)
        precision_at_5_mean = np.mean(precision_at_5_values)
        precision_at_5_std = np.std(precision_at_5_values)
        visual_pass_rate = np.mean(visual_checks) if visual_checks else None
        
        logger.info("=" * 70)
        logger.info("BATCH EVALUATION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total Queries Evaluated: {len(valid_results)}")
        logger.info(f"Top-1 Accuracy: {top_1_accuracy:.4f} ({int(top_1_accuracy*len(valid_results))}/{len(valid_results)} correct)")
        logger.info(f"Precision@5 Mean: {precision_at_5_mean:.4f} ± {precision_at_5_std:.4f}")
        logger.info(f"Visual Check Pass Rate: {visual_pass_rate:.4f}" if visual_pass_rate else "Visual Check Pass Rate: N/A")
        logger.info("=" * 70)
        
        return {
            'total_queries': len(valid_results),
            'top_1_accuracy': top_1_accuracy,
            'precision_at_5_mean': precision_at_5_mean,
            'precision_at_5_std': precision_at_5_std,
            'visual_pass_rate': visual_pass_rate
        }
    else:
        logger.warning("No valid results to aggregate")
        return None

if __name__ == '__main__':
    metrics = run_batch_evaluation(num_queries=100)
