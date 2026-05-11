# ============================================================================
# Evaluation - Precision@5, Top-1 Accuracy
# ============================================================================

import os
import sqlite3
import numpy as np
import logging
import csv
import argparse
from pathlib import Path

import config
from search import search_fingerprint

logger = logging.getLogger(__name__)

# ============================================================================
# 1. Parse Query Result
# ============================================================================
def get_ground_truth_id(filename):
    """
    Extract ground truth ID from filename.
    Assumes format: <id>_*.bmp or similar
    """
    basename = Path(filename).stem
    parts = basename.split('_')
    
    if len(parts) > 0:
        # Try to extract ID from filename
        try:
            return int(parts[0])
        except:
            return basename
    
    return basename

# ============================================================================
# 2. Calculate Metrics for Single Query
# ============================================================================
def evaluate_single_query(query_path, ground_truth_id, search_result):
    """
    Calculate metrics for a single query.
    Returns: dict with metrics
    """
    if search_result is None or not search_result['results']:
        return {
            'query': query_path,
            'ground_truth_id': ground_truth_id,
            'top_1_match': None,
            'top_1_match_id': None,
            'top_1_accuracy': 0,
            'top_1_score': 0,
            'precision_at_5': 0,
            'error': 'No results'
        }
    
    results = search_result['results']
    
    # Top-1 evaluation
    top_1_result = results[0]
    top_1_filename = top_1_result['filename']
    top_1_id = get_ground_truth_id(top_1_filename)
    top_1_match = (top_1_id == ground_truth_id)
    
    # Precision@5
    top_5_matches = sum(1 for r in results[:5] 
                       if get_ground_truth_id(r['filename']) == ground_truth_id)
    precision_at_5 = top_5_matches / min(5, len(results)) if results else 0
    
    return {
        'query': query_path,
        'ground_truth_id': ground_truth_id,
        'top_1_match': top_1_filename,
        'top_1_match_id': top_1_id,
        'top_1_accuracy': 1 if top_1_match else 0,
        'top_1_score': top_1_result['similarity'],
        'precision_at_5': precision_at_5
    }

# ============================================================================
# 3. Evaluate on Test Set
# ============================================================================
def evaluate_test_set(test_folder, top_k=config.TOP_K, db_path=config.DB_PATH):
    """
    Evaluate on all test images in a folder.
    Returns: list of evaluation results
    """
    # Get test images
    test_files = [f for f in os.listdir(test_folder) 
                  if f.lower().endswith(('.bmp', '.BMP'))]
    
    logger.info(f"Evaluating {len(test_files)} test images")
    
    results = []
    
    for idx, filename in enumerate(test_files):
        query_path = os.path.join(test_folder, filename)
        ground_truth_id = get_ground_truth_id(filename)
        
        try:
            # Search
            search_result = search_fingerprint(
                query_path,
                top_k=top_k,
                db_path=db_path
            )
            
            # Evaluate
            metrics = evaluate_single_query(query_path, ground_truth_id, search_result)
            results.append(metrics)
            
            logger.info(f"[{idx+1}/{len(test_files)}] Top-1: {metrics['top_1_accuracy']}, "
                       f"P@5: {metrics['precision_at_5']:.2f}")
        
        except Exception as e:
            logger.error(f"Error evaluating {filename}: {e}")
            results.append({
                'query': query_path,
                'ground_truth_id': ground_truth_id,
                'error': str(e)
            })
    
    return results

# ============================================================================
# 4. Compute Aggregated Metrics
# ============================================================================
def compute_aggregated_metrics(evaluation_results):
    """Compute overall metrics from evaluation results"""
    valid_results = [r for r in evaluation_results if 'error' not in r or r['error'] is None]
    
    if not valid_results:
        logger.warning("No valid results for aggregation")
        return {}
    
    top_1_accuracies = [r['top_1_accuracy'] for r in valid_results]
    precision_at_5_values = [r['precision_at_5'] for r in valid_results]
    
    metrics = {
        'total_queries': len(valid_results),
        'top_1_accuracy': np.mean(top_1_accuracies),
        'precision_at_5_mean': np.mean(precision_at_5_values),
        'precision_at_5_std': np.std(precision_at_5_values)
    }
    
    return metrics

# ============================================================================
# 5. Save Results to CSV
# ============================================================================
def save_evaluation_results(evaluation_results, output_path=config.EVALUATION_CSV):
    """Save evaluation results to CSV"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Filter out error entries for CSV
    valid_results = [r for r in evaluation_results if 'error' not in r or r['error'] is None]
    
    if not valid_results:
        logger.warning("No valid results to save")
        return
    
    fieldnames = [
        'query', 'ground_truth_id', 'top_1_match', 'top_1_match_id',
        'top_1_accuracy', 'top_1_score',
        'precision_at_5'
    ]
    
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in valid_results:
                writer.writerow({k: result.get(k, '') for k in fieldnames})
        
        logger.info(f"Results saved to: {output_path}")
    
    except Exception as e:
        logger.error(f"Error saving results: {e}")

# ============================================================================
# 6. Print Evaluation Report
# ============================================================================
def print_evaluation_report(aggregated_metrics, evaluation_results):
    """Print evaluation report"""
    print(f"\n{'='*80}")
    print(f"EVALUATION REPORT")
    print(f"{'='*80}")
    
    print(f"\nAggregate Metrics:")
    print(f"  Total Queries: {aggregated_metrics.get('total_queries', 0)}")
    print(f"  Top-1 Accuracy: {aggregated_metrics.get('top_1_accuracy', 0):.4f}")
    print(f"  Precision@5 (mean): {aggregated_metrics.get('precision_at_5_mean', 0):.4f} "
          f"(±{aggregated_metrics.get('precision_at_5_std', 0):.4f})")
    
    print(f"\nDetailed Results:")
    print(f"{'Query':<40} {'GT_ID':<10} {'Top-1':<10} {'P@5':<8}")
    print(f"{'-'*68}")
    
    valid_results = [r for r in evaluation_results if 'error' not in r or r['error'] is None]
    for result in valid_results[:10]:  # Show first 10
        query_name = Path(result['query']).name[:35]
        gt_id = str(result['ground_truth_id'])[:8]
        top1_acc = str(result['top_1_accuracy'])
        p_at_5 = f"{result['precision_at_5']:.2f}"
        
        print(f"{query_name:<40} {gt_id:<10} {top1_acc:<10} {p_at_5:<8}")
    
    if len(valid_results) > 10:
        print(f"... and {len(valid_results) - 10} more")
    
    print(f"{'='*68}\n")

# ============================================================================
# 7. Main - CLI Evaluation
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
    
    parser = argparse.ArgumentParser(description='Evaluate Fingerprint Recognition System')
    parser.add_argument('--test-folder', type=str, required=True, help='Test images folder')
    parser.add_argument('--top-k', type=int, default=config.TOP_K, help='Top-k for evaluation')
    parser.add_argument('--db', type=str, default=config.DB_PATH, help='Database path')
    parser.add_argument('--output', type=str, default=config.EVALUATION_CSV, help='Output CSV path')
    
    args = parser.parse_args()
    
    # Evaluate
    logger.info("Starting evaluation...")
    results = evaluate_test_set(args.test_folder, top_k=args.top_k, db_path=args.db)
    
    # Compute aggregated metrics
    aggregated = compute_aggregated_metrics(results)
    
    # Save results
    save_evaluation_results(results, output_path=args.output)
    
    # Print report
    print_evaluation_report(aggregated, results)
    
    logger.info("Evaluation completed")
