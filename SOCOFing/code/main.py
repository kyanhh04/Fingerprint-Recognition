# ============================================================================
# Main CLI - Orchestrator for Full Pipeline
# ============================================================================

import argparse
import logging
import sys
import os

import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# ============================================================================
# 1. Build Database Command
# ============================================================================
def cmd_build_database(args):
    """Build database from BMP folder"""
    logger.info("=" * 80)
    logger.info("COMMAND: Build Database")
    logger.info("=" * 80)
    
    from build_database import build_database_from_folder, get_database_stats
    
    try:
        processed, failed = build_database_from_folder(
            data_folder=args.data_folder,
            db_path=args.db
        )
        
        logger.info(f"Database built: {processed} processed, {failed} failed")
        
        # Print stats
        total_images, total_features = get_database_stats(args.db)
        logger.info(f"Database stats: {total_images} images, {total_features} features")
        
        return 0
    
    except Exception as e:
        logger.error(f"Build database failed: {e}")
        return 1

# ============================================================================
# 2. Search Command
# ============================================================================
def cmd_search(args):
    """Search for fingerprint matches"""
    logger.info("=" * 80)
    logger.info("COMMAND: Search")
    logger.info("=" * 80)
    
    from search import search_fingerprint, print_search_results
    
    try:
        if not os.path.exists(args.query):
            logger.error(f"Query image not found: {args.query}")
            return 1
        
        logger.info(f"Searching for: {args.query}")
        logger.info(f"Top-K: {args.top_k}, Rotation: {not args.no_rotation}")
        
        results = search_fingerprint(
            query_image_path=args.query,
            top_k=args.top_k,
            use_rotation=not args.no_rotation,
            db_path=args.db
        )
        
        print_search_results(results)
        
        return 0
    
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return 1

# ============================================================================
# 3. Evaluate Command
# ============================================================================
def cmd_evaluate(args):
    """Evaluate system on test set"""
    logger.info("=" * 80)
    logger.info("COMMAND: Evaluate")
    logger.info("=" * 80)
    
    from evaluate import evaluate_test_set, compute_aggregated_metrics, save_evaluation_results, print_evaluation_report
    
    try:
        if not os.path.exists(args.test_folder):
            logger.error(f"Test folder not found: {args.test_folder}")
            return 1
        
        logger.info(f"Evaluating on test set: {args.test_folder}")
        
        # Evaluate
        results = evaluate_test_set(
            test_folder=args.test_folder,
            top_k=args.top_k,
            db_path=args.db
        )
        
        # Compute aggregated metrics
        aggregated = compute_aggregated_metrics(results)
        
        # Save results
        save_evaluation_results(results, output_path=args.output)
        logger.info(f"Results saved to: {args.output}")
        
        # Print report
        print_evaluation_report(aggregated, results)
        
        return 0
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1

# ============================================================================
# 4. Preprocess Command (Standalone)
# ============================================================================
def cmd_preprocess(args):
    """Preprocess single image (for debugging)"""
    logger.info("=" * 80)
    logger.info("COMMAND: Preprocess")
    logger.info("=" * 80)
    
    from preprocess import preprocess_image, save_preprocessed_images
    
    try:
        if not os.path.exists(args.image):
            logger.error(f"Image not found: {args.image}")
            return 1
        
        logger.info(f"Preprocessing: {args.image}")
        
        preprocessed = preprocess_image(args.image, apply_enhancements=True)
        
        if preprocessed is None:
            logger.error("Preprocessing failed")
            return 1
        
        # Save stages
        output_dir = args.output_dir or "output/preprocess"
        image_id = os.path.splitext(os.path.basename(args.image))[0]
        save_preprocessed_images(preprocessed, output_dir, image_id)
        
        logger.info(f"Preprocessing complete. Saved to: {output_dir}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return 1

# ============================================================================
# 5. Extract Features Command (Standalone)
# ============================================================================
def cmd_extract(args):
    """Extract features from single image (for debugging)"""
    logger.info("=" * 80)
    logger.info("COMMAND: Extract Features")
    logger.info("=" * 80)
    
    from preprocess import preprocess_image
    from extract_features import extract_features
    
    try:
        if not os.path.exists(args.image):
            logger.error(f"Image not found: {args.image}")
            return 1
        
        logger.info(f"Extracting features: {args.image}")
        
        # Preprocess
        preprocessed = preprocess_image(args.image, apply_enhancements=True)
        if preprocessed is None:
            logger.error("Preprocessing failed")
            return 1
        
        # Extract
        features = extract_features(preprocessed)
        
        logger.info("Features extracted:")
        logger.info(f"  Minutiae: {features['minutiae']['count']} "
                   f"(endings={features['minutiae']['endings']}, "
                   f"bifurcations={features['minutiae']['bifurcations']})")
        logger.info(f"  Orientation histogram: {len(features['orientation_hist'])} bins")
        if features['lbp_hist'] is not None:
            logger.info(f"  LBP histogram: {len(features['lbp_hist'])} bins")
        logger.info(f"  Total feature vector size: {len(features['feature_vector'])}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        return 1

# ============================================================================
# 6. Web App Command
# ============================================================================
def cmd_webapp(args):
    """Launch Streamlit web app"""
    logger.info("=" * 80)
    logger.info("COMMAND: Web App")
    logger.info("=" * 80)
    
    import subprocess
    
    try:
        logger.info("Launching Streamlit app...")
        subprocess.run(["streamlit", "run", "app.py"], check=True)
        return 0
    
    except FileNotFoundError:
        logger.error("Streamlit not found. Install with: pip install streamlit")
        return 1
    except Exception as e:
        logger.error(f"Web app failed: {e}")
        return 1

# ============================================================================
# 7. Main CLI Parser
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Fingerprint Recognition Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  
  # 1. Build database
  python main.py build --data-folder data_processed_bmp/
  
  # 2. Search
  python main.py search --query path/to/query.bmp --top-k 5
  
  # 3. Evaluate
  python main.py evaluate --test-folder path/to/test/ --top-k 5
  
  # 4. Preprocess (debug)
  python main.py preprocess --image path/to/image.bmp
  
  # 5. Extract features (debug)
  python main.py extract --image path/to/image.bmp
  
  # 6. Web app
  python main.py webapp
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # ========== Build Database ==========
    build_parser = subparsers.add_parser('build', help='Build database from BMP folder')
    build_parser.add_argument('--data-folder', type=str, default=config.DATA_FOLDER,
                             help=f'BMP data folder (default: {config.DATA_FOLDER})')
    build_parser.add_argument('--db', type=str, default=config.DB_PATH,
                             help=f'Database path (default: {config.DB_PATH})')
    build_parser.set_defaults(func=cmd_build_database)
    
    # ========== Search ==========
    search_parser = subparsers.add_parser('search', help='Search for fingerprint matches')
    search_parser.add_argument('--query', type=str, required=True,
                              help='Query image path')
    search_parser.add_argument('--top-k', type=int, default=config.TOP_K,
                              help=f'Return top-k matches (default: {config.TOP_K})')
    search_parser.add_argument('--no-rotation', action='store_true',
                              help='Disable rotation re-ranking')
    search_parser.add_argument('--db', type=str, default=config.DB_PATH,
                              help=f'Database path (default: {config.DB_PATH})')
    search_parser.set_defaults(func=cmd_search)
    
    # ========== Evaluate ==========
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate on test set')
    eval_parser.add_argument('--test-folder', type=str, required=True,
                            help='Test images folder')
    eval_parser.add_argument('--top-k', type=int, default=config.TOP_K,
                            help=f'Top-k for evaluation (default: {config.TOP_K})')
    eval_parser.add_argument('--db', type=str, default=config.DB_PATH,
                            help=f'Database path (default: {config.DB_PATH})')
    eval_parser.add_argument('--output', type=str, default=config.EVALUATION_CSV,
                            help=f'Output CSV path (default: {config.EVALUATION_CSV})')
    eval_parser.set_defaults(func=cmd_evaluate)
    
    # ========== Preprocess ==========
    prep_parser = subparsers.add_parser('preprocess', help='Preprocess single image (debug)')
    prep_parser.add_argument('--image', type=str, required=True,
                            help='Image path')
    prep_parser.add_argument('--output-dir', type=str, default=None,
                            help='Output directory for preprocessed stages')
    prep_parser.set_defaults(func=cmd_preprocess)
    
    # ========== Extract ==========
    ext_parser = subparsers.add_parser('extract', help='Extract features from image (debug)')
    ext_parser.add_argument('--image', type=str, required=True,
                           help='Image path')
    ext_parser.set_defaults(func=cmd_extract)
    
    # ========== Web App ==========
    app_parser = subparsers.add_parser('webapp', help='Launch Streamlit web app')
    app_parser.set_defaults(func=cmd_webapp)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    return args.func(args)

# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    sys.exit(main())
