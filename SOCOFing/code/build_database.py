# ============================================================================
# Build Database - Store Fingerprints & Features in SQLite
# ============================================================================

import os
import sqlite3
import numpy as np
import logging
from datetime import datetime

import config
import preprocess
import extract_features

logger = logging.getLogger(__name__)

# ============================================================================
# 1. Initialize Database
# ============================================================================
def init_database(db_path=config.DB_PATH):
    """Create database schema"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Images table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE NOT NULL,
            filepath TEXT NOT NULL,
            image_size INTEGER,
            feature_size INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Features table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS features (
            feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL UNIQUE,
            minutiae_count INTEGER,
            bifurcations INTEGER,
            endings INTEGER,
            feature_vector BLOB NOT NULL,
            orientation_hist BLOB,
            lbp_hist BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (image_id) REFERENCES images(image_id)
        )
    ''')
    
    conn.commit()
    logger.info(f"Database initialized: {db_path}")
    return conn

# ============================================================================
# 2. Insert Image & Features
# ============================================================================
def insert_image_and_features(conn, image_path, filename, features_dict):
    """Insert image metadata and features into database"""
    cursor = conn.cursor()
    
    try:
        # Insert image metadata
        cursor.execute('''
            INSERT INTO images (filename, filepath, image_size, feature_size)
            VALUES (?, ?, ?, ?)
        ''', (
            filename,
            image_path,
            config.IMAGE_SIZE,
            len(features_dict['feature_vector']) if features_dict['feature_vector'] is not None else 0
        ))
        
        image_id = cursor.lastrowid
        
        # Serialize feature vectors
        feature_vector_blob = features_dict['feature_vector'].astype(np.float32).tobytes()
        orientation_hist_blob = features_dict['orientation_hist'].astype(np.float32).tobytes()
        lbp_hist_blob = features_dict['lbp_hist'].astype(np.float32).tobytes() if features_dict['lbp_hist'] is not None else None
        
        # Insert features
        minutiae_data = features_dict['minutiae']
        cursor.execute('''
            INSERT INTO features (image_id, minutiae_count, bifurcations, endings, 
                                 feature_vector, orientation_hist, lbp_hist)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            image_id,
            minutiae_data['count'],
            minutiae_data['bifurcations'],
            minutiae_data['endings'],
            feature_vector_blob,
            orientation_hist_blob,
            lbp_hist_blob
        ))
        
        conn.commit()
        logger.debug(f"Inserted: {filename} (ID={image_id})")
        return image_id
        
    except sqlite3.IntegrityError as e:
        logger.warning(f"Image already exists: {filename}")
        return None
    except Exception as e:
        logger.error(f"Error inserting {filename}: {e}")
        conn.rollback()
        return None

# ============================================================================
# 3. Load Feature Vector from Database
# ============================================================================
def load_feature_vector(conn, image_id):
    """Load feature vector from database"""
    cursor = conn.cursor()
    cursor.execute('SELECT feature_vector FROM features WHERE image_id = ?', (image_id,))
    row = cursor.fetchone()
    
    if row is None:
        return None
    
    feature_vector = np.frombuffer(row[0], dtype=np.float32)
    return feature_vector

# ============================================================================
# 4. Build Database from BMP Folder
# ============================================================================
def build_database_from_folder(data_folder=config.DATA_FOLDER, db_path=config.DB_PATH):
    """
    Scan folder, preprocess images, extract features, store in database.
    """
    # Initialize database
    conn = init_database(db_path)
    
    # Get list of BMP files
    bmp_files = [f for f in os.listdir(data_folder) if f.lower().endswith(('.bmp', '.BMP'))]
    logger.info(f"Found {len(bmp_files)} BMP files")
    
    processed = 0
    failed = 0
    
    for idx, filename in enumerate(bmp_files):
        image_path = os.path.join(data_folder, filename)
        
        try:
            # Preprocess
            preprocessed = preprocess.preprocess_image(image_path, apply_enhancements=True)
            if preprocessed is None:
                failed += 1
                continue
            
            # Extract features
            features_dict = extract_features.extract_features(preprocessed)
            
            # Insert into database
            image_id = insert_image_and_features(conn, image_path, filename, features_dict)
            if image_id is not None:
                processed += 1
                logger.info(f"[{idx+1}/{len(bmp_files)}] Processed: {filename} (ID={image_id})")
                
                # Checkpoint every N images
                if processed % config.DB_CHECKPOINT_SIZE == 0:
                    conn.commit()
                    logger.info(f"Checkpoint: {processed} images processed")
        
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            failed += 1
            continue
    
    conn.commit()
    conn.close()
    
    logger.info(f"Database building completed: {processed} processed, {failed} failed")
    return processed, failed

# ============================================================================
# 5. Get Database Statistics
# ============================================================================
def get_database_stats(db_path=config.DB_PATH):
    """Print database statistics"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM images')
    total_images = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM features')
    total_features = cursor.fetchone()[0]
    
    conn.close()
    
    logger.info(f"Database stats: images={total_images}, features={total_features}")
    return total_images, total_features

# ============================================================================
# 6. Export Database as CSV (for backup)
# ============================================================================
def export_database_csv(db_path=config.DB_PATH, output_path="database_export.csv"):
    """Export database metadata to CSV"""
    import pandas as pd
    
    conn = sqlite3.connect(db_path)
    query = '''
        SELECT i.image_id, i.filename, f.minutiae_count, f.bifurcations, f.endings
        FROM images i
        LEFT JOIN features f ON i.image_id = f.image_id
    '''
    df = pd.read_sql_query(query, conn)
    df.to_csv(output_path, index=False)
    conn.close()
    
    logger.info(f"Database exported to: {output_path}")

# ============================================================================
# 7. Main - Build Database
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
    
    logger.info("Starting database build...")
    processed, failed = build_database_from_folder()
    logger.info(f"Completed: {processed} processed, {failed} failed")
    
    # Print stats
    get_database_stats()
