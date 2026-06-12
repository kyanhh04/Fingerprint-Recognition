# ============================================================================
# Build Database - Store Fingerprints & Features in SQLite
# ============================================================================

import os
import sqlite3
import numpy as np
import logging
import hashlib
from pathlib import Path
import cv2

import config
import preprocess
import extract_features

logger = logging.getLogger(__name__)

EXTRACTOR_VERSION = "minutiae-orientation-v1"


def parse_fingerprint_filename(filename):
    """
    Parse SOCOFing filenames such as:
    100__M_Left_index_finger.BMP
    """
    stem = Path(filename).stem
    parts = stem.split("__", 1)
    subject_id = parts[0] if parts else None
    metadata = {
        "subject_id": subject_id,
        "gender": None,
        "hand": None,
        "finger": None,
    }

    if len(parts) < 2:
        return metadata

    tokens = parts[1].split("_")
    if tokens:
        metadata["gender"] = tokens[0] or None
    if len(tokens) > 1:
        metadata["hand"] = tokens[1] or None
    if len(tokens) > 2:
        metadata["finger"] = tokens[2] or None

    return metadata


def compute_file_hash(file_path, chunk_size=1024 * 1024):
    """Compute SHA-256 hash for duplicate-file detection."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_image_dimensions(file_path):
    """Return image width and height without storing the raw image in DB."""
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None
    height, width = image.shape[:2]
    return width, height


def add_column_if_missing(cursor, table_name, column_name, column_definition):
    cursor.execute(f"PRAGMA table_info({table_name})")
    existing_columns = {row[1] for row in cursor.fetchall()}
    if column_name not in existing_columns:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")


def table_exists(cursor, table_name):
    cursor.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    )
    return cursor.fetchone() is not None

# ============================================================================
# 1. Initialize Database
# ============================================================================
def init_database(db_path=config.DB_PATH, data_folder=config.DATA_FOLDER):
    """Create database schema"""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()
    
    # Images table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE NOT NULL,
            filepath TEXT NOT NULL,
            file_hash TEXT,
            file_size INTEGER,
            image_width INTEGER,
            image_height INTEGER,
            image_size INTEGER,
            feature_size INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Add columns when migrating an existing database created by the old schema.
    add_column_if_missing(cursor, "images", "file_hash", "TEXT")
    add_column_if_missing(cursor, "images", "file_size", "INTEGER")
    add_column_if_missing(cursor, "images", "image_width", "INTEGER")
    add_column_if_missing(cursor, "images", "image_height", "INTEGER")
    
    # Labels parsed from the dataset filename.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fingerprint_labels (
            label_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL UNIQUE,
            subject_id TEXT,
            gender TEXT,
            hand TEXT,
            finger TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE
        )
    ''')

    # Extracted fingerprint features.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fingerprint_features (
            feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL UNIQUE,
            minutiae_count INTEGER,
            bifurcations INTEGER,
            endings INTEGER,
            minutiae_vector BLOB,
            feature_vector BLOB NOT NULL,
            feature_dim INTEGER NOT NULL,
            orientation_hist BLOB,
            orientation_dim INTEGER,
            lbp_hist BLOB,
            lbp_dim INTEGER,
            extractor_version TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE
        )
    ''')

    # Optional search audit table.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS match_results (
            match_id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_image_id INTEGER,
            candidate_image_id INTEGER NOT NULL,
            similarity_score REAL NOT NULL,
            metric TEXT NOT NULL,
            is_match INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (query_image_id) REFERENCES images(image_id) ON DELETE SET NULL,
            FOREIGN KEY (candidate_image_id) REFERENCES images(image_id) ON DELETE CASCADE
        )
    ''')

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_file_hash ON images(file_hash)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_labels_subject ON fingerprint_labels(subject_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_labels_hand_finger ON fingerprint_labels(hand, finger)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_match_query ON match_results(query_image_id)")

    conn.commit()

    migrate_legacy_database(conn, data_folder=data_folder)
    
    conn.commit()
    logger.info(f"Database initialized: {db_path}")
    return conn


def migrate_legacy_database(conn, data_folder=config.DATA_FOLDER):
    """
    Populate the redesigned tables from a database built with the old schema.
    The old `features` table is left in place as legacy data; new code reads
    from `fingerprint_features`.
    """
    cursor = conn.cursor()

    # Populate labels and repair stale filepath/metadata using the current data folder.
    cursor.execute("SELECT image_id, filename, filepath FROM images")
    image_rows = cursor.fetchall()

    for image_id, filename, _ in image_rows:
        current_path = os.path.abspath(os.path.join(data_folder, filename))
        file_hash = None
        file_size = None
        image_width = None
        image_height = None

        if os.path.exists(current_path):
            file_hash = compute_file_hash(current_path)
            file_size = os.path.getsize(current_path)
            image_width, image_height = get_image_dimensions(current_path)
        else:
            current_path = os.path.abspath(current_path)

        cursor.execute('''
            UPDATE images
            SET filepath = ?,
                file_hash = COALESCE(?, file_hash),
                file_size = COALESCE(?, file_size),
                image_width = COALESCE(?, image_width),
                image_height = COALESCE(?, image_height)
            WHERE image_id = ?
        ''', (current_path, file_hash, file_size, image_width, image_height, image_id))

        label = parse_fingerprint_filename(filename)
        cursor.execute('''
            INSERT INTO fingerprint_labels (image_id, subject_id, gender, hand, finger)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(image_id) DO UPDATE SET
                subject_id = excluded.subject_id,
                gender = excluded.gender,
                hand = excluded.hand,
                finger = excluded.finger
        ''', (
            image_id,
            label["subject_id"],
            label["gender"],
            label["hand"],
            label["finger"],
        ))

    if table_exists(cursor, "features"):
        cursor.execute('''
            SELECT image_id, minutiae_count, bifurcations, endings,
                   feature_vector, orientation_hist, lbp_hist
            FROM features
        ''')

        for row in cursor.fetchall():
            image_id, minutiae_count, bifurcations, endings, feature_blob, orientation_blob, lbp_blob = row
            if feature_blob is None:
                continue

            feature_dim = len(feature_blob) // np.dtype(np.float32).itemsize
            orientation_dim = len(orientation_blob) // np.dtype(np.float32).itemsize if orientation_blob else None
            lbp_dim = len(lbp_blob) // np.dtype(np.float32).itemsize if lbp_blob else None
            minutiae_blob = None

            if feature_dim >= 261:
                feature_vector = np.frombuffer(feature_blob, dtype=np.float32)
                minutiae_blob = feature_vector[:261].astype(np.float32).tobytes()

            cursor.execute('''
                INSERT INTO fingerprint_features (
                    image_id, minutiae_count, bifurcations, endings,
                    minutiae_vector, feature_vector, feature_dim,
                    orientation_hist, orientation_dim, lbp_hist, lbp_dim,
                    extractor_version
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(image_id) DO UPDATE SET
                    minutiae_count = excluded.minutiae_count,
                    bifurcations = excluded.bifurcations,
                    endings = excluded.endings,
                    minutiae_vector = excluded.minutiae_vector,
                    feature_vector = excluded.feature_vector,
                    feature_dim = excluded.feature_dim,
                    orientation_hist = excluded.orientation_hist,
                    orientation_dim = excluded.orientation_dim,
                    lbp_hist = excluded.lbp_hist,
                    lbp_dim = excluded.lbp_dim,
                    extractor_version = excluded.extractor_version
            ''', (
                image_id,
                minutiae_count,
                bifurcations,
                endings,
                minutiae_blob,
                feature_blob,
                feature_dim,
                orientation_blob,
                orientation_dim,
                lbp_blob,
                lbp_dim,
                EXTRACTOR_VERSION,
            ))

    conn.commit()

# ============================================================================
# 2. Insert Image & Features
# ============================================================================
def insert_image_and_features(conn, image_path, filename, features_dict):
    """Insert image metadata and features into database"""
    cursor = conn.cursor()
    
    try:
        absolute_path = os.path.abspath(image_path)
        file_hash = compute_file_hash(absolute_path)
        file_size = os.path.getsize(absolute_path)
        image_width, image_height = get_image_dimensions(absolute_path)
        feature_dim = len(features_dict['feature_vector']) if features_dict['feature_vector'] is not None else 0
        orientation_dim = len(features_dict['orientation_hist']) if features_dict['orientation_hist'] is not None else None
        lbp_dim = len(features_dict['lbp_hist']) if features_dict['lbp_hist'] is not None else None

        # Insert image metadata
        cursor.execute('''
            INSERT INTO images (
                filename, filepath, file_hash, file_size, image_width, image_height,
                image_size, feature_size
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(filename) DO UPDATE SET
                filepath = excluded.filepath,
                file_hash = excluded.file_hash,
                file_size = excluded.file_size,
                image_width = excluded.image_width,
                image_height = excluded.image_height,
                image_size = excluded.image_size,
                feature_size = excluded.feature_size
        ''', (
            filename,
            absolute_path,
            file_hash,
            file_size,
            image_width,
            image_height,
            config.IMAGE_SIZE,
            feature_dim
        ))

        cursor.execute("SELECT image_id FROM images WHERE filename = ?", (filename,))
        image_id = cursor.fetchone()[0]

        label = parse_fingerprint_filename(filename)
        cursor.execute('''
            INSERT INTO fingerprint_labels (image_id, subject_id, gender, hand, finger)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(image_id) DO UPDATE SET
                subject_id = excluded.subject_id,
                gender = excluded.gender,
                hand = excluded.hand,
                finger = excluded.finger
        ''', (
            image_id,
            label["subject_id"],
            label["gender"],
            label["hand"],
            label["finger"],
        ))
        
        # Serialize feature vectors
        feature_vector_blob = features_dict['feature_vector'].astype(np.float32).tobytes()
        minutiae_vector_blob = features_dict['minutiae_vector'].astype(np.float32).tobytes() if features_dict['minutiae_vector'] is not None else None
        orientation_hist_blob = features_dict['orientation_hist'].astype(np.float32).tobytes()
        lbp_hist_blob = features_dict['lbp_hist'].astype(np.float32).tobytes() if features_dict['lbp_hist'] is not None else None
        
        # Insert features
        minutiae_data = features_dict['minutiae']
        cursor.execute('''
            INSERT INTO fingerprint_features (
                image_id, minutiae_count, bifurcations, endings,
                minutiae_vector, feature_vector, feature_dim,
                orientation_hist, orientation_dim, lbp_hist, lbp_dim,
                extractor_version
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(image_id) DO UPDATE SET
                minutiae_count = excluded.minutiae_count,
                bifurcations = excluded.bifurcations,
                endings = excluded.endings,
                minutiae_vector = excluded.minutiae_vector,
                feature_vector = excluded.feature_vector,
                feature_dim = excluded.feature_dim,
                orientation_hist = excluded.orientation_hist,
                orientation_dim = excluded.orientation_dim,
                lbp_hist = excluded.lbp_hist,
                lbp_dim = excluded.lbp_dim,
                extractor_version = excluded.extractor_version
        ''', (
            image_id,
            minutiae_data['count'],
            minutiae_data['bifurcations'],
            minutiae_data['endings'],
            minutiae_vector_blob,
            feature_vector_blob,
            feature_dim,
            orientation_hist_blob,
            orientation_dim,
            lbp_hist_blob,
            lbp_dim,
            EXTRACTOR_VERSION
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
    cursor.execute('SELECT feature_vector FROM fingerprint_features WHERE image_id = ?', (image_id,))
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
    conn = init_database(db_path, data_folder=data_folder)
    
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
    
    cursor.execute('SELECT COUNT(*) FROM fingerprint_features')
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
        SELECT i.image_id, i.filename, l.subject_id, l.gender, l.hand, l.finger,
               f.minutiae_count, f.bifurcations, f.endings, f.feature_dim
        FROM images i
        LEFT JOIN fingerprint_labels l ON i.image_id = l.image_id
        LEFT JOIN fingerprint_features f ON i.image_id = f.image_id
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
