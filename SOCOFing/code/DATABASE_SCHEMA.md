# Database Schema

Database file: `fingerprints.db`

The database is organized into separate groups of data:

```text
images
-> raw file metadata

fingerprint_labels
-> label parsed from SOCOFing filename

fingerprint_features
-> extracted fingerprint features used for matching

match_results
-> optional search/audit results
```

## `images`

Stores file-level information.

| Column | Meaning |
|---|---|
| `image_id` | Primary key |
| `filename` | Image filename |
| `filepath` | Current local path |
| `file_hash` | SHA-256 hash for exact duplicate detection |
| `file_size` | File size in bytes |
| `image_width` | Raw image width |
| `image_height` | Raw image height |
| `image_size` | Configured processing size |
| `feature_size` | Final feature vector dimension |
| `created_at` | Insert timestamp |

## `fingerprint_labels`

Stores dataset labels parsed from filenames such as `100__M_Left_index_finger.BMP`.

| Column | Meaning |
|---|---|
| `label_id` | Primary key |
| `image_id` | Foreign key to `images` |
| `subject_id` | Person ID |
| `gender` | `M` or `F` |
| `hand` | `Left` or `Right` |
| `finger` | Finger type, for example `index`, `thumb` |

## `fingerprint_features`

Stores extracted biometric features.

| Column | Meaning |
|---|---|
| `feature_id` | Primary key |
| `image_id` | Foreign key to `images` |
| `minutiae_count` | Total detected minutiae |
| `bifurcations` | Ridge branching points |
| `endings` | Ridge ending points |
| `minutiae_vector` | Minutiae vector stored as `float32` BLOB |
| `feature_vector` | Final matching vector stored as `float32` BLOB |
| `feature_dim` | Final vector dimension |
| `orientation_hist` | Orientation histogram stored as `float32` BLOB |
| `orientation_dim` | Orientation vector dimension |
| `lbp_hist` | Optional LBP histogram BLOB |
| `lbp_dim` | Optional LBP vector dimension |
| `extractor_version` | Feature extraction version |

Current vector dimensions:

```text
minutiae_vector: 261
orientation_hist: 256
feature_vector: 517
lbp_hist: NULL because LBP_ENABLED = False
```

## `match_results`

Optional table for storing search results.

| Column | Meaning |
|---|---|
| `match_id` | Primary key |
| `query_image_id` | Query image ID, nullable for uploaded images |
| `candidate_image_id` | Candidate image from database |
| `similarity_score` | Matching score |
| `metric` | Similarity metric |
| `is_match` | Manual or evaluated match flag |
| `created_at` | Search timestamp |

## Example Query

```sql
SELECT
    i.image_id,
    i.filename,
    l.subject_id,
    l.gender,
    l.hand,
    l.finger,
    f.minutiae_count,
    f.bifurcations,
    f.endings,
    f.feature_dim,
    f.orientation_dim,
    length(f.feature_vector) AS feature_vector_bytes,
    length(f.orientation_hist) AS orientation_hist_bytes
FROM images i
JOIN fingerprint_labels l ON i.image_id = l.image_id
JOIN fingerprint_features f ON i.image_id = f.image_id
LIMIT 5;
```

New code reads extracted feature data from `fingerprint_features`.
