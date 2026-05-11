#!/usr/bin/env python3
"""
Test script to verify quality check functionality
"""

import sys
import logging
import preprocess
import config

logging.basicConfig(level=logging.INFO)

def test_quality_check(image_path):
    """Test quality check on an image"""
    print("=" * 80)
    print(f"Testing Quality Check: {image_path}")
    print("=" * 80)
    
    # Preprocess
    print("\n1. Preprocessing...")
    preprocessed = preprocess.preprocess_image(image_path, apply_enhancements=True)
    
    if preprocessed is None:
        print("❌ Preprocessing failed!")
        return
    
    print("✓ Preprocessing successful")
    
    # Quality check
    print("\n2. Quality Check...")
    is_valid, quality_score, reason = preprocess.check_fingerprint_quality(preprocessed)
    
    print(f"\nResults:")
    print(f"  Valid: {is_valid}")
    print(f"  Score: {quality_score:.4f}")
    print(f"  Reason: {reason}")
    
    if is_valid:
        print("\n✅ Image passed quality check!")
    else:
        print("\n❌ Image failed quality check!")
    
    print("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_quality.py <image_path>")
        print("\nExample:")
        print("  python test_quality.py ../Real/100__M_Left_index_finger.BMP")
        sys.exit(1)
    
    test_quality_check(sys.argv[1])
