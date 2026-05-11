#!/usr/bin/env python3
"""
Test script to verify quality check rejects landscape images
"""

import sys
import logging
import cv2
import numpy as np
import preprocess

logging.basicConfig(level=logging.INFO)

def create_test_landscape():
    """Create a synthetic landscape-like image"""
    # Create a 512x512 image with gradient (sky to ground)
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Sky (blue gradient)
    for i in range(200):
        img[i, :] = [200 - i//2, 150 - i//3, 100]
    
    # Mountains (dark)
    for i in range(200, 350):
        img[i, :] = [50, 40, 30]
    
    # Grass (green)
    for i in range(350, 512):
        img[i, :] = [100, 180 - (i-350)//2, 50]
    
    # Add some noise for texture
    noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img

def test_image(image_path_or_array, name="Test Image"):
    """Test quality check on an image"""
    print("=" * 80)
    print(f"Testing: {name}")
    print("=" * 80)
    
    # Save if it's an array
    if isinstance(image_path_or_array, np.ndarray):
        test_path = "test_landscape_temp.bmp"
        cv2.imwrite(test_path, image_path_or_array)
        image_path = test_path
    else:
        image_path = image_path_or_array
    
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
    
    print(f"\n{'='*80}")
    print(f"RESULTS:")
    print(f"{'='*80}")
    print(f"  Valid: {'✅ YES' if is_valid else '❌ NO'}")
    print(f"  Score: {quality_score:.4f} / 1.0")
    print(f"  Reason: {reason}")
    print(f"{'='*80}\n")
    
    return is_valid, quality_score

if __name__ == "__main__":
    print("\n" + "="*80)
    print("FINGERPRINT QUALITY CHECK - LANDSCAPE TEST")
    print("="*80 + "\n")
    
    # Test 1: Synthetic landscape
    print("\n### TEST 1: Synthetic Landscape Image ###")
    landscape = create_test_landscape()
    is_valid1, score1 = test_image(landscape, "Synthetic Landscape")
    
    # Test 2: Real image if provided
    if len(sys.argv) > 1:
        print("\n### TEST 2: Provided Image ###")
        is_valid2, score2 = test_image(sys.argv[1], sys.argv[1])
    
    # Test 3: Real fingerprint for comparison
    import os
    fingerprint_path = "../Real/100__M_Left_index_finger.BMP"
    if os.path.exists(fingerprint_path):
        print("\n### TEST 3: Real Fingerprint (for comparison) ###")
        is_valid3, score3 = test_image(fingerprint_path, "Real Fingerprint")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Synthetic Landscape: {'REJECTED ✅' if not is_valid1 else 'ACCEPTED ❌'} (score: {score1:.2f})")
    if len(sys.argv) > 1:
        print(f"Provided Image: {'REJECTED' if not is_valid2 else 'ACCEPTED'} (score: {score2:.2f})")
    if os.path.exists(fingerprint_path):
        print(f"Real Fingerprint: {'ACCEPTED ✅' if is_valid3 else 'REJECTED ❌'} (score: {score3:.2f})")
    print("="*80 + "\n")
