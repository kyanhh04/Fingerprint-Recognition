#!/usr/bin/env python3
# Test script to verify exclude_filename logic

import os

# Test cases
test_cases = [
    # (exclude_filename, db_filename, should_exclude)
    ("100__M_Left_index_finger.BMP", "100__M_Left_index_finger.BMP", True),
    ("temp_query.bmp", "100__M_Left_index_finger.BMP", False),
    ("100__M_Left_index_finger.BMP", "101__M_Left_index_finger.BMP", False),
    ("/path/to/100__M_Left_index_finger.BMP", "100__M_Left_index_finger.BMP", True),
    ("100__M_Left_index_finger.BMP", "/full/path/100__M_Left_index_finger.BMP", True),
]

print("Testing exclude_filename logic:")
print("=" * 80)

for exclude_filename, db_filename, expected in test_cases:
    if exclude_filename:
        exclude_base = os.path.basename(exclude_filename)
        filename_base = os.path.basename(db_filename)
        
        should_exclude = (exclude_base == filename_base or exclude_base == db_filename)
    else:
        should_exclude = False
    
    status = "✓" if should_exclude == expected else "✗"
    print(f"{status} exclude='{exclude_filename}' vs db='{db_filename}'")
    print(f"   Expected: {expected}, Got: {should_exclude}")
    print()

print("=" * 80)
