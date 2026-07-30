#!/usr/bin/env python3
"""Temp script to read specific sections of unified_screener_v9_final.py"""

with open("unified_screener_v9_final.py", "r") as f:
    lines = f.readlines()

# Print specific line ranges
ranges = [
    (995, 1010),   # Core score calculation
    (1720, 1740),  # Swing ranking
    (1200, 1240),  # Market regime breakout
    (1580, 1656),  # Confidence scoring
    (613, 670),    # VCP detection
    (506, 613),    # Existing holdings
    (1061, 1123),  # Bulk deal filtering
    (1298, 1328),  # Macro calendar
    (290, 300),    # Fallback universe
]

for start, end in ranges:
    print(f"\n{'='*60}")
    print(f"Lines {start}-{end}")
    print('='*60)
    for i in range(start-1, min(end, len(lines))):
        print(f"{i+1:4d} | {lines[i].rstrip()}")
