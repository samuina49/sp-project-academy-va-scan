"""
Parse Big-Vul Dataset to our format
Quick script to extract Python/JavaScript samples from Big-Vul CSV

Usage:
    python parse_bigvul.py --input function.csv --output bigvul_parsed.json --max_samples 5000
"""

import pandas as pd
import json
import argparse
from pathlib import Path
from typing import List, Dict
import re

# Map CWE to OWASP (same as data_prep.py)
OWASP_MAPPING = {
    "A01": [22, 23, 35, 59, 200, 201, 219, 264, 275, 276, 284, 285, 352, 359, 377, 402, 425, 441, 497, 538, 540, 548, 552, 566, 601, 639, 651, 668, 706, 862, 863, 913, 922, 1275],
    "A03": [20, 74, 75, 77, 78, 79, 80, 83, 87, 88, 89, 90, 91, 94, 95, 113, 116, 138, 184, 470, 564, 610, 643, 652, 917, 943],
    "A04": [73, 183, 209, 213, 235, 256, 257, 269, 280, 311, 312, 313, 316, 419, 430, 434, 453, 472, 501, 522, 525, 539, 579, 598, 602, 642, 656, 657, 799, 807, 840, 841, 1021, 1173],
    "A05": [2, 11, 13, 15, 16, 260, 315, 520, 526, 537, 541, 547, 611, 614, 756, 776, 942, 1004, 1032, 1174]
}

def get_owasp(cwe_id):
    """Map CWE to OWASP category"""
    for owasp, cwes in OWASP_MAPPING.items():
        if cwe_id in cwes:
            return owasp
    return "A05"  # Default

def detect_language(code):
    """Simple heuristic to detect language"""
    if 'def ' in code or 'import ' in code or 'print(' in code:
        return 'python'
    elif 'function ' in code or 'const ' in code or 'var ' in code:
        return 'javascript'
    return 'unknown'

def parse_bigvul(csv_path: str, max_samples: int = 5000) -> List[Dict]:
    """Parse Big-Vul CSV and extract samples"""
    print(f"Loading Big-Vul from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Total rows: {len(df)}")
    
    # Filter for quality
    df = df[df['target'].notna()]
    df = df[df['code'].notna()]
    df = df[df['code'].str.len() > 50]
    df = df[df['code'].str.len() < 5000]
    
    print(f"After filtering: {len(df)} rows")
    
    samples = []
    
    for idx, row in df.iterrows():
        code = str(row['code'])
        
        # Detect language
        language = detect_language(code)
        if language == 'unknown':
            continue
        
        # Get CWE
        cwe_id = int(row.get('cwe_id', 0)) if pd.notna(row.get('cwe_id')) else 0
        owasp = get_owasp(cwe_id)
        
        sample = {
            'code': code,
            'language': language,
            'cwe_id': cwe_id,
            'owasp_category': owasp,
            'is_vulnerable': bool(row['target']),
            'source': 'big-vul',
            'project': row.get('project', 'unknown')
        }
        
        samples.append(sample)
        
        if len(samples) >= max_samples:
            break
    
    print(f"✓ Extracted {len(samples)} samples")
    
    # Distribution
    vulnerable = sum(1 for s in samples if s['is_vulnerable'])
    safe = len(samples) - vulnerable
    print(f"Vulnerable: {vulnerable}, Safe: {safe}")
    
    python_count = sum(1 for s in samples if s['language'] == 'python')
    js_count = sum(1 for s in samples if s['language'] == 'javascript')
    print(f"Python: {python_count}, JavaScript: {js_count}")
    
    return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='function.csv', help='Path to Big-Vul CSV')
    parser.add_argument('--output', default='backend/ml/data/bigvul_parsed.json', help='Output JSON path')
    parser.add_argument('--max_samples', type=int, default=5000, help='Maximum samples to extract')
    
    args = parser.parse_args()
    
    # Parse
    samples = parse_bigvul(args.input, args.max_samples)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'samples': samples,
        'metadata': {
            'source': 'big-vul',
            'total_samples': len(samples),
            'vulnerable': sum(1 for s in samples if s['is_vulnerable']),
            'safe': sum(1 for s in samples if not s['is_vulnerable'])
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Saved to {output_path}")
    print("\nNext steps:")
    print("1. Run data_prep.py to merge with other datasets")
    print("2. Start training with ml/train.py")

if __name__ == "__main__":
    main()
