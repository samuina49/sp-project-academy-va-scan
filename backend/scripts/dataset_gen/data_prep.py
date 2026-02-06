"""
Production-Ready Data Pipeline for Vulnerability Detection Dataset
Handles SARD/Juliet Test Suite parsing, code normalization, and TypeScript transpilation

Author: AI Vulnerability Scanner Project
Usage: python data_prep.py --input_dir ./sard_dataset --output_dir ./processed_data
"""

import os
import re
import json
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter
import argparse

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

# OWASP to CWE mapping
OWASP_CWE_MAPPING = {
    "A01": [  # Broken Access Control
        22, 23, 35, 59, 200, 201, 219, 264, 275, 276, 284, 285, 352, 359,
        377, 402, 425, 441, 497, 538, 540, 548, 552, 566, 601, 639, 651,
        668, 706, 862, 863, 913, 922, 1275
    ],
    "A03": [  # Injection
        20, 74, 75, 77, 78, 79, 80, 83, 87, 88, 89, 90, 91, 94, 95, 113,
        116, 138, 184, 470, 564, 610, 643, 652, 917, 943
    ],
    "A04": [  # Insecure Design
        73, 183, 209, 213, 235, 256, 257, 269, 280, 311, 312, 313, 316,
        419, 430, 434, 453, 472, 501, 522, 525, 539, 579, 598, 602, 642,
        656, 657, 799, 807, 840, 841, 1021, 1173
    ],
    "A05": [  # Security Misconfiguration
        2, 11, 13, 15, 16, 260, 315, 520, 526, 537, 541, 547, 611, 614,
        756, 776, 942, 1004, 1032, 1174
    ]
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_owasp_category(cwe_id: int) -> Optional[str]:
    """Map CWE ID to OWASP category"""
    for owasp, cwe_list in OWASP_CWE_MAPPING.items():
        if cwe_id in cwe_list:
            return owasp
    return None


def detect_language_from_file(file_path: Path) -> Optional[str]:
    """Detect language from file extension"""
    ext = file_path.suffix.lower()
    if ext == '.py':
        return 'python'
    elif ext == '.js':
        return 'javascript'
    elif ext == '.ts':
        return 'typescript'
    return None


def remove_comments_and_blanks(code: str, language: str) -> str:
    """Remove comments and blank lines from code"""
    lines = []
    
    if language in ['python']:
        for line in code.split('\n'):
            # Remove single-line comments
            if '#' in line:
                line = line[:line.index('#')]
            # Skip blank lines
            if line.strip():
                lines.append(line)
    
    elif language in ['javascript', 'typescript']:
        code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)  # Single-line
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # Multi-line
        lines = [line for line in code.split('\n') if line.strip()]
    
    return '\n'.join(lines)


def normalize_code(code: str) -> str:
    """
    Normalize code to prevent overfitting on variable/function names
    Replaces identifiers with generic names like var_1, func_1
    """
    # Track unique identifiers
    var_counter = 1
    func_counter = 1
    identifier_map = {}
    
    # Regex patterns for identifiers (simplified)
    var_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
    
    # Reserved keywords to skip
    reserved = {
        'if', 'else', 'for', 'while', 'def', 'class', 'import', 'from',
        'return', 'print', 'True', 'False', 'None', 'and', 'or', 'not',
        'function', 'var', 'let', 'const', 'async', 'await', 'try', 'catch'
    }
    
    def replace_identifier(match):
        nonlocal var_counter, func_counter
        identifier = match.group(1)
        
        # Skip reserved words and common functions
        if identifier in reserved or identifier.startswith('__'):
            return identifier
        
        # Check if already mapped
        if identifier in identifier_map:
            return identifier_map[identifier]
        
        # Determine if function or variable (basic heuristic)
        # Check next non-whitespace character
        start_pos = match.end()
        if start_pos < len(code) and code[start_pos:start_pos+1] == '(':
            new_name = f'func_{func_counter}'
            func_counter += 1
        else:
            new_name = f'var_{var_counter}'
            var_counter += 1
        
        identifier_map[identifier] = new_name
        return new_name
    
    normalized = re.sub(var_pattern, replace_identifier, code)
    return normalized


def handle_typescript(file_path: Path) -> Tuple[str, bool]:
    """
    Transpile TypeScript to JavaScript using tsc
    
    Args:
        file_path: Path to .ts file
    
    Returns:
        (transpiled_code, success)
    """
    try:
        # Create temp directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            output_file = temp_dir / file_path.with_suffix('.js').name
            
            # Run TypeScript compiler
            result = subprocess.run(
                ['tsc', str(file_path), '--outDir', str(temp_dir), '--target', 'ES2020'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Check if compilation succeeded
            if result.returncode == 0 and output_file.exists():
                with open(output_file, 'r', encoding='utf-8') as f:
                    return f.read(), True
            else:
                print(f"⚠️  TypeScript compilation failed for {file_path}")
                print(f"Error: {result.stderr}")
                return "", False
                
    except subprocess.TimeoutExpired:
        print(f"⚠️  TypeScript compilation timeout for {file_path}")
        return "", False
    except Exception as e:
        print(f"⚠️  Error transpiling {file_path}: {e}")
        return "", False


# ============================================================================
# DATASET PARSING
# ============================================================================

def parse_juliet_manifest(manifest_path: Path) -> List[Dict]:
    """
    Parse Juliet Test Suite manifest XML file
    
    Expected structure:
    <testcases>
        <testcase>
            <file>path/to/file.py</file>
            <cwe>89</cwe>
            <bad>true</bad>
        </testcase>
    </testcases>
    """
    samples = []
    
    try:
        tree = ET.parse(manifest_path)
        root = tree.getroot()
        
        for testcase in root.findall('.//testcase'):
            file_elem = testcase.find('file')
            cwe_elem = testcase.find('cwe')
            bad_elem = testcase.find('bad')
            
            if file_elem is None or cwe_elem is None:
                continue
            
            file_path = Path(manifest_path.parent) / file_elem.text
            cwe_id = int(cwe_elem.text.replace('CWE-', ''))
            is_vulnerable = bad_elem.text.lower() == 'true' if bad_elem is not None else False
            
            # Check if file exists
            if not file_path.exists():
                continue
            
            # Determine language from extension
            ext = file_path.suffix.lower()
            if ext == '.py':
                language = 'python'
            elif ext == '.js':
                language = 'javascript'
            elif ext == '.ts':
                language = 'typescript'
            else:
                continue  # Skip unsupported languages
            
            # Read code
            try:
                if language == 'typescript':
                    code, success = handle_typescript(file_path)
                    if not success:
                        continue
                    language = 'javascript'  # After transpilation
                else:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        code = f.read()
                
                # Get OWASP category
                owasp_category = get_owasp_category(cwe_id)
                if owasp_category is None:
                    continue  # Skip if not in our target categories
                
                samples.append({
                    'code': code,
                    'language': language,
                    'cwe_id': cwe_id,
                    'owasp_category': owasp_category,
                    'is_vulnerable': is_vulnerable,
                    'source_file': str(file_path)
                })
                
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
    
    except Exception as e:
        print(f"Error parsing manifest {manifest_path}: {e}")
    
    return samples




def load_linevul_dataset(linevul_dir: Path) -> List[Dict]:
    """
    Load existing LineVul dataset from data/linevul/
    LineVul format: JSON files with vulnerability annotations
    """
    samples = []
    
    if not linevul_dir.exists():
        print(f"⚠️  LineVul directory not found: {linevul_dir}")
        return samples
    
    print(f"Loading LineVul dataset from {linevul_dir}...")
    
    # Find JSON files
    json_files = list(linevul_dir.glob('**/*.json'))
    
    for json_file in tqdm(json_files, desc="Loading LineVul"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Parse LineVul format (adapt based on actual structure)
            if isinstance(data, list):
                for item in data:
                    if 'code' in item and 'target' in item:
                        # Detect CWE from metadata
                        cwe_id = item.get('cwe_id', 79)  # Default to XSS if not specified
                        owasp = get_owasp_category(cwe_id)
                        
                        if owasp:
                            samples.append({
                                'code': item['code'],
                                'language': item.get('language', 'python'),
                                'cwe_id': cwe_id,
                                'owasp_category': owasp,
                                'is_vulnerable': bool(item['target']),
                                'source_file': str(json_file),
                                'source': 'linevul'
                            })
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    print(f"✓ Loaded {len(samples)} samples from LineVul")
    return samples


def load_production_dataset(prod_dir: Path) -> List[Dict]:
    """
    Load and scan production code from data/production/
    Uses basic heuristics to identify potential vulnerabilities
    """
    samples = []
    
    if not prod_dir.exists():
        print(f"⚠️  Production directory not found: {prod_dir}")
        return samples
    
    print(f"Scanning production code from {prod_dir}...")
    
    # Find Python and JavaScript files
    code_files = []
    code_files.extend(prod_dir.glob('**/*.py'))
    code_files.extend(prod_dir.glob('**/*.js'))
    
    # Limit to avoid overwhelming
    code_files = list(code_files)[:1000]  # Max 1000 files
    
    for code_file in tqdm(code_files, desc="Scanning production"):
        try:
            language = detect_language_from_file(code_file)
            if not language:
                continue
            
            with open(code_file, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
            
            # Skip if too large or too small
            if len(code) < 100 or len(code) > 10000:
                continue
            
            # Basic vulnerability detection (use as "safe" samples mostly)
            is_vulnerable = False
            cwe_id = 0
            
            # Simple pattern matching for obvious vulnerabilities
            if 'eval(' in code or 'exec(' in code:
                is_vulnerable = True
                cwe_id = 95  # Code Injection
            elif re.search(r'password\s*=\s*["\']', code, re.IGNORECASE):
                is_vulnerable = True
                cwe_id = 798  # Hardcoded credentials
            elif 'hashlib.md5' in code or 'hashlib.sha1' in code:
                is_vulnerable = True
                cwe_id = 327  # Weak crypto
            
            # Even if not vulnerable, use as negative samples
            owasp = get_owasp_category(cwe_id) if cwe_id else 'A05'  # Default category
            
            if owasp:
                samples.append({
                    'code': code,
                    'language': language,
                    'cwe_id': cwe_id,
                    'owasp_category': owasp,
                    'is_vulnerable': is_vulnerable,
                    'source_file': str(code_file),
                    'source': 'production'
                })
                
        except Exception as e:
            continue
    
    print(f"✓ Scanned {len(samples)} files from production")
    return samples


def parse_sard_dataset(sard_dir: Path) -> List[Dict]:
    """
    Parse SARD (Software Assurance Reference Dataset)
    Structure may vary - adapt as needed
    """
    samples = []
    
    # Find all manifest files
    manifest_files = list(sard_dir.glob('**/manifest.xml'))
    
    print(f"Found {len(manifest_files)} manifest files")
    
    for manifest in tqdm(manifest_files, desc="Parsing manifests"):
        samples.extend(parse_juliet_manifest(manifest))
    
    return samples


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def preprocess_dataset(samples: List[Dict]) -> List[Dict]:
    """
    Preprocess all samples: clean, normalize, filter
    """
    processed = []
    
    print("Preprocessing samples...")
    for sample in tqdm(samples):
        try:
            # Clean code
            code = remove_comments_and_blanks(sample['code'], sample['language'])
            
            # Skip if too short or too long
            if len(code) < 50 or len(code) > 5000:
                continue
            
            # Normalize code
            code = normalize_code(code)
            
            sample['code'] = code
            processed.append(sample)
            
        except Exception as e:
            print(f"Error preprocessing sample: {e}")
            continue
    
    return processed


def calculate_class_weights(labels: List[int]) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced dataset
    
    Args:
        labels: List of binary labels (0 or 1)
    
    Returns:
        Dict mapping class to weight
    """
    unique_classes = np.unique(labels)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=labels
    )
    
    weight_dict = {str(int(cls)): float(weight) for cls, weight in zip(unique_classes, class_weights)}
    
    print("\n📊 Class Distribution:")
    counter = Counter(labels)
    for cls in unique_classes:
        print(f"  Class {cls}: {counter[cls]} samples (weight: {weight_dict[str(int(cls))]:.4f})")
    
    return weight_dict


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(args):
    """Main data preparation pipeline"""
    input_dir = Path(args.input_dir) if args.input_dir else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("VULNERABILITY DATASET PREPARATION PIPELINE")
    print("Multi-Source: SARD/Juliet + LineVul + Production")
    print("="*60)
    
    all_samples = []
    
    # Step 1: Parse PRIMARY dataset (SARD/Juliet)
    if args.input_dir:
        print("\n[Step 1/5] Loading PRIMARY dataset (SARD/Juliet)...")
        if args.dataset_type == 'juliet':
            samples = parse_juliet_manifest(Path(args.input_dir) / 'manifest.xml')
        elif args.dataset_type == 'sard':
            samples = parse_sard_dataset(Path(args.input_dir))
        else:
            raise ValueError(f"Unknown dataset type: {args.dataset_type}")
        
        print(f"✓ Loaded {len(samples)} samples from {args.dataset_type}")
        all_samples.extend(samples)
    
    # Step 2: Load EXISTING datasets (if available)
    print("\n[Step 2/5] Loading EXISTING datasets...")
    
    # Load LineVul
    if args.include_linevul:
        linevul_dir = Path(args.existing_data_dir) / "linevul"
        linevul_samples = load_linevul_dataset(linevul_dir)
        all_samples.extend(linevul_samples)
    
    # Load Production code
    if args.include_production:
        prod_dir = Path(args.existing_data_dir) / "production"
        prod_samples = load_production_dataset(prod_dir)
        all_samples.extend(prod_samples)
    
    print(f"✓ Total samples after merging: {len(all_samples)}")
    
    if len(all_samples) == 0:
        print("❌ No samples loaded! Check your input directories.")
        return
    
    # Step 3: Preprocess
    print("\n[Step 3/5] Preprocessing...")
    samples = preprocess_dataset(all_samples)
    print(f"✓ {len(samples)} samples after preprocessing")
    
    # Step 4: Calculate class weights
    print("\n[Step 4/5] Calculating class weights...")
    labels = [int(s['is_vulnerable']) for s in samples]
    class_weights = calculate_class_weights(labels)
    
    # Step 5: Save processed data
    print("\n[Step 5/5] Saving processed dataset...")
    
    # Split into train/val/test
    np.random.shuffle(samples)
    n = len(samples)
    train_size = int(n * 0.7)
    val_size = int(n * 0.15)
    
    splits = {
        'train': samples[:train_size],
        'val': samples[train_size:train_size+val_size],
        'test': samples[train_size+val_size:]
    }
    
    for split_name, split_data in splits.items():
        output_file = output_dir / f"{split_name}_dataset.json"
        with open(output_file, 'w') as f:
            json.dump({
                'samples': split_data,
                'class_weights': class_weights,
                'metadata': {
                    'total_samples': len(split_data),
                    'dataset_type': args.dataset_type,
                    'owasp_categories': list(OWASP_CWE_MAPPING.keys())
                }
            }, f, indent=2)
        print(f"✓ Saved {split_name}: {len(split_data)} samples → {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("DATASET PREPARATION COMPLETE!")
    print("="*60)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"📊 Total samples: {len(samples)}")
    print(f"   - Train: {len(splits['train'])}")
    print(f"   - Val: {len(splits['val'])}")
    print(f"   - Test: {len(splits['test'])}")
    
    # OWASP distribution
    print(f"\n🎯 OWASP Distribution:")
    owasp_counts = Counter([s['owasp_category'] for s in samples])
    for owasp, count in sorted(owasp_counts.items()):
        print(f"   {owasp}: {count} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare vulnerability detection dataset from multiple sources")
    parser.add_argument(
        '--input_dir',
        type=str,
        default=None,
        help='Path to SARD/Juliet dataset directory (optional if using existing data only)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./backend/ml/data',
        help='Output directory for processed dataset'
    )
    parser.add_argument(
        '--dataset_type',
        type=str,
        choices=['sard', 'juliet'],
        default='sard',
        help='Type of primary dataset (sard or juliet)'
    )
    parser.add_argument(
        '--existing_data_dir',
        type=str,
        default='./backend/data',
        help='Path to existing data directory containing linevul/ and production/'
    )
    parser.add_argument(
        '--include_linevul',
        action='store_true',
        default=True,
        help='Include LineVul dataset from existing data'
    )
    parser.add_argument(
        '--include_production',
        action='store_true',
        default=True,
        help='Include production code from existing data'
    )
    
    args = parser.parse_args()
    main(args)
