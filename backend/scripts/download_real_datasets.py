"""
Download Real Vulnerability Datasets from Verified Sources
===========================================================
Downloads high-quality datasets from:
1. CleanVul (Hugging Face) - 6K-8K functions, 90-97% correctness
2. PyVul (GitHub) - Python vulnerabilities
3. CodeXGLUE (Hugging Face) - Defect detection

Author: AI-based Vulnerability Scanner
Date: February 6, 2026
"""

import json
import logging
from pathlib import Path
from tqdm import tqdm
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base paths
BACKEND_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BACKEND_DIR / "data" / "raw_datasets"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_cleanvul():
    """
    Download CleanVul dataset from Hugging Face.
    High-quality vulnerability dataset with 90.6-97.3% correctness.
    """
    logger.info("=" * 70)
    logger.info("📦 Downloading CleanVul Dataset")
    logger.info("=" * 70)
    
    try:
        from datasets import load_dataset
        
        logger.info("📥 Loading CleanVul from Hugging Face...")
        logger.info("   Dataset: yikun-li/CleanVul")
        
        # Download dataset
        dataset = load_dataset("yikun-li/CleanVul")
        
        logger.info(f"✅ Loaded dataset with splits: {list(dataset.keys())}")
        
        # Convert to our format
        samples = []
        
        # Process train/test splits
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            logger.info(f"   Processing {split_name}: {len(split_data)} samples")
            
            for item in tqdm(split_data, desc=f"Converting {split_name}"):
                # Extract fields
                try:
                    # CleanVul has func_before/func_after
                    code_before = item.get('func_before', '')
                    code_after = item.get('func_after', '')
                    
                    # Detect language from extension
                    ext = item.get('extension', 'js').lower()
                    lang_map = {'js': 'javascript', 'py': 'python', 'c': 'c', 
                                'cpp': 'cpp', 'java': 'java', 'jsx': 'javascript'}
                    lang = lang_map.get(ext, 'javascript')
                    
                    # Use before as vulnerable, after as safe
                    if code_before and len(code_before) > 20:
                        samples.append({
                            'code': code_before,
                            'label': 1,  # vulnerable
                            'language': lang,
                            'vulnerability_type': item.get('cwe_id', 'unknown'),
                            'source': 'cleanvul',
                            'metadata': {
                                'commit_url': item.get('commit_url', ''),
                                'cve_id': item.get('cve_id', ''),
                                'score': item.get('vulnerability_score', 0),
                                'split': split_name
                            }
                        })
                    
                    if code_after and len(code_after) > 20:
                        samples.append({
                            'code': code_after,
                            'label': 0,  # safe (fixed)
                            'language': lang,
                            'vulnerability_type': 'none',
                            'source': 'cleanvul',
                            'metadata': {
                                'commit_url': item.get('commit_url', ''),
                                'cve_id': item.get('cve_id', ''),
                                'score': item.get('vulnerability_score', 0),
                                'split': split_name,
                                'is_fixed_version': True
                            }
                        })
                except Exception as e:
                    logger.debug(f"Error processing item: {e}")
                    continue
        
        # Save to JSON
        output_file = RAW_DATA_DIR / 'cleanvul_dataset.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Saved {len(samples)} samples to {output_file}")
        
        # Statistics
        vulnerable = sum(1 for s in samples if s['label'] == 1)
        safe = len(samples) - vulnerable
        logger.info(f"📊 Vulnerable: {vulnerable}, Safe: {safe}")
        
        return samples
        
    except Exception as e:
        logger.error(f"❌ CleanVul download failed: {e}")
        logger.info("💡 Make sure 'datasets' is installed: pip install datasets")
        return []


def download_codexglue_defect():
    """
    Download CodeXGLUE Defect Detection dataset.
    Microsoft's code understanding benchmark.
    """
    logger.info("=" * 70)
    logger.info("📦 Downloading CodeXGLUE Defect Detection")
    logger.info("=" * 70)
    
    try:
        from datasets import load_dataset
        
        logger.info("📥 Loading from Hugging Face...")
        logger.info("   Dataset: code_x_glue_cc_defect_detection")
        
        # Download dataset
        dataset = load_dataset("code_x_glue_cc_defect_detection")
        
        logger.info(f"✅ Loaded dataset with splits: {list(dataset.keys())}")
        
        # Convert to our format
        samples = []
        
        for split_name in ['train', 'validation', 'test']:
            if split_name not in dataset:
                continue
                
            split_data = dataset[split_name]
            logger.info(f"   Processing {split_name}: {len(split_data)} samples")
            
            for item in tqdm(split_data, desc=f"Converting {split_name}"):
                try:
                    code = item.get('func', '')
                    if not code or len(code) < 20:
                        continue
                    
                    samples.append({
                        'code': code,
                        'label': int(item.get('target', 0)),
                        'language': 'c',  # CodeXGLUE defect detection is C
                        'vulnerability_type': 'defect' if item.get('target', 0) == 1 else 'none',
                        'source': 'codexglue_defect',
                        'metadata': {
                            'split': split_name,
                            'project': item.get('project', ''),
                            'commit_id': item.get('commit_id', '')
                        }
                    })
                except Exception as e:
                    logger.debug(f"Error processing item: {e}")
                    continue
        
        # Save to JSON
        output_file = RAW_DATA_DIR / 'codexglue_defect_detection.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Saved {len(samples)} samples to {output_file}")
        
        # Statistics
        vulnerable = sum(1 for s in samples if s['label'] == 1)
        safe = len(samples) - vulnerable
        logger.info(f"📊 Defective: {vulnerable}, Clean: {safe}")
        
        return samples
        
    except Exception as e:
        logger.error(f"❌ CodeXGLUE download failed: {e}")
        logger.info("💡 Try: pip install datasets")
        return []


def download_pyvul():
    """
    Download PyVul dataset from GitHub.
    Python vulnerability benchmark.
    """
    logger.info("=" * 70)
    logger.info("📦 Downloading PyVul Dataset")
    logger.info("=" * 70)
    
    try:
        import requests
        
        logger.info("📥 Downloading from GitHub...")
        
        # PyVul has function-level dataset
        urls = [
            "https://raw.githubusercontent.com/billquan/PyVul/main/dataset/function_level_dataset.out"
        ]
        
        samples = []
        
        for url in urls:
            logger.info(f"   Fetching: {url}")
            response = requests.get(url, timeout=60)
            
            if response.status_code == 200:
                # Parse the dataset (JSON lines format)
                lines = response.text.strip().split('\n')
                logger.info(f"   Got {len(lines)} lines")
                
                for line in tqdm(lines, desc="Processing PyVul"):
                    try:
                        # Parse JSON
                        item = json.loads(line)
                        
                        # Get code before and after
                        code_before = item.get('code_before', '')
                        code_after = item.get('code_after', '')
                        
                        # Use before as vulnerable, after as safe
                        if code_before and len(code_before) > 20:
                            samples.append({
                                'code': code_before,
                                'label': 1,  # vulnerable
                                'language': 'python',
                                'vulnerability_type': item.get('cve_id', 'unknown'),
                                'source': 'pyvul',
                                'metadata': {
                                    'function_name': item.get('function_name', ''),
                                    'commit_message': item.get('commit_message', '')
                                }
                            })
                        
                        if code_after and len(code_after) > 20:
                            samples.append({
                                'code': code_after,
                                'label': 0,  # safe (fixed)
                                'language': 'python',
                                'vulnerability_type': 'none',
                                'source': 'pyvul',
                                'metadata': {
                                    'function_name': item.get('function_name', ''),
                                    'commit_message': item.get('commit_message', ''),
                                    'is_fixed_version': True
                                }
                            })
                    except json.JSONDecodeError as e:
                        logger.debug(f"Error parsing JSON: {e}")
                        continue
                    except Exception as e:
                        logger.debug(f"Error processing item: {e}")
                        continue
            else:
                logger.warning(f"   Failed to download: HTTP {response.status_code}")
        
        if samples:
            # Save to JSON
            output_file = RAW_DATA_DIR / 'pyvul_dataset.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(samples, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Saved {len(samples)} samples to {output_file}")
            
            # Statistics
            vulnerable = sum(1 for s in samples if s['label'] == 1)
            safe = len(samples) - vulnerable
            logger.info(f"📊 Vulnerable: {vulnerable}, Safe: {safe}")
        else:
            logger.warning("⚠️ No samples extracted from PyVul")
        
        return samples
        
    except Exception as e:
        logger.error(f"❌ PyVul download failed: {e}")
        return []


def main():
    """Main download function"""
    logger.info("=" * 70)
    logger.info("🚀 REAL DATASET DOWNLOADER")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Will download:")
    logger.info("  1. CleanVul (6K-8K samples, 90-97% accuracy)")
    logger.info("  2. CodeXGLUE Defect Detection (21K+ samples)")
    logger.info("  3. PyVul (1K+ Python samples)")
    logger.info("")
    
    total_samples = 0
    
    # Download CleanVul
    cleanvul_samples = download_cleanvul()
    total_samples += len(cleanvul_samples)
    
    print()
    
    # Download CodeXGLUE
    codexglue_samples = download_codexglue_defect()
    total_samples += len(codexglue_samples)
    
    print()
    
    # Download PyVul
    pyvul_samples = download_pyvul()
    total_samples += len(pyvul_samples)
    
    print()
    logger.info("=" * 70)
    logger.info("📊 DOWNLOAD SUMMARY")
    logger.info("=" * 70)
    logger.info(f"✅ CleanVul: {len(cleanvul_samples)} samples")
    logger.info(f"✅ CodeXGLUE:{len(codexglue_samples)} samples")
    logger.info(f"✅ PyVul: {len(pyvul_samples)} samples")
    logger.info(f"")
    logger.info(f"🎉 Total: {total_samples} samples downloaded!")
    logger.info(f"📁 Saved to: {RAW_DATA_DIR}")
    logger.info("=" * 70)
    logger.info("")
    logger.info("🎯 NEXT STEPS:")
    logger.info("1. Process datasets:")
    logger.info("   python scripts/enhanced_dataset_pipeline.py")
    logger.info("2. Train model:")
    logger.info("   python training/train_enhanced.py --epochs 100 --batch-size 32")


if __name__ == "__main__":
    main()
