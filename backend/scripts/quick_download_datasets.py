"""
Quick Dataset Downloader using Hugging Face & API
=================================================
This script downloads high-quality datasets using Hugging Face datasets library
and public APIs (no manual download needed).

Requirements:
    pip install datasets requests tqdm pandas

Usage:
    python scripts/quick_download_datasets.py

Author: AI Vulnerability Scanner
Date: February 6, 2026
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try importing optional dependencies
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("⚠️ Hugging Face datasets not installed. Run: pip install datasets")

try:
    import requests
    from tqdm import tqdm
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("⚠️ requests/tqdm not installed. Run: pip install requests tqdm")

# Paths
BACKEND_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BACKEND_DIR / "data" / "raw_datasets"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


class QuickDatasetDownloader:
    """Download datasets using APIs and Hugging Face."""
    
    def __init__(self):
        self.stats = {
            'total_samples': 0,
            'datasets': {}
        }
    
    def download_all(self):
        """Download all available datasets."""
        logger.info("=" * 70)
        logger.info("🚀 Quick Dataset Downloader")
        logger.info("=" * 70)
        
        # Check dependencies
        if not HF_AVAILABLE:
            logger.error("❌ Cannot proceed without 'datasets' library")
            logger.info("📦 Install with: pip install datasets requests tqdm pandas")
            return
        
        datasets_to_download = [
            ("CodeSearchNet", self.download_codesearchnet),
            ("Big Clone Bench", self.download_bigclonebench),
            ("Code Defects", self.download_code_defects),
        ]
        
        for name, func in datasets_to_download:
            try:
                logger.info(f"\n{'='*70}")
                logger.info(f"📦 Downloading: {name}")
                logger.info('='*70)
                func()
            except Exception as e:
                logger.error(f"❌ Failed to download {name}: {e}")
        
        self._print_summary()
    
    def download_codesearchnet(self):
        """
        Download CodeSearchNet dataset (Python code).
        Contains real Python functions from GitHub.
        """
        output_dir = RAW_DATA_DIR / "codesearchnet"
        output_dir.mkdir(exist_ok=True)
        
        logger.info("📥 Downloading CodeSearchNet Python subset...")
        
        try:
            # Download Python subset only (smaller, faster)
            dataset = load_dataset(
                "code_search_net",
                "python",
                split="train[:10000]"  # First 10,000 samples
            )
            
            logger.info(f"✅ Downloaded {len(dataset)} Python samples")
            
            # Convert to unified format
            unified_data = []
            for idx, item in enumerate(dataset):
                code = item.get('func_code_string', '') or item.get('whole_func_string', '')
                
                if not code or len(code) < 50:
                    continue
                
                unified_item = {
                    'code': code,
                    'label': 0,  # CodeSearchNet doesn't have vulnerability labels, mark as safe
                    'language': 'python',
                    'vulnerability_type': 'safe',
                    'source': 'codesearchnet',
                    'metadata': {
                        'func_name': item.get('func_name', 'unknown'),
                        'repo': item.get('repo', 'unknown'),
                        'docstring': item.get('func_documentation_string', '')[:100]
                    }
                }
                
                unified_data.append(unified_item)
                
                if len(unified_data) % 1000 == 0:
                    logger.info(f"  Processed {len(unified_data)} samples...")
            
            # Save
            output_file = output_dir / "codesearchnet_python.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(unified_data, f, indent=2, ensure_ascii=False)
            
            self.stats['total_samples'] += len(unified_data)
            self.stats['datasets']['CodeSearchNet'] = len(unified_data)
            
            logger.info(f"✅ Saved {len(unified_data)} samples to: {output_file}")
            
        except Exception as e:
            logger.error(f"❌ CodeSearchNet download failed: {e}")
            raise
    
    def download_bigclonebench(self):
        """
        Download BigCloneBench or similar code dataset.
        Note: May not have vulnerability labels, but good for "safe" code samples.
        """
        output_dir = RAW_DATA_DIR / "bigclonebench"
        output_dir.mkdir(exist_ok=True)
        
        logger.info("📥 Attempting BigCloneBench download...")
        
        try:
            # BigCloneBench might not be on HF, try code_contests instead
            dataset = load_dataset(
                "deepmind/code_contests",
                split="train[:1000]"  # Small subset
            )
            
            logger.info(f"✅ Downloaded {len(dataset)} coding problems")
            
            unified_data = []
            for item in dataset:
                # Extract Python solutions
                solutions = item.get('solutions', {})
                python_code = solutions.get('python', [''])[0] if 'python' in solutions else ''
                
                if not python_code or len(python_code) < 50:
                    continue
                
                unified_item = {
                    'code': python_code,
                    'label': 0,  # Competition code assumed safe
                    'language': 'python',
                    'vulnerability_type': 'safe',
                    'source': 'code_contests',
                    'metadata': {
                        'name': item.get('name', 'unknown'),
                        'difficulty': item.get('difficulty', 'unknown')
                    }
                }
                
                unified_data.append(unified_item)
            
            # Save
            output_file = output_dir / "code_contests.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(unified_data, f, indent=2, ensure_ascii=False)
            
            self.stats['total_samples'] += len(unified_data)
            self.stats['datasets']['Code Contests'] = len(unified_data)
            
            logger.info(f"✅ Saved {len(unified_data)} samples to: {output_file}")
            
        except Exception as e:
            logger.warning(f"⚠️ BigCloneBench not available: {e}")
    
    def download_code_defects(self):
        """
        Download code defect datasets from Hugging Face.
        """
        output_dir = RAW_DATA_DIR / "code_defects"
        output_dir.mkdir(exist_ok=True)
        
        logger.info("📥 Searching for code defect datasets...")
        
        # Try multiple possible dataset names
        dataset_attempts = [
            ("code_x_glue_cc_defect_detection", "default"),
            ("mbpp", "default"),  # Python programming problems
        ]
        
        for dataset_name, config in dataset_attempts:
            try:
                logger.info(f"  Trying: {dataset_name}...")
                
                if config == "default":
                    dataset = load_dataset(dataset_name, split="train[:5000]")
                else:
                    dataset = load_dataset(dataset_name, config, split="train[:5000]")
                
                logger.info(f"  ✅ Found {len(dataset)} samples!")
                
                # Save raw dataset info
                output_file = output_dir / f"{dataset_name}.json"
                
                # Convert to list of dicts
                data_list = []
                for item in dataset:
                    # Try to extract code
                    code = None
                    if 'func' in item:
                        code = item['func']
                    elif 'code' in item:
                        code = item['code']
                    elif 'text' in item:
                        code = item['text']
                    
                    if not code:
                        continue
                    
                    # Try to extract label
                    label = item.get('target', item.get('label', 0))
                    
                    data_list.append({
                        'code': code,
                        'label': int(label) if label is not None else 0,
                        'source': dataset_name,
                        'raw_item': str(item)[:200]  # Debug info
                    })
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data_list, f, indent=2, ensure_ascii=False)
                
                self.stats['total_samples'] += len(data_list)
                self.stats['datasets'][dataset_name] = len(data_list)
                
                logger.info(f"  ✅ Saved {len(data_list)} samples")
                
                break  # Success, stop trying
                
            except Exception as e:
                logger.warning(f"  ⚠️ {dataset_name} failed: {e}")
                continue
    
    def _print_summary(self):
        """Print download summary."""
        logger.info("\n" + "=" * 70)
        logger.info("📊 DOWNLOAD SUMMARY")
        logger.info("=" * 70)
        logger.info(f"✅ Total samples downloaded: {self.stats['total_samples']:,}")
        logger.info("")
        
        for dataset_name, count in self.stats['datasets'].items():
            logger.info(f"  📦 {dataset_name}: {count:,} samples")
        
        logger.info("\n" + "=" * 70)
        logger.info(f"📁 Data saved to: {RAW_DATA_DIR}")
        logger.info("=" * 70)
        
        logger.info("\n🎯 NEXT STEPS:")
        logger.info("1. Review downloaded data quality")
        logger.info("2. Add vulnerable code datasets manually:")
        logger.info("   - Devign: git clone https://github.com/saikat107/Devign")
        logger.info("   - CVEFixes: git clone https://github.com/secureIT-project/CVEfixes")
        logger.info("3. Run: python scripts/parse_quality_datasets.py")
        logger.info("4. Train model: python training/train_with_real_data.py")


def main():
    """Main entry point."""
    
    # Check dependencies
    if not HF_AVAILABLE:
        print("\n" + "="*70)
        print("❌ ERROR: Required dependencies not installed")
        print("="*70)
        print("\n📦 Please install required packages:")
        print("\n  pip install datasets requests tqdm pandas\n")
        print("Then run this script again.")
        print("="*70)
        return
    
    downloader = QuickDatasetDownloader()
    
    try:
        downloader.download_all()
        print("\n✅ Dataset download complete!")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Download interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Download failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
