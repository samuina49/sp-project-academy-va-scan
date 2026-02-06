"""
Filter datasets to only Python and JavaScript samples.
"""
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/raw_datasets")
FILTERED_DIR = RAW_DATA_DIR / "python_js_only"
FILTERED_DIR.mkdir(exist_ok=True)

def filter_dataset(input_file: str, output_file: str, target_languages: set):
    """Filter dataset by language."""
    logger.info(f"\n📂 Processing {input_file}")
    
    input_path = RAW_DATA_DIR / input_file
    if not input_path.exists():
        logger.warning(f"  ⚠️  File not found: {input_path}")
        return 0
    
    # Load data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"  📊 Total samples: {len(data):,}")
    
    # Filter by language
    filtered = [
        sample for sample in data
        if sample.get('language', '').lower() in target_languages
    ]
    
    logger.info(f"  ✅ Filtered samples: {len(filtered):,}")
    
    if filtered:
        # Count by language
        lang_counts = {}
        for sample in filtered:
            lang = sample.get('language', 'unknown').lower()
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        
        logger.info(f"  📈 Languages: {dict(sorted(lang_counts.items()))}")
        
        # Save
        output_path = FILTERED_DIR / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered, f, indent=2, ensure_ascii=False)
        
        logger.info(f"  💾 Saved to {output_path}")
    
    return len(filtered)


def main():
    logger.info("=" * 70)
    logger.info("🔍 FILTERING PYTHON & JAVASCRIPT DATASETS")
    logger.info("=" * 70)
    
    target_languages = {'python', 'javascript', 'js'}
    total_filtered = 0
    
    # Filter each dataset
    datasets = [
        ('cleanvul_dataset.json', 'cleanvul_py_js.json'),
        ('codexglue_defect_detection.json', 'codexglue_py_js.json'),
        ('pyvul_dataset.json', 'pyvul_py_js.json'),
        ('mock_vulnerabilities.json', 'mock_py_js.json'),
    ]
    
    # Also check synthetic
    synthetic_path = RAW_DATA_DIR / "synthetic_large" / "synthetic_vulnerabilities.json"
    if synthetic_path.exists():
        datasets.append(('synthetic_large/synthetic_vulnerabilities.json', 'synthetic_py_js.json'))
    
    for input_file, output_file in datasets:
        count = filter_dataset(input_file, output_file, target_languages)
        total_filtered += count
    
    logger.info("\n" + "=" * 70)
    logger.info(f"✅ FILTERING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"📊 Total filtered samples: {total_filtered:,}")
    logger.info(f"📁 Output directory: {FILTERED_DIR}")
    logger.info("\n🎯 NEXT STEPS:")
    logger.info("1. Process filtered data:")
    logger.info(f"   python scripts/enhanced_dataset_pipeline.py --data-dir {FILTERED_DIR}")
    logger.info("2. Train model:")
    logger.info("   python training/train_enhanced.py --epochs 50 --batch-size 32")
    logger.info("")


if __name__ == "__main__":
    main()
