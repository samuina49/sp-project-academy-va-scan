import json

# Load CodeXGLUE dataset
with open('data/raw_datasets/code_x_glue_cc_defect_detection.json') as f:
    data = json.load(f)

print(f"✅ Total samples: {len(data):,}")

if data:
    print(f"📋 Sample fields: {list(data[0].keys())}")
    
    # Count labels
    vuln = sum(1 for s in data if s.get('label') == 1)
    safe = len(data) - vuln
    print(f"🔴 Vulnerable: {vuln:,} ({100*vuln/len(data):.1f}%)")
    print(f"🟢 Safe: {safe:,} ({100*safe/len(data):.1f}%)")
    
    # Check languages
    langs = {}
    for s in data:
        lang = s.get('language', 'unknown')
        langs[lang] = langs.get(lang, 0) + 1
    print(f"🌐 Languages: {langs}")
    
    # Sample code
    print(f"\n📝 Sample code preview:")
    print(f"  {data[0].get('code', '')[:100]}...")
