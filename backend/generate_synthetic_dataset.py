"""
Generate Synthetic Training Dataset
====================================
Create a larger dataset based on mock vulnerability patterns
for testing the training pipeline with more data.

This is for testing/demo purposes only.
Real production models should use real-world datasets.
"""
import json
import random
from pathlib import Path

# Base directory
output_dir = Path("data/raw_datasets/synthetic_large")
output_dir.mkdir(parents=True, exist_ok=True)

# Vulnerability templates
VULN_TEMPLATES = {
    'sql_injection': [
        ('def search_user(username):\n    query = f"SELECT * FROM users WHERE name = \'{username}\'"\n    return db.execute(query)', True),
        ('def search_user(username):\n    query = "SELECT * FROM users WHERE name = %s"\n    return db.execute(query, (username,))', False),
    ],
    'xss': [
        ('function renderComment(text) {\n  document.getElementById("comment").innerHTML = text;\n}', True),
        ('function renderComment(text) {\n  document.getElementById("comment").textContent = text;\n}', False),
    ],
    'command_injection': [
        ('import os\ndef backup_file(filename):\n    os.system(f"cp {filename} /backup/")', True),
        ('import subprocess\ndef backup_file(filename):\n    subprocess.run(["cp", filename, "/backup/"], check=True)', False),
    ],
    'path_traversal': [
        ('def read_file(filename):\n    with open(filename) as f:\n        return f.read()', True),
        ('def read_file(filename):\n    safe_path = os.path.basename(filename)\n    with open(safe_path) as f:\n        return f.read()', False),
    ],
    'insecure_deserialization': [
        ('import pickle\ndef load_data(data):\n    return pickle.loads(data)', True),
        ('import json\ndef load_data(data):\n    return json.loads(data)', False),
    ],
}

# Generate samples
samples = []
sample_id = 1

# Generate multiple variations of each template
for vuln_type, templates in VULN_TEMPLATES.items():
    for template_code, is_vulnerable in templates:
        # Generate 50 variations per template
        for i in range(50):
            code = template_code
            
            # Add random variations
            # Add random comments
            if random.random() > 0.5:
                code = f"# Auto-generated sample {sample_id}\n" + code
            
            # Add random whitespace
            if random.random() > 0.7:
                code = code.replace('\n', '\n\n', random.randint(1, 2))
            
            # Random variable names (simple variations)
            if random.random() > 0.6:
                variations = {
                    'filename': ['file_name', 'fname', 'filepath'],
                    'username': ['user_name', 'uname', 'user'],
                    'data': ['raw_data', 'input_data', 'payload'],
                    'text': ['content', 'msg', 'message'],
                }
                for orig, replacements in variations.items():
                    if orig in code and random.random() > 0.5:
                        code = code.replace(orig, random.choice(replacements))
            
            # Determine language
            language = 'javascript' if 'function' in code else 'python'
            
            sample = {
                'code': code,
                'label': 1 if is_vulnerable else 0,
                'language': language,
                'vulnerability_type': vuln_type if is_vulnerable else 'none',
                'source': 'synthetic_generated',
                'metadata': {
                    'template_id': sample_id,
                    'variation': i
                }
            }
            
            samples.append(sample)
            sample_id += 1

# Shuffle
random.shuffle(samples)

# Save
output_file = output_dir / 'synthetic_vulnerabilities.json'
with open(output_file, 'w') as f:
    json.dump(samples, f, indent=2)

print(f"✅ Generated {len(samples)} samples")
print(f"💾 Saved to: {output_file}")

# Statistics
vulnerable = sum(1 for s in samples if s['label'] == 1)
safe = len(samples) - vulnerable
print(f"\n📊 Statistics:")
print(f"  Vulnerable: {vulnerable} ({100*vulnerable/len(samples):.1f}%)")
print(f"  Safe: {safe} ({100*safe/len(samples):.1f}%)")

# Count by type
vuln_counts = {}
for s in samples:
    if s['label'] == 1:
        vtype = s['vulnerability_type']
        vuln_counts[vtype] = vuln_counts.get(vtype, 0) + 1

print(f"\n🔍 Vulnerability types:")
for vtype, count in sorted(vuln_counts.items()):
    print(f"  {vtype}: {count}")

# Count by language
py_count = sum(1 for s in samples if s['language'] == 'python')
js_count = sum(1 for s in samples if s['language'] == 'javascript')
print(f"\n🌐 Languages:")
print(f"  Python: {py_count}")
print(f"  JavaScript: {js_count}")
