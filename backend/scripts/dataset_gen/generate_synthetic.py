"""
Synthetic Vulnerability Dataset Generator
Generates training data for the GNN+LSTM model
"""
import random
import json
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime


class VulnerabilityGenerator:
    """Generate synthetic vulnerable code samples"""
    
    def __init__(self):
        self.sample_id = 0
    
    def _get_next_id(self) -> str:
        self.sample_id += 1
        return f"sample_{self.sample_id:06d}"
    
    # ==================== SQL Injection ====================
    
    def generate_sql_injection(self) -> Dict:
        """Generate SQL injection vulnerable code"""
        templates = [
            {
                "code": '''import sqlite3
user_input = input("Enter user ID: ")
query = f"SELECT * FROM users WHERE id = {user_input}"
conn.execute(query)''',
                "line": 3
            },
            {
                "code": '''cursor.execute("SELECT * FROM products WHERE name = '" + product_name + "'")''',
                "line": 1
            },
            {
                "code": '''sql = "DELETE FROM items WHERE id = %s" % item_id
db.execute(sql)''',
                "line": 1
            },
            {
                "code": '''query = "UPDATE users SET role='admin' WHERE username='" + username + "'"
cursor.execute(query)''',
                "line": 1
            },
        ]
        
        template = random.choice(templates)
        return {
            "id": self._get_next_id(),
            "language": "python",
            "code": template["code"],
            "vulnerabilities": [{
                "cwe_id": "CWE-89",
                "severity": "HIGH",
                "line": template["line"],
                "message": "SQL Injection via string formatting",
                "owasp_category": "A03:2021-Injection",
                "confidence": 0.95
            }],
            "metadata": {
                "source": "synthetic",
                "created_at": datetime.now().isoformat()
            }
        }
    
    # ==================== Command Injection ====================
    
    def generate_command_injection(self) -> Dict:
        """Generate command injection vulnerable code"""
        templates = [
            {
                "code": '''import os
hostname = input("Enter hostname: ")
os.system(f"ping {hostname}")''',
                "line": 3
            },
            {
                "code": '''subprocess.call(user_input, shell=True)''',
                "line": 1
            },
            {
                "code": '''os.popen("nslookup " + domain).read()''',
                "line": 1
            },
        ]
        
        template = random.choice(templates)
        return {
            "id": self._get_next_id(),
            "language": "python",
            "code": template["code"],
            "vulnerabilities": [{
                "cwe_id": "CWE-78",
                "severity": "CRITICAL",
                "line": template["line"],
                "message": "Command injection vulnerability",
                "owasp_category": "A03:2021-Injection",
                "confidence": 0.98
            }],
            "metadata": {
                "source": "synthetic",
                "created_at": datetime.now().isoformat()
            }
        }
    
    # ==================== XSS ====================
    
    def generate_xss(self) -> Dict:
        """Generate XSS vulnerable code"""
        templates = [
            {
                "code": '''return render(request, 'page.html', {
    'user_comment': request.GET['comment']
})''',
                "line": 2
            },
            {
                "code": '''html = f"<div>{user_input}</div>"
return HttpResponse(html)''',
                "line": 1
            },
        ]
        
        template = random.choice(templates)
        return {
            "id": self._get_next_id(),
            "language": "python",
            "code": template["code"],
            "vulnerabilities": [{
                "cwe_id": "CWE-79",
                "severity": "MEDIUM",
                "line": template["line"],
                "message": "Cross-Site Scripting (XSS) vulnerability",
                "owasp_category": "A03:2021-Injection",
                "confidence": 0.85
            }],
            "metadata": {
                "source": "synthetic",
                "created_at": datetime.now().isoformat()
            }
        }
    
    # ==================== Hardcoded Secrets ====================
    
    def generate_hardcoded_secret(self) -> Dict:
        """Generate hardcoded secret code"""
        templates = [
            {
                "code": '''API_KEY = "sk_live_1234567890abcdef"
SECRET_KEY = "django-insecure-hardcoded-key"''',
                "line": 1
            },
            {
                "code": '''password = "P@ssw0rd123"
conn.login(username, password)''',
                "line": 1
            },
        ]
        
        template = random.choice(templates)
        return {
            "id": self._get_next_id(),
            "language": "python",
            "code": template["code"],
            "vulnerabilities": [{
                "cwe_id": "CWE-798",
                "severity": "CRITICAL",
                "line": template["line"],
                "message": "Hardcoded credentials detected",
                "owasp_category": "A02:2021-Cryptographic Failures",
                "confidence": 0.99
            }],
            "metadata": {
                "source": "synthetic",
                "created_at": datetime.now().isoformat()
            }
        }
    
    # ==================== Weak Cryptography ====================
    
    def generate_weak_crypto(self) -> Dict:
        """Generate weak cryptography code"""
        templates = [
            {
                "code": '''import hashlib
password_hash = hashlib.md5(password.encode()).hexdigest()''',
                "line": 2
            },
            {
                "code": '''import random
token = random.randint(1000, 9999)''',
                "line": 2
            },
        ]
        
        template = random.choice(templates)
        return {
            "id": self._get_next_id(),
            "language": "python",
            "code": template["code"],
            "vulnerabilities": [{
                "cwe_id": "CWE-327",
                "severity": "HIGH",
                "line": template["line"],
                "message": "Weak cryptographic algorithm",
                "owasp_category": "A02:2021-Cryptographic Failures",
                "confidence": 0.95
            }],
            "metadata": {
                "source": "synthetic",
                "created_at": datetime.now().isoformat()
            }
        }
    
    # ==================== Path Traversal ====================
    
    def generate_path_traversal(self) -> Dict:
        """Generate path traversal vulnerable code"""
        templates = [
            {
                "code": '''filename = request.GET['file']
with open(f"/var/www/files/{filename}", 'r') as f:
    content = f.read()''',
                "line": 2
            },
            {
                "code": '''path = "../" + user_path
file_content = open(path).read()''',
                "line": 2
            },
        ]
        
        template = random.choice(templates)
        return {
            "id": self._get_next_id(),
            "language": "python",
            "code": template["code"],
            "vulnerabilities": [{
                "cwe_id": "CWE-22",
                "severity": "HIGH",
                "line": template["line"],
                "message": "Path traversal vulnerability",
                "owasp_category": "A01:2021-Broken Access Control",
                "confidence": 0.90
            }],
            "metadata": {
                "source": "synthetic",
                "created_at": datetime.now().isoformat()
            }
        }
    
    # ==================== Main Generator ====================
    
    def generate_dataset(self, n_samples: int = 1000) -> List[Dict]:
        """Generate complete balanced dataset"""
        print(f"Generating {n_samples} synthetic samples...")
        
        generators = [
            self.generate_sql_injection,
            self.generate_command_injection,
            self.generate_xss,
            self.generate_hardcoded_secret,
            self.generate_weak_crypto,
            self.generate_path_traversal,
        ]
        
        dataset = []
        samples_per_type = n_samples // len(generators)
        
        for generator in generators:
            print(f"  Generating {generator.__name__}...")
            for _ in range(samples_per_type):
                dataset.append(generator())
        
        # Shuffle dataset
        random.shuffle(dataset)
        
        print(f"✓ Generated {len(dataset)} samples")
        return dataset


def split_dataset(samples: List[Dict], train_ratio=0.7, val_ratio=0.15) -> Tuple[List, List, List]:
    """Split dataset into train/val/test"""
    random.shuffle(samples)
    
    n = len(samples)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train = samples[:train_size]
    val = samples[train_size:train_size+val_size]
    test = samples[train_size+val_size:]
    
    return train, val, test


def save_dataset(samples: List[Dict], output_path: Path):
    """Save dataset to JSON file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    dataset = {
        "version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "total_samples": len(samples),
        "samples": samples
    }
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"✓ Saved {len(samples)} samples to {output_path}")


if __name__ == "__main__":
    # Generate synthetic dataset
    generator = VulnerabilityGenerator()
    all_samples = generator.generate_dataset(n_samples=3000)
    
    # Split into train/val/test
    train, val, test = split_dataset(all_samples)
    
    # Save datasets
    data_dir = Path("backend/ml/data")
    save_dataset(train, data_dir / "train_dataset.json")
    save_dataset(val, data_dir / "val_dataset.json")
    save_dataset(test, data_dir / "test_dataset.json")
    
    print("\n=== Dataset Summary ===")
    print(f"Training:   {len(train)} samples")
    print(f"Validation: {len(val)} samples")
    print(f"Test:       {len(test)} samples")
    print(f"Total:      {len(all_samples)} samples")
