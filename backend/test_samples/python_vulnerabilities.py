"""
ตัวอย่างโค้ด Python ที่มีช่องโหว่แต่ละประเภท
ใช้สำหรับทดสอบว่า Scanner เจอช่องโหว่ได้ครบถ้วนและตรงบรรทัดไหม
"""

# =============================================================================
# 1. CODE INJECTION (CRITICAL) - บรรทัด 11-12
# =============================================================================
user_input = input("Enter command: ")
eval(user_input)  # ⚠️ CRITICAL: Code Injection via eval()
exec(user_input)  # ⚠️ CRITICAL: Code Injection via exec()


# =============================================================================
# 2. COMMAND INJECTION (HIGH/CRITICAL) - บรรทัด 17-19
# =============================================================================
import os
import subprocess
os.system("rm -rf /")  # ⚠️ HIGH: Command Injection via os.system()
subprocess.call(["ls", "-la"])  # ⚠️ MEDIUM: Command Execution via subprocess


# =============================================================================
# 3. SQL INJECTION (HIGH) - บรรทัด 24-27
# =============================================================================
import sqlite3
username = request.GET['username']
cursor.execute("SELECT * FROM users WHERE name = '%s'" % username)  # ⚠️ HIGH: SQL Injection
cursor.execute("DELETE FROM users WHERE id = " + user_id)  # ⚠️ HIGH: SQL Injection


# =============================================================================
# 4. PATH TRAVERSAL (HIGH) - บรรทัด 32-33
# =============================================================================
filename = request.GET['file']
open("../../../etc/passwd", "r")  # ⚠️ HIGH: Path Traversal


# =============================================================================
# 5. HARDCODED CREDENTIALS (MEDIUM) - บรรทัด 38-40
# =============================================================================
password = "admin123"  # ⚠️ MEDIUM: Hardcoded Password
api_key = "sk_test_1234567890abcdef"  # ⚠️ MEDIUM: Hardcoded API Key  
db_password = "P@ssw0rd!"  # ⚠️ MEDIUM: Hardcoded Password


# =============================================================================
# 6. WEAK CRYPTOGRAPHY (LOW/MEDIUM) - บรรทัด 45-48
# =============================================================================
import hashlib
hashlib.md5(password.encode())  # ⚠️ MEDIUM: Weak Hash (MD5)
hashlib.sha1(data.encode())  # ⚠️ LOW: Weak Hash (SHA1)


# =============================================================================
# 7. UNSAFE DESERIALIZATION (HIGH) - บรรทัด 53-54
# =============================================================================
import pickle
pickle.loads(untrusted_data)  # ⚠️ HIGH: Unsafe Deserialization


# =============================================================================
# 8. RANDOM (MEDIUM) - บรรทัด 59-60
# =============================================================================
import random
token = random.random()  # ⚠️ MEDIUM: Weak Random Generator


# =============================================================================
# สรุปช่องโหว่ที่ควรเจอ:
# =============================================================================
# ✓ Code Injection (eval, exec) - CRITICAL - บรรทัด 11, 12
# ✓ Command Injection (os.system) - HIGH - บรรทัด 19
# ✓ Command Execution (subprocess) - MEDIUM - บรรทัด 20  
# ✓ SQL Injection - HIGH - บรรทัด 27, 28
# ✓ Path Traversal - HIGH - บรรทัด 33
# ✓ Hardcoded Password - MEDIUM - บรรทัด 38, 39, 40
# ✓ Weak Hash MD5 - MEDIUM - บรรทัด 47
# ✓ Weak Hash SHA1 - LOW - บรรทัด 48
# ✓ Unsafe Pickle - HIGH - บรรทัด 54
# ✓ Weak Random - MEDIUM - บรรทัด 60
#
# รวมทั้งหมด: ประมาณ 12-14 findings
# =============================================================================
