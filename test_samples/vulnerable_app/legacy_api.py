import os
import sqlite3
import pickle
from flask import Flask, request

app = Flask(__name__)

# Vulnerability 1: Hardcoded credentials (CWE-798)
DB_USER = "admin"
DB_PASSWORD = "supersecretpassword123"

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    
    # Vulnerability 2: SQL Injection (CWE-89)
    # Concatenating user input directly into the query string
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE username = '" + username + "' AND password = '" + password + "'"
    cursor.execute(query)
    user = cursor.fetchone()
    
    if user:
        return "Login successful"
    return "Login failed"

@app.route('/admin/system_check', methods=['GET'])
def system_check():
    ip_address = request.args.get('ip')
    
    # Vulnerability 3: Command Injection (CWE-78)
    # Passing user input directly to a shell command
    os.system("ping -c 1 " + ip_address)
    
    return "System check complete"

@app.route('/deserialize', methods=['POST'])
def deserialize_data():
    data = request.data
    
    # Vulnerability 4: Insecure Deserialization (CWE-502)
    # Using pickle on untrusted data
    obj = pickle.loads(data)
    
    return "Data processed"

@app.route('/read_log', methods=['GET'])
def read_log():
    filename = request.args.get('file')
    
    # Vulnerability 5: Path Traversal (CWE-22)
    # Reading file based on user input without validation
    with open('/var/logs/' + filename, 'r') as f:
        return f.read()

if __name__ == '__main__':
    # Vulnerability 6: Debug Mode Enabled (CWE-489)
    app.run(debug=True, host='0.0.0.0')
