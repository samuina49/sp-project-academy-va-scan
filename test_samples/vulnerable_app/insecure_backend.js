const express = require('express');
const app = express();
const fs = require('fs');

app.use(express.urlencoded({ extended: true }));

// Vulnerability 1: Reflected XSS (CWE-79)
app.get('/search', (req, res) => {
    const query = req.query.q;
    // Sending raw user input back in HTML
    res.send(`<h1>Search results for: ${query}</h1>`);
});

// Vulnerability 2: Server-Side Request Forgery / SSRF (CWE-918)
app.get('/proxy', (req, res) => {
    const url = req.query.url;
    // Fetching arbitrary URLs supplied by user
    fetch(url)
        .then(response => response.text())
        .then(data => res.send(data));
});

// Vulnerability 3: Code Injection / Eval (CWE-94)
app.post('/calc', (req, res) => {
    const expression = req.body.expression;
    // Dangerous use of eval
    const result = eval(expression);
    res.send(`Result: ${result}`);
});

// Vulnerability 4: Hardcoded Secrets (CWE-798)
const AWS_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE";
const AWS_SECRET_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";

function connectToCloud() {
    console.log("Connecting with " + AWS_ACCESS_KEY);
    // ... logic
}

app.listen(3000, () => {
    console.log('Server running on port 3000');
});
