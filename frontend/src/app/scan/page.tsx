'use client';

import React, { useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import { useTheme } from '@/components/ThemeProvider';
import dynamic from 'next/dynamic';
import { 
  Play, 
  FileCode, 
  Upload, 
  Code2, 
  AlertTriangle,
  CheckCircle,
  Loader2,
  FileArchive,
  Sparkles,
  ChevronDown
} from 'lucide-react';
import { scanCodeHybrid, scanZip } from '@/lib/api';

// Dynamic import for Monaco Editor
const Editor = dynamic(() => import('@monaco-editor/react'), { 
  ssr: false,
  loading: () => (
    <div className="h-[400px] bg-slate-100 dark:bg-slate-800 rounded-lg flex items-center justify-center">
      <Loader2 className="w-6 h-6 animate-spin text-slate-400" />
    </div>
  )
});

type Language = 'python' | 'javascript' | 'typescript';
type InputMode = 'code' | 'file' | 'zip';

interface LanguageInfo {
  name: string;
  icon: string;
  extensions: string[];
  color: string;
}

const LANGUAGES: Record<Language, LanguageInfo> = {
  python: {
    name: 'Python',
    icon: '🐍',
    extensions: ['.py'],
    color: 'from-blue-500 to-yellow-500'
  },
  javascript: {
    name: 'JavaScript',
    icon: '⚡',
    extensions: ['.js', '.jsx'],
    color: 'from-yellow-400 to-yellow-600'
  },
  typescript: {
    name: 'TypeScript',
    icon: '📘',
    extensions: ['.ts', '.tsx'],
    color: 'from-blue-500 to-blue-700'
  }
};

const CODE_EXAMPLES: Record<Language, string> = {
  python: `# Paste your Python code here to scan for vulnerabilities
import os
import subprocess

def execute_command(user_input):
    # Example: Command injection vulnerability
    os.system(f"echo {user_input}")
    
def read_file(filename):
    # Example: Path traversal vulnerability
    with open(filename, 'r') as f:
        return f.read()
`,
  javascript: `// Paste your JavaScript code here to scan for vulnerabilities
const express = require('express');
const app = express();

app.get('/user', (req, res) => {
  // Example: SQL injection vulnerability
  const query = "SELECT * FROM users WHERE id = " + req.query.id;
  db.query(query);
});

app.get('/search', (req, res) => {
  // Example: XSS vulnerability
  res.send("<h1>Results for: " + req.query.q + "</h1>");
});
`,
  typescript: `// Paste your TypeScript code here to scan for vulnerabilities
import { Request, Response } from 'express';
import { exec } from 'child_process';

export function handleCommand(req: Request, res: Response): void {
  const cmd: string = req.body.command;
  // Example: Command injection vulnerability
  exec(cmd, (error, stdout, stderr) => {
    res.json({ output: stdout });
  });
}

export function renderTemplate(req: Request): string {
  // Example: Template injection vulnerability
  return \`<div>\${req.query.content}</div>\`;
}
`
};

export default function ScanPage() {
  const router = useRouter();
  const { theme } = useTheme();
  
  // State
  const [inputMode, setInputMode] = useState<InputMode>('code');
  const [language, setLanguage] = useState<Language>('python');
  const [code, setCode] = useState(CODE_EXAMPLES.python);
  const [fileName, setFileName] = useState('');
  const [zipFile, setZipFile] = useState<File | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [scanProgress, setScanProgress] = useState(0);
  const [scanStep, setScanStep] = useState('');
  const [showLanguageDropdown, setShowLanguageDropdown] = useState(false);

  // Handle language change
  const handleLanguageChange = (lang: Language) => {
    setLanguage(lang);
    if (inputMode === 'code' && code === CODE_EXAMPLES[language]) {
      setCode(CODE_EXAMPLES[lang]);
    }
    setShowLanguageDropdown(false);
  };

  // Handle file upload
  const handleFileUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const content = event.target?.result as string;
      setCode(content);
      setFileName(file.name);
      
      // Auto-detect language
      const ext = file.name.split('.').pop()?.toLowerCase();
      if (ext === 'py') setLanguage('python');
      else if (ext === 'js' || ext === 'jsx') setLanguage('javascript');
      else if (ext === 'ts' || ext === 'tsx') setLanguage('typescript');
    };
    reader.readAsText(file);
  }, []);

  // Handle ZIP upload
  const handleZipUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.name.endsWith('.zip')) {
      setZipFile(file);
    }
  }, []);

  // Scan progress simulation
  const simulateProgress = (steps: { label: string; duration: number }[]) => {
    let currentProgress = 0;
    let stepIndex = 0;

    const interval = setInterval(() => {
      if (stepIndex >= steps.length) {
        clearInterval(interval);
        return;
      }

      const step = steps[stepIndex];
      setScanStep(step.label);
      
      const increment = 100 / steps.reduce((a, b) => a + b.duration, 0);
      currentProgress += increment;
      setScanProgress(Math.min(Math.round(currentProgress), 95));
      
      if (currentProgress >= (stepIndex + 1) * (100 / steps.length)) {
        stepIndex++;
      }
    }, 200);

    return () => clearInterval(interval);
  };

  // Handle scan
  const handleScan = async () => {
    if (inputMode === 'zip' && !zipFile) return;
    if (inputMode !== 'zip' && !code.trim()) return;

    setIsScanning(true);
    setScanProgress(0);

    const scanSteps = [
      { label: '🔍 Parsing source code...', duration: 1 },
      { label: '🧠 Running AI analysis...', duration: 2 },
      { label: '🔎 Detecting vulnerability patterns...', duration: 2 },
      { label: '📊 Generating security report...', duration: 1 },
    ];

    const cleanup = simulateProgress(scanSteps);

    try {
      let result;

      if (inputMode === 'zip' && zipFile) {
        result = await scanZip(zipFile);
      } else {
        result = await scanCodeHybrid({
          code,
          language,
          filename: fileName || 'untitled'
        });
      }

      // Store result for report page - use correct keys that report page expects
      sessionStorage.setItem('scanResults', JSON.stringify(result));
      sessionStorage.setItem('scannedCode', code);
      sessionStorage.setItem('scannedLanguage', language);
      
      // Update scan history
      const history = JSON.parse(localStorage.getItem('scanHistory') || '[]');
      
      // Calculate vulnerability count based on response type
      let vulnerabilityCount = 0;
      if ('findings' in result && result.findings) {
        vulnerabilityCount = result.findings.length;
      } else if ('file_results' in result && result.file_results) {
        vulnerabilityCount = result.file_results.reduce(
          (acc: number, r: any) => acc + (r.findings?.length || 0), 0
        );
      } else if ('summary' in result && result.summary) {
        vulnerabilityCount = result.summary.total_findings;
      }
      
      history.unshift({
        id: Date.now(),
        fileName: inputMode === 'zip' ? zipFile?.name : (fileName || `${language}_scan`),
        date: new Date().toISOString(),
        vulnerabilities: vulnerabilityCount,
        type: inputMode === 'zip' ? 'project' : 'file',
        language: inputMode === 'zip' ? 'mixed' : language
      });
      localStorage.setItem('scanHistory', JSON.stringify(history.slice(0, 50)));
      
      // Update totals
      const totalScans = parseInt(localStorage.getItem('totalScans') || '0') + 1;
      const totalVulns = parseInt(localStorage.getItem('totalVulnerabilities') || '0') + vulnerabilityCount;
      
      // Calculate files analyzed based on response type
      let filesAnalyzed = 1;
      if ('file_results' in result && result.file_results) {
        filesAnalyzed = result.file_results.length;
      }
      const totalFiles = parseInt(localStorage.getItem('totalFilesAnalyzed') || '0') + filesAnalyzed;
      
      localStorage.setItem('totalScans', String(totalScans));
      localStorage.setItem('totalVulnerabilities', String(totalVulns));
      localStorage.setItem('totalFilesAnalyzed', String(totalFiles));

      setScanProgress(100);
      setScanStep('✅ Scan complete!');
      
      setTimeout(() => {
        router.push('/report');
      }, 500);

    } catch (error: any) {
      console.error('Scan error:', error);
      setScanStep('❌ Scan failed');
      alert(error.response?.data?.detail || 'Scan failed. Please try again.');
    } finally {
      cleanup();
      setTimeout(() => {
        setIsScanning(false);
        setScanProgress(0);
        setScanStep('');
      }, 1000);
    }
  };

  return (
    <div className="min-h-screen p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-3xl font-bold text-slate-900 dark:text-white flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            New Vulnerability Scan
          </h1>
          <p className="text-slate-600 dark:text-slate-400 mt-2">
            Scan your code for security vulnerabilities using our AI-powered hybrid detection engine
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Scan Area */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="lg:col-span-2"
          >
            <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-xl overflow-hidden">
              {/* Mode Selector */}
              <div className="flex border-b border-slate-200 dark:border-slate-700">
                {[
                  { id: 'code', label: 'Code', icon: Code2 },
                  { id: 'file', label: 'File', icon: FileCode },
                  { id: 'zip', label: 'Project', icon: FileArchive },
                ].map((mode) => (
                  <button
                    key={mode.id}
                    onClick={() => setInputMode(mode.id as InputMode)}
                    className={`flex-1 flex items-center justify-center gap-2 py-4 text-sm font-medium transition-all ${
                      inputMode === mode.id
                        ? 'bg-indigo-50 dark:bg-indigo-900/30 text-indigo-600 dark:text-indigo-400 border-b-2 border-indigo-500'
                        : 'text-slate-500 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-700/50'
                    }`}
                  >
                    <mode.icon className="w-4 h-4" />
                    {mode.label}
                  </button>
                ))}
              </div>

              <div className="p-6">
                {/* Language Selector (for code/file mode) */}
                {inputMode !== 'zip' && (
                  <div className="mb-4 relative">
                    <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                      Programming Language
                    </label>
                    <button
                      onClick={() => setShowLanguageDropdown(!showLanguageDropdown)}
                      className="w-full flex items-center justify-between px-4 py-3 bg-slate-50 dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-xl text-left"
                    >
                      <div className="flex items-center gap-3">
                        <span className="text-xl">{LANGUAGES[language].icon}</span>
                        <span className="font-medium text-slate-900 dark:text-white">
                          {LANGUAGES[language].name}
                        </span>
                      </div>
                      <ChevronDown className={`w-5 h-5 text-slate-400 transition-transform ${showLanguageDropdown ? 'rotate-180' : ''}`} />
                    </button>
                    
                    <AnimatePresence>
                      {showLanguageDropdown && (
                        <motion.div
                          initial={{ opacity: 0, y: -10 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: -10 }}
                          className="absolute top-full left-0 right-0 mt-2 bg-white dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-xl shadow-lg z-10 overflow-hidden"
                        >
                          {Object.entries(LANGUAGES).map(([key, lang]) => (
                            <button
                              key={key}
                              onClick={() => handleLanguageChange(key as Language)}
                              className={`w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-slate-50 dark:hover:bg-slate-600 transition-colors ${
                                language === key ? 'bg-indigo-50 dark:bg-indigo-900/30' : ''
                              }`}
                            >
                              <span className="text-xl">{lang.icon}</span>
                              <span className="font-medium text-slate-900 dark:text-white">{lang.name}</span>
                              {language === key && (
                                <CheckCircle className="w-4 h-4 text-indigo-500 ml-auto" />
                              )}
                            </button>
                          ))}
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                )}

                {/* File Upload Area */}
                {inputMode === 'file' && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="mb-4"
                  >
                    <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed border-slate-300 dark:border-slate-600 rounded-xl cursor-pointer bg-slate-50 dark:bg-slate-700/50 hover:bg-slate-100 dark:hover:bg-slate-700 transition-all">
                      <FileCode className="w-10 h-10 text-slate-400 mb-2" />
                      {fileName ? (
                        <p className="text-sm font-medium text-indigo-600 dark:text-indigo-400">{fileName}</p>
                      ) : (
                        <>
                          <p className="text-sm font-medium text-slate-600 dark:text-slate-300">
                            Click to upload or drag and drop
                          </p>
                          <p className="text-xs text-slate-400 mt-1">.py, .js, .jsx, .ts, .tsx</p>
                        </>
                      )}
                      <input
                        type="file"
                        className="hidden"
                        accept=".py,.js,.jsx,.ts,.tsx"
                        onChange={handleFileUpload}
                        disabled={isScanning}
                      />
                    </label>
                  </motion.div>
                )}

                {/* ZIP Upload Area */}
                {inputMode === 'zip' && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="mb-4"
                  >
                    <label className="flex flex-col items-center justify-center w-full h-40 border-2 border-dashed border-indigo-300 dark:border-indigo-700 rounded-xl cursor-pointer bg-indigo-50 dark:bg-indigo-900/20 hover:bg-indigo-100 dark:hover:bg-indigo-900/30 transition-all">
                      <Upload className="w-12 h-12 text-indigo-500 mb-3" />
                      {zipFile ? (
                        <div className="text-center">
                          <p className="text-sm font-medium text-indigo-600 dark:text-indigo-400">
                            {zipFile.name}
                          </p>
                          <p className="text-xs text-slate-500 mt-1">
                            {(zipFile.size / 1024 / 1024).toFixed(2)} MB
                          </p>
                        </div>
                      ) : (
                        <>
                          <p className="text-sm font-medium text-indigo-600 dark:text-indigo-400">
                            Upload Project ZIP File
                          </p>
                          <p className="text-xs text-slate-400 mt-1">
                            Scan entire project with all dependencies
                          </p>
                        </>
                      )}
                      <input
                        type="file"
                        className="hidden"
                        accept=".zip"
                        onChange={handleZipUpload}
                        disabled={isScanning}
                      />
                    </label>
                  </motion.div>
                )}

                {/* Code Editor */}
                {inputMode !== 'zip' && (
                  <div className="border border-slate-200 dark:border-slate-600 rounded-xl overflow-hidden">
                    <Editor
                      height="400px"
                      language={language === 'typescript' ? 'typescript' : language}
                      value={code}
                      onChange={(value) => setCode(value || '')}
                      theme={theme === 'dark' ? 'vs-dark' : 'light'}
                      options={{
                        minimap: { enabled: false },
                        fontSize: 14,
                        fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
                        padding: { top: 16, bottom: 16 },
                        scrollBeyondLastLine: false,
                        lineNumbers: 'on',
                        glyphMargin: false,
                        folding: true,
                        lineDecorationsWidth: 10,
                        automaticLayout: true,
                      }}
                    />
                  </div>
                )}

                {/* Scan Progress */}
                <AnimatePresence>
                  {isScanning && (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      className="mt-6 p-4 bg-indigo-50 dark:bg-indigo-900/30 rounded-xl border border-indigo-200 dark:border-indigo-800"
                    >
                      <div className="flex items-center justify-between mb-3">
                        <span className="text-sm font-medium text-indigo-700 dark:text-indigo-300">
                          {scanStep}
                        </span>
                        <span className="text-sm font-bold text-indigo-600 dark:text-indigo-400">
                          {scanProgress}%
                        </span>
                      </div>
                      <div className="h-2 bg-indigo-200 dark:bg-indigo-800 rounded-full overflow-hidden">
                        <motion.div
                          className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full"
                          initial={{ width: 0 }}
                          animate={{ width: `${scanProgress}%` }}
                          transition={{ duration: 0.3 }}
                        />
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Scan Button */}
                <div className="mt-6">
                  <button
                    onClick={handleScan}
                    disabled={isScanning || (inputMode === 'zip' && !zipFile) || (inputMode !== 'zip' && !code.trim())}
                    className="w-full flex items-center justify-center gap-3 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white px-6 py-4 rounded-xl font-semibold text-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-indigo-500/30"
                  >
                    {isScanning ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        Scanning...
                      </>
                    ) : (
                      <>
                        <Play className="w-5 h-5 fill-current" />
                        Start Security Scan
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Sidebar Info */}
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="space-y-6"
          >
            {/* Detection Capabilities */}
            <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200 dark:border-slate-700 p-6 shadow-lg">
              <h3 className="font-semibold text-slate-900 dark:text-white mb-4 flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-amber-500" />
                Detection Capabilities
              </h3>
              <div className="space-y-3">
                {[
                  { name: 'SQL Injection', color: 'bg-red-500' },
                  { name: 'XSS Attacks', color: 'bg-orange-500' },
                  { name: 'Command Injection', color: 'bg-purple-500' },
                  { name: 'Path Traversal', color: 'bg-blue-500' },
                  { name: 'Insecure Deserialization', color: 'bg-pink-500' },
                  { name: 'SSRF Vulnerabilities', color: 'bg-green-500' },
                ].map((vuln) => (
                  <div key={vuln.name} className="flex items-center gap-3">
                    <div className={`w-2 h-2 rounded-full ${vuln.color}`} />
                    <span className="text-sm text-slate-600 dark:text-slate-300">{vuln.name}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Hybrid Engine Info */}
            <div className="bg-gradient-to-br from-slate-900 to-slate-800 rounded-2xl p-6 text-white shadow-lg">
              <h3 className="font-semibold mb-3 flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-indigo-400" />
                Hybrid AI Engine
              </h3>
              <p className="text-sm text-slate-300 mb-4">
                Our detection engine combines GNN (Graph Neural Network) and LSTM models for accurate vulnerability detection.
              </p>
              <div className="space-y-2 text-sm">
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-emerald-400" />
                  <span className="text-slate-300">OWASP Top 10 Coverage</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-emerald-400" />
                  <span className="text-slate-300">Pattern + ML Detection</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-emerald-400" />
                  <span className="text-slate-300">Remediation Suggestions</span>
                </div>
              </div>
            </div>

            {/* Tips */}
            <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-2xl p-6">
              <h3 className="font-semibold text-amber-800 dark:text-amber-200 mb-3">💡 Pro Tips</h3>
              <ul className="text-sm text-amber-700 dark:text-amber-300 space-y-2">
                <li>• Use ZIP upload for multi-file projects</li>
                <li>• Include all dependencies for thorough scan</li>
                <li>• Check OWASP mappings in reports</li>
              </ul>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
