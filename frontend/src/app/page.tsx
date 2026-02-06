'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { 
  Shield, 
  ShieldCheck,
  ShieldAlert,
  Activity, 
  ArrowRight,
  Zap,
  Brain,
  Code2,
  FileCode,
  FolderArchive,
  Scan,
  GitBranch,
  Cpu,
  FileWarning,
  Lock,
  Unlock,
  Terminal,
  Globe,
  Database,
  Sparkles,
  Gauge,
  FileSearch
} from 'lucide-react';
import { useI18n } from '@/lib/i18n';

export default function DashboardPage() {
  const { language } = useI18n();

  // Text content based on language
  const t = {
    title: 'AI-Based Vulnerability Scanner',
    subtitle: language === 'th' 
      ? 'ระบบตรวจสอบช่องโหว่ความปลอดภัยของเว็บแอปพลิเคชันด้วยปัญญาประดิษฐ์' 
      : 'A Web Application Security Vulnerability Scanning System Using Artificial Intelligence',
    startScan: language === 'th' ? 'เริ่มสแกนโค้ด' : 'Start Scanning',
    uploadProject: language === 'th' ? 'อัพโหลดโปรเจค' : 'Upload Project',
    features: language === 'th' ? 'จุดเด่นของระบบ' : 'Key Features',
    capabilities: language === 'th' ? 'ความสามารถในการตรวจจับ' : 'Detection Capabilities',
    howItWorks: language === 'th' ? 'วิธีการทำงาน' : 'How It Works',
    supportedLang: language === 'th' ? 'ภาษาที่รองรับ' : 'Supported Languages',
    owaspCoverage: 'OWASP Top 10 Coverage',
    step1Title: language === 'th' ? 'อัพโหลดโค้ด' : 'Upload Code',
    step1Desc: language === 'th' ? 'วางโค้ดหรืออัพโหลดไฟล์/โปรเจค ZIP' : 'Paste code or upload file/ZIP project',
    step2Title: language === 'th' ? 'วิเคราะห์โค้ด' : 'Hybrid Analysis',
    step2Desc: language === 'th' ? 'ระบบวิเคราะห์ด้วย AI Model (GNN+BiLSTM) ร่วมกับ Pattern-Matching' : 'AI Model (GNN+BiLSTM) combined with Pattern-Matching engines',
    step3Title: language === 'th' ? 'รายงานผล' : 'Get Report',
    step3Desc: language === 'th' ? 'รับรายงานช่องโหว่พร้อมวิธีแก้ไข' : 'Receive vulnerability report with remediation',
    readyToScan: language === 'th' ? 'พร้อมตรวจสอบโค้ดของคุณแล้วหรือยัง?' : 'Ready to scan your code?',
  };

  const features = [
    { 
      icon: Zap, 
      title: language === 'th' ? 'Hybrid AI + Pattern Detection' : 'Hybrid AI + Pattern Detection',
      desc: language === 'th' 
        ? 'ผสมผสาน GNN+BiLSTM AI Model กับ Pattern-Matching (Semgrep/Bandit) เพื่อความแม่นยำสูงสุด' 
        : 'Combines GNN+BiLSTM AI Model with Pattern-Matching engines for maximum accuracy',
      color: 'from-purple-500 to-indigo-600'
    },
    { 
      icon: Gauge, 
      title: language === 'th' ? 'วิเคราะห์เร็ว' : 'Fast Analysis',
      desc: language === 'th' 
        ? 'สแกนโค้ดหลายพันบรรทัดภายในไม่กี่วินาที' 
        : 'Scan thousands of lines of code in seconds',
      color: 'from-emerald-500 to-teal-600'
    },
    { 
      icon: FileSearch, 
      title: language === 'th' ? 'รองรับ ZIP Project' : 'ZIP Project Support',
      desc: language === 'th' 
        ? 'อัพโหลดทั้งโปรเจคเพื่อสแกนทุกไฟล์พร้อมกัน' 
        : 'Upload entire project to scan all files at once',
      color: 'from-blue-500 to-cyan-600'
    },
    { 
      icon: Sparkles, 
      title: language === 'th' ? 'คำแนะนำการแก้ไข' : 'Remediation Advice',
      desc: language === 'th' 
        ? 'รับคำแนะนำวิธีแก้ไขช่องโหว่พร้อมตัวอย่างโค้ด' 
        : 'Get fix recommendations with code examples',
      color: 'from-amber-500 to-orange-600'
    },
  ];

  const owaspCategories = [
    { id: 'A01', name: 'Broken Access Control', icon: Unlock, color: 'text-red-500' },
    { id: 'A02', name: 'Cryptographic Failures', icon: Lock, color: 'text-orange-500' },
    { id: 'A03', name: 'Injection', icon: Terminal, color: 'text-red-600' },
    { id: 'A04', name: 'Insecure Design', icon: GitBranch, color: 'text-yellow-500' },
    { id: 'A05', name: 'Security Misconfiguration', icon: Database, color: 'text-amber-500' },
    { id: 'A06', name: 'Vulnerable Components', icon: FileWarning, color: 'text-orange-600' },
    { id: 'A07', name: 'Auth Failures', icon: ShieldAlert, color: 'text-red-500' },
    { id: 'A08', name: 'Data Integrity Failures', icon: FileCode, color: 'text-purple-500' },
    { id: 'A09', name: 'Logging Failures', icon: Activity, color: 'text-blue-500' },
    { id: 'A10', name: 'SSRF', icon: Globe, color: 'text-indigo-500' },
  ];

  const vulnerabilityTypes = [
    { name: 'SQL Injection', cwe: 'CWE-89' },
    { name: 'XSS (Cross-Site Scripting)', cwe: 'CWE-79' },
    { name: 'Command Injection', cwe: 'CWE-78' },
    { name: 'Path Traversal', cwe: 'CWE-22' },
    { name: 'Code Injection', cwe: 'CWE-94' },
    { name: 'SSRF', cwe: 'CWE-918' },
    { name: 'Insecure Deserialization', cwe: 'CWE-502' },
    { name: 'Hardcoded Credentials', cwe: 'CWE-798' },
    { name: 'Weak Cryptography', cwe: 'CWE-327' },
    { name: 'Sensitive Data Exposure', cwe: 'CWE-200' },
  ];

  return (
    <div className="min-h-screen p-6 lg:p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        
        {/* Hero Section */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center py-8"
        >
          <div className="inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600 shadow-2xl shadow-indigo-500/30 mb-6">
            <Shield className="w-10 h-10 text-white" />
          </div>
          <h1 className="text-3xl lg:text-4xl font-bold text-slate-900 dark:text-white mb-3">
            {t.title}
          </h1>
          <p className="text-lg text-slate-600 dark:text-slate-400 max-w-2xl mx-auto mb-8">
            {t.subtitle}
          </p>
          
          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/scan">
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="flex items-center justify-center gap-3 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white px-8 py-4 rounded-xl font-semibold text-lg shadow-xl shadow-indigo-500/25 transition-all"
              >
                <Scan className="w-5 h-5" />
                {t.startScan}
                <ArrowRight className="w-5 h-5" />
              </motion.button>
            </Link>
            <Link href="/scan?mode=zip">
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="flex items-center justify-center gap-3 bg-white dark:bg-slate-800 border-2 border-slate-200 dark:border-slate-700 hover:border-indigo-500 dark:hover:border-indigo-500 text-slate-900 dark:text-white px-8 py-4 rounded-xl font-semibold text-lg transition-all"
              >
                <FolderArchive className="w-5 h-5" />
                {t.uploadProject}
              </motion.button>
            </Link>
          </div>
        </motion.div>

        {/* Key Features */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4"
        >
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 + index * 0.05 }}
              className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200 dark:border-slate-700 p-6 shadow-lg hover:shadow-xl transition-shadow"
            >
              <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${feature.color} flex items-center justify-center shadow-lg mb-4`}>
                <feature.icon className="w-6 h-6 text-white" />
              </div>
              <h3 className="font-semibold text-slate-900 dark:text-white mb-2">{feature.title}</h3>
              <p className="text-sm text-slate-500 dark:text-slate-400">{feature.desc}</p>
            </motion.div>
          ))}
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* How It Works */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200 dark:border-slate-700 p-6 shadow-lg"
          >
            <h2 className="text-lg font-semibold text-slate-900 dark:text-white mb-6 flex items-center gap-2">
              <Zap className="w-5 h-5 text-amber-500" />
              {t.howItWorks}
            </h2>
            
            <div className="space-y-6">
              {[
                { step: 1, title: t.step1Title, desc: t.step1Desc, icon: FileCode, color: 'from-blue-500 to-blue-600' },
                { step: 2, title: t.step2Title, desc: t.step2Desc, icon: Cpu, color: 'from-purple-500 to-purple-600' },
                { step: 3, title: t.step3Title, desc: t.step3Desc, icon: ShieldCheck, color: 'from-emerald-500 to-emerald-600' },
              ].map((item) => (
                <div key={item.step} className="flex items-start gap-4">
                  <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${item.color} flex items-center justify-center shadow-lg flex-shrink-0`}>
                    <item.icon className="w-6 h-6 text-white" />
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-xs font-bold text-slate-400">STEP {item.step}</span>
                    </div>
                    <h3 className="font-semibold text-slate-900 dark:text-white">{item.title}</h3>
                    <p className="text-sm text-slate-500 dark:text-slate-400">{item.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Supported Languages */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.25 }}
            className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200 dark:border-slate-700 p-6 shadow-lg"
          >
            <h2 className="text-lg font-semibold text-slate-900 dark:text-white mb-6 flex items-center gap-2">
              <Code2 className="w-5 h-5 text-blue-500" />
              {t.supportedLang}
            </h2>
            
            <div className="grid grid-cols-1 gap-4">
              {[
                { lang: 'Python', icon: '🐍', ext: '.py', desc: language === 'th' ? 'Flask, Django, FastAPI และอื่นๆ' : 'Flask, Django, FastAPI and more', color: 'from-blue-500 to-yellow-500' },
                { lang: 'JavaScript', icon: '⚡', ext: '.js, .jsx', desc: language === 'th' ? 'Node.js, Express, React' : 'Node.js, Express, React', color: 'from-yellow-400 to-yellow-600' },
                { lang: 'TypeScript', icon: '📘', ext: '.ts, .tsx', desc: language === 'th' ? 'Type-safe JavaScript' : 'Type-safe JavaScript', color: 'from-blue-500 to-blue-700' },
              ].map((item) => (
                <div key={item.lang} className="flex items-center gap-4 p-4 rounded-xl bg-slate-50 dark:bg-slate-700/50 border border-slate-100 dark:border-slate-600">
                  <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${item.color} flex items-center justify-center text-2xl shadow-lg`}>
                    {item.icon}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <h3 className="font-semibold text-slate-900 dark:text-white">{item.lang}</h3>
                      <span className="text-xs px-2 py-0.5 rounded bg-slate-200 dark:bg-slate-600 text-slate-600 dark:text-slate-300">{item.ext}</span>
                    </div>
                    <p className="text-sm text-slate-500 dark:text-slate-400">{item.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        </div>

        {/* OWASP Top 10 Coverage */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl p-6 shadow-xl border border-indigo-100 dark:border-slate-700"
        >
          <h2 className="text-lg font-semibold text-slate-900 dark:text-white mb-6 flex items-center gap-2">
            <Shield className="w-5 h-5 text-indigo-500 dark:text-indigo-400" />
            {t.owaspCoverage}
          </h2>
          
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
            {owaspCategories.map((cat) => (
              <div key={cat.id} className="p-3 rounded-xl bg-white dark:bg-white/5 border border-indigo-200 dark:border-white/10 hover:bg-indigo-50 dark:hover:bg-white/10 transition-colors">
                <div className="flex items-center gap-2 mb-2">
                  <cat.icon className={`w-4 h-4 ${cat.color}`} />
                  <span className="text-xs font-bold text-indigo-600 dark:text-indigo-400">{cat.id}</span>
                </div>
                <p className="text-xs text-slate-600 dark:text-slate-300 line-clamp-2">{cat.name}</p>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Vulnerability Types Detection */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.35 }}
          className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200 dark:border-slate-700 p-6 shadow-lg"
        >
          <h2 className="text-lg font-semibold text-slate-900 dark:text-white mb-6 flex items-center gap-2">
            <ShieldAlert className="w-5 h-5 text-red-500" />
            {t.capabilities}
          </h2>
          
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
            {vulnerabilityTypes.map((vuln) => (
              <div key={vuln.cwe} className="p-3 rounded-xl bg-slate-50 dark:bg-slate-700/50 border border-slate-100 dark:border-slate-600 hover:border-red-300 dark:hover:border-red-700 transition-colors">
                <p className="text-sm font-medium text-slate-900 dark:text-white mb-1">{vuln.name}</p>
                <span className="text-xs px-2 py-0.5 rounded bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400">{vuln.cwe}</span>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Quick Start CTA */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="text-center py-8"
        >
          <p className="text-slate-600 dark:text-slate-400 mb-4">
            {t.readyToScan}
          </p>
          <Link href="/scan">
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="inline-flex items-center gap-2 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white px-8 py-4 rounded-xl font-semibold shadow-xl shadow-indigo-500/25 transition-all"
            >
              <Scan className="w-5 h-5" />
              {t.startScan}
              <ArrowRight className="w-5 h-5" />
            </motion.button>
          </Link>
        </motion.div>
      </div>
    </div>
  );
}

