'use client';

import React from 'react';
import Link from 'next/link';
import { useI18n } from '@/lib/i18n';

export default function Footer() {
  const { t } = useI18n();
  
  return (
    <footer className="bg-white dark:bg-slate-900 border-t border-border dark:border-slate-700 py-8 mt-16 transition-colors duration-300">
      <div className="mx-auto max-w-6xl px-6">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          {/* Left: Project Info */}
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 flex items-center justify-center bg-primary-100 dark:bg-primary-900/50 rounded-lg">
              <svg className="w-4 h-4 text-primary-600 dark:text-primary-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z" />
              </svg>
            </div>
            <div>
              <p className="text-sm font-medium text-text-primary dark:text-white">
                {t.footer.projectName}
              </p>
              <p className="text-xs text-text-muted dark:text-slate-500">
                {t.footer.copyright}
              </p>
            </div>
          </div>

          {/* Right: Links */}
          <div className="flex items-center gap-6 text-sm text-text-secondary dark:text-slate-400">
            <Link href="/" className="hover:text-primary-600 dark:hover:text-primary-400 transition-colors">
              {t.navbar.scanner}
            </Link>
            <Link href="/report" className="hover:text-primary-600 dark:hover:text-primary-400 transition-colors">
              {t.navbar.report}
            </Link>
            <a 
              href="https://github.com" 
              target="_blank" 
              rel="noopener noreferrer"
              className="hover:text-primary-600 dark:hover:text-primary-400 transition-colors flex items-center gap-1"
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.17 6.839 9.49.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.604-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.464-1.11-1.464-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.831.092-.646.35-1.086.636-1.336-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0112 6.836c.85.004 1.705.115 2.504.337 1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48C19.138 20.167 22 16.418 22 12c0-5.523-4.477-10-10-10z" />
              </svg>
              GitHub
            </a>
          </div>
        </div>

        {/* Tech Stack Badge */}
        <div className="mt-6 pt-6 border-t border-gray-100 dark:border-slate-700 flex flex-wrap justify-center gap-3">
          {['Next.js', 'FastAPI', 'PyTorch', 'GNN+LSTM'].map((tech) => (
            <span 
              key={tech}
              className="px-3 py-1 bg-gray-50 dark:bg-slate-800 text-text-muted dark:text-slate-400 text-xs rounded-full"
            >
              {tech}
            </span>
          ))}
        </div>
      </div>
    </footer>
  );
}
