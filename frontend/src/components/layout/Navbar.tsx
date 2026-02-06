'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion } from 'framer-motion';
import { useState, useEffect } from 'react';
import { checkHealth } from '@/lib/api';
import ThemeToggle from '../ThemeToggle';
import LanguageToggle from '../LanguageToggle';
import { useI18n } from '@/lib/i18n';

export default function Navbar() {
  const pathname = usePathname();
  const [isOnline, setIsOnline] = useState(false);
  const { t } = useI18n();

  useEffect(() => {
    const checkSystem = async () => {
      try {
        const health = await checkHealth();
        setIsOnline(health.status === 'healthy');
      } catch {
        setIsOnline(false);
      }
    };
    checkSystem();
    const interval = setInterval(checkSystem, 30000);
    return () => clearInterval(interval);
  }, []);

  const navItems = [
    { href: '/', label: t.navbar.scanner, icon: '🔍' },
    { href: '/report', label: t.navbar.report, icon: '📊' },
  ];

  return (
    <motion.nav 
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="fixed top-0 left-0 right-0 z-50 bg-white/80 dark:bg-slate-900/90 backdrop-blur-md border-b border-border dark:border-slate-700 transition-colors duration-300"
    >
      <div className="mx-auto max-w-6xl px-6 h-16 flex items-center justify-between">
        {/* Logo */}
        <Link href="/" className="group flex items-center gap-3">
          <motion.div 
            whileHover={{ scale: 1.05 }}
            className="relative w-10 h-10 flex items-center justify-center bg-gradient-to-br from-primary-500 to-primary-700 rounded-xl shadow-soft dark:shadow-glow"
          >
            <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z" />
            </svg>
          </motion.div>
          <div>
            <h1 className="text-lg font-semibold text-text-primary dark:text-white">
              {t.navbar.title}
            </h1>
            <p className="text-xs text-text-muted dark:text-slate-400">{t.navbar.subtitle}</p>
          </div>
        </Link>

        {/* Navigation Links */}
        <div className="flex items-center gap-1">
          {navItems.map((item) => (
            <Link key={item.href} href={item.href}>
              <motion.div
                whileHover={{ y: -2 }}
                whileTap={{ scale: 0.98 }}
                className={`relative px-4 py-2 rounded-lg font-medium text-sm transition-colors ${
                  pathname === item.href
                    ? 'text-primary-600 dark:text-primary-400 bg-primary-50 dark:bg-primary-900/30'
                    : 'text-text-secondary dark:text-slate-300 hover:text-text-primary dark:hover:text-white hover:bg-gray-50 dark:hover:bg-slate-800'
                }`}
              >
                <span className="mr-1.5">{item.icon}</span>
                {item.label}
                {pathname === item.href && (
                  <motion.div
                    layoutId="activeTab"
                    className="absolute bottom-0 left-2 right-2 h-0.5 bg-primary-500 rounded-full"
                    transition={{ type: "spring", stiffness: 400, damping: 30 }}
                  />
                )}
              </motion.div>
            </Link>
          ))}
        </div>

        {/* Status Indicator & Theme Toggle */}
        <div className="flex items-center gap-3">
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex items-center gap-2 px-3 py-1.5 bg-gray-50 dark:bg-slate-800 rounded-full"
          >
            <motion.div
              animate={{ scale: isOnline ? [1, 1.2, 1] : 1 }}
              transition={{ duration: 2, repeat: Infinity }}
              className={`w-2 h-2 rounded-full ${isOnline ? 'bg-success' : 'bg-danger'}`}
            />
            <span className="text-xs font-medium text-text-secondary dark:text-slate-300">
              {isOnline ? t.navbar.systemOnline : t.navbar.offline}
            </span>
          </motion.div>
          
          {/* Language Toggle */}
          <LanguageToggle />
          
          {/* Theme Toggle */}
          <ThemeToggle />
        </div>
      </div>
    </motion.nav>
  );
}
