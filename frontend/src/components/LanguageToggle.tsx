'use client';

import { useI18n, Language } from '@/lib/i18n';
import { motion } from 'framer-motion';

export default function LanguageToggle() {
  const { language, setLanguage } = useI18n();

  const toggleLanguage = () => {
    setLanguage(language === 'en' ? 'th' : 'en');
  };

  return (
    <motion.button
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      onClick={toggleLanguage}
      className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-100 dark:bg-slate-700 hover:bg-gray-200 dark:hover:bg-slate-600 rounded-full transition-colors"
      title={language === 'en' ? 'Switch to Thai' : 'เปลี่ยนเป็นภาษาอังกฤษ'}
    >
      <span className="text-base">{language === 'en' ? '🇺🇸' : '🇹🇭'}</span>
      <span className="text-xs font-medium text-text-secondary dark:text-slate-300 uppercase">
        {language}
      </span>
    </motion.button>
  );
}
