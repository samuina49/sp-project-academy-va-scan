/**
 * Internationalization (i18n) utilities
 * Simple hook for language switching
 */

'use client';

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type Language = 'en' | 'th';

interface I18nStore {
  language: Language;
  setLanguage: (lang: Language) => void;
}

export interface Translations {
  navbar: {
    title: string;
    subtitle: string;
    scanner: string;
    report: string;
    systemOnline: string;
    offline: string;
  };
  footer: {
    projectName: string;
    copyright: string;
  };
}

const translations: Record<Language, Translations> = {
  en: {
    navbar: {
      title: 'VA Scanner',
      subtitle: 'AI-Based Vulnerability Scanner',
      scanner: 'Scanner',
      report: 'Report',
      systemOnline: 'System Online',
      offline: 'Offline',
    },
    footer: {
      projectName: 'AI-Based Vulnerability Scanner',
      copyright: '© 2024 SP Project Academy. All rights reserved.',
    },
  },
  th: {
    navbar: {
      title: 'VA Scanner',
      subtitle: 'ระบบตรวจสอบช่องโหว่ด้วย AI',
      scanner: 'สแกนเนอร์',
      report: 'รายงาน',
      systemOnline: 'ระบบออนไลน์',
      offline: 'ออฟไลน์',
    },
    footer: {
      projectName: 'ระบบตรวจสอบช่องโหว่ด้วย AI',
      copyright: '© 2024 SP Project Academy สงวนลิขสิทธิ์',
    },
  },
};

// Create zustand store with persistence
const useI18nStore = create<I18nStore>()(
  persist(
    (set) => ({
      language: 'en' as Language,
      setLanguage: (lang: Language) => set({ language: lang }),
    }),
    {
      name: 'va-scanner-language', // localStorage key
    }
  )
);

/**
 * Hook for accessing i18n state, actions and translations
 */
export function useI18n() {
  const store = useI18nStore();
  const t: Translations = translations[store.language];
  return { ...store, t };
}

/**
 * I18n Provider Component (placeholder for compatibility)
 * The actual state is managed by zustand store
 */
export function I18nProvider({ children }: { children: React.ReactNode }) {
  return children;
}

/**
 * Get translation helper
 * @param translations - Object with language keys
 */
export function t<T extends Record<Language, string>>(
  translationsObj: T,
  language: Language
): string {
  return translationsObj[language];
}

export default useI18n;
