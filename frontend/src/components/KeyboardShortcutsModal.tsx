'use client';

import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Keyboard } from 'lucide-react';
import { formatShortcut, useGlobalShortcuts } from '@/hooks/useKeyboardShortcuts';

export default function KeyboardShortcutsModal() {
  const [isOpen, setIsOpen] = useState(false);
  const shortcuts = useGlobalShortcuts();

  useEffect(() => {
    const handleOpen = () => setIsOpen(true);
    const handleClose = () => setIsOpen(false);

    window.addEventListener('open-shortcuts-modal', handleOpen);
    window.addEventListener('close-modals', handleClose);

    return () => {
      window.removeEventListener('open-shortcuts-modal', handleOpen);
      window.removeEventListener('close-modals', handleClose);
    };
  }, []);

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
        {/* Backdrop */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="absolute inset-0 bg-black/50 backdrop-blur-sm"
          onClick={() => setIsOpen(false)}
        />

        {/* Modal */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 20 }}
          className="relative bg-white dark:bg-slate-900 rounded-2xl shadow-2xl max-w-2xl w-full max-h-[80vh] overflow-hidden border border-slate-200 dark:border-slate-700"
        >
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-slate-200 dark:border-slate-700">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
                <Keyboard className="w-5 h-5 text-white" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-slate-900 dark:text-white">
                  Keyboard Shortcuts
                </h2>
                <p className="text-sm text-slate-500 dark:text-slate-400">
                  Navigate faster with these shortcuts
                </p>
              </div>
            </div>
            <button
              onClick={() => setIsOpen(false)}
              className="p-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
            >
              <X className="w-5 h-5 text-slate-500" />
            </button>
          </div>

          {/* Content */}
          <div className="p-6 overflow-y-auto max-h-[calc(80vh-120px)]">
            <div className="space-y-3">
              {shortcuts.map((shortcut, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-4 rounded-xl bg-slate-50 dark:bg-slate-800/50 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
                >
                  <span className="text-slate-700 dark:text-slate-300">
                    {shortcut.description}
                  </span>
                  <kbd className="px-3 py-1.5 rounded-lg bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 text-sm font-mono text-slate-700 dark:text-slate-300 shadow-sm">
                    {formatShortcut(shortcut)}
                  </kbd>
                </div>
              ))}
            </div>

            {/* Additional Tips */}
            <div className="mt-6 p-4 rounded-xl bg-indigo-50 dark:bg-indigo-900/20 border border-indigo-200 dark:border-indigo-800">
              <p className="text-sm text-indigo-900 dark:text-indigo-300">
                <strong>Pro Tip:</strong> Press <kbd className="px-2 py-0.5 rounded bg-white dark:bg-slate-900 border border-indigo-300 dark:border-indigo-700 text-xs font-mono">?</kbd> anytime to view this dialog
              </p>
            </div>
          </div>

          {/* Footer */}
          <div className="p-4 border-t border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50">
            <button
              onClick={() => setIsOpen(false)}
              className="w-full py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white font-medium transition-colors"
            >
              Got it!
            </button>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
