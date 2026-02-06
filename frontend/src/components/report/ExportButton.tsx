/**
 * Export Button Component
 * Provides UI for exporting reports in various formats
 */

'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Download, Check, FileJson, FileSpreadsheet, FileCode, FileText } from 'lucide-react';
import { exportReport, exportFormats, type ExportFormat } from '@/lib/export';
import toast from 'react-hot-toast';

interface ExportButtonProps {
  report: any;
  className?: string;
}

export default function ExportButton({ report, className = '' }: ExportButtonProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [exporting, setExporting] = useState<ExportFormat | null>(null);

  const handleExport = async (format: ExportFormat) => {
    setExporting(format);
    
    try {
      await exportReport(report, format);
      toast.success(`Report exported as ${format.toUpperCase()}`);
      setTimeout(() => setIsOpen(false), 500);
    } catch (error) {
      toast.error('Failed to export report');
      console.error('Export error:', error);
    } finally {
      setTimeout(() => setExporting(null), 1000);
    }
  };

  const icons = {
    json: FileJson,
    csv: FileSpreadsheet,
    html: FileCode,
    md: FileText,
  };

  return (
    <div className={`relative ${className}`}>
      {/* Trigger Button */}
      <motion.button
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white font-medium transition-colors shadow-lg shadow-indigo-500/25"
      >
        <Download className="w-4 h-4" />
        Export Report
      </motion.button>

      {/* Dropdown Menu */}
      <AnimatePresence>
        {isOpen && (
          <>
            {/* Backdrop */}
            <div
              className="fixed inset-0 z-40"
              onClick={() => setIsOpen(false)}
            />

            {/* Menu */}
            <motion.div
              initial={{ opacity: 0, y: -10, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -10, scale: 0.95 }}
              transition={{ duration: 0.15 }}
              className="absolute right-0 mt-2 w-72 bg-white dark:bg-slate-800 rounded-xl shadow-2xl border border-slate-200 dark:border-slate-700 overflow-hidden z-50"
            >
              <div className="p-3">
                <p className="text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2 px-2">
                  Export Format
                </p>
                <div className="space-y-1">
                  {exportFormats.map((format) => {
                    const Icon = icons[format.value];
                    const isExporting = exporting === format.value;

                    return (
                      <motion.button
                        key={format.value}
                        whileHover={{ x: 4 }}
                        onClick={() => handleExport(format.value)}
                        disabled={isExporting}
                        className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors text-left disabled:opacity-50 disabled:cursor-not-allowed group"
                      >
                        <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-slate-100 dark:bg-slate-700 group-hover:bg-indigo-100 dark:group-hover:bg-indigo-900/30 transition-colors">
                          {isExporting ? (
                            <div className="w-4 h-4 border-2 border-indigo-600 border-t-transparent rounded-full animate-spin" />
                          ) : (
                            <Icon className="w-5 h-5 text-slate-600 dark:text-slate-400 group-hover:text-indigo-600 dark:group-hover:text-indigo-400" />
                          )}
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <span className="font-medium text-slate-900 dark:text-white">
                              {format.label}
                            </span>
                            {format.icon && (
                              <span className="text-base">{format.icon}</span>
                            )}
                          </div>
                          <p className="text-xs text-slate-500 dark:text-slate-400">
                            {format.description}
                          </p>
                        </div>
                        {isExporting && (
                          <Check className="w-4 h-4 text-green-600" />
                        )}
                      </motion.button>
                    );
                  })}
                </div>
              </div>

              {/* Footer Tip */}
              <div className="px-3 py-2 bg-slate-50 dark:bg-slate-900/50 border-t border-slate-200 dark:border-slate-700">
                <p className="text-xs text-slate-600 dark:text-slate-400">
                  💡 <strong>Tip:</strong> Use CSV for spreadsheet analysis
                </p>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}
