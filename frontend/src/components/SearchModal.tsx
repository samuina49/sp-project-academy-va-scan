/**
 * Enhanced Search Component
 * Fuzzy search with keyboard shortcuts and highlights
 */

'use client';

import { useState, useEffect, useRef, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, X, ArrowUp, ArrowDown, Loader2 } from 'lucide-react';
import { useKeyboardShortcuts } from '@/hooks/useKeyboardShortcuts';
import { useFocusTrap, announceToScreenReader } from '@/lib/accessibility';

interface SearchResult<T = any> {
  item: T;
  matches: Array<{ key: string; indices: number[][] }>;
  score: number;
}

interface SearchProps<T = any> {
  data: T[];
  onSelect?: (item: T) => void;
  placeholder?: string;
  searchKeys?: string[];
  minSearchLength?: number;
  maxResults?: number;
  isOpen: boolean;
  onClose: () => void;
}

export default function SearchModal<T extends Record<string, any>>({
  data,
  onSelect,
  placeholder = 'Search...',
  searchKeys = ['name', 'title'],
  minSearchLength = 2,
  maxResults = 10,
  isOpen,
  onClose,
}: SearchProps<T>) {
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [isSearching, setIsSearching] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const modalRef = useFocusTrap(isOpen);

  // Simple fuzzy search implementation
  const results = useMemo(() => {
    if (query.length < minSearchLength) return [];

    setIsSearching(true);
    const lowerQuery = query.toLowerCase();

    const filtered = data
      .map((item) => {
        let score = 0;
        const matches: Array<{ key: string; indices: number[][] }> = [];

        searchKeys.forEach((key) => {
          const value = String(item[key] || '').toLowerCase();
          const index = value.indexOf(lowerQuery);

          if (index !== -1) {
            // Exact match bonus
            if (value === lowerQuery) score += 100;
            // Start of string bonus
            else if (index === 0) score += 50;
            // Contains match
            else score += 10;

            matches.push({
              key,
              indices: [[index, index + lowerQuery.length]],
            });
          }
        });

        return { item, matches, score };
      })
      .filter((result) => result.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, maxResults);

    setIsSearching(false);
    return filtered;
  }, [query, data, searchKeys, minSearchLength, maxResults]);

  // Reset selected index when results change
  useEffect(() => {
    setSelectedIndex(0);
  }, [results]);

  // Handle keyboard navigation
  useKeyboardShortcuts([
    {
      key: 'ArrowDown',
      description: 'Next result',
      action: () => {
        setSelectedIndex((prev) => (prev + 1) % results.length);
      },
      disabled: !isOpen || results.length === 0,
    },
    {
      key: 'ArrowUp',
      description: 'Previous result',
      action: () => {
        setSelectedIndex((prev) => (prev - 1 + results.length) % results.length);
      },
      disabled: !isOpen || results.length === 0,
    },
    {
      key: 'Enter',
      description: 'Select result',
      action: () => {
        if (results[selectedIndex]) {
          handleSelect(results[selectedIndex].item);
        }
      },
      disabled: !isOpen || results.length === 0,
    },
    {
      key: 'Escape',
      description: 'Close search',
      action: onClose,
      disabled: !isOpen,
    },
  ]);

  // Focus input when modal opens
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
      announceToScreenReader('Search opened');
    }
  }, [isOpen]);

  // Announce results to screen reader
  useEffect(() => {
    if (results.length > 0) {
      announceToScreenReader(`${results.length} results found`);
    } else if (query.length >= minSearchLength) {
      announceToScreenReader('No results found');
    }
  }, [results.length, query, minSearchLength]);

  const handleSelect = (item: T) => {
    onSelect?.(item);
    onClose();
    setQuery('');
  };

  const highlightMatch = (text: string, indices: number[][]): React.ReactNode => {
    if (!indices || indices.length === 0) return text;

    const parts: React.ReactNode[] = [];
    let lastIndex = 0;

    indices.forEach(([start, end]) => {
      // Text before match
      if (start > lastIndex) {
        parts.push(text.substring(lastIndex, start));
      }
      // Highlighted match
      parts.push(
        <mark key={start} className="bg-yellow-200 dark:bg-yellow-700 text-slate-900 dark:text-white">
          {text.substring(start, end)}
        </mark>
      );
      lastIndex = end;
    });

    // Remaining text
    if (lastIndex < text.length) {
      parts.push(text.substring(lastIndex));
    }

    return <>{parts}</>;
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-start justify-center pt-20 px-4">
        {/* Backdrop */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="absolute inset-0 bg-black/50 backdrop-blur-sm"
          onClick={onClose}
        />

        {/* Search Modal */}
        <motion.div
          ref={modalRef}
          initial={{ opacity: 0, scale: 0.95, y: -20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: -20 }}
          className="relative w-full max-w-2xl bg-white dark:bg-slate-900 rounded-2xl shadow-2xl border border-slate-200 dark:border-slate-700 overflow-hidden"
        >
          {/* Search Input */}
          <div className="flex items-center gap-3 p-4 border-b border-slate-200 dark:border-slate-700">
            <Search className="w-5 h-5 text-slate-400" />
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={placeholder}
              className="flex-1 bg-transparent text-slate-900 dark:text-white placeholder:text-slate-400 outline-none text-lg"
              aria-label="Search"
              aria-autocomplete="list"
              aria-controls="search-results"
              aria-expanded={results.length > 0}
              aria-activedescendant={results[selectedIndex] ? `result-${selectedIndex}` : undefined}
            />
            {isSearching && <Loader2 className="w-5 h-5 text-indigo-500 animate-spin" />}
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
              aria-label="Close search"
            >
              <X className="w-5 h-5 text-slate-500" />
            </button>
          </div>

          {/* Results */}
          <div
            id="search-results"
            role="listbox"
            className="max-h-[400px] overflow-y-auto"
          >
            {query.length < minSearchLength && (
              <div className="p-8 text-center text-slate-500 dark:text-slate-400">
                <Search className="w-12 h-12 mx-auto mb-3 opacity-30" />
                <p>Type at least {minSearchLength} characters to search</p>
              </div>
            )}

            {query.length >= minSearchLength && results.length === 0 && !isSearching && (
              <div className="p-8 text-center text-slate-500 dark:text-slate-400">
                <p>No results found for "{query}"</p>
              </div>
            )}

            {results.map((result, index) => {
              const isSelected = index === selectedIndex;
              const primaryMatch = result.matches[0];

              return (
                <motion.button
                  key={index}
                  id={`result-${index}`}
                  role="option"
                  aria-selected={isSelected}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: index * 0.05 }}
                  onClick={() => handleSelect(result.item)}
                  onMouseEnter={() => setSelectedIndex(index)}
                  className={`w-full flex items-center gap-4 p-4 text-left transition-colors ${
                    isSelected
                      ? 'bg-indigo-50 dark:bg-indigo-900/20 border-l-4 border-indigo-500'
                      : 'hover:bg-slate-50 dark:hover:bg-slate-800/50 border-l-4 border-transparent'
                  }`}
                >
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-slate-900 dark:text-white truncate">
                      {primaryMatch
                        ? highlightMatch(
                            String(result.item[primaryMatch.key]),
                            primaryMatch.indices
                          )
                        : String(result.item[searchKeys[0]])}
                    </div>
                    {searchKeys[1] && result.item[searchKeys[1]] && (
                      <div className="text-sm text-slate-500 dark:text-slate-400 truncate">
                        {String(result.item[searchKeys[1]])}
                      </div>
                    )}
                  </div>
                  {isSelected && (
                    <div className="text-xs text-indigo-600 dark:text-indigo-400 flex items-center gap-1">
                      <kbd className="px-2 py-1 rounded bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600">
                        Enter
                      </kbd>
                    </div>
                  )}
                </motion.button>
              );
            })}
          </div>

          {/* Footer */}
          {results.length > 0 && (
            <div className="flex items-center justify-between px-4 py-2 bg-slate-50 dark:bg-slate-800/50 border-t border-slate-200 dark:border-slate-700 text-xs text-slate-600 dark:text-slate-400">
              <div className="flex items-center gap-4">
                <span className="flex items-center gap-1">
                  <kbd className="px-1.5 py-0.5 rounded bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600">
                    <ArrowUp className="w-3 h-3" />
                  </kbd>
                  <kbd className="px-1.5 py-0.5 rounded bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600">
                    <ArrowDown className="w-3 h-3" />
                  </kbd>
                  Navigate
                </span>
                <span className="flex items-center gap-1">
                  <kbd className="px-1.5 py-0.5 rounded bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600">
                    Enter
                  </kbd>
                  Select
                </span>
                <span className="flex items-center gap-1">
                  <kbd className="px-1.5 py-0.5 rounded bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600">
                    Esc
                  </kbd>
                  Close
                </span>
              </div>
              <span>{results.length} results</span>
            </div>
          )}
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
