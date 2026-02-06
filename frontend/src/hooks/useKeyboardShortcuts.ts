/**
 * Keyboard Shortcuts Hook
 * Provides system-wide keyboard navigation and shortcuts
 */

import { useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';

interface ShortcutConfig {
  key: string;
  ctrlKey?: boolean;
  shiftKey?: boolean;
  altKey?: boolean;
  metaKey?: boolean;
  description: string;
  action: () => void;
  disabled?: boolean;
}

export const useKeyboardShortcuts = (shortcuts: ShortcutConfig[]) => {
  const handleKeyPress = useCallback(
    (event: KeyboardEvent) => {
      const activeElement = document.activeElement;
      const isInputField =
        activeElement?.tagName === 'INPUT' ||
        activeElement?.tagName === 'TEXTAREA' ||
        activeElement?.getAttribute('contenteditable') === 'true';

      for (const shortcut of shortcuts) {
        if (shortcut.disabled) continue;

        const keyMatch = event.key.toLowerCase() === shortcut.key.toLowerCase();
        const ctrlMatch = shortcut.ctrlKey === undefined || event.ctrlKey === shortcut.ctrlKey;
        const shiftMatch = shortcut.shiftKey === undefined || event.shiftKey === shortcut.shiftKey;
        const altMatch = shortcut.altKey === undefined || event.altKey === shortcut.altKey;
        const metaMatch = shortcut.metaKey === undefined || event.metaKey === shortcut.metaKey;

        if (keyMatch && ctrlMatch && shiftMatch && altMatch && metaMatch) {
          // Allow specific shortcuts in input fields (like Ctrl+S)
          if (isInputField && !shortcut.ctrlKey && !shortcut.metaKey) {
            continue;
          }

          event.preventDefault();
          shortcut.action();
          break;
        }
      }
    },
    [shortcuts]
  );

  useEffect(() => {
    document.addEventListener('keydown', handleKeyPress);
    return () => document.removeEventListener('keydown', handleKeyPress);
  }, [handleKeyPress]);
};

/**
 * Global keyboard shortcuts for navigation
 */
export const useGlobalShortcuts = () => {
  const router = useRouter();

  const shortcuts: ShortcutConfig[] = [
    {
      key: 'h',
      description: 'Go to home',
      action: () => router.push('/'),
    },
    {
      key: 's',
      ctrlKey: true,
      description: 'Start new scan',
      action: () => router.push('/scan'),
    },
    {
      key: 'r',
      ctrlKey: true,
      description: 'View reports',
      action: () => router.push('/report'),
    },
    {
      key: '/',
      description: 'Show keyboard shortcuts',
      action: () => {
        // This will be handled by a modal
        const event = new CustomEvent('open-shortcuts-modal');
        window.dispatchEvent(event);
      },
    },
    {
      key: 'Escape',
      description: 'Close modals',
      action: () => {
        const event = new CustomEvent('close-modals');
        window.dispatchEvent(event);
      },
    },
  ];

  useKeyboardShortcuts(shortcuts);

  return shortcuts;
};

/**
 * Format shortcut for display
 */
export const formatShortcut = (shortcut: ShortcutConfig): string => {
  const keys: string[] = [];
  
  if (shortcut.ctrlKey) keys.push('Ctrl');
  if (shortcut.metaKey) keys.push('⌘');
  if (shortcut.shiftKey) keys.push('Shift');
  if (shortcut.altKey) keys.push('Alt');
  
  keys.push(shortcut.key.toUpperCase());
  
  return keys.join(' + ');
};
