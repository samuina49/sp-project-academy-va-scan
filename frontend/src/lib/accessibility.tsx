'use client';

/**
 * Accessibility Utilities and Hooks
 * Enhances keyboard navigation, ARIA labels, and screen reader support
 */

import { useEffect, useRef, RefObject } from 'react';

/**
 * Focus Management Hook
 * Manages focus trapping for modals and dialogs
 */
export function useFocusTrap(isActive: boolean): RefObject<HTMLDivElement> {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isActive || !containerRef.current) return;

    const container = containerRef.current;
    const focusableElements = container.querySelectorAll<HTMLElement>(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );

    const firstElement = focusableElements[0];
    const lastElement = focusableElements[focusableElements.length - 1];

    // Focus first element
    firstElement?.focus();

    const handleTabKey = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return;

      if (e.shiftKey) {
        if (document.activeElement === firstElement) {
          e.preventDefault();
          lastElement?.focus();
        }
      } else {
        if (document.activeElement === lastElement) {
          e.preventDefault();
          firstElement?.focus();
        }
      }
    };

    document.addEventListener('keydown', handleTabKey);
    return () => document.removeEventListener('keydown', handleTabKey);
  }, [isActive]);

  return containerRef;
}

/**
 * Announce to Screen Readers
 */
export function announceToScreenReader(message: string, priority: 'polite' | 'assertive' = 'polite') {
  const announcement = document.createElement('div');
  announcement.setAttribute('role', 'status');
  announcement.setAttribute('aria-live', priority);
  announcement.setAttribute('aria-atomic', 'true');
  announcement.className = 'sr-only';
  announcement.textContent = message;

  document.body.appendChild(announcement);

  setTimeout(() => {
    document.body.removeChild(announcement);
  }, 1000);
}

/**
 * Skip to Content Link
 */
export function SkipToContent() {
  return (
    <a
      href="#main-content"
      className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:px-4 focus:py-2 focus:bg-indigo-600 focus:text-white focus:rounded-lg focus:shadow-lg"
    >
      Skip to main content
    </a>
  );
}

/**
 * Visually Hidden Component
 * Hides content visually but keeps it accessible to screen readers
 */
interface VisuallyHiddenProps {
  children: React.ReactNode;
  as?: keyof JSX.IntrinsicElements;
}

export function VisuallyHidden({ children, as: Component = 'span' }: VisuallyHiddenProps) {
  return (
    <Component className="sr-only">
      {children}
    </Component>
  );
}

/**
 * Generate Unique ID for A11y
 */
let idCounter = 0;

export function useA11yId(prefix = 'a11y'): string {
  const idRef = useRef<string>();

  if (!idRef.current) {
    idRef.current = `${prefix}-${++idCounter}`;
  }

  return idRef.current;
}

/**
 * Live Region Hook
 * Announces dynamic content changes to screen readers
 */
export function useLiveRegion(message: string, priority: 'polite' | 'assertive' = 'polite') {
  useEffect(() => {
    if (message) {
      announceToScreenReader(message, priority);
    }
  }, [message, priority]);
}

/**
 * Keyboard Navigation Helper
 */
export const KEY_CODES = {
  ENTER: 'Enter',
  SPACE: ' ',
  ESCAPE: 'Escape',
  ARROW_UP: 'ArrowUp',
  ARROW_DOWN: 'ArrowDown',
  ARROW_LEFT: 'ArrowLeft',
  ARROW_RIGHT: 'ArrowRight',
  TAB: 'Tab',
  HOME: 'Home',
  END: 'End',
} as const;

/**
 * Check if user prefers reduced motion
 */
export function usePrefersReducedMotion(): boolean {
  const prefersReducedMotion = useRef(false);

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    prefersReducedMotion.current = mediaQuery.matches;

    const handleChange = (e: MediaQueryListEvent) => {
      prefersReducedMotion.current = e.matches;
    };

    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  return prefersReducedMotion.current;
}

/**
 * Focus Visible Hook
 * Only show focus outlines for keyboard navigation
 */
export function useFocusVisible() {
  useEffect(() => {
    let hadKeyboardEvent = false;

    const detectKeyboard = () => {
      hadKeyboardEvent = true;
    };

    const detectMouse = () => {
      hadKeyboardEvent = false;
    };

    const updateFocusVisible = () => {
      if (hadKeyboardEvent) {
        document.body.classList.add('focus-visible');
      } else {
        document.body.classList.remove('focus-visible');
      }
    };

    document.addEventListener('keydown', detectKeyboard);
    document.addEventListener('mousedown', detectMouse);
    document.addEventListener('focusin', updateFocusVisible);

    return () => {
      document.removeEventListener('keydown', detectKeyboard);
      document.removeEventListener('mousedown', detectMouse);
      document.removeEventListener('focusin', updateFocusVisible);
    };
  }, []);
}

/**
 * ARIA Label Generators
 */
export const ariaLabels = {
  button: {
    close: 'Close',
    menu: 'Open menu',
    expand: 'Expand section',
    collapse: 'Collapse section',
    more: 'Show more options',
    submit: 'Submit form',
  },
  status: {
    loading: 'Loading',
    success: 'Success',
    error: 'Error',
    warning: 'Warning',
  },
  navigation: {
    main: 'Main navigation',
    breadcrumb: 'Breadcrumb navigation',
    pagination: 'Pagination navigation',
  },
};

/**
 * Roving Tab Index Hook
 * For keyboard navigation in lists/grids
 */
export function useRovingTabIndex(itemsCount: number) {
  const [focusedIndex, setFocusedIndex] = useState(0);

  const handleKeyDown = (e: React.KeyboardEvent, index: number) => {
    switch (e.key) {
      case KEY_CODES.ARROW_DOWN:
      case KEY_CODES.ARROW_RIGHT:
        e.preventDefault();
        setFocusedIndex((prev) => (prev + 1) % itemsCount);
        break;
      case KEY_CODES.ARROW_UP:
      case KEY_CODES.ARROW_LEFT:
        e.preventDefault();
        setFocusedIndex((prev) => (prev - 1 + itemsCount) % itemsCount);
        break;
      case KEY_CODES.HOME:
        e.preventDefault();
        setFocusedIndex(0);
        break;
      case KEY_CODES.END:
        e.preventDefault();
        setFocusedIndex(itemsCount - 1);
        break;
    }
  };

  return {
    focusedIndex,
    setFocusedIndex,
    handleKeyDown,
    getTabIndex: (index: number) => (index === focusedIndex ? 0 : -1),
  };
}

/**
 * Color Contrast Checker
 */
export function checkColorContrast(
  foreground: string,
  background: string
): { ratio: number; aa: boolean; aaa: boolean } {
  // Simplified contrast calculation
  // For production, use a library like polished or color
  const luminance = (color: string) => {
    // This is a simplified version
    const rgb = parseInt(color.slice(1), 16);
    const r = ((rgb >> 16) & 0xff) / 255;
    const g = ((rgb >> 8) & 0xff) / 255;
    const b = (rgb & 0xff) / 255;

    const sRGB = [r, g, b].map((val) => {
      if (val <= 0.03928) return val / 12.92;
      return Math.pow((val + 0.055) / 1.055, 2.4);
    });

    return 0.2126 * sRGB[0] + 0.7152 * sRGB[1] + 0.0722 * sRGB[2];
  };

  const l1 = luminance(foreground);
  const l2 = luminance(background);
  const ratio = (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);

  return {
    ratio,
    aa: ratio >= 4.5,
    aaa: ratio >= 7,
  };
}

import { useState } from 'react';
