/**
 * Enhanced Tooltip Component
 * Accessible tooltips with keyboard support
 */

'use client';

import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { createPortal } from 'react-dom';
import { useA11yId } from '@/lib/accessibility';

interface TooltipProps {
  content: React.ReactNode;
  children: React.ReactNode;
  placement?: 'top' | 'bottom' | 'left' | 'right';
  delay?: number;
  className?: string;
}

export default function Tooltip({
  content,
  children,
  placement = 'top',
  delay = 300,
  className = '',
}: TooltipProps) {
  const [isVisible, setIsVisible] = useState(false);
  const [coords, setCoords] = useState({ x: 0, y: 0 });
  const timeoutRef = useRef<NodeJS.Timeout>();
  const targetRef = useRef<HTMLDivElement>(null);
  const tooltipId = useA11yId('tooltip');

  const calculatePosition = () => {
    if (!targetRef.current) return;

    const rect = targetRef.current.getBoundingClientRect();
    const tooltipWidth = 200; // Approximate
    const tooltipHeight = 40; // Approximate
    const gap = 8;

    let x = 0;
    let y = 0;

    switch (placement) {
      case 'top':
        x = rect.left + rect.width / 2 - tooltipWidth / 2;
        y = rect.top - tooltipHeight - gap;
        break;
      case 'bottom':
        x = rect.left + rect.width / 2 - tooltipWidth / 2;
        y = rect.bottom + gap;
        break;
      case 'left':
        x = rect.left - tooltipWidth - gap;
        y = rect.top + rect.height / 2 - tooltipHeight / 2;
        break;
      case 'right':
        x = rect.right + gap;
        y = rect.top + rect.height / 2 - tooltipHeight / 2;
        break;
    }

    setCoords({ x, y });
  };

  const handleMouseEnter = () => {
    calculatePosition();
    timeoutRef.current = setTimeout(() => {
      setIsVisible(true);
    }, delay);
  };

  const handleMouseLeave = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    setIsVisible(false);
  };

  const handleFocus = () => {
    calculatePosition();
    setIsVisible(true);
  };

  const handleBlur = () => {
    setIsVisible(false);
  };

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  const getAnimationVariants = () => {
    const distance = 8;
    const variants = {
      top: { y: distance },
      bottom: { y: -distance },
      left: { x: distance },
      right: { x: -distance },
    };

    return {
      hidden: {
        opacity: 0,
        ...variants[placement],
      },
      visible: {
        opacity: 1,
        x: 0,
        y: 0,
      },
    };
  };

  const tooltipElement = isVisible && typeof window !== 'undefined' ? (
    createPortal(
      <AnimatePresence>
        <motion.div
          initial="hidden"
          animate="visible"
          exit="hidden"
          variants={getAnimationVariants()}
          transition={{ duration: 0.15, ease: 'easeOut' }}
          role="tooltip"
          id={tooltipId}
          className={`fixed z-50 px-3 py-2 text-sm rounded-lg bg-slate-900 dark:bg-slate-700 text-white shadow-xl pointer-events-none max-w-xs ${className}`}
          style={{
            left: `${coords.x}px`,
            top: `${coords.y}px`,
          }}
        >
          {content}
          
          {/* Arrow */}
          <div
            className={`absolute w-2 h-2 bg-slate-900 dark:bg-slate-700 transform rotate-45 ${
              placement === 'top' ? 'bottom-[-4px] left-1/2 -translate-x-1/2' :
              placement === 'bottom' ? 'top-[-4px] left-1/2 -translate-x-1/2' :
              placement === 'left' ? 'right-[-4px] top-1/2 -translate-y-1/2' :
              'left-[-4px] top-1/2 -translate-y-1/2'
            }`}
          />
        </motion.div>
      </AnimatePresence>,
      document.body
    )
  ) : null;

  return (
    <>
      <div
        ref={targetRef}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        onFocus={handleFocus}
        onBlur={handleBlur}
        aria-describedby={isVisible ? tooltipId : undefined}
        className="inline-flex"
      >
        {children}
      </div>
      {tooltipElement}
    </>
  );
}

/**
 * Info Tooltip Icon
 */
import { HelpCircle } from 'lucide-react';

interface InfoTooltipProps {
  content: React.ReactNode;
  className?: string;
}

export function InfoTooltip({ content, className = '' }: InfoTooltipProps) {
  return (
    <Tooltip content={content} placement="top">
      <button
        className={`inline-flex items-center justify-center w-4 h-4 rounded-full text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-300 transition-colors ${className}`}
        aria-label="More information"
      >
        <HelpCircle className="w-full h-full" />
      </button>
    </Tooltip>
  );
}
