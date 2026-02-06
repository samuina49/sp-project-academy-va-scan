/**
 * Skeleton Loading Components
 * Provides placeholder UI while content loads
 */

import { cn } from '@/lib/utils';

interface SkeletonProps {
  className?: string;
  variant?: 'text' | 'circular' | 'rectangular' | 'rounded';
  width?: string | number;
  height?: string | number;
  animation?: 'pulse' | 'wave' | 'none';
}

export function Skeleton({
  className,
  variant = 'rectangular',
  width,
  height,
  animation = 'pulse',
}: SkeletonProps) {
  const baseClasses = 'bg-slate-200 dark:bg-slate-700';
  
  const variantClasses = {
    text: 'rounded h-4',
    circular: 'rounded-full',
    rectangular: 'rounded-lg',
    rounded: 'rounded-xl',
  };

  const animationClasses = {
    pulse: 'animate-pulse',
    wave: 'animate-shimmer bg-gradient-to-r from-slate-200 via-slate-300 to-slate-200 dark:from-slate-700 dark:via-slate-600 dark:to-slate-700',
    none: '',
  };

  const style: React.CSSProperties = {};
  if (width) style.width = typeof width === 'number' ? `${width}px` : width;
  if (height) style.height = typeof height === 'number' ? `${height}px` : height;

  return (
    <div
      className={cn(
        baseClasses,
        variantClasses[variant],
        animationClasses[animation],
        className
      )}
      style={style}
    />
  );
}

// Card Skeleton
export function CardSkeleton() {
  return (
    <div className="p-6 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800">
      <div className="flex items-start gap-4">
        <Skeleton variant="circular" width={48} height={48} />
        <div className="flex-1 space-y-3">
          <Skeleton variant="text" width="60%" />
          <Skeleton variant="text" width="80%" />
          <Skeleton variant="text" width="40%" />
        </div>
      </div>
    </div>
  );
}

// Table Skeleton
export function TableSkeleton({ rows = 5 }: { rows?: number }) {
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-4 pb-3 border-b border-slate-200 dark:border-slate-700">
        <Skeleton variant="text" width="20%" />
        <Skeleton variant="text" width="30%" />
        <Skeleton variant="text" width="25%" />
        <Skeleton variant="text" width="25%" />
      </div>
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="flex items-center gap-4">
          <Skeleton variant="text" width="20%" />
          <Skeleton variant="text" width="30%" />
          <Skeleton variant="text" width="25%" />
          <Skeleton variant="text" width="25%" />
        </div>
      ))}
    </div>
  );
}

// Dashboard Stats Skeleton
export function StatsSkeleton({ count = 4 }: { count?: number }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {Array.from({ length: count }).map((_, i) => (
        <div
          key={i}
          className="p-6 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800"
        >
          <div className="flex items-center justify-between mb-4">
            <Skeleton variant="circular" width={40} height={40} />
            <Skeleton variant="rectangular" width={60} height={24} />
          </div>
          <Skeleton variant="text" width="50%" className="mb-2" />
          <Skeleton variant="text" width="70%" height={32} />
        </div>
      ))}
    </div>
  );
}

// Code Editor Skeleton
export function CodeEditorSkeleton() {
  return (
    <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 p-6">
      <div className="space-y-3">
        {Array.from({ length: 15 }).map((_, i) => (
          <Skeleton
            key={i}
            variant="text"
            width={`${Math.random() * 40 + 60}%`}
            animation="wave"
          />
        ))}
      </div>
    </div>
  );
}

// Report Skeleton
export function ReportSkeleton() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="space-y-2">
          <Skeleton variant="text" width={200} height={32} />
          <Skeleton variant="text" width={300} />
        </div>
        <Skeleton variant="rounded" width={120} height={40} />
      </div>

      {/* Stats */}
      <StatsSkeleton />

      {/* Findings */}
      <div className="space-y-4">
        {Array.from({ length: 3 }).map((_, i) => (
          <div
            key={i}
            className="p-6 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800"
          >
            <div className="flex items-start gap-4 mb-4">
              <Skeleton variant="circular" width={32} height={32} />
              <div className="flex-1 space-y-2">
                <Skeleton variant="text" width="40%" />
                <Skeleton variant="text" width="80%" />
              </div>
            </div>
            <Skeleton variant="rectangular" width="100%" height={80} />
          </div>
        ))}
      </div>
    </div>
  );
}
