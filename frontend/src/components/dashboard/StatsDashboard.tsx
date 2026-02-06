/**
 * Statistics Dashboard Component
 * Displays visual analytics and metrics
 */

'use client';

import { motion } from 'framer-motion';
import {
  TrendingUp,
  TrendingDown,
  Shield,
  AlertTriangle,
  CheckCircle,
  Activity,
  Clock,
  FileCode,
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface StatCardProps {
  title: string;
  value: string | number;
  change?: number;
  changeLabel?: string;
  icon: React.ComponentType<{ className?: string }>;
  color: string;
  trend?: 'up' | 'down' | 'neutral';
}

export function StatCard({
  title,
  value,
  change,
  changeLabel,
  icon: Icon,
  color,
  trend = 'neutral',
}: StatCardProps) {
  const trendIcon = trend === 'up' ? TrendingUp : trend === 'down' ? TrendingDown : Activity;
  const TrendIcon = trendIcon;

  const trendColor =
    trend === 'up'
      ? 'text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-900/20'
      : trend === 'down'
      ? 'text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20'
      : 'text-slate-600 dark:text-slate-400 bg-slate-50 dark:bg-slate-800/50';

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -4, transition: { duration: 0.2 } }}
      className="group relative overflow-hidden rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-6 shadow-sm hover:shadow-xl transition-all duration-300"
    >
      {/* Background Gradient */}
      <div className={cn('absolute inset-0 opacity-0 group-hover:opacity-5 transition-opacity', color)} />

      {/* Content */}
      <div className="relative">
        <div className="flex items-center justify-between mb-4">
          <div className={cn('w-12 h-12 rounded-xl flex items-center justify-center', color)}>
            <Icon className="w-6 h-6 text-white" />
          </div>
          {change !== undefined && (
            <div className={cn('flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium', trendColor)}>
              <TrendIcon className="w-3 h-3" />
              {Math.abs(change)}%
            </div>
          )}
        </div>

        <div className="space-y-1">
          <p className="text-sm font-medium text-slate-600 dark:text-slate-400">{title}</p>
          <p className="text-3xl font-bold text-slate-900 dark:text-white">{value}</p>
          {changeLabel && (
            <p className="text-xs text-slate-500 dark:text-slate-400">{changeLabel}</p>
          )}
        </div>
      </div>
    </motion.div>
  );
}

interface DashboardStatsProps {
  stats: {
    totalScans: number;
    vulnerabilitiesFound: number;
    criticalIssues: number;
    avgScanTime: number;
    scansToday?: number;
    fixRate?: number;
  };
}

export function DashboardStats({ stats }: DashboardStatsProps) {
  const statCards: StatCardProps[] = [
    {
      title: 'Total Scans',
      value: stats.totalScans.toLocaleString(),
      change: stats.scansToday ? 12 : undefined,
      changeLabel: stats.scansToday ? `${stats.scansToday} today` : undefined,
      icon: FileCode,
      color: 'bg-gradient-to-br from-blue-500 to-blue-600',
      trend: 'up',
    },
    {
      title: 'Vulnerabilities Found',
      value: stats.vulnerabilitiesFound.toLocaleString(),
      change: stats.fixRate ? -15 : undefined,
      changeLabel: stats.fixRate ? `${stats.fixRate}% fixed` : undefined,
      icon: AlertTriangle,
      color: 'bg-gradient-to-br from-amber-500 to-orange-600',
      trend: 'down',
    },
    {
      title: 'Critical Issues',
      value: stats.criticalIssues,
      change: -8,
      changeLabel: 'vs last week',
      icon: Shield,
      color: 'bg-gradient-to-br from-red-500 to-red-600',
      trend: 'down',
    },
    {
      title: 'Avg Scan Time',
      value: `${stats.avgScanTime}s`,
      change: 5,
      changeLabel: 'faster than avg',
      icon: Clock,
      color: 'bg-gradient-to-br from-green-500 to-emerald-600',
      trend: 'up',
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {statCards.map((stat, index) => (
        <StatCard key={index} {...stat} />
      ))}
    </div>
  );
}

// Progress Ring Component
interface ProgressRingProps {
  percentage: number;
  size?: number;
  strokeWidth?: number;
  color?: string;
}

export function ProgressRing({
  percentage,
  size = 120,
  strokeWidth = 8,
  color = '#6366f1',
}: ProgressRingProps) {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (percentage / 100) * circumference;

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg width={size} height={size} className="transform -rotate-90">
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="currentColor"
          strokeWidth={strokeWidth}
          fill="none"
          className="text-slate-200 dark:text-slate-700"
        />
        {/* Progress circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={color}
          strokeWidth={strokeWidth}
          fill="none"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          className="transition-all duration-500 ease-out"
        />
      </svg>
      <div className="absolute flex flex-col items-center justify-center">
        <span className="text-2xl font-bold text-slate-900 dark:text-white">
          {percentage}%
        </span>
      </div>
    </div>
  );
}

// Security Score Component
interface SecurityScoreProps {
  score: number;
  maxScore?: number;
}

export function SecurityScore({ score, maxScore = 100 }: SecurityScoreProps) {
  const percentage = (score / maxScore) * 100;
  const getScoreColor = () => {
    if (percentage >= 80) return '#10b981'; // green
    if (percentage >= 60) return '#f59e0b'; // amber
    return '#ef4444'; // red
  };

  const getScoreLabel = () => {
    if (percentage >= 80) return 'Excellent';
    if (percentage >= 60) return 'Good';
    if (percentage >= 40) return 'Fair';
    return 'Poor';
  };

  const ScoreIcon = percentage >= 80 ? CheckCircle : percentage >= 60 ? Shield : AlertTriangle;

  return (
    <motion.div
      initial={{ scale: 0.9, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="flex flex-col items-center p-8 rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800"
    >
      <ProgressRing percentage={percentage} color={getScoreColor()} />
      <div className="mt-6 text-center">
        <div className="flex items-center justify-center gap-2 mb-2">
          <ScoreIcon className="w-5 h-5" style={{ color: getScoreColor() }} />
          <h3 className="text-xl font-bold text-slate-900 dark:text-white">
            {getScoreLabel()}
          </h3>
        </div>
        <p className="text-sm text-slate-600 dark:text-slate-400">
          Security Score: {score} / {maxScore}
        </p>
      </div>
    </motion.div>
  );
}
