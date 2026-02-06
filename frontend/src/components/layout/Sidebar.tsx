'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState, useEffect } from 'react';
import { 
  LayoutDashboard, 
  ScanSearch, 
  FileText, 
  Settings, 
  Shield,
  Menu,
  X,
  Activity,
  Moon,
  Sun,
  ChevronRight,
  Zap
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useTheme } from '@/components/ThemeProvider';
import { useI18n } from '@/lib/i18n';
import { checkHealth } from '@/lib/api';

export default function Sidebar() {
  const pathname = usePathname();
  const { theme, setTheme } = useTheme();
  const { language, setLanguage } = useI18n();
  const [isOpen, setIsOpen] = useState(false);
  const [isOnline, setIsOnline] = useState(false);
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    const checkSystem = async () => {
      try {
        const health = await checkHealth();
        setIsOnline(health.status === 'healthy');
      } catch {
        setIsOnline(false);
      }
    };
    checkSystem();
    const interval = setInterval(checkSystem, 30000);
    return () => clearInterval(interval);
  }, []);

  const navItems = [
    { href: '/', label: 'Dashboard', icon: LayoutDashboard, description: 'Overview & Stats' },
    { href: '/scan', label: 'New Scan', icon: ScanSearch, description: 'Analyze Code' },
    { href: '/report', label: 'Reports', icon: FileText, description: 'View Results' },
    { href: '/settings', label: 'Settings', icon: Settings, description: 'Configuration' },
  ];

  return (
    <>
      {/* Mobile Toggle */}
      <button
        className="lg:hidden fixed top-4 left-4 z-50 p-2.5 bg-gradient-to-br from-primary-600 to-primary-700 text-white rounded-xl shadow-lg"
        onClick={() => setIsOpen(!isOpen)}
      >
        {isOpen ? <X size={20} /> : <Menu size={20} />}
      </button>

      {/* Overlay for mobile */}
      {isOpen && (
        <div 
          className="lg:hidden fixed inset-0 bg-black/50 z-30 backdrop-blur-sm"
          onClick={() => setIsOpen(false)}
        />
      )}

      <aside
        className={cn(
          "fixed top-0 left-0 z-40 h-screen bg-gradient-to-b from-slate-900 via-slate-900 to-slate-950 text-white transition-all duration-300 lg:translate-x-0 border-r border-slate-800/50",
          isOpen ? "translate-x-0" : "-translate-x-full",
          collapsed ? "w-20" : "w-64"
        )}
      >
        <div className="flex flex-col h-full">
          {/* Logo Section */}
          <div className="h-16 flex items-center justify-between px-4 border-b border-slate-800/50">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center shadow-lg shadow-primary-500/20">
                <Shield className="w-5 h-5 text-white" />
              </div>
              {!collapsed && (
                <div>
                  <h1 className="font-bold text-sm tracking-tight">VulnScanner</h1>
                  <p className="text-[10px] text-slate-500">AI-Powered Security</p>
                </div>
              )}
            </div>
            <button
              onClick={() => setCollapsed(!collapsed)}
              className="hidden lg:flex p-1.5 rounded-lg hover:bg-slate-800 transition-colors"
            >
              <ChevronRight className={cn("w-4 h-4 text-slate-500 transition-transform", collapsed && "rotate-180")} />
            </button>
          </div>

          {/* System Status */}
          <div className={cn("px-4 py-3 border-b border-slate-800/50", collapsed && "px-2")}>
            <div className={cn("flex items-center gap-2", collapsed && "justify-center")}>
              <div className={cn(
                "w-2 h-2 rounded-full",
                isOnline ? "bg-emerald-500 shadow-lg shadow-emerald-500/50 animate-pulse" : "bg-red-500 shadow-lg shadow-red-500/50"
              )} />
              {!collapsed && (
                <span className="text-xs text-slate-400">
                  {isOnline ? 'Backend Online' : 'Connecting...'}
                </span>
              )}
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 py-4 px-3 space-y-1 overflow-y-auto">
            {!collapsed && (
              <div className="px-3 mb-3 text-[10px] font-semibold text-slate-600 uppercase tracking-widest">
                Main Menu
              </div>
            )}
            
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href || (item.href !== '/' && pathname?.startsWith(item.href));
              
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  onClick={() => setIsOpen(false)}
                  className={cn(
                    "flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-200 group relative",
                    collapsed && "justify-center px-2",
                    isActive 
                      ? "bg-gradient-to-r from-primary-600/20 to-primary-600/5 text-primary-400" 
                      : "text-slate-400 hover:text-white hover:bg-slate-800/50"
                  )}
                >
                  {isActive && (
                    <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-6 bg-primary-500 rounded-r-full" />
                  )}
                  <Icon size={20} className={cn(
                    "shrink-0 transition-colors",
                    isActive ? "text-primary-400" : "text-slate-500 group-hover:text-primary-400"
                  )} />
                  {!collapsed && (
                    <div className="flex-1 min-w-0">
                      <span className="font-medium text-sm block">{item.label}</span>
                      <span className="text-[10px] text-slate-600 truncate block">{item.description}</span>
                    </div>
                  )}
                </Link>
              );
            })}
          </nav>

          {/* Language Toggle */}
          <div className={cn("px-4 py-2 border-t border-slate-800/50", collapsed && "px-2")}>
            <button
              onClick={() => setLanguage(language === 'en' ? 'th' : 'en')}
              className={cn(
                "w-full flex items-center gap-3 px-3 py-2.5 rounded-xl bg-slate-800/50 hover:bg-slate-800 transition-colors",
                collapsed && "justify-center px-2"
              )}
            >
              <span className="text-lg">{language === 'th' ? '🇹🇭' : '🇺🇸'}</span>
              {!collapsed && (
                <span className="text-sm text-slate-400">
                  {language === 'th' ? 'ภาษาไทย' : 'English'}
                </span>
              )}
            </button>
          </div>

          {/* Theme Toggle */}
          <div className={cn("px-4 py-2", collapsed && "px-2")}>
            <button
              onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              className={cn(
                "w-full flex items-center gap-3 px-3 py-2.5 rounded-xl bg-slate-800/50 hover:bg-slate-800 transition-colors",
                collapsed && "justify-center px-2"
              )}
            >
              {theme === 'dark' ? (
                <Sun size={18} className="text-amber-400" />
              ) : (
                <Moon size={18} className="text-slate-400" />
              )}
              {!collapsed && (
                <span className="text-sm text-slate-400">
                  {theme === 'dark' ? (language === 'th' ? 'โหมดสว่าง' : 'Light Mode') : (language === 'th' ? 'โหมดมืด' : 'Dark Mode')}
                </span>
              )}
            </button>
          </div>

          {/* Footer Info */}
          <div className={cn("p-4 border-t border-slate-800/50 bg-slate-900/50", collapsed && "p-2")}>
            <div className={cn("flex items-center gap-3", collapsed && "justify-center")}>
              <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-emerald-500 to-emerald-700 flex items-center justify-center text-white shadow-lg">
                <Zap size={18} />
              </div>
              {!collapsed && (
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-slate-200 truncate">Hybrid AI Model</p>
                  <p className="text-[10px] text-slate-500 truncate flex items-center gap-1">
                    <Activity size={10} />
                    GNN + LSTM Architecture
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}

