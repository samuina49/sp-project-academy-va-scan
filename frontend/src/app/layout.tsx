import type { Metadata, Viewport } from 'next'
import Sidebar from '@/components/layout/Sidebar'
import { ThemeProvider } from '@/components/ThemeProvider'
import { I18nProvider } from '@/lib/i18n'
import { ErrorBoundary } from '@/components/ErrorBoundary'
import KeyboardShortcutsModal from '@/components/KeyboardShortcutsModal'
import { SkipToContent } from '@/lib/accessibility'
import './globals.css'
import { Toaster } from 'react-hot-toast'

export const metadata: Metadata = {
    title: 'VulnScanner - AI-based Vulnerability Scanner',
    description: 'Advanced vulnerability detection using hybrid GNN+LSTM deep learning architecture',
    keywords: ['vulnerability scanner', 'security', 'AI', 'code analysis', 'GNN', 'LSTM'],
    authors: [{ name: 'VulnScanner Team' }],
}

export const viewport: Viewport = {
    width: 'device-width',
    initialScale: 1,
    themeColor: [
        { media: '(prefers-color-scheme: light)', color: '#ffffff' },
        { media: '(prefers-color-scheme: dark)', color: '#0f172a' }
    ],
}

export default function RootLayout({
    children,
}: {
    children: React.ReactNode
}) {
    return (
        <html lang="en" suppressHydrationWarning>
            <body className="antialiased min-h-screen bg-slate-50 dark:bg-slate-950 transition-colors duration-300">
                <SkipToContent />
                
                <ErrorBoundary>
                    <ThemeProvider>
                        <I18nProvider>
                            <div className="flex h-screen overflow-hidden">
                                <Sidebar />
                                
                                <div className="flex-1 flex flex-col relative overflow-hidden lg:ml-64 transition-all duration-300">
                                    <main 
                                        id="main-content" 
                                        className="flex-1 overflow-x-hidden overflow-y-auto"
                                        tabIndex={-1}
                                    >
                                        <Toaster 
                                            position="top-right"
                                            toastOptions={{
                                                className: 'dark:bg-slate-800 dark:text-white',
                                                duration: 4000,
                                                success: {
                                                    iconTheme: {
                                                        primary: '#10b981',
                                                        secondary: '#ffffff',
                                                    },
                                                },
                                                error: {
                                                    iconTheme: {
                                                        primary: '#ef4444',
                                                        secondary: '#ffffff',
                                                    },
                                                },
                                            }}
                                        />
                                        {children}
                                    </main>
                                </div>
                            </div>

                            {/* Global Keyboard Shortcuts Modal */}
                            <KeyboardShortcutsModal />
                        </I18nProvider>
                    </ThemeProvider>
                </ErrorBoundary>
            </body>
        </html>
    )
}
