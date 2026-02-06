'use client';

import React from 'react';
import Card from '../ui/Card';
import Badge from '../ui/Badge';
import type { FileScanResult } from '@/types/api';
import styles from './ScanResults.module.css';

interface ScanResultsProps {
    results: FileScanResult[];
}

export default function ScanResults({ results }: ScanResultsProps) {
    const totalFindings = results.reduce((acc, file) => acc + file.findings.length, 0);

    const severityCounts = results.reduce((acc, file) => {
        file.findings.forEach(finding => {
            const severity = finding.severity?.toLowerCase() || 'info';
            acc[severity] = (acc[severity] || 0) + 1;
        });
        return acc;
    }, {} as Record<string, number>);

    const getSeverityVariant = (severity: string): 'critical' | 'high' | 'medium' | 'low' | 'info' => {
        const sev = severity.toLowerCase();
        if (sev.includes('critical')) return 'critical';
        if (sev.includes('error') || sev.includes('high')) return 'high';
        if (sev.includes('warning') || sev.includes('medium')) return 'medium';
        if (sev.includes('info') || sev.includes('low')) return 'low';
        return 'info';
    };

    if (totalFindings === 0) {
        return (
            <Card className={styles.emptyState}>
                <div className={styles.emptyIcon}>✓</div>
                <h3>No Vulnerabilities Found</h3>
                <p>Great! Your code looks secure.</p>
            </Card>
        );
    }

    return (
        <div className={styles.container}>
            {/* Summary Stats */}
            <div className={styles.stats}>
                <Card className={styles.statCard}>
                    <div className={styles.statValue}>{totalFindings}</div>
                    <div className={styles.statLabel}>Total Issues</div>
                </Card>

                {Object.entries(severityCounts).map(([severity, count]) => (
                    <Card key={severity} className={styles.statCard}>
                        <div className={styles.statValue}>{count}</div>
                        <Badge variant={getSeverityVariant(severity)}>
                            {severity.toUpperCase()}
                        </Badge>
                    </Card>
                ))}
            </div>

            {/* Findings List */}
            <div className={styles.findingsList}>
                {results.map((file, fileIdx) => (
                    file.findings.length > 0 && (
                        <Card key={fileIdx} className={styles.fileCard}>
                            <div className={styles.fileHeader}>
                                <div className={styles.fileName}>
                                    <svg className={styles.fileIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                    </svg>
                                    {file.file_path}
                                </div>
                                <Badge variant="info">
                                    {file.findings.length} {file.findings.length === 1 ? 'issue' : 'issues'}
                                </Badge>
                            </div>

                            <div className={styles.findings}>
                                {file.findings.map((finding, findingIdx) => (
                                    <div key={findingIdx} className={styles.finding}>
                                        <div className={styles.findingHeader}>
                                            <Badge variant={getSeverityVariant(finding.severity || 'info')}>
                                                {finding.severity}
                                            </Badge>
                                            <span className={styles.ruleId}>{finding.rule_id}</span>
                                            <span className={styles.line}>Line {finding.start_line}</span>
                                        </div>

                                        <p className={styles.message}>{finding.message}</p>

                                        {finding.code_snippet && (
                                            <pre className={styles.codeSnippet}>
                                                <code>{finding.code_snippet}</code>
                                            </pre>
                                        )}

                                        <div className={styles.metadata}>
                                            {finding.cwe_id && (
                                                <span className={styles.tag}>{finding.cwe_id}</span>
                                            )}
                                            {finding.owasp_category && (
                                                <span className={styles.tag}>{finding.owasp_category}</span>
                                            )}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </Card>
                    )
                ))}
            </div>
        </div>
    );
}
