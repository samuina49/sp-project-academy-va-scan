'use client';

import { useEffect, useState } from 'react';
import Card from '@/components/ui/Card';
import styles from './page.module.css';

type ScanStep = {
    id: number;
    label: string;
    description: string;
    status: 'pending' | 'active' | 'complete';
    icon: string;
};

export default function ScanningPage() {
    const [steps, setSteps] = useState<ScanStep[]>([
        { id: 1, label: 'Uploading', description: 'Receiving files...', status: 'active', icon: '📤' },
        { id: 2, label: 'Transpiling TypeScript', description: 'Converting to JavaScript...', status: 'pending', icon: '⚡' },
        { id: 3, label: 'Graph Extraction', description: 'Building AST & CPG...', status: 'pending', icon: '🕸️' },
        { id: 4, label: 'AI Inference', description: 'GNN+LSTM Analysis...', status: 'pending', icon: '🤖' },
        { id: 5, label: 'Report Generation', description: 'Finalizing results...', status: 'pending', icon: '📊' }
    ]);

    useEffect(() => {
        const intervals = steps.map((step, index) =>
            setTimeout(() => {
                setSteps(prev => prev.map(s =>
                    s.id === step.id ? { ...s, status: 'active' } :
                        s.id < step.id ? { ...s, status: 'complete' } : s
                ));
            }, (index + 1) * 2000)
        );

        return () => intervals.forEach(clearTimeout);
    }, []);

    return (
        <div className={styles.container}>
            <div className={styles.content}>
                <h2 className={styles.title}>Scanning in Progress...</h2>

                <Card className={styles.card}>
                    <div className={styles.steps}>
                        {steps.map((step, index) => (
                            <div key={step.id} className={styles.stepWrapper}>
                                {index < steps.length - 1 && (
                                    <div className={`${styles.connector} ${step.status === 'complete' ? styles.activeConnector : ''}`} />
                                )}

                                <div className={styles.step}>
                                    <div className={`${styles.iconCircle} ${styles[`${step.status}Circle`]}`}>
                                        {step.icon}
                                    </div>

                                    <div className={styles.stepContent}>
                                        <h3 className={`${styles.stepLabel} ${step.status === 'active' ? styles.activeLabel : ''}`}>
                                            {step.label}
                                        </h3>
                                        <p className={styles.stepDescription}>{step.description}</p>

                                        {step.status === 'active' && (
                                            <div className={styles.activeIndicator}>
                                                <div className={styles.dot}></div>
                                                <span>Processing...</span>
                                            </div>
                                        )}
                                    </div>

                                    {step.status === 'complete' && (
                                        <div className={styles.checkmark}>✓</div>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>

                    <div className={styles.aiInfo}>
                        <div className={styles.pulseDot}></div>
                        <p>Deep learning models analyzing code patterns...</p>
                    </div>
                </Card>
            </div>
        </div>
    );
}
