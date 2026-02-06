import React from 'react';
import styles from './Badge.module.css';

interface BadgeProps {
    children: React.ReactNode;
    variant?: 'critical' | 'high' | 'medium' | 'low' | 'info' | 'success';
    className?: string;
}

export default function Badge({ children, variant = 'info', className = '' }: BadgeProps) {
    return (
        <span className={`${styles.badge} ${styles[variant]} ${className}`}>
            {children}
        </span>
    );
}
