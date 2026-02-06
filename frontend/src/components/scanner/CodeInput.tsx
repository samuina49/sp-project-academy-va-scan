'use client';

import React, { useEffect, useRef } from 'react';
import Prism from 'prismjs';
import 'prismjs/themes/prism-tomorrow.css';
import 'prismjs/components/prism-python';
import 'prismjs/components/prism-javascript';
import 'prismjs/components/prism-typescript';
import styles from './CodeInput.module.css';

interface CodeInputProps {
    value: string;
    onChange: (value: string) => void;
    language: 'python' | 'javascript' | 'typescript';
    placeholder?: string;
    minHeight?: string;
}

export default function CodeInput({
    value,
    onChange,
    language,
    placeholder = 'Paste your code here...',
    minHeight = '400px'
}: CodeInputProps) {
    const textareaRef = useRef<HTMLTextAreaElement>(null);
    const highlightRef = useRef<HTMLPreElement>(null);

    useEffect(() => {
        if (highlightRef.current) {
            const codeElement = highlightRef.current.querySelector('code');
            if (codeElement) {
                Prism.highlightElement(codeElement);
            }
        }
    }, [value, language]);

    const handleScroll = (e: React.UIEvent<HTMLTextAreaElement>) => {
        if (highlightRef.current) {
            highlightRef.current.scrollTop = e.currentTarget.scrollTop;
            highlightRef.current.scrollLeft = e.currentTarget.scrollLeft;
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === 'Tab') {
            e.preventDefault();
            const start = e.currentTarget.selectionStart;
            const end = e.currentTarget.selectionEnd;
            const newValue = value.substring(0, start) + '    ' + value.substring(end);
            onChange(newValue);
            setTimeout(() => {
                if (textareaRef.current) {
                    textareaRef.current.selectionStart = textareaRef.current.selectionEnd = start + 4;
                }
            }, 0);
        }
    };

    const lines = value.split('\n').length;

    return (
        <div className={styles.container} style={{ minHeight }}>
            <div className={styles.lineNumbers}>
                {Array.from({ length: Math.max(lines, 20) }, (_, i) => (
                    <div key={i + 1}>{i + 1}</div>
                ))}
            </div>

            <div className={styles.editorWrapper}>
                <pre
                    ref={highlightRef}
                    className={styles.highlight}
                    suppressHydrationWarning
                >
                    <code className={`language-${language}`}>{value || ' '}</code>
                </pre>

                <textarea
                    ref={textareaRef}
                    value={value}
                    onChange={(e) => onChange(e.target.value)}
                    onScroll={handleScroll}
                    onKeyDown={handleKeyDown}
                    placeholder={placeholder}
                    className={styles.textarea}
                    spellCheck={false}
                />
            </div>
        </div>
    );
}
