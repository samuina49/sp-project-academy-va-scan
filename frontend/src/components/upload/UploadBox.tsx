'use client';

import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import Card from '../ui/Card';
import Badge from '../ui/Badge';
import styles from './UploadBox.module.css';

interface UploadBoxProps {
    onFileSelect: (file: File) => void;
}

export default function UploadBox({ onFileSelect }: UploadBoxProps) {
    const [isDragActive, setIsDragActive] = useState(false);
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [showTranspileBadge, setShowTranspileBadge] = useState(false);

    const onDrop = useCallback((acceptedFiles: File[]) => {
        if (acceptedFiles.length > 0) {
            const file = acceptedFiles[0];
            setSelectedFile(file);

            // Check if TypeScript file
            const isTypeScript = file.name.endsWith('.ts') || file.name.endsWith('.tsx');
            setShowTranspileBadge(isTypeScript);
            onFileSelect(file);
        }
        setIsDragActive(false);
    }, [onFileSelect]);

    const { getRootProps, getInputProps } = useDropzone({
        onDrop,
        accept: {
            'application/zip': ['.zip'],
            'text/x-python': ['.py'],
            'application/javascript': ['.js'],
            'application/typescript': ['.ts', '.tsx']
        },
        multiple: false,
        onDragEnter: () => setIsDragActive(true),
        onDragLeave: () => setIsDragActive(false)
    });

    return (
        <div className={styles.container}>

            {/* Upload Area */}
            <Card className={styles.uploadCard}>
                <div
                    {...getRootProps()}
                    className={`${styles.dropzone} ${isDragActive ? styles.active : ''}`}
                >
                    <input {...getInputProps()} />

                    {/* Upload Icon */}
                    <div className={styles.icon}>
                        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                        </svg>
                    </div>

                    {selectedFile ? (
                        <div className={styles.fileInfo}>
                            <p className={styles.fileName}>📁 {selectedFile.name}</p>
                            <p className={styles.fileSize}>{(selectedFile.size / 1024).toFixed(2)} KB</p>

                            {showTranspileBadge && (
                                <div className={styles.transpileBadge}>
                                    <Badge variant="success">⚡ Auto-Transpilation Active</Badge>
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className={styles.instructions}>
                            <h3 className={styles.instructionsTitle}>
                                {isDragActive ? 'Drop your file here' : 'Upload Source Code'}
                            </h3>
                            <p className={styles.instructionsText}>
                                Drag & drop or click to browse
                            </p>
                            <div className={styles.formats}>
                                <span>.zip</span>
                                <span>.py</span>
                                <span>.js</span>
                                <span>.ts</span>
                            </div>
                        </div>
                    )}
                </div>
            </Card>
        </div>
    );
}
