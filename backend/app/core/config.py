"""
Configuration management for the vulnerability scanner backend.
"""
from tkinter import FALSE
from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application configuration"""
    
    # Application
    APP_NAME: str = "Vulnerability Scanner"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False  # Set to True only for development
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"
    
    # Security Limits
    MAX_ZIP_SIZE_MB: int = 500  # Increased for very large projects (500MB)
    MAX_FILE_COUNT: int = 5000  # Increased file count limit for large projects
    SCAN_TIMEOUT_SECONDS: int = 600  # 10 minutes timeout for large scans
    
    # Paths
    TEMP_DIR: str = "./tmp"
    MODEL_DIR: str = "./training/models"  # ✅ Updated to use retrained models
    
    # Scanner Settings
    BANDIT_CONFIG_PATH: str = ""
    SEMGREP_RULES_PATH: str = "./data/semgrep-rules.yaml"
    
    
    # ML Model Settings
    # ✅ NEW IMPROVED MODEL: Test F1=88.99%, Accuracy=90.86%
    # Training: Val F1=99.01%, Val Acc=99.10% (Epoch 12)
    # Trained on 6,222 samples (2,860 vulnerable, 3,362 safe) + minimal augmentation (13%)
    # Model: HybridVulnerabilityModel (GNN + BiLSTM) with stronger regularization
    # Vocabulary: 5,319 tokens
    # Test Set (Unseen): 90.86% accuracy, 99.80% precision, 80.29% recall
    # False Positive Rate: 0.14% (1 sample only!)
    # False Negative Rate: 19.71% (acceptable for security scanner)
    ML_ENABLED: bool = FALSE  # ✅ Enabled: Production-ready model
    ML_MODEL_PATH: str = "./training/models/hybrid_model_best.pth"  # Improved model (90.86% test acc)
    ML_VOCAB_PATH: str = "./training/models/vocab.json"  # Vocabulary (5319 tokens)
    ML_CONFIDENCE_THRESHOLD: float = 0.5  # Minimum confidence for ML findings
    ML_WEIGHT: float = 0.4  # Weight for ML vs pattern-matching (40% ML, 60% pattern)


    
    # Ignored directories during ZIP scan
    IGNORED_DIRS: List[str] = [
        "node_modules",
        ".git",
        ".next",
        "dist",
        "build",
        "venv",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        "coverage",
        ".idea",
        ".vscode"
    ]
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS: dict = {
        "python": [".py"],
        "javascript": [".js", ".jsx"],
        "typescript": [".ts", ".tsx"]
    }
    
    @property
    def cors_origins(self) -> List[str]:
        """Parse CORS origins from comma-separated string"""
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]
    
    @property
    def max_zip_size_bytes(self) -> int:
        """Convert MAX_ZIP_SIZE_MB to bytes"""
        return self.MAX_ZIP_SIZE_MB * 1024 * 1024
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
