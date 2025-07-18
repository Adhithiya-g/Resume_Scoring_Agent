"""
Configuration and Setup Module for Resume Scoring Agent
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
def setup_logging(log_level="INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('resume_agent.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_RESUMES_DIR = DATA_DIR / "sample_resumes"
UTILS_DIR = PROJECT_ROOT / "utils"

# Default configurations
DEFAULT_PREPROCESSOR_CONFIG = {
    'use_spacy': False,  # Set to False to avoid spaCy dependency issues
    'remove_stopwords': True,
    'use_lemmatization': True,
    'min_word_length': 2,
    'custom_stopwords': {
        'resume', 'cv', 'curriculum', 'vitae', 'experience',
        'responsibilities', 'skills', 'education', 'work'
    }
}

DEFAULT_SCORER_CONFIG = {
    'max_features': 5000,
    'min_df': 1,
    'max_df': 0.95,
    'ngram_range': (1, 2),
    'use_idf': True,
    'sublinear_tf': True
}

DEFAULT_VISUALIZATION_CONFIG = {
    'color_scheme': 'viridis',
    'figure_size': (12, 8),
    'top_n_display': 15
}

# File type configurations
SUPPORTED_FILE_TYPES = {
    'pdf': ['.pdf'],
    'word': ['.docx', '.doc'],
    'text': ['.txt']
}

def get_all_supported_extensions():
    """Get all supported file extensions"""
    extensions = []
    for file_type, exts in SUPPORTED_FILE_TYPES.items():
        extensions.extend(exts)
    return extensions

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [DATA_DIR, SAMPLE_RESUMES_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'scikit-learn',
        'nltk', 'matplotlib', 'plotly', 'seaborn',
        'PyPDF2', 'pdfplumber', 'python-docx', 'wordcloud'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def download_nltk_data():
    """Download required NLTK data"""
    import nltk
    
    required_nltk_data = [
        'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'
    ]
    
    for data_name in required_nltk_data:
        try:
            nltk.data.find(f'tokenizers/{data_name}')
        except LookupError:
            try:
                nltk.download(data_name, quiet=True)
                print(f"Downloaded NLTK data: {data_name}")
            except Exception as e:
                print(f"Failed to download {data_name}: {e}")

def setup_environment():
    """Setup the complete environment"""
    print("Setting up Resume Scoring Agent environment...")
    
    # Setup logging
    setup_logging()
    
    # Ensure directories exist
    ensure_directories()
    print("✓ Directories created")
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print(f"⚠️  Missing packages: {', '.join(missing)}")
        print("Please install them using: pip install -r requirements.txt")
    else:
        print("✓ All dependencies installed")
    
    # Download NLTK data
    try:
        download_nltk_data()
        print("✓ NLTK data downloaded")
    except Exception as e:
        print(f"⚠️  NLTK data download failed: {e}")
    
    print("Environment setup complete!")

if __name__ == "__main__":
    setup_environment()
