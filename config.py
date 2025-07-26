import os
import sys
import logging
from pathlib import Path

def setup_logging(log_level="INFO"):
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('resume_agent.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_RESUMES_DIR = DATA_DIR / "sample_resumes"
UTILS_DIR = PROJECT_ROOT / "utils"

DEFAULT_PREPROCESSOR_CONFIG = {
    'use_spacy': False,
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

SUPPORTED_FILE_TYPES = {
    'pdf': ['.pdf'],
    'word': ['.docx', '.doc'],
    'text': ['.txt']
}

def get_all_supported_extensions():
    extensions = []
    for file_type, exts in SUPPORTED_FILE_TYPES.items():
        extensions.extend(exts)
    return extensions

def ensure_directories():
    directories = [DATA_DIR, SAMPLE_RESUMES_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def check_dependencies():
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

def clean_corrupted_nltk_data():
    """Clean up corrupted NLTK data files"""
    import nltk
    import os
    import shutil
    
    print("Cleaning up potentially corrupted NLTK data...")
    
    # Get NLTK data directory
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    
    # Remove potentially corrupted files
    corrupted_files = [
        os.path.join(nltk_data_dir, 'corpora', 'omw-1.4.zip'),
        os.path.join(nltk_data_dir, 'corpora', 'omw-1.4'),
    ]
    
    for file_path in corrupted_files:
        if os.path.exists(file_path):
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                else:
                    shutil.rmtree(file_path)
                print(f"Removed corrupted file: {file_path}")
            except Exception as e:
                print(f"Could not remove {file_path}: {e}")

def download_nltk_data():
    import nltk
    
    required_nltk_data = [
        'punkt', 'punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'
    ]
    
    for data_name in required_nltk_data:
        try:
            # Try to find existing data first
            if data_name in ['punkt', 'punkt_tab']:
                nltk.data.find(f'tokenizers/{data_name}')
            elif data_name in ['stopwords', 'wordnet']:
                nltk.data.find(f'corpora/{data_name}')
            else:
                nltk.data.find(f'taggers/{data_name}')
        except (LookupError, Exception):
            try:
                nltk.download(data_name, quiet=True)
                print(f"Downloaded NLTK data: {data_name}")
            except Exception as e:
                print(f"Failed to download {data_name}: {e}")
                # Continue with other downloads even if one fails
                continue

def setup_environment():
    print("Setting up Resume Scoring Agent environment...")
    
    # Clean corrupted NLTK data first
    try:
        clean_corrupted_nltk_data()
    except Exception as e:
        print(f"Warning: Could not clean NLTK data: {e}")
    
    setup_logging()
    ensure_directories()
    print("✓ Directories created")
    
    missing = check_dependencies()
    if missing:
        print(f"⚠️  Missing packages: {', '.join(missing)}")
        print("Please install them using: pip install -r requirements.txt")
    else:
        print("✓ All dependencies installed")
    
    try:
        download_nltk_data()
        print("✓ NLTK data downloaded")
    except Exception as e:
        print(f"⚠️  NLTK data download failed: {e}")
    
    print("Environment setup complete!")

if __name__ == "__main__":
    setup_environment()
