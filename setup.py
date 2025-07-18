#!/usr/bin/env python
"""
Setup script for Resume Scoring Agent
Run this script to install dependencies and setup the environment
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def install_dependencies():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    
    # Install main requirements
    success = run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing requirements"
    )
    
    if not success:
        print("Trying alternative installation methods...")
        
        # Try installing packages individually
        packages = [
            "streamlit", "pandas", "numpy", "scikit-learn",
            "nltk", "matplotlib", "plotly", "seaborn",
            "PyPDF2", "pdfplumber", "python-docx", "wordcloud",
            "openpyxl", "fpdf2"
        ]
        
        for package in packages:
            run_command(
                f"{sys.executable} -m pip install {package}",
                f"Installing {package}"
            )
    
    return success

def setup_nltk():
    """Setup NLTK data"""
    print("\nSetting up NLTK data...")
    
    nltk_setup_script = '''
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
nltk_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']

for data in nltk_data:
    try:
        nltk.download(data, quiet=True)
        print(f"✓ Downloaded {data}")
    except Exception as e:
        print(f"❌ Failed to download {data}: {e}")

print("NLTK setup complete!")
'''
    
    try:
        with open("temp_nltk_setup.py", "w") as f:
            f.write(nltk_setup_script)
        
        run_command(
            f"{sys.executable} temp_nltk_setup.py",
            "Setting up NLTK data"
        )
        
        # Clean up temporary file
        os.remove("temp_nltk_setup.py")
        
    except Exception as e:
        print(f"❌ NLTK setup failed: {e}")

def create_directories():
    """Create necessary directories"""
    print("\nCreating project directories...")
    
    directories = [
        "data",
        "data/sample_resumes",
        "utils",
        "outputs",
        "visualizations"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def test_installation():
    """Test if the installation was successful"""
    print("\nTesting installation...")
    
    test_script = '''
try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import sklearn
    import nltk
    import matplotlib.pyplot as plt
    import plotly.express as px
    import PyPDF2
    import pdfplumber
    from docx import Document
    from wordcloud import WordCloud
    
    print("✓ All dependencies imported successfully!")
    
    # Test NLTK data
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    
    test_text = "This is a test sentence."
    tokens = word_tokenize(test_text)
    stops = set(stopwords.words('english'))
    
    print("✓ NLTK data working correctly!")
    print("Installation test passed! 🎉")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Some dependencies may not be installed correctly.")
except Exception as e:
    print(f"❌ Test failed: {e}")
'''
    
    try:
        with open("test_installation.py", "w") as f:
            f.write(test_script)
        
        run_command(
            f"{sys.executable} test_installation.py",
            "Testing installation"
        )
        
        # Clean up
        os.remove("test_installation.py")
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")

def main():
    """Main setup function"""
    print("🚀 Resume Scoring Agent Setup")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    install_dependencies()
    
    # Setup NLTK
    setup_nltk()
    
    # Test installation
    test_installation()
    
    print("\n" + "=" * 50)
    print("Setup complete! 🎉")
    print("\nTo run the application:")
    print("streamlit run app.py")
    print("\nTo test with sample data:")
    print("python -c \"from config import setup_environment; setup_environment()\"")

if __name__ == "__main__":
    main()
