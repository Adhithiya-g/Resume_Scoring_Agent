#!/usr/bin/env python3
"""
NLTK Data Cleanup Script
Fix corrupted NLTK data files that cause BadZipFile errors
"""

import os
import shutil
import nltk

def clean_nltk_data():
    """Clean up corrupted NLTK data files"""
    print("Cleaning up corrupted NLTK data...")
    
    # Get NLTK data directory
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    print(f"NLTK data directory: {nltk_data_dir}")
    
    # Files that commonly get corrupted
    potential_corrupted = [
        'corpora/omw-1.4.zip',
        'corpora/omw-1.4',
        'corpora/wordnet.zip',
        'tokenizers/punkt.zip',
        'tokenizers/punkt_tab.zip'
    ]
    
    removed_files = []
    for file_rel_path in potential_corrupted:
        file_path = os.path.join(nltk_data_dir, file_rel_path)
        if os.path.exists(file_path):
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                else:
                    shutil.rmtree(file_path)
                removed_files.append(file_path)
                print(f"✓ Removed: {file_path}")
            except Exception as e:
                print(f"✗ Could not remove {file_path}: {e}")
    
    if removed_files:
        print(f"\nRemoved {len(removed_files)} potentially corrupted files")
    else:
        print("No corrupted files found")
    
    return removed_files

def download_essential_data():
    """Download essential NLTK data"""
    print("\nDownloading essential NLTK data...")
    
    essential_data = ['punkt', 'stopwords', 'wordnet']
    
    for data_name in essential_data:
        try:
            print(f"Downloading {data_name}...")
            nltk.download(data_name, quiet=False)
            print(f"✓ Downloaded {data_name}")
        except Exception as e:
            print(f"✗ Failed to download {data_name}: {e}")

def main():
    print("NLTK Data Cleanup and Repair Tool")
    print("=" * 40)
    
    # Clean corrupted files
    removed_files = clean_nltk_data()
    
    # Download essential data
    download_essential_data()
    
    print("\n" + "=" * 40)
    print("Cleanup complete!")
    print("\nYou can now try running your Streamlit app again:")
    print("streamlit run app.py")

if __name__ == "__main__":
    main()
