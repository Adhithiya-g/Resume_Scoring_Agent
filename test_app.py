#!/usr/bin/env python3
"""
Test script to verify the Resume Scoring Agent functionality
"""

import os
import sys
from pathlib import Path

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import streamlit as st
        print("✓ Streamlit imported")
    except ImportError as e:
        print(f"✗ Streamlit import failed: {e}")
        return False
    
    try:
        from utils.extractor import TextExtractor
        print("✓ TextExtractor imported")
    except ImportError as e:
        print(f"✗ TextExtractor import failed: {e}")
        return False
    
    try:
        from utils.preprocessor import TextPreprocessor
        print("✓ TextPreprocessor imported")
    except ImportError as e:
        print(f"✗ TextPreprocessor import failed: {e}")
        return False
    
    try:
        from utils.scorer import ResumeScorer
        print("✓ ResumeScorer imported")
    except ImportError as e:
        print(f"✗ ResumeScorer import failed: {e}")
        return False
    
    try:
        from app import ResumeAgent
        print("✓ ResumeAgent imported")
    except ImportError as e:
        print(f"✗ ResumeAgent import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality with sample data"""
    print("\nTesting basic functionality...")
    
    try:
        from utils.preprocessor import TextPreprocessor
        from utils.scorer import ResumeScorer
        
        # Test text preprocessing
        preprocessor = TextPreprocessor()
        sample_text = "This is a test resume with Python programming skills and data science experience."
        processed = preprocessor.preprocess_text(sample_text)
        print(f"✓ Text preprocessing works: '{sample_text[:30]}...' -> '{processed[:30]}...'")
        
        # Test scoring
        scorer = ResumeScorer()
        job_desc = "We need a Python developer with data science skills"
        resumes = {"resume1": "Python expert with machine learning experience", 
                   "resume2": "Java developer with web skills"}
        
        # Fit the scorer with job description and resumes
        scorer.fit_transform_documents(job_desc, resumes)
        scores_dict = scorer.compute_similarity_scores()
        scores = list(scores_dict.values())
        print(f"✓ Scoring works: {len(scores)} scores calculated")
        print(f"  Sample scores: {[f'{score:.2f}' for score in scores]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_sample_files():
    """Test if sample files exist"""
    print("\nTesting sample files...")
    
    data_dir = Path("data")
    sample_resumes_dir = data_dir / "sample_resumes"
    
    if data_dir.exists():
        print("✓ Data directory exists")
    else:
        print("✗ Data directory missing")
        return False
    
    if sample_resumes_dir.exists():
        print("✓ Sample resumes directory exists")
        sample_files = list(sample_resumes_dir.glob("*.txt"))
        print(f"  Found {len(sample_files)} sample resume files")
        for file in sample_files:
            print(f"    - {file.name}")
    else:
        print("✗ Sample resumes directory missing")
        return False
    
    job_desc_file = data_dir / "sample_job_description.txt"
    if job_desc_file.exists():
        print("✓ Sample job description exists")
    else:
        print("✗ Sample job description missing")
        return False
    
    return True

def main():
    """Run all tests"""
    print("Resume Scoring Agent - Test Suite")
    print("=" * 50)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test basic functionality
    if not test_basic_functionality():
        all_passed = False
    
    # Test sample files
    if not test_sample_files():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All tests passed! The Resume Scoring Agent is ready to use.")
        print("\nTo start the application, run:")
        print("streamlit run app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
