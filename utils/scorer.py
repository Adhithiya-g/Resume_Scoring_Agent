"""
Scoring Module for Resume Scoring Agent
Handles TF-IDF vectorization and cosine similarity computation
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResumeScorer:
    """Handles TF-IDF vectorization and similarity scoring between job descriptions and resumes"""
    
    def __init__(self, 
                 max_features: int = 5000,
                 min_df: float = 1,
                 max_df: float = 0.95,
                 ngram_range: Tuple[int, int] = (1, 2),
                 use_idf: bool = True,
                 sublinear_tf: bool = True):
        """
        Initialize the resume scorer
        
        Args:
            max_features: Maximum number of features for TF-IDF
            min_df: Minimum document frequency for terms
            max_df: Maximum document frequency for terms
            ngram_range: Range of n-grams to consider
            use_idf: Whether to use inverse document frequency
            sublinear_tf: Whether to use sublinear TF scaling
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.use_idf = use_idf
        self.sublinear_tf = sublinear_tf
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            use_idf=self.use_idf,
            sublinear_tf=self.sublinear_tf,
            stop_words='english'
        )
        
        self.tfidf_matrix = None
        self.feature_names = None
        self.job_description = ""
        self.resume_texts = {}
        self.similarity_scores = {}
        self.keyword_matches = {}
    
    def fit_transform_documents(self, 
                              job_description: str, 
                              resume_texts: Dict[str, str],
                              use_simple_preprocessing: bool = True) -> np.ndarray:
        """
        Fit TF-IDF vectorizer and transform documents
        Critical: TF-IDF vectorizer must be trained on combined corpus for accurate similarity
        
        Args:
            job_description: Job description text
            resume_texts: Dictionary mapping resume names to text content
            use_simple_preprocessing: Whether to use simplified preprocessing for better scores
            
        Returns:
            TF-IDF matrix
        """
        from .preprocessor import TextPreprocessor
        
        self.job_description = job_description
        self.resume_texts = resume_texts
        
        # Initialize preprocessor for simple preprocessing if requested
        if use_simple_preprocessing:
            preprocessor = TextPreprocessor()
            
            # Apply consistent preprocessing to all documents
            processed_jd = preprocessor.preprocess_simple(job_description)
            processed_resumes = {
                name: preprocessor.preprocess_simple(text) 
                for name, text in resume_texts.items()
            }
            
            # Combine all documents for corpus training
            corpus = [processed_jd] + list(processed_resumes.values())
            
            logger.info(f"Preprocessing applied - JD sample: {processed_jd[:200]}")
            logger.info(f"Resume sample: {list(processed_resumes.values())[0][:200] if processed_resumes else 'No resumes'}")
        else:
            # Use raw text
            corpus = [job_description] + list(resume_texts.values())
        
        # Fit and transform the combined corpus
        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
            self.feature_names = self.vectorizer.get_feature_names_out()
            logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
            logger.info(f"Vocabulary size: {len(self.feature_names)}")
            return self.tfidf_matrix
        except Exception as e:
            logger.error(f"Error in TF-IDF transformation: {str(e)}")
            raise
    
    def compute_similarity_scores(self, normalize_scores: bool = True) -> Dict[str, float]:
        """
        Compute cosine similarity scores between job description and resumes
        
        Args:
            normalize_scores: Whether to normalize scores as percentages
        
        Returns:
            Dictionary mapping resume names to similarity scores
        """
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF matrix not computed. Call fit_transform_documents first.")
        
        # Job description is the first document (index 0)
        job_vector = self.tfidf_matrix[0:1]
        resume_vectors = self.tfidf_matrix[1:]
        
        # Compute cosine similarities
        similarities = cosine_similarity(job_vector, resume_vectors)[0]
        
        # Map similarities to resume names
        resume_names = list(self.resume_texts.keys())
        
        if normalize_scores:
            # Convert to percentages and apply score enhancement for display
            self.similarity_scores = {
                name: float(score * 100) for name, score in zip(resume_names, similarities)
            }
        else:
            self.similarity_scores = {
                name: float(score) for name, score in zip(resume_names, similarities)
            }
        
        logger.info(f"Computed similarity scores for {len(self.similarity_scores)} resumes")
        for name, score in self.similarity_scores.items():
            category = self.categorize_score(score if normalize_scores else score * 100)
            logger.info(f"  {name}: {score:.3f}{'%' if normalize_scores else ''} ({category})")
        
        return self.similarity_scores
    
    def categorize_score(self, score: float) -> str:
        """
        Categorize similarity scores into meaningful ranges
        
        Args:
            score: Similarity score (0-100 if percentage)
            
        Returns:
            Score category description
        """
        if score >= 60:
            return "Excellent Match"
        elif score >= 40:
            return "Good Match"
        elif score >= 20:
            return "Average Match"
        else:
            return "Below Average"
    
    def get_ranked_resumes(self, threshold: float = 0.0) -> List[Tuple[str, float]]:
        """
        Get resumes ranked by similarity score
        
        Args:
            threshold: Minimum similarity score threshold
            
        Returns:
            List of tuples (resume_name, score) sorted by score descending
        """
        if not self.similarity_scores:
            raise ValueError("Similarity scores not computed. Call compute_similarity_scores first.")
        
        # Filter by threshold and sort
        filtered_scores = {
            name: score for name, score in self.similarity_scores.items() 
            if score >= threshold
        }
        
        ranked = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked
    
    def extract_top_keywords(self, 
                           document_index: int, 
                           top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Extract top keywords from a document based on TF-IDF scores
        
        Args:
            document_index: Index of the document (0 for job description, 1+ for resumes)
            top_n: Number of top keywords to return
            
        Returns:
            List of tuples (keyword, tfidf_score)
        """
        if self.tfidf_matrix is None or self.feature_names is None:
            raise ValueError("TF-IDF matrix not computed.")
        
        # Get TF-IDF scores for the document
        doc_tfidf = self.tfidf_matrix[document_index].toarray()[0]
        
        # Get top features
        top_indices = doc_tfidf.argsort()[-top_n:][::-1]
        top_keywords = [
            (self.feature_names[i], doc_tfidf[i]) 
            for i in top_indices if doc_tfidf[i] > 0
        ]
        
        return top_keywords
    
    def find_keyword_matches(self, 
                           resume_name: str, 
                           top_n: int = 50) -> Dict[str, Any]:
        """
        Find matching keywords between job description and a specific resume
        
        Args:
            resume_name: Name of the resume to analyze
            top_n: Number of top keywords to consider
            
        Returns:
            Dictionary containing match analysis
        """
        if resume_name not in self.resume_texts:
            raise ValueError(f"Resume '{resume_name}' not found")
        
        # Get resume index (job description is at index 0)
        resume_names = list(self.resume_texts.keys())
        resume_index = resume_names.index(resume_name) + 1
        
        # Get top keywords for both documents
        job_keywords = dict(self.extract_top_keywords(0, top_n))
        resume_keywords = dict(self.extract_top_keywords(resume_index, top_n))
        
        # Find common keywords
        common_keywords = set(job_keywords.keys()) & set(resume_keywords.keys())
        
        # Calculate match details
        matches = {}
        for keyword in common_keywords:
            matches[keyword] = {
                'job_tfidf': job_keywords[keyword],
                'resume_tfidf': resume_keywords[keyword],
                'avg_tfidf': (job_keywords[keyword] + resume_keywords[keyword]) / 2
            }
        
        # Sort by average TF-IDF score
        sorted_matches = dict(
            sorted(matches.items(), key=lambda x: x[1]['avg_tfidf'], reverse=True)
        )
        
        match_analysis = {
            'resume_name': resume_name,
            'total_matches': len(common_keywords),
            'match_percentage': len(common_keywords) / len(job_keywords) * 100 if job_keywords else 0,
            'top_matches': list(sorted_matches.keys())[:10],
            'detailed_matches': sorted_matches,
            'job_only_keywords': list(set(job_keywords.keys()) - common_keywords)[:10],
            'resume_only_keywords': list(set(resume_keywords.keys()) - common_keywords)[:10]
        }
        
        return match_analysis
    
    def analyze_skill_overlap(self, resume_name: str) -> Dict[str, Any]:
        """
        Analyze skill overlap between job description and resume
        
        Args:
            resume_name: Name of the resume to analyze
            
        Returns:
            Dictionary containing skill overlap analysis
        """
        # Define skill patterns (can be extended)
        skill_patterns = {
            'programming': r'\b(?:python|java|javascript|c\+\+|c#|php|ruby|go|rust|swift)\b',
            'web_tech': r'\b(?:html|css|react|angular|vue|node|express|django|flask)\b',
            'databases': r'\b(?:sql|mysql|postgresql|mongodb|redis|elasticsearch)\b',
            'cloud': r'\b(?:aws|azure|gcp|docker|kubernetes|jenkins|terraform)\b',
            'data_science': r'\b(?:pandas|numpy|sklearn|tensorflow|pytorch|matplotlib)\b',
            'tools': r'\b(?:git|jira|confluence|slack|excel|powerbi|tableau)\b'
        }
        
        job_skills = {}
        resume_skills = {}
        
        # Extract skills from both texts
        job_text = self.job_description.lower()
        resume_text = self.resume_texts[resume_name].lower()
        
        for category, pattern in skill_patterns.items():
            job_matches = set(re.findall(pattern, job_text, re.IGNORECASE))
            resume_matches = set(re.findall(pattern, resume_text, re.IGNORECASE))
            
            job_skills[category] = job_matches
            resume_skills[category] = resume_matches
        
        # Calculate overlap
        overlap_analysis = {}
        for category in skill_patterns.keys():
            job_set = job_skills[category]
            resume_set = resume_skills[category]
            common = job_set & resume_set
            
            overlap_analysis[category] = {
                'job_skills': list(job_set),
                'resume_skills': list(resume_set),
                'common_skills': list(common),
                'overlap_percentage': len(common) / len(job_set) * 100 if job_set else 0
            }
        
        # Overall analysis
        all_job_skills = set()
        all_resume_skills = set()
        all_common_skills = set()
        
        for category in skill_patterns.keys():
            all_job_skills.update(job_skills[category])
            all_resume_skills.update(resume_skills[category])
            all_common_skills.update(overlap_analysis[category]['common_skills'])
        
        overall_overlap = len(all_common_skills) / len(all_job_skills) * 100 if all_job_skills else 0
        
        return {
            'resume_name': resume_name,
            'overall_overlap_percentage': overall_overlap,
            'total_job_skills': len(all_job_skills),
            'total_resume_skills': len(all_resume_skills),
            'total_common_skills': len(all_common_skills),
            'category_analysis': overlap_analysis,
            'missing_skills': list(all_job_skills - all_common_skills),
            'extra_skills': list(all_resume_skills - all_job_skills)
        }
    
    def generate_comprehensive_report(self, 
                                    top_n_resumes: int = 10,
                                    include_keyword_analysis: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive scoring report
        
        Args:
            top_n_resumes: Number of top resumes to include in detailed analysis
            include_keyword_analysis: Whether to include keyword match analysis
            
        Returns:
            Comprehensive report dictionary
        """
        if not self.similarity_scores:
            raise ValueError("Similarity scores not computed.")
        
        # Get ranked resumes
        ranked_resumes = self.get_ranked_resumes()
        
        # Basic statistics
        scores = list(self.similarity_scores.values())
        report = {
            'summary': {
                'total_resumes': len(self.resume_texts),
                'avg_similarity': np.mean(scores),
                'max_similarity': np.max(scores),
                'min_similarity': np.min(scores),
                'std_similarity': np.std(scores)
            },
            'rankings': ranked_resumes,
            'top_performers': ranked_resumes[:top_n_resumes],
            'detailed_analysis': {}
        }
        
        # Detailed analysis for top performers
        for i, (resume_name, score) in enumerate(ranked_resumes[:top_n_resumes]):
            analysis = {
                'rank': i + 1,
                'similarity_score': score,
                'score_percentile': (len([s for s in scores if s < score]) / len(scores)) * 100
            }
            
            if include_keyword_analysis:
                try:
                    keyword_matches = self.find_keyword_matches(resume_name)
                    skill_overlap = self.analyze_skill_overlap(resume_name)
                    
                    analysis['keyword_matches'] = keyword_matches
                    analysis['skill_overlap'] = skill_overlap
                except Exception as e:
                    logger.error(f"Error analyzing {resume_name}: {str(e)}")
            
            report['detailed_analysis'][resume_name] = analysis
        
        return report
    
    def export_results_to_dataframe(self) -> pd.DataFrame:
        """
        Export results to a pandas DataFrame
        
        Returns:
            DataFrame with resume names, scores, and rankings
        """
        if not self.similarity_scores:
            raise ValueError("Similarity scores not computed.")
        
        ranked_resumes = self.get_ranked_resumes()
        
        df_data = []
        for rank, (resume_name, score) in enumerate(ranked_resumes, 1):
            df_data.append({
                'Rank': rank,
                'Resume': resume_name,
                'Similarity_Score': round(score, 4),
                'Score_Percentage': round(score * 100, 2)
            })
        
        return pd.DataFrame(df_data)
    
    def debug_similarity_computation(self, job_description: str, resume_texts: Dict[str, str]):
        """
        Debug method to analyze similarity computation step by step
        Follows the testing specification for validation
        """
        from .preprocessor import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        
        print("=" * 60)
        print("🧪 SIMILARITY COMPUTATION DEBUG")
        print("=" * 60)
        
        # Show original texts
        print("📝 Job Description (first 200 chars):")
        print(job_description[:200])
        print()
        
        if resume_texts:
            first_resume_name = list(resume_texts.keys())[0]
            print(f"📄 Resume Sample - {first_resume_name} (first 200 chars):")
            print(resume_texts[first_resume_name][:200])
            print()
        
        # Show preprocessed texts
        processed_jd = preprocessor.preprocess_simple(job_description)
        print("🔧 Preprocessed Job Description (first 200 chars):")
        print(processed_jd[:200])
        print()
        
        if resume_texts:
            processed_resume = preprocessor.preprocess_simple(resume_texts[first_resume_name])
            print(f"🔧 Preprocessed Resume - {first_resume_name} (first 200 chars):")
            print(processed_resume[:200])
            print()
        
        # Compute and show scores
        self.fit_transform_documents(job_description, resume_texts, use_simple_preprocessing=True)
        scores = self.compute_similarity_scores(normalize_scores=False)
        
        print("📊 Similarity Scores:")
        for name, score in scores.items():
            category = self.categorize_score(score * 100)
            print(f"  {name}: {score:.4f} ({score*100:.2f}%) - {category}")
        print()
        
        # Show top matching keywords
        if self.feature_names is not None and len(resume_texts) > 0:
            print("🔍 Top Job Description Keywords (TF-IDF):")
            jd_keywords = self.extract_top_keywords(0, 10)
            for keyword, tfidf_score in jd_keywords[:10]:
                print(f"  {keyword}: {tfidf_score:.4f}")
            print()
            
            # Show keyword overlap for first resume
            first_resume_name = list(resume_texts.keys())[0]
            keyword_analysis = self.find_keyword_matches(first_resume_name, 50)
            print(f"🎯 Keyword Matches with {first_resume_name}:")
            common_keywords = keyword_analysis.get('common_keywords', [])
            print(f"  Total matches: {len(common_keywords)}")
            print(f"  Top matches: {', '.join(common_keywords[:10])}")
        
        print("=" * 60)


def score_resumes_pipeline(job_description: str, 
                          resume_texts: Dict[str, str],
                          scorer_config: Optional[Dict[str, Any]] = None) -> ResumeScorer:
    """
    Complete pipeline for scoring resumes against a job description
    Uses improved preprocessing and TF-IDF configuration for better similarity scores
    
    Args:
        job_description: Job description text
        resume_texts: Dictionary mapping resume names to text content
        scorer_config: Configuration for the scorer
        
    Returns:
        Configured ResumeScorer with computed scores
    """
    # Initialize scorer with config
    if scorer_config is None:
        scorer_config = {}
    
    # Adjust max_df for small datasets to avoid TF-IDF errors
    num_documents = len(resume_texts) + 1  # +1 for job description
    if num_documents <= 3:
        scorer_config['max_df'] = 1.0
    elif 'max_df' not in scorer_config:
        scorer_config['max_df'] = 0.95
    
    scorer = ResumeScorer(**scorer_config)
    
    # Run the scoring pipeline with improved preprocessing
    scorer.fit_transform_documents(
        job_description, 
        resume_texts, 
        use_simple_preprocessing=True  # Use improved preprocessing
    )
    scorer.compute_similarity_scores(normalize_scores=True)  # Normalize as percentages
    
    logger.info("Resume scoring pipeline completed successfully")
    return scorer


# Example usage and testing
if __name__ == "__main__":
    # Test the scorer
    sample_job_description = """
    We are looking for a Senior Data Scientist with expertise in Python, 
    machine learning, and statistical analysis. The ideal candidate should 
    have experience with pandas, scikit-learn, and data visualization tools.
    """
    
    sample_resumes = {
        "resume1.pdf": "Data scientist with 5 years experience in Python, machine learning, pandas, and scikit-learn.",
        "resume2.pdf": "Software engineer with Java and web development experience.",
        "resume3.pdf": "Experienced data analyst with Python, statistical analysis, and data visualization skills."
    }
    
    # Test scoring
    scorer = score_resumes_pipeline(sample_job_description, sample_resumes)
    ranked = scorer.get_ranked_resumes()
    
    print("Ranked Resumes:")
    for name, score in ranked:
        print(f"{name}: {score:.3f}")
    
    # Test keyword matching
    top_resume = ranked[0][0]
    matches = scorer.find_keyword_matches(top_resume)
    print(f"\nTop matches for {top_resume}:")
    print(f"Total matches: {matches['total_matches']}")
    print(f"Top keywords: {matches['top_matches'][:5]}")
