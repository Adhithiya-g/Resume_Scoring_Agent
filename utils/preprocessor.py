import re
import string
import logging
from typing import List, Set, Optional, Dict, Any
import nltk
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    nltk.data.find('tokenizers/punkt')
except (LookupError, Exception):
    try:
        nltk.download('punkt')
    except:
        pass

try:
    nltk.data.find('tokenizers/punkt_tab')
except (LookupError, Exception):
    try:
        nltk.download('punkt_tab')
    except:
        pass

try:
    nltk.data.find('corpora/stopwords')
except (LookupError, Exception):
    try:
        nltk.download('stopwords')
    except:
        pass

try:
    nltk.data.find('corpora/wordnet')
except (LookupError, Exception):
    try:
        nltk.download('wordnet')
    except:
        pass

# Removed omw-1.4 check as it causes BadZipFile errors

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer


class TextPreprocessor:
    
    def __init__(self, 
                 use_spacy: bool = True,
                 remove_stopwords: bool = True,
                 use_lemmatization: bool = True,
                 min_word_length: int = 2,
                 custom_stopwords: Optional[Set[str]] = None):
        self.use_spacy = use_spacy
        self.remove_stopwords = remove_stopwords
        self.use_lemmatization = use_lemmatization
        self.min_word_length = min_word_length
        
        # Initialize stopwords with fallback
        self.stop_words = set()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            # Fallback stopwords if NLTK data not available
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
                'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once'
            }
        
        try:
            self.stop_words.update(ENGLISH_STOP_WORDS)
        except:
            pass
        
        # Add custom stopwords
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)
        
        # Add common resume/job description stopwords
        resume_stopwords = {
            'resume', 'cv', 'curriculum', 'vitae', 'profile', 'summary',
            'objective', 'career', 'professional', 'work', 'job', 'position',
            'role', 'responsibilities', 'duties', 'tasks', 'achievements',
            'accomplishments', 'skills', 'education', 'experience',
            'employment', 'company', 'organization', 'team', 'project',
            'years', 'year', 'month', 'months', 'week', 'weeks', 'day', 'days'
        }
        self.stop_words.update(resume_stopwords)
        
        # Initialize lemmatizer and stemmer
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        # Initialize spaCy model if requested
        self.nlp = None
        if self.use_spacy:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
            except ImportError:
                logger.warning("spaCy not installed. Install with: pip install spacy")
                self.use_spacy = False
            except OSError:
                logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
                self.use_spacy = False
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning operations
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S*@\S*\s?', '', text)
        
        # Remove phone numbers (basic pattern)
        text = re.sub(r'[\+]?[1-9]?[0-9]{3}[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}', '', text)
        
        # Remove special characters but keep spaces and alphanumeric
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def remove_numbers(self, text: str) -> str:
        """Remove standalone numbers from text"""
        return re.sub(r'\b\d+\b', '', text)
    
    def tokenize(self, text: str) -> List[str]:
        if self.use_spacy and self.nlp:
            doc = self.nlp(text)
            return [token.text for token in doc if not token.is_punct and not token.is_space]
        else:
            try:
                return word_tokenize(text)
            except LookupError:
                # Fallback to simple split if NLTK data is not available
                import re
                # Simple tokenization as fallback
                tokens = re.findall(r'\b\w+\b', text.lower())
                return tokens
    
    def remove_stopwords_from_tokens(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered tokens
        """
        if not self.remove_stopwords:
            return tokens
        
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens using WordNet lemmatizer or spaCy
        
        Args:
            tokens: List of tokens to lemmatize
            
        Returns:
            Lemmatized tokens
        """
        if self.use_spacy and self.nlp:
            doc = self.nlp(" ".join(tokens))
            return [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
        else:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """
        Stem tokens using Porter stemmer
        
        Args:
            tokens: List of tokens to stem
            
        Returns:
            Stemmed tokens
        """
        return [self.stemmer.stem(token) for token in tokens]
    
    def filter_tokens(self, tokens: List[str]) -> List[str]:
        """
        Filter tokens by length and content
        
        Args:
            tokens: List of tokens to filter
            
        Returns:
            Filtered tokens
        """
        filtered = []
        for token in tokens:
            # Check minimum length
            if len(token) < self.min_word_length:
                continue
            
            # Skip if all digits
            if token.isdigit():
                continue
            
            # Skip if all punctuation
            if all(c in string.punctuation for c in token):
                continue
            
            filtered.append(token)
        
        return filtered
    
    def extract_skills(self, text: str) -> List[str]:
        """
        Extract potential skills from text using pattern matching
        
        Args:
            text: Text to extract skills from
            
        Returns:
            List of potential skills
        """
        # Common skill patterns (extend as needed)
        skill_patterns = [
            r'\b(?:python|java|javascript|c\+\+|sql|html|css|react|angular|vue)\b',
            r'\b(?:machine learning|deep learning|artificial intelligence|data science)\b',
            r'\b(?:aws|azure|gcp|docker|kubernetes|jenkins)\b',
            r'\b(?:excel|powerbi|tableau|salesforce|sap)\b',
            r'\b(?:project management|agile|scrum|kanban)\b'
        ]
        
        skills = []
        text_lower = text.lower()
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            skills.extend(matches)
        
        return list(set(skills))  # Remove duplicates
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities using spaCy
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dictionary of entity types and their values
        """
        entities = {}
        
        if not self.use_spacy or not self.nlp:
            return entities
        
        doc = self.nlp(text)
        
        for ent in doc.ents:
            entity_type = ent.label_
            entity_text = ent.text.strip()
            
            if entity_type not in entities:
                entities[entity_type] = []
            
            entities[entity_type].append(entity_text)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def preprocess_text(self, text: str, return_string: bool = True) -> str | List[str]:
        """
        Complete preprocessing pipeline
        
        Args:
            text: Raw text to preprocess
            return_string: Whether to return processed text as string or token list
            
        Returns:
            Processed text (string or list of tokens)
        """
        if not text or not text.strip():
            return "" if return_string else []

        # Step 1: Clean text
        cleaned_text = self.clean_text(text)
        
        # Step 2: Tokenize
        tokens = self.tokenize(cleaned_text)
        
        # Step 3: Filter tokens
        tokens = self.filter_tokens(tokens)
        
        # Step 4: Remove stopwords
        tokens = self.remove_stopwords_from_tokens(tokens)
        
        # Step 5: Lemmatize or stem
        if self.use_lemmatization:
            tokens = self.lemmatize_tokens(tokens)
        else:
            tokens = self.stem_tokens(tokens)
        
        # Final filtering
        tokens = [token for token in tokens if len(token) >= self.min_word_length]

        if return_string:
            return " ".join(tokens)
        else:
            return tokens
    
    def preprocess_simple(self, text: str) -> str:
        import re
        
        if not text or not text.strip():
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Tokenize and remove stopwords
        tokens = text.split()
        
        # Use fallback stopwords if NLTK data is not available
        try:
            from nltk.corpus import stopwords
            english_stopwords = set(stopwords.words('english'))
        except:
            # Fallback basic stopwords list
            english_stopwords = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
                'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once'
            }
        
        tokens = [word for word in tokens if word not in english_stopwords and len(word) > 2]
        
        return ' '.join(tokens)
    
    def preprocess_documents(self, documents: Dict[str, str]) -> Dict[str, str]:
        """
        Preprocess multiple documents
        
        Args:
            documents: Dictionary mapping document names to text content
            
        Returns:
            Dictionary of preprocessed documents
        """
        preprocessed = {}
        
        for doc_name, text in documents.items():
            try:
                preprocessed[doc_name] = self.preprocess_text(text)
                logger.info(f"Preprocessed document: {doc_name}")
            except Exception as e:
                logger.error(f"Error preprocessing {doc_name}: {str(e)}")
                preprocessed[doc_name] = ""
        
        return preprocessed
    
    def get_text_statistics(self, text: str) -> Dict[str, Any]:
        """
        Get statistical information about text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of text statistics
        """
        if not text:
            return {}
        
        # Basic statistics
        stats = {
            'character_count': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(sent_tokenize(text)),
            'avg_word_length': sum(len(word) for word in text.split()) / len(text.split()) if text.split() else 0
        }
        
        # Processed statistics
        processed_text = self.preprocess_text(text, return_string=False)
        stats['processed_word_count'] = len(processed_text)
        stats['unique_words'] = len(set(processed_text))
        stats['lexical_diversity'] = len(set(processed_text)) / len(processed_text) if processed_text else 0
        
        return stats


def create_custom_preprocessor(config: Dict[str, Any]) -> TextPreprocessor:
    """
    Create a custom preprocessor based on configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured TextPreprocessor instance
    """
    return TextPreprocessor(
        use_spacy=config.get('use_spacy', True),
        remove_stopwords=config.get('remove_stopwords', True),
        use_lemmatization=config.get('use_lemmatization', True),
        min_word_length=config.get('min_word_length', 2),
        custom_stopwords=config.get('custom_stopwords', None)
    )


# Example usage and testing
if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = TextPreprocessor()
    
    sample_text = """
    John Doe is a skilled Software Engineer with 5 years of experience in Python, 
    JavaScript, and machine learning. He has worked on various projects involving 
    data analysis and web development. Contact: john.doe@email.com, +1-555-0123.
    """
    
    # Test preprocessing
    processed = preprocessor.preprocess_text(sample_text)
    print("Original:", sample_text[:100])
    print("Processed:", processed)
    
    # Test entity extraction
    entities = preprocessor.extract_entities(sample_text)
    print("Entities:", entities)
    
    # Test statistics
    stats = preprocessor.get_text_statistics(sample_text)
    print("Statistics:", stats)
