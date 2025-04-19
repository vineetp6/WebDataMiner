import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk import ne_chunk, pos_tag
from nltk.tree import Tree
from typing import Dict, List, Tuple, Union, Any, Optional
import re
import spacy
import pandas as pd
import logging
from collections import Counter
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('text_analyzer')

# Initialize NLTK resources
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('chunkers/maxent_ne_chunker')
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)

# Initialize spaCy model (will load on first use)
_spacy_nlp = None

def get_spacy_nlp():
    """
    Get or initialize the spaCy NLP model.
    
    Returns:
        spacy.Language: Initialized spaCy NLP model
    """
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            _spacy_nlp = spacy.load('en_core_web_sm')
        except:
            # If model not found, download it
            logger.info("Downloading spaCy model...")
            import os
            os.system('python -m spacy download en_core_web_sm')
            _spacy_nlp = spacy.load('en_core_web_sm')
    return _spacy_nlp

def extract_text_from_data(data: Union[Dict, List[Dict], pd.DataFrame, str]) -> str:
    """
    Extract text content from various data formats.
    
    Args:
        data: Input data (can be dictionary, list, DataFrame, or string)
        
    Returns:
        str: Extracted text content
    """
    # Handle string directly
    if isinstance(data, str):
        return data
        
    # Handle dictionary
    elif isinstance(data, dict):
        # Extract content field if it exists
        return data.get('content', '')
        
    # Handle list of dictionaries
    elif isinstance(data, list):
        # Combine content from all dictionaries in the list
        contents = []
        for item in data:
            if isinstance(item, dict):
                content = item.get('content', '')
                if content:
                    contents.append(content)
        return ' '.join(contents)
        
    # Handle DataFrame
    elif isinstance(data, pd.DataFrame):
        # Check if 'content' column exists
        if 'content' in data.columns:
            return ' '.join(data['content'].astype(str).tolist())
        else:
            # Try to find a column that might contain text
            for col in data.columns:
                if data[col].dtype == 'object':
                    return ' '.join(data[col].astype(str).tolist())
    
    # Return empty string if no text content found
    return ''

def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text to clean
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_entities(
    data: Union[Dict, List[Dict], pd.DataFrame, str],
    entity_types: List[str] = None,
    min_confidence: float = 0.5
) -> Dict[str, List[Tuple[str, int]]]:
    """
    Extract named entities from text.
    
    Args:
        data: Input data
        entity_types: Types of entities to extract (e.g., 'PERSON', 'ORGANIZATION')
        min_confidence: Minimum confidence score for entity extraction
        
    Returns:
        Dict: Dictionary mapping entity types to lists of (entity, count) tuples
    """
    # Default entity types if none provided
    if entity_types is None:
        entity_types = ['PERSON', 'ORGANIZATION', 'LOCATION', 'GPE']
    
    # Convert all entity types to uppercase for consistency
    entity_types = [etype.upper() for etype in entity_types]
    
    # Extract text from data
    text = extract_text_from_data(data)
    
    if not text:
        return {etype: [] for etype in entity_types}
    
    # Use spaCy for entity recognition
    nlp = get_spacy_nlp()
    
    # Process the text with spaCy
    doc = nlp(text)
    
    # Map spaCy entity types to standard types
    spacy_to_std = {
        'PERSON': 'PERSON',
        'ORG': 'ORGANIZATION',
        'GPE': 'GPE',  # Geopolitical entity
        'LOC': 'LOCATION',
        'PRODUCT': 'PRODUCT',
        'EVENT': 'EVENT',
        'WORK_OF_ART': 'WORK_OF_ART',
        'DATE': 'DATE',
        'TIME': 'TIME',
        'MONEY': 'MONEY',
        'PERCENT': 'PERCENT',
        'LANGUAGE': 'LANGUAGE',
        'NORP': 'NORP'  # Nationalities or religious or political groups
    }
    
    # Extract entities and count occurrences
    entities = {}
    for etype in entity_types:
        entities[etype] = []
    
    entity_counts = Counter()
    
    for ent in doc.ents:
        # Map spaCy entity type to standard type
        std_type = spacy_to_std.get(ent.label_, ent.label_)
        
        # Skip if entity type not in requested types
        if std_type not in entity_types:
            continue
        
        # Only include if confidence is above threshold
        if hasattr(ent, 'score') and ent.score < min_confidence:
            continue
        
        # Count this entity
        entity_text = ent.text.strip()
        if entity_text:
            entity_counts[(std_type, entity_text)] += 1
    
    # Organize entities by type
    for (etype, entity), count in entity_counts.items():
        if etype in entities:
            entities[etype].append((entity, count))
    
    # Sort entities by count within each type
    for etype in entities:
        entities[etype] = sorted(entities[etype], key=lambda x: x[1], reverse=True)
    
    return entities

def extract_keywords(
    data: Union[Dict, List[Dict], pd.DataFrame, str],
    max_keywords: int = 20,
    use_phrases: bool = True
) -> List[Tuple[str, float]]:
    """
    Extract keywords from text using TF-IDF.
    
    Args:
        data: Input data
        max_keywords: Maximum number of keywords to extract
        use_phrases: Whether to extract multi-word phrases
        
    Returns:
        List[Tuple[str, float]]: List of (keyword, score) tuples
    """
    # Extract text from data
    text = extract_text_from_data(data)
    
    if not text:
        return []
    
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Tokenize sentences for better processing
    sentences = sent_tokenize(cleaned_text)
    
    # Set up TF-IDF vectorizer
    ngram_range = (1, 2) if use_phrases else (1, 1)
    
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        stop_words='english',
        max_features=max_keywords * 2,  # Get more than we need for filtering
        token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'  # Only words with letters
    )
    
    # Calculate TF-IDF
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()
        
        # Sum TF-IDF scores across sentences
        tfidf_scores = np.sum(tfidf_matrix.toarray(), axis=0)
        
        # Create (keyword, score) pairs
        keyword_scores = [(feature_names[i], tfidf_scores[i]) for i in range(len(feature_names))]
        
        # Sort by score in descending order and take top keywords
        keyword_scores = sorted(keyword_scores, key=lambda x: x[1], reverse=True)[:max_keywords]
        
        return keyword_scores
        
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        
        # Fallback to NLTK for keyword extraction
        words = word_tokenize(cleaned_text)
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word.lower() not in stop_words and len(word) > 2]
        
        fdist = FreqDist(words)
        keywords = [(word, score / len(words)) for word, score in fdist.most_common(max_keywords)]
        
        return keywords

def categorize_text(
    data: Union[Dict, List[Dict], pd.DataFrame, str],
    taxonomy: str = 'general_topics',
    custom_categories: List[str] = None,
    threshold: float = 0.3
) -> List[Tuple[str, float]]:
    """
    Categorize text into predefined or custom categories.
    
    Args:
        data: Input data
        taxonomy: Categorization taxonomy to use
        custom_categories: Custom categories (if taxonomy is 'custom')
        threshold: Confidence threshold for categorization
        
    Returns:
        List[Tuple[str, float]]: List of (category, confidence) tuples
    """
    # Extract text from data
    text = extract_text_from_data(data)
    
    if not text:
        return []
    
    # Clean and prepare the text
    cleaned_text = clean_text(text)
    
    # Get categories based on taxonomy
    if taxonomy == 'custom' and custom_categories:
        categories = custom_categories
    else:
        # Default taxonomies
        if taxonomy == 'news_categories':
            categories = [
                'Politics', 'Business', 'Technology', 'Science', 'Health', 
                'Sports', 'Entertainment', 'Education', 'Environment', 'World'
            ]
        elif taxonomy == 'academic_subjects':
            categories = [
                'Mathematics', 'Physics', 'Chemistry', 'Biology', 'Computer Science',
                'Engineering', 'Medicine', 'Economics', 'Psychology', 'Sociology',
                'History', 'Literature', 'Philosophy', 'Arts', 'Law'
            ]
        else:  # general_topics
            categories = [
                'Technology', 'Business', 'Health', 'Entertainment', 'Sports',
                'Science', 'Politics', 'Environment', 'Education', 'Travel',
                'Food', 'Fashion', 'Arts', 'Culture', 'Lifestyle'
            ]
    
    # Extract keywords from the text
    keywords = extract_keywords(cleaned_text, max_keywords=50, use_phrases=True)
    keyword_text = ' '.join([kw for kw, _ in keywords])
    
    # Calculate similarity between text and each category
    category_scores = []
    
    for category in categories:
        # Simple approach: calculate similarity based on keyword presence
        category_lower = category.lower()
        
        # Check for category word in keywords
        direct_match = category_lower in cleaned_text.lower()
        
        # Calculate similarity score
        word_matches = 0
        for word in category.lower().split():
            if word in cleaned_text.lower():
                word_matches += 1
        
        # Calculate base similarity score
        base_score = word_matches / len(category.split()) if category.split() else 0
        
        # Boost score for direct matches
        if direct_match:
            base_score = max(base_score, 0.5)
        
        # Only include categories with scores above threshold
        if base_score > threshold:
            category_scores.append((category, min(base_score, 0.99)))
    
    # If we have spaCy available, use it for more advanced categorization
    try:
        nlp = get_spacy_nlp()
        doc = nlp(cleaned_text)
        
        # Try to identify categories based on entities and key phrases
        for category in categories:
            found = False
            for ent in doc.ents:
                if category.lower() in ent.text.lower():
                    found = True
                    break
            
            if found and (category, 0.0) not in category_scores:
                category_scores.append((category, 0.7))
    except:
        pass
    
    # Sort categories by confidence score
    category_scores = sorted(category_scores, key=lambda x: x[1], reverse=True)
    
    # If no categories found, try to infer from content
    if not category_scores:
        # Create a simple mapping of keywords to potential categories
        keyword_to_category = {
            'technology': 'Technology', 'tech': 'Technology', 'software': 'Technology',
            'business': 'Business', 'economy': 'Business', 'market': 'Business',
            'health': 'Health', 'medical': 'Health', 'disease': 'Health',
            'sport': 'Sports', 'game': 'Sports', 'team': 'Sports',
            'science': 'Science', 'research': 'Science', 'study': 'Science',
            'politic': 'Politics', 'government': 'Politics', 'election': 'Politics',
            'art': 'Arts', 'music': 'Arts', 'film': 'Entertainment',
            'environment': 'Environment', 'climate': 'Environment', 'nature': 'Environment'
        }
        
        # Check for keywords in text
        for keyword, category in keyword_to_category.items():
            if keyword in cleaned_text.lower() and category in categories:
                category_scores.append((category, 0.6))
        
        # Remove duplicates and sort again
        seen_categories = set()
        unique_scores = []
        for cat, score in category_scores:
            if cat not in seen_categories:
                seen_categories.add(cat)
                unique_scores.append((cat, score))
        
        category_scores = sorted(unique_scores, key=lambda x: x[1], reverse=True)
    
    return category_scores

def analyze_text_statistics(
    data: Union[Dict, List[Dict], pd.DataFrame, str]
) -> Dict[str, Any]:
    """
    Analyze text statistics like word count, sentence count, etc.
    
    Args:
        data: Input data
        
    Returns:
        Dict: Dictionary of text statistics
    """
    # Extract text from data
    text = extract_text_from_data(data)
    
    if not text:
        return {"error": "No text content available for analysis"}
    
    # Tokenize text
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    
    # Calculate statistics
    word_count = len(words)
    sentence_count = len(sentences)
    
    # Average word length
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
    
    # Average sentence length
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    
    # Remove stopwords and calculate lexical density
    stop_words = set(stopwords.words('english'))
    non_stop_words = [word for word in words if word.lower() not in stop_words]
    lexical_density = len(non_stop_words) / word_count if word_count > 0 else 0
    
    # Count punctuation
    punct_count = sum(1 for char in text if char in string.punctuation)
    
    # Word frequency distribution
    fdist = FreqDist(word.lower() for word in words if word.isalpha())
    most_common_words = fdist.most_common(10)
    
    # Readability score (very simplified Flesch Reading Ease)
    readability = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length / 5)
    readability = max(0, min(100, readability))  # Clamp to 0-100 range
    
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": round(avg_word_length, 2),
        "avg_sentence_length": round(avg_sentence_length, 2),
        "lexical_density": round(lexical_density, 2),
        "punctuation_count": punct_count,
        "most_common_words": most_common_words,
        "readability_score": round(readability, 2)
    }

if __name__ == "__main__":
    # Test the text analyzer with a sample text
    test_text = """
    Web crawler applications are essential tools in the modern digital landscape.
    Developed by technology companies like Google, these automated programs systematically browse the World Wide Web.
    They collect information from websites, index content, and make it available for search engines.
    Organizations use web crawlers for competitive intelligence and data mining purposes.
    The development of advanced web crawlers requires expertise in programming languages such as Python or Java.
    """
    
    print("Testing entity extraction...")
    entities = extract_entities(test_text)
    for entity_type, entity_list in entities.items():
        if entity_list:
            print(f"{entity_type}:")
            for entity, count in entity_list:
                print(f"  - {entity}: {count}")
    
    print("\nTesting keyword extraction...")
    keywords = extract_keywords(test_text, max_keywords=10)
    for keyword, score in keywords:
        print(f"  - {keyword}: {score:.4f}")
    
    print("\nTesting text categorization...")
    categories = categorize_text(test_text)
    for category, confidence in categories:
        print(f"  - {category}: {confidence:.4f}")
    
    print("\nTesting text statistics...")
    stats = analyze_text_statistics(test_text)
    for key, value in stats.items():
        if key != "most_common_words":
            print(f"  - {key}: {value}")
        else:
            print(f"  - most_common_words:")
            for word, count in value:
                print(f"    - {word}: {count}")
