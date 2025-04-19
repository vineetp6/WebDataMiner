import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import spacy
import pandas as pd
from typing import Dict, List, Union, Any, Optional
import logging
from collections import Counter
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sentiment_analyzer')

# Initialize NLTK resources
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

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
    Clean and normalize text for sentiment analysis.
    
    Args:
        text: Input text to clean
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep punctuation for sentiment analysis
    text = re.sub(r'[^\w\s.,!?:;]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def analyze_sentiment(
    data: Union[Dict, List[Dict], pd.DataFrame, str],
    model: str = 'vader',
    detailed: bool = False
) -> Dict[str, Any]:
    """
    Analyze sentiment of text.
    
    Args:
        data: Input data
        model: Sentiment analysis model to use ('vader', 'textblob', or 'spacy')
        detailed: Whether to return detailed breakdown
        
    Returns:
        Dict: Dictionary containing sentiment analysis results
    """
    # Extract text from data
    text = extract_text_from_data(data)
    
    if not text:
        return {
            'overall': 'neutral',
            'polarity': 0.0,
            'breakdown': {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0} if detailed else {}
        }
    
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Analyze sentiment based on selected model
    if model == 'vader':
        return _analyze_with_vader(cleaned_text, detailed)
    elif model == 'textblob':
        return _analyze_with_textblob(cleaned_text, detailed)
    elif model == 'spacy':
        return _analyze_with_spacy(cleaned_text, detailed)
    else:
        # Default to VADER if model not recognized
        logger.warning(f"Unknown sentiment model '{model}', using VADER instead")
        return _analyze_with_vader(cleaned_text, detailed)

def _analyze_with_vader(text: str, detailed: bool) -> Dict[str, Any]:
    """
    Analyze sentiment using VADER (Valence Aware Dictionary and sEntiment Reasoner).
    
    Args:
        text: Text to analyze
        detailed: Whether to return detailed breakdown
        
    Returns:
        Dict: Dictionary containing sentiment analysis results
    """
    # Initialize the VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Get sentiment scores
    sentiment_scores = sia.polarity_scores(text)
    
    # Determine overall sentiment
    compound_score = sentiment_scores['compound']
    
    if compound_score >= 0.05:
        overall_sentiment = 'positive'
    elif compound_score <= -0.05:
        overall_sentiment = 'negative'
    else:
        overall_sentiment = 'neutral'
    
    # Create result dictionary
    result = {
        'overall': overall_sentiment,
        'polarity': compound_score,
    }
    
    # Add detailed breakdown if requested
    if detailed:
        result['breakdown'] = {
            'positive': sentiment_scores['pos'],
            'neutral': sentiment_scores['neu'],
            'negative': sentiment_scores['neg']
        }
    
    return result

def _analyze_with_textblob(text: str, detailed: bool) -> Dict[str, Any]:
    """
    Analyze sentiment using TextBlob.
    
    Args:
        text: Text to analyze
        detailed: Whether to return detailed breakdown
        
    Returns:
        Dict: Dictionary containing sentiment analysis results
    """
    # Create TextBlob object
    blob = TextBlob(text)
    
    # Get sentiment polarity (-1 to 1)
    polarity = blob.sentiment.polarity
    
    # Get subjectivity (0 to 1)
    subjectivity = blob.sentiment.subjectivity
    
    # Determine overall sentiment
    if polarity >= 0.05:
        overall_sentiment = 'positive'
    elif polarity <= -0.05:
        overall_sentiment = 'negative'
    else:
        overall_sentiment = 'neutral'
    
    # Create result dictionary
    result = {
        'overall': overall_sentiment,
        'polarity': polarity,
    }
    
    # Add detailed breakdown if requested
    if detailed:
        # Calculate approximate scores for positive, negative, and neutral
        if polarity > 0:
            pos_score = polarity
            neg_score = 0
        else:
            pos_score = 0
            neg_score = -polarity
        
        neu_score = 1.0 - (pos_score + neg_score)
        
        result['breakdown'] = {
            'positive': pos_score,
            'neutral': neu_score,
            'negative': neg_score,
            'subjectivity': subjectivity
        }
    
    return result

def _analyze_with_spacy(text: str, detailed: bool) -> Dict[str, Any]:
    """
    Analyze sentiment using spaCy (with a basic approach).
    
    Args:
        text: Text to analyze
        detailed: Whether to return detailed breakdown
        
    Returns:
        Dict: Dictionary containing sentiment analysis results
    """
    # Use TextBlob as a fallback since spaCy doesn't have built-in sentiment analysis
    logger.info("Using TextBlob as a fallback for spaCy sentiment analysis")
    return _analyze_with_textblob(text, detailed)

def analyze_emotions(
    data: Union[Dict, List[Dict], pd.DataFrame, str],
    extended: bool = False
) -> Dict[str, Any]:
    """
    Analyze emotions in text.
    
    Args:
        data: Input data
        extended: Whether to use extended emotion set
        
    Returns:
        Dict: Dictionary containing emotion analysis results
    """
    # Extract text from data
    text = extract_text_from_data(data)
    
    if not text:
        if extended:
            return {
                'dominant': 'neutral',
                'scores': {
                    'Joy': 0.0, 'Sadness': 0.0, 'Anger': 0.0, 'Fear': 0.0, 
                    'Surprise': 0.0, 'Disgust': 0.0, 'Trust': 0.0, 'Anticipation': 0.0
                }
            }
        else:
            return {
                'dominant': 'neutral',
                'scores': {'Joy': 0.0, 'Sadness': 0.0, 'Anger': 0.0, 'Fear': 0.0, 'Surprise': 0.0}
            }
    
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Define emotion lexicons (simplified for illustration)
    emotion_lexicons = {
        'Joy': ['happy', 'joy', 'delighted', 'glad', 'pleased', 'happiness', 'cheerful', 'content', 'thrilled', 'enjoy'],
        'Sadness': ['sad', 'unhappy', 'sorrow', 'depressed', 'gloomy', 'miserable', 'grief', 'mourning', 'despair', 'melancholy'],
        'Anger': ['angry', 'mad', 'furious', 'enraged', 'irritated', 'annoyed', 'rage', 'outrage', 'wrath', 'hatred'],
        'Fear': ['afraid', 'scared', 'frightened', 'terrified', 'anxious', 'worried', 'panic', 'horror', 'terror', 'dread'],
        'Surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'stunned', 'unexpected', 'wonder', 'startled', 'incredible', 'wow']
    }
    
    # Add extended emotions if requested
    if extended:
        emotion_lexicons.update({
            'Disgust': ['disgusted', 'revolted', 'repulsed', 'nauseous', 'sickened', 'gross', 'distaste', 'aversion', 'revulsion', 'yuck'],
            'Trust': ['trust', 'believe', 'faith', 'confidence', 'reliable', 'dependable', 'loyal', 'honest', 'credible', 'authentic'],
            'Anticipation': ['anticipate', 'expect', 'await', 'foresee', 'hope', 'looking forward', 'keen', 'excited', 'eager', 'enthusiasm']
        })
    
    # Count emotion words
    word_tokens = cleaned_text.lower().split()
    emotion_counts = {emotion: 0.0 for emotion in emotion_lexicons}
    
    for word in word_tokens:
        for emotion, words in emotion_lexicons.items():
            if any(emotion_word in word for emotion_word in words):
                emotion_counts[emotion] += 1
    
    # Normalize scores
    total_emotion_words = sum(emotion_counts.values())
    
    if total_emotion_words > 0:
        emotion_scores = {emotion: count / (total_emotion_words + 1) for emotion, count in emotion_counts.items()}
    else:
        # If no emotion words found, assign small baseline values
        emotion_scores = {emotion: 0.2 / len(emotion_lexicons) for emotion in emotion_lexicons}
    
    # Find dominant emotion
    dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
    
    return {
        'dominant': dominant_emotion,
        'scores': emotion_scores
    }

if __name__ == "__main__":
    # Test the sentiment analyzer with sample texts
    positive_text = "I love this product! It's amazing and works perfectly. Highly recommended."
    negative_text = "This is terrible. I'm very disappointed and frustrated with the poor quality."
    neutral_text = "The product arrived yesterday. It contains several components and a user manual."
    
    print("Testing VADER sentiment analysis...")
    print("Positive text:", analyze_sentiment(positive_text, model='vader')['overall'])
    print("Negative text:", analyze_sentiment(negative_text, model='vader')['overall'])
    print("Neutral text:", analyze_sentiment(neutral_text, model='vader')['overall'])
    
    print("\nTesting TextBlob sentiment analysis...")
    print("Positive text:", analyze_sentiment(positive_text, model='textblob')['overall'])
    print("Negative text:", analyze_sentiment(negative_text, model='textblob')['overall'])
    print("Neutral text:", analyze_sentiment(neutral_text, model='textblob')['overall'])
    
    print("\nTesting emotion analysis...")
    emotions = analyze_emotions(positive_text, extended=True)
    print("Dominant emotion:", emotions['dominant'])
    print("Emotion scores:", emotions['scores'])
