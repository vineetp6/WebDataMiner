import re
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Optional
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_filter')

# Initialize NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def filter_text_data(
    data: Union[str, Dict, List[Dict]], 
    include_terms: List[str] = None, 
    exclude_terms: List[str] = None,
    case_sensitive: bool = False
) -> Union[str, Dict, List[Dict]]:
    """
    Filter text data based on inclusion and exclusion terms.
    
    Args:
        data: Text data or dictionary/list of dictionaries containing text data
        include_terms: List of terms that should be included in the filtered result
        exclude_terms: List of terms that should be excluded from the filtered result
        case_sensitive: Whether string matching should be case-sensitive
        
    Returns:
        Filtered data in the same format as the input
    """
    # Handle empty filter lists
    include_terms = include_terms or []
    exclude_terms = exclude_terms or []
    
    # Strip whitespace from filter terms
    include_terms = [term.strip() for term in include_terms if term and term.strip()]
    exclude_terms = [term.strip() for term in exclude_terms if term and term.strip()]
    
    # If no filter terms provided, return the original data
    if not include_terms and not exclude_terms:
        return data
    
    # Handle different data types
    if isinstance(data, str):
        return _filter_text_string(data, include_terms, exclude_terms, case_sensitive)
    
    elif isinstance(data, dict):
        return _filter_dict(data, include_terms, exclude_terms, case_sensitive)
    
    elif isinstance(data, list):
        return [_filter_dict(item, include_terms, exclude_terms, case_sensitive) 
                if isinstance(item, dict) else item 
                for item in data]
    
    # Return data as is if it's not a recognized type
    return data

def _filter_text_string(
    text: str, 
    include_terms: List[str], 
    exclude_terms: List[str],
    case_sensitive: bool
) -> str:
    """
    Filter a text string based on inclusion and exclusion terms.
    
    Args:
        text: Text string to filter
        include_terms: Terms that should be included
        exclude_terms: Terms that should be excluded
        case_sensitive: Whether string matching should be case-sensitive
        
    Returns:
        str: Filtered text string
    """
    if not text:
        return text
    
    # Process include terms
    if include_terms:
        filtered_lines = []
        lines = text.split('\n')
        
        for line in lines:
            flags = 0 if case_sensitive else re.IGNORECASE
            if any(re.search(r'\b' + re.escape(term) + r'\b', line, flags) for term in include_terms):
                filtered_lines.append(line)
        
        text = '\n'.join(filtered_lines)
    
    # Process exclude terms
    if exclude_terms and text:
        flags = 0 if case_sensitive else re.IGNORECASE
        for term in exclude_terms:
            text = re.sub(r'\b' + re.escape(term) + r'\b.*?\n', '', text, flags=flags)
    
    return text

def _filter_dict(
    data_dict: Dict, 
    include_terms: List[str], 
    exclude_terms: List[str],
    case_sensitive: bool
) -> Dict:
    """
    Filter a dictionary containing text data.
    
    Args:
        data_dict: Dictionary containing text data
        include_terms: Terms that should be included
        exclude_terms: Terms that should be excluded
        case_sensitive: Whether string matching should be case-sensitive
        
    Returns:
        Dict: Filtered dictionary
    """
    result = data_dict.copy()
    
    # Filter 'content' field if it exists and is a string
    if 'content' in result and isinstance(result['content'], str):
        result['content'] = _filter_text_string(
            result['content'], 
            include_terms, 
            exclude_terms, 
            case_sensitive
        )
    
    # Filter 'search_results' field if it exists
    if 'search_results' in result and isinstance(result['search_results'], list):
        filtered_results = []
        
        for item in result['search_results']:
            # Skip non-dict items
            if not isinstance(item, dict):
                filtered_results.append(item)
                continue
                
            # Check if the term should be included
            term = item.get('term', '')
            context = item.get('context', '')
            
            if include_terms and term not in include_terms:
                continue
                
            if exclude_terms and term in exclude_terms:
                continue
                
            # Filter context text if present
            if context:
                filtered_context = _filter_text_string(
                    context, 
                    include_terms, 
                    exclude_terms, 
                    case_sensitive
                )
                
                # Only include if context still contains content after filtering
                if filtered_context.strip():
                    item_copy = item.copy()
                    item_copy['context'] = filtered_context
                    filtered_results.append(item_copy)
            else:
                filtered_results.append(item)
                
        result['search_results'] = filtered_results
    
    return result

def apply_advanced_filters(
    data: Union[Dict, List[Dict]], 
    filters: Dict[str, Any]
) -> Union[Dict, List[Dict]]:
    """
    Apply advanced filters to the data.
    
    Args:
        data: Dictionary or list of dictionaries containing data
        filters: Dictionary of filter settings
            - min_word_count: Minimum word count
            - max_word_count: Maximum word count
            - regex_pattern: Custom regex pattern to match
            - date_filter: Date range for filtering
            
    Returns:
        Filtered data in the same format as the input
    """
    if not filters:
        return data
    
    # Extract filter options
    min_word_count = filters.get('min_word_count')
    max_word_count = filters.get('max_word_count')
    regex_pattern = filters.get('regex_pattern')
    date_filter = filters.get('date_filter')
    
    # Apply filters to different data types
    if isinstance(data, dict):
        return _apply_advanced_filters_to_dict(data, min_word_count, max_word_count, regex_pattern, date_filter)
    
    elif isinstance(data, list):
        return [_apply_advanced_filters_to_dict(item, min_word_count, max_word_count, regex_pattern, date_filter) 
                if isinstance(item, dict) else item 
                for item in data]
    
    # Return data as is if it's not a recognized type
    return data

def _apply_advanced_filters_to_dict(
    data_dict: Dict, 
    min_word_count: Optional[int], 
    max_word_count: Optional[int],
    regex_pattern: Optional[str],
    date_filter: Optional[List[datetime]]
) -> Dict:
    """
    Apply advanced filters to a dictionary.
    
    Args:
        data_dict: Dictionary containing data
        min_word_count: Minimum word count
        max_word_count: Maximum word count
        regex_pattern: Custom regex pattern to match
        date_filter: Date range for filtering
        
    Returns:
        Dict: Filtered dictionary
    """
    result = data_dict.copy()
    
    # Filter content field if it exists
    if 'content' in result and isinstance(result['content'], str):
        content = result['content']
        
        # Apply word count filter
        if min_word_count is not None or max_word_count is not None:
            word_count = len(word_tokenize(content))
            
            if min_word_count is not None and word_count < min_word_count:
                result['content'] = ''
            
            if max_word_count is not None and word_count > max_word_count:
                # Truncate content to max words
                words = word_tokenize(content)
                result['content'] = ' '.join(words[:max_word_count])
        
        # Apply regex pattern if provided
        if regex_pattern and result['content']:
            try:
                pattern = re.compile(regex_pattern)
                matches = pattern.findall(result['content'])
                if matches:
                    # Join all matched content
                    result['content'] = '\n'.join(matches)
                else:
                    result['content'] = ''
            except re.error as e:
                logger.error(f"Invalid regex pattern: {e}")
        
    # Apply date filter if provided
    if date_filter and len(date_filter) == 2 and 'crawl_time' in result:
        try:
            # Convert timestamp to datetime if it's a number
            if isinstance(result['crawl_time'], (int, float)):
                crawl_datetime = datetime.fromtimestamp(result['crawl_time'])
            # If it's already a datetime object
            elif isinstance(result['crawl_time'], datetime):
                crawl_datetime = result['crawl_time']
            # If it's a string, try to parse it
            elif isinstance(result['crawl_time'], str):
                try:
                    crawl_datetime = datetime.fromisoformat(result['crawl_time'])
                except:
                    crawl_datetime = datetime.strptime(result['crawl_time'], "%Y-%m-%d %H:%M:%S")
            else:
                # Skip date filtering if we can't parse the date
                crawl_datetime = None
            
            # Apply date range filter
            if crawl_datetime:
                start_date, end_date = date_filter
                if not (start_date <= crawl_datetime.date() <= end_date):
                    # Clear content if outside date range
                    result['content'] = ''
        except Exception as e:
            logger.error(f"Error applying date filter: {e}")
    
    return result

def remove_stopwords(text: str) -> str:
    """
    Remove stopwords from text.
    
    Args:
        text: Input text
        
    Returns:
        str: Text with stopwords removed
    """
    if not text:
        return text
        
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [w for w in word_tokens if w.lower() not in stop_words]
    return ' '.join(filtered_text)

def clean_html(text: str) -> str:
    """
    Remove HTML tags from text.
    
    Args:
        text: HTML text
        
    Returns:
        str: Clean text without HTML tags
    """
    if not text:
        return text
        
    # Remove HTML tags
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

if __name__ == "__main__":
    # Test the filters
    test_text = """
    This is a sample text about web crawling.
    Web crawlers, also known as spiders, are automated programs.
    They browse the World Wide Web for various purposes.
    Search engines use crawlers to update their index.
    """
    
    # Test basic filtering
    filtered = filter_text_data(
        test_text,
        include_terms=["crawler", "spider"],
        exclude_terms=["index"],
        case_sensitive=False
    )
    
    print("Basic filtering result:")
    print(filtered)
    
    # Test advanced filtering
    test_dict = {
        "content": test_text,
        "crawl_time": datetime.now().timestamp()
    }
    
    advanced_filtered = apply_advanced_filters(
        test_dict,
        {
            "min_word_count": 10,
            "regex_pattern": r".*crawler.*",
            "date_filter": [datetime.now().date(), datetime.now().date()]
        }
    )
    
    print("\nAdvanced filtering result:")
    print(advanced_filtered["content"])
