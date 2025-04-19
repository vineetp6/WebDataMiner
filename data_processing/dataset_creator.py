import pandas as pd
import json
import csv
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from typing import Dict, List, Union, Any, Tuple, Optional
from datetime import datetime
import logging
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dataset_creator')

# Initialize NLTK resources
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)

def create_dataset(
    data: Union[Dict, List[Dict]],
    include_metadata: bool = True,
    include_links: bool = True,
    processing_options: Dict[str, bool] = None
) -> pd.DataFrame:
    """
    Create a dataset from crawled data.
    
    Args:
        data: Crawled data as a dictionary or list of dictionaries
        include_metadata: Whether to include metadata fields in the dataset
        include_links: Whether to include extracted links in the dataset
        processing_options: Text processing options
            - remove_stopwords: Whether to remove stopwords
            - lemmatize: Whether to lemmatize text
            - lowercase: Whether to convert text to lowercase
            - remove_punct: Whether to remove punctuation
            - remove_html: Whether to remove HTML tags
            
    Returns:
        pd.DataFrame: Dataset as a pandas DataFrame
    """
    # Initialize processing options
    if processing_options is None:
        processing_options = {
            'remove_stopwords': False,
            'lemmatize': False,
            'lowercase': False,
            'remove_punct': False,
            'remove_html': True
        }
    
    # Convert single dictionary to list for consistent processing
    if isinstance(data, dict):
        data_list = [data]
    else:
        data_list = data
    
    # Skip if data is empty
    if not data_list:
        return pd.DataFrame()
    
    # Process the data into a format suitable for a DataFrame
    processed_data = []
    
    for item in data_list:
        # Skip if not a dictionary
        if not isinstance(item, dict):
            continue
        
        # Create a new dict for the processed item
        processed_item = {}
        
        # Extract and process content
        content = item.get('content', '')
        if content:
            # Apply text processing options
            content = process_text(content, processing_options)
        
        processed_item['content'] = content
        
        # Include metadata if requested
        if include_metadata:
            processed_item['url'] = item.get('url', '')
            processed_item['status'] = item.get('status', '')
            processed_item['depth'] = item.get('depth', 0)
            
            # Process timestamp if available
            if 'crawl_time' in item:
                try:
                    # Handle different timestamp formats
                    if isinstance(item['crawl_time'], (int, float)):
                        crawl_time = datetime.fromtimestamp(item['crawl_time'])
                        processed_item['crawl_time'] = crawl_time.strftime('%Y-%m-%d %H:%M:%S')
                    elif isinstance(item['crawl_time'], str):
                        processed_item['crawl_time'] = item['crawl_time']
                    else:
                        processed_item['crawl_time'] = str(item['crawl_time'])
                except Exception as e:
                    logger.error(f"Error processing timestamp: {e}")
                    processed_item['crawl_time'] = str(item.get('crawl_time', ''))
        
        # Include links if requested
        if include_links:
            links = item.get('links', [])
            if isinstance(links, list):
                processed_item['links'] = json.dumps(links)
            else:
                processed_item['links'] = str(links)
        
        # Include search results
        search_results = item.get('search_results', [])
        if isinstance(search_results, list):
            # Extract search terms and contexts
            search_terms = [result.get('term', '') for result in search_results if isinstance(result, dict)]
            search_contexts = [result.get('context', '') for result in search_results if isinstance(result, dict)]
            
            processed_item['search_terms'] = json.dumps(search_terms) if search_terms else ''
            processed_item['search_contexts'] = json.dumps(search_contexts) if search_contexts else ''
        
        # Add other fields that might be useful
        if 'title' in item:
            processed_item['title'] = item['title']
        
        if 'error' in item:
            processed_item['error'] = item['error']
        
        # Add to the list of processed items
        processed_data.append(processed_item)
    
    # Create DataFrame
    df = pd.DataFrame(processed_data)
    
    return df

def process_text(text: str, options: Dict[str, bool]) -> str:
    """
    Process text according to specified options.
    
    Args:
        text: Text to process
        options: Processing options
            
    Returns:
        str: Processed text
    """
    if not text:
        return text
    
    # Remove HTML tags if requested
    if options.get('remove_html', False):
        text = BeautifulSoup(text, 'html.parser').get_text()
    
    # Convert to lowercase if requested
    if options.get('lowercase', False):
        text = text.lower()
    
    # Tokenize for further processing
    tokens = word_tokenize(text)
    
    # Remove stopwords if requested
    if options.get('remove_stopwords', False):
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token.lower() not in stop_words]
    
    # Lemmatize if requested
    if options.get('lemmatize', False):
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Remove punctuation if requested
    if options.get('remove_punct', False):
        tokens = [token for token in tokens if token.isalnum()]
    
    # Rejoin tokens into text
    processed_text = ' '.join(tokens)
    
    # Clean up extra whitespace
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    
    return processed_text

def export_dataset(
    dataset: Union[pd.DataFrame, Dict, List],
    file_path: str,
    format_type: str = 'CSV'
) -> str:
    """
    Export a dataset to a file.
    
    Args:
        dataset: Dataset to export (DataFrame, dict, or list)
        file_path: Path to save the file
        format_type: Export format (CSV, JSON, Excel, or Pandas DataFrame)
            
    Returns:
        str: Path to the exported file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)) or '.', exist_ok=True)
    
    # Convert dict/list to DataFrame if necessary
    if not isinstance(dataset, pd.DataFrame):
        if isinstance(dataset, dict):
            dataset = pd.DataFrame([dataset])
        elif isinstance(dataset, list):
            dataset = pd.DataFrame(dataset)
        else:
            raise ValueError("Dataset must be a DataFrame, dict, or list")
    
    # Export in the requested format
    format_type = format_type.lower()
    
    if format_type == 'csv':
        dataset.to_csv(file_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    elif format_type == 'json':
        dataset.to_json(file_path, orient='records', lines=False, indent=2)
    elif format_type == 'excel' or format_type == 'xlsx':
        dataset.to_excel(file_path, index=False)
    elif format_type == 'pandas dataframe':
        # Just return the path as we don't need to do anything for this format
        pass
    else:
        raise ValueError(f"Unsupported format type: {format_type}")
    
    logger.info(f"Dataset exported to {file_path} in {format_type} format")
    return file_path

def get_dataset_stats(dataset: pd.DataFrame) -> Dict[str, Any]:
    """
    Get statistics about a dataset.
    
    Args:
        dataset: Dataset as a pandas DataFrame
            
    Returns:
        Dict: Statistics about the dataset
    """
    if dataset is None or dataset.empty:
        return {"error": "Dataset is empty"}
    
    stats = {
        "row_count": len(dataset),
        "column_count": len(dataset.columns),
        "columns": list(dataset.columns),
        "memory_usage": dataset.memory_usage(deep=True).sum(),
        "null_values": dataset.isnull().sum().to_dict(),
    }
    
    # Add statistics for text columns
    text_columns = [col for col in dataset.columns if dataset[col].dtype == 'object']
    
    for col in text_columns:
        if col in ['content', 'title']:
            non_empty = dataset[col].astype(str).str.strip().str.len() > 0
            stats[f"{col}_stats"] = {
                "non_empty_count": non_empty.sum(),
                "avg_length": dataset.loc[non_empty, col].astype(str).str.len().mean(),
                "max_length": dataset.loc[non_empty, col].astype(str).str.len().max(),
                "min_length": dataset.loc[non_empty, col].astype(str).str.len().min(),
            }
    
    return stats

if __name__ == "__main__":
    # Test dataset creation
    test_data = [
        {
            "url": "https://example.com",
            "content": "<p>This is a sample HTML content for testing.</p>",
            "links": ["https://example.com/link1", "https://example.com/link2"],
            "crawl_time": datetime.now().timestamp(),
            "search_results": [
                {"term": "sample", "context": "This is a sample HTML content"},
                {"term": "testing", "context": "content for testing."}
            ]
        },
        {
            "url": "https://example.com/page2",
            "content": "<p>Another page with different content to test with.</p>",
            "links": ["https://example.com/link3"],
            "crawl_time": datetime.now().timestamp(),
            "search_results": [
                {"term": "test", "context": "content to test with."}
            ]
        }
    ]
    
    # Create dataset with default options
    dataset = create_dataset(test_data)
    print("Dataset shape:", dataset.shape)
    print("Dataset columns:", dataset.columns.tolist())
    print(dataset.head())
    
    # Export as CSV and JSON for testing
    temp_csv = "temp_dataset.csv"
    temp_json = "temp_dataset.json"
    
    export_dataset(dataset, temp_csv, "CSV")
    export_dataset(dataset, temp_json, "JSON")
    
    print(f"Exported dataset to {temp_csv} and {temp_json}")
    
    # Get dataset stats
    stats = get_dataset_stats(dataset)
    print("\nDataset statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Clean up test files
    try:
        os.remove(temp_csv)
        os.remove(temp_json)
    except:
        pass
