import re
from urllib.parse import urlparse
import pandas as pd
from typing import Dict, List, Union, Any, Optional
import json
import logging
import os
import time
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('helpers')

def validate_url(url: str) -> bool:
    """
    Validate if the provided URL is correctly formatted and has a proper domain.
    
    Args:
        url: URL string to validate
        
    Returns:
        bool: True if URL is valid, False otherwise
    """
    if not url or not isinstance(url, str):
        return False
    
    # Check if URL has a scheme (http/https)
    if not (url.startswith('http://') or url.startswith('https://')):
        return False
    
    try:
        result = urlparse(url)
        
        # Check for scheme and netloc (domain)
        if not all([result.scheme, result.netloc]):
            return False
        
        # Verify domain has at least one dot (e.g., example.com)
        if '.' not in result.netloc:
            return False
            
        # Additional check to prevent single-character domains
        domain_parts = result.netloc.split('.')
        if any(len(part) < 1 for part in domain_parts):
            return False
            
        return True
    except Exception as e:
        logger.error(f"URL validation error: {e}")
        return False

def truncate_text(text: str, max_length: int = 100, add_ellipsis: bool = True) -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length of returned text
        add_ellipsis: Whether to add ellipsis if text is truncated
        
    Returns:
        str: Truncated text
    """
    if not text:
        return ""
    
    text = str(text).strip()
    
    if len(text) <= max_length:
        return text
    
    truncated = text[:max_length].strip()
    
    if add_ellipsis:
        truncated += "..."
        
    return truncated

def format_timestamp(timestamp: Union[int, float, str, datetime]) -> str:
    """
    Format a timestamp into a readable date string.
    
    Args:
        timestamp: Timestamp to format (can be numeric, string, or datetime)
        
    Returns:
        str: Formatted timestamp string
    """
    try:
        # Handle numeric timestamp (Unix timestamp)
        if isinstance(timestamp, (int, float)):
            dt = datetime.fromtimestamp(timestamp)
        # Handle datetime object
        elif isinstance(timestamp, datetime):
            dt = timestamp
        # Handle string timestamp
        elif isinstance(timestamp, str):
            try:
                # Try parsing as ISO format
                dt = datetime.fromisoformat(timestamp)
            except:
                # Try parsing as common format
                dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        else:
            return str(timestamp)
        
        # Format the datetime
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    
    except Exception as e:
        logger.error(f"Error formatting timestamp: {e}")
        return str(timestamp)

def format_crawl_results(results: Union[Dict, List[Dict]]) -> Dict:
    """
    Format crawl results for display.
    
    Args:
        results: Crawl results (single dict or list of dicts)
        
    Returns:
        Dict: Formatted results for display
    """
    if not results:
        return {"message": "No results found"}
    
    # Handle single page result (dict)
    if isinstance(results, dict):
        return _format_single_page_result(results)
    
    # Handle multiple page results (list)
    if isinstance(results, list):
        return _format_multiple_page_results(results)
    
    return {"message": f"Unsupported result format: {type(results).__name__}"}

def _format_single_page_result(result: Dict) -> Dict:
    """Format a single page crawl result."""
    formatted = {}
    
    # Add URL
    formatted["url"] = result.get("url", "Unknown URL")
    
    # Check for error
    if "error" in result:
        formatted["status"] = "Error"
        formatted["error"] = result["error"]
        return formatted
    
    # Add content preview
    if "content" in result:
        formatted["content_preview"] = truncate_text(result["content"], 200)
        formatted["content_length"] = len(result["content"])
    
    # Add links
    if "links" in result:
        links = result["links"]
        formatted["links_count"] = len(links)
        formatted["links_preview"] = [truncate_text(link, 70) for link in links[:5]]
        if len(links) > 5:
            formatted["links_preview"].append(f"... and {len(links) - 5} more")
    
    # Add search results
    if "search_results" in result:
        search_results = result["search_results"]
        formatted["search_results_count"] = len(search_results)
        
        if search_results:
            formatted["search_results_preview"] = []
            for i, sr in enumerate(search_results[:3]):
                if isinstance(sr, dict):
                    term = sr.get("term", "")
                    context = sr.get("context", "")
                    formatted["search_results_preview"].append({
                        "term": term,
                        "context": truncate_text(context, 100)
                    })
            
            if len(search_results) > 3:
                formatted["search_results_preview"].append(
                    f"... and {len(search_results) - 3} more matches"
                )
    
    # Add metadata
    if "crawl_time" in result:
        formatted["crawl_time"] = format_timestamp(result["crawl_time"])
    
    if "depth" in result:
        formatted["depth"] = result["depth"]
    
    return formatted

def _format_multiple_page_results(results: List[Dict]) -> Dict:
    """Format multiple page crawl results."""
    formatted = {
        "pages_count": len(results),
        "pages": []
    }
    
    # Process each page
    for i, result in enumerate(results[:5]):  # Show first 5 pages
        page_result = _format_single_page_result(result)
        formatted["pages"].append(page_result)
    
    # Add summary if more than 5 pages
    if len(results) > 5:
        formatted["pages"].append(f"... and {len(results) - 5} more pages")
    
    # Calculate totals
    total_links = 0
    total_search_results = 0
    successful_pages = 0
    
    for result in results:
        if "error" not in result:
            successful_pages += 1
            total_links += len(result.get("links", []))
            total_search_results += len(result.get("search_results", []))
    
    formatted["summary"] = {
        "successful_pages": successful_pages,
        "failed_pages": len(results) - successful_pages,
        "total_links": total_links,
        "total_search_results": total_search_results
    }
    
    return formatted

def get_file_extension_from_format(format_type: str) -> str:
    """
    Get file extension from format type.
    
    Args:
        format_type: Format type (e.g., 'CSV', 'JSON', 'Excel')
        
    Returns:
        str: File extension
    """
    format_map = {
        'csv': 'csv',
        'json': 'json',
        'excel': 'xlsx',
        'pandas dataframe': 'csv'
    }
    
    return format_map.get(format_type.lower(), 'txt')

def ensure_directory_exists(path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def extract_domain(url: str) -> str:
    """
    Extract domain from URL.
    
    Args:
        url: URL to extract domain from
        
    Returns:
        str: Domain name
    """
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # Remove 'www.' if present
        if domain.startswith('www.'):
            domain = domain[4:]
            
        return domain
    except:
        return url

def create_filename_from_url(url: str, prefix: str = "", suffix: str = "", ext: str = "txt") -> str:
    """
    Create a valid filename from a URL.
    
    Args:
        url: URL to convert to filename
        prefix: Optional prefix for the filename
        suffix: Optional suffix for the filename
        ext: File extension
        
    Returns:
        str: Valid filename
    """
    # Extract domain
    domain = extract_domain(url)
    
    # Replace invalid filename characters
    domain = re.sub(r'[\\/*?:"<>|]', "_", domain)
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Assemble filename
    filename = f"{prefix}{domain}_{timestamp}{suffix}.{ext}"
    
    return filename

def get_mime_type(file_extension: str) -> str:
    """
    Get MIME type for a file extension.
    
    Args:
        file_extension: File extension (without dot)
        
    Returns:
        str: MIME type
    """
    mime_types = {
        'csv': 'text/csv',
        'json': 'application/json',
        'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'html': 'text/html',
        'txt': 'text/plain',
        'pdf': 'application/pdf',
        'xml': 'application/xml'
    }
    
    return mime_types.get(file_extension.lower(), 'application/octet-stream')

def safe_json_serialize(obj: Any) -> Any:
    """
    Safely serialize an object to JSON, handling non-serializable types.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.to_dict()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return str(obj)

def format_execution_time(start_time: float) -> str:
    """
    Format execution time from start time to now.
    
    Args:
        start_time: Start time (as returned by time.time())
        
    Returns:
        str: Formatted execution time
    """
    execution_time = time.time() - start_time
    
    # Format based on duration
    if execution_time < 1:
        # Convert to milliseconds
        return f"{execution_time * 1000:.0f} ms"
    elif execution_time < 60:
        return f"{execution_time:.2f} seconds"
    else:
        minutes = int(execution_time // 60)
        seconds = execution_time % 60
        return f"{minutes} min {seconds:.0f} sec"

if __name__ == "__main__":
    # Test the helper functions
    test_url = "https://www.example.com/path/to/page?query=param#fragment"
    
    print(f"URL validation: {validate_url(test_url)}")
    print(f"Domain extraction: {extract_domain(test_url)}")
    print(f"Truncated text: {truncate_text('This is a long text that needs to be truncated for display purposes', 20)}")
    print(f"Filename from URL: {create_filename_from_url(test_url, prefix='crawl_', ext='json')}")
    
    # Test timestamp formatting
    print(f"Formatted timestamp: {format_timestamp(time.time())}")
    
    # Test crawl results formatting
    test_result = {
        "url": test_url,
        "content": "This is some sample content from the crawled page.",
        "links": ["https://example.com/link1", "https://example.com/link2"],
        "search_results": [
            {"term": "sample", "context": "This is some sample content"},
            {"term": "content", "context": "sample content from the crawled"}
        ],
        "crawl_time": time.time()
    }
    
    formatted = format_crawl_results(test_result)
    print("\nFormatted crawl results:")
    for key, value in formatted.items():
        print(f"  {key}: {value}")
