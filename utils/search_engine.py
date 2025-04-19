import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
import logging
import urllib.parse
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('search_engine')

def search_duckduckgo(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """
    Search DuckDuckGo for the given query and return relevant websites.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        
    Returns:
        List[Dict[str, str]]: List of search results with title, url, and description
    """
    # Encode the query for URL
    encoded_query = urllib.parse.quote_plus(query)
    
    # DuckDuckGo search URL
    search_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
    
    try:
        # Set a user agent to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Send the request
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract search results
        results = []
        
        # Find all result containers
        result_elements = soup.select('.result')
        
        for result in result_elements[:max_results]:
            try:
                # Extract title
                title_element = result.select_one('.result__a')
                title = title_element.get_text().strip() if title_element else "No title"
                
                # Extract URL
                url = None
                if title_element and title_element.has_attr('href'):
                    href = title_element['href']
                    # Try to extract the actual URL from DuckDuckGo's redirect URL
                    try:
                        # Force href to be a string
                        href_str = str(href)
                        # Parse the URL
                        parsed = urllib.parse.urlparse(href_str)
                        if parsed.query:
                            query_params = urllib.parse.parse_qs(parsed.query)
                            if 'uddg' in query_params and query_params['uddg'] and query_params['uddg'][0]:
                                url = query_params['uddg'][0]
                            else:
                                url = href_str
                        else:
                            url = href_str
                    except Exception as e:
                        logger.error(f"Error parsing URL: {e}")
                        url = str(href)
                
                if not url:
                    url = "No URL"
                
                # Extract description/snippet
                snippet_element = result.select_one('.result__snippet')
                snippet = snippet_element.get_text().strip() if snippet_element else "No description"
                
                # Skip ads and non-standard results
                if (url != "No URL" and isinstance(url, str) and 
                    not url.startswith("https://duckduckgo.com")):
                    results.append({
                        'title': title,
                        'url': url,
                        'description': snippet
                    })
            except Exception as e:
                logger.error(f"Error parsing search result: {e}")
                continue
        
        return results
    
    except Exception as e:
        logger.error(f"Error searching DuckDuckGo: {e}")
        return []

if __name__ == "__main__":
    # Test the search function
    query = "python web crawling"
    print(f"Searching for: {query}")
    
    results = search_duckduckgo(query, max_results=5)
    
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']}")
        print(f"   URL: {result['url']}")
        print(f"   Description: {result['description']}")