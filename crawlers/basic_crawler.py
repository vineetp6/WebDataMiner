import requests
from bs4 import BeautifulSoup
import trafilatura
import time
import re
from urllib.parse import urljoin, urlparse
import random
from typing import Dict, List, Union, Set, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('basic_crawler')

# User agent list to rotate and avoid being blocked
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
]

def get_random_user_agent() -> str:
    """Return a random user agent from the list to avoid detection."""
    return random.choice(USER_AGENTS)

def validate_url(url: str) -> bool:
    """
    Validate if the provided URL is correctly formatted and has a proper domain.
    
    Args:
        url: URL string to validate
        
    Returns:
        bool: True if URL is valid, False otherwise
    """
    if not url or url == "https://" or url == "http://":
        return False
        
    try:
        result = urlparse(url)
        # Check scheme, netloc (domain), and make sure netloc has more than just www.
        if not all([result.scheme, result.netloc]):
            return False
            
        # Make sure it has a valid TLD (at least one dot in the domain)
        if '.' not in result.netloc:
            return False
            
        # Make sure netloc has actual content beyond the TLD
        domain_parts = result.netloc.split('.')
        if len(domain_parts) < 2 or not all(domain_parts):
            return False
            
        return True
    except:
        return False

def get_domain(url: str) -> str:
    """
    Extract the domain from a URL.
    
    Args:
        url: URL to extract domain from
        
    Returns:
        str: Domain name
    """
    parsed_uri = urlparse(url)
    return '{uri.netloc}'.format(uri=parsed_uri)

def crawl_url(url: str, timeout: int = 10) -> str:
    """
    Crawl a URL and extract its content using trafilatura.
    
    Args:
        url: URL to crawl
        timeout: Request timeout in seconds
        
    Returns:
        str: Extracted text content
    """
    try:
        # Use trafilatura to download and extract content
        # Note: trafilatura.fetch_url doesn't accept the headers parameter
        # We'll use requests to handle custom headers instead
        downloaded = trafilatura.fetch_url(url)
        
        if downloaded:
            # Extract main text content
            text = trafilatura.extract(downloaded)
            
            # If trafilatura fails, try with BeautifulSoup as backup
            if not text:
                response = requests.get(
                    url, 
                    headers={'User-Agent': get_random_user_agent()}, 
                    timeout=timeout
                )
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.extract()
                
                # Get text
                text = soup.get_text(separator='\n')
                
                # Remove blank lines
                text = '\n'.join([line.strip() for line in text.split('\n') if line.strip()])
            
            return text or "No content extracted."
        
        return "Failed to download the page."
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error for {url}: {e}")
        return f"Error: {str(e)}"
    except Exception as e:
        logger.error(f"General error crawling {url}: {e}")
        return f"Error: {str(e)}"

def extract_links(base_url: str, html_content: str) -> List[str]:
    """
    Extract links from HTML content.
    
    Args:
        base_url: Base URL for resolving relative links
        html_content: HTML content to extract links from
        
    Returns:
        List[str]: List of extracted URLs
    """
    try:
        # We'll use requests to download the HTML content again since we need the raw HTML
        response = requests.get(
            base_url, 
            headers={'User-Agent': get_random_user_agent()}
        )
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        links = []
        
        # Extract links from anchor tags
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href'].strip()
            if href and not href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                # Resolve relative URLs
                absolute_url = urljoin(base_url, href)
                links.append(absolute_url)
        
        # Remove duplicates and return
        return list(set(links))
    
    except Exception as e:
        logger.error(f"Error extracting links from {base_url}: {e}")
        return []

def filter_links_same_domain(links: List[str], base_url: str) -> List[str]:
    """
    Filter links to keep only those from the same domain as the base URL.
    
    Args:
        links: List of links to filter
        base_url: Base URL to compare domains against
        
    Returns:
        List[str]: Filtered list of links
    """
    base_domain = get_domain(base_url)
    return [link for link in links if get_domain(link) == base_domain]

def crawl_multiple_pages(
    start_url: str, 
    search_terms: List[str] = None, 
    max_pages: int = 5, 
    depth: int = 1,
    timeout: int = 10,
    same_domain_only: bool = True
) -> List[Dict]:
    """
    Crawl multiple pages starting from a URL, following links up to a specified depth.
    
    Args:
        start_url: Starting URL
        search_terms: Optional list of terms to search for in the content
        max_pages: Maximum number of pages to crawl
        depth: Maximum depth of link following
        timeout: Request timeout in seconds
        same_domain_only: Whether to restrict crawling to the same domain
        
    Returns:
        List[Dict]: List of dictionaries containing crawled data for each page
    """
    if not validate_url(start_url):
        return [{"url": start_url, "error": "Invalid URL format"}]
    
    # Set to track visited URLs to avoid duplicates
    visited_urls = set()
    # Queue of URLs to visit, with their depths
    url_queue = [(start_url, 0)]  # (url, depth)
    # Results list
    results = []
    
    while url_queue and len(visited_urls) < max_pages:
        # Get the next URL and its depth from the queue
        current_url, current_depth = url_queue.pop(0)
        
        # Skip if already visited or beyond max depth
        if current_url in visited_urls or current_depth > depth:
            continue
        
        # Mark URL as visited
        visited_urls.add(current_url)
        
        # Log progress
        logger.info(f"Crawling ({len(visited_urls)}/{max_pages}): {current_url}")
        
        try:
            # Crawl the current URL
            content = crawl_url(current_url, timeout)
            
            # Extract links for further crawling
            links = extract_links(current_url, content)
            
            # Filter links to stay in the same domain if required
            if same_domain_only:
                links = filter_links_same_domain(links, start_url)
            
            # Filter out already visited links
            links = [link for link in links if link not in visited_urls]
            
            # Process search terms if provided
            search_results = []
            if search_terms and content:
                for term in search_terms:
                    if not term:  # Skip empty terms
                        continue
                    
                    # Find all occurrences of the term in the content
                    pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
                    matches = pattern.finditer(content)
                    
                    for match in matches:
                        # Get some context around the match
                        start_pos = max(0, match.start() - 50)
                        end_pos = min(len(content), match.end() + 50)
                        context = content[start_pos:end_pos]
                        
                        search_results.append({
                            'term': term,
                            'context': context,
                            'position': match.start()
                        })
            
            # Add current page results
            results.append({
                'url': current_url,
                'content': content,
                'links': links,
                'search_results': search_results,
                'depth': current_depth
            })
            
            # Add links to the queue for the next depth level
            if current_depth < depth:
                for link in links:
                    if link not in visited_urls and len(url_queue) + len(visited_urls) < max_pages:
                        url_queue.append((link, current_depth + 1))
            
            # Add a small delay to be respectful to the server
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error crawling {current_url}: {e}")
            results.append({
                'url': current_url,
                'error': str(e),
                'depth': current_depth
            })
    
    return results

if __name__ == "__main__":
    # Test the crawler with a sample URL
    test_url = "https://en.wikipedia.org/wiki/Web_crawler"
    print(f"Testing crawler with URL: {test_url}")
    
    content = crawl_url(test_url)
    print(f"Content length: {len(content)} characters")
    print(f"First 500 characters: {content[:500]}")
    
    links = extract_links(test_url, content)
    print(f"Extracted {len(links)} links")
    for i, link in enumerate(links[:5], 1):
        print(f"  {i}. {link}")
