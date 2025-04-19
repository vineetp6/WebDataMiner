import os
import sys
import logging
import tempfile
import json
from typing import List, Dict, Any, Optional, Union
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from scrapy.http import Response
import re
from urllib.parse import urlparse
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('scrapy_crawler')

class ScrapyLinkExtractor:
    """
    Helper class to extract links using Scrapy's LinkExtractor.
    """
    def __init__(self, allow_domains=None, deny_domains=None, allow_urls=None, deny_urls=None):
        self.extractor = LinkExtractor(
            allow_domains=allow_domains,
            deny_domains=deny_domains,
            allow=allow_urls,
            deny=deny_urls,
            unique=True
        )
    
    def extract_links(self, response):
        """Extract links from a Scrapy response."""
        return [link.url for link in self.extractor.extract_links(response)]

class WebCrawlerSpider(CrawlSpider):
    """
    Scrapy spider to crawl websites and extract content based on search terms.
    """
    name = 'web_crawler'
    
    def __init__(
        self, 
        start_url=None, 
        allowed_domains=None, 
        search_terms=None, 
        max_pages=10, 
        max_depth=1,
        *args, 
        **kwargs
    ):
        super(WebCrawlerSpider, self).__init__(*args, **kwargs)
        
        # Set start URLs
        self.start_urls = [start_url] if start_url else []
        
        # Set allowed domains based on start URLs or provided domains
        if allowed_domains:
            self.allowed_domains = allowed_domains if isinstance(allowed_domains, list) else [allowed_domains]
        elif start_url:
            domain = urlparse(start_url).netloc
            self.allowed_domains = [domain]
        else:
            self.allowed_domains = []
        
        # Set search terms
        self.search_terms = search_terms or []
        
        # Set crawling limits
        self.max_pages = max_pages
        self.max_depth = max_depth
        
        # Initialize counters and storage
        self.pages_crawled = 0
        self.crawl_results = []
        
        # Configure rules for following links
        self.rules = (
            Rule(
                LinkExtractor(allow_domains=self.allowed_domains),
                callback='parse_item',
                follow=True,
                process_links='process_links',
                cb_kwargs={'depth': 0}
            ),
        )
        
        # Initialize the rules
        self._compile_rules()
    
    def process_links(self, links):
        """Process and limit the number of links to follow."""
        # Return a limited number of links to stay within max_pages
        remaining = max(0, self.max_pages - self.pages_crawled)
        return links[:remaining]
    
    def parse_start_url(self, response):
        """Parse the start URL."""
        return self.parse_item(response, depth=0)
    
    def parse_item(self, response, depth):
        """
        Parse each crawled page, extract content, links, and search for terms.
        
        Args:
            response: The HTTP response
            depth: Current crawl depth
            
        Returns:
            dict: Extracted data from the page
        """
        # Increment page counter
        self.pages_crawled += 1
        logger.info(f"Crawling page {self.pages_crawled}/{self.max_pages}: {response.url}")
        
        # Extract main content from the page
        content = self.extract_content(response)
        
        # Extract links for further crawling
        link_extractor = ScrapyLinkExtractor(allow_domains=self.allowed_domains)
        links = link_extractor.extract_links(response)
        
        # Search for terms in content
        search_results = []
        if content and self.search_terms:
            for term in self.search_terms:
                if not term:  # Skip empty terms
                    continue
                
                # Find all occurrences of the term in the content
                pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
                matches = pattern.finditer(content)
                
                for match in matches:
                    # Get context around the match
                    start_pos = max(0, match.start() - 50)
                    end_pos = min(len(content), match.end() + 50)
                    context = content[start_pos:end_pos]
                    
                    search_results.append({
                        'term': term,
                        'context': context,
                        'position': match.start()
                    })
        
        # Store results
        result = {
            'url': response.url,
            'content': content,
            'links': links,
            'search_results': search_results,
            'depth': depth,
            'status': response.status,
            'headers': dict(response.headers),
            'crawl_time': time.time()
        }
        
        self.crawl_results.append(result)
        
        # Follow links if we haven't reached max depth and max pages
        if depth < self.max_depth and self.pages_crawled < self.max_pages:
            for link in links:
                # Check if we've reached the page limit
                if self.pages_crawled >= self.max_pages:
                    break
                
                # Yield a request to follow the link
                yield scrapy.Request(
                    link, 
                    callback=self.parse_item,
                    cb_kwargs={'depth': depth + 1}
                )
        
        yield result
    
    def extract_content(self, response):
        """
        Extract main content from the response.
        
        Args:
            response: The HTTP response
            
        Returns:
            str: Extracted text content
        """
        # Try to get the main content
        main_content = ' '.join(response.css('body p::text').getall())
        
        # If main content is too short, get all text
        if len(main_content) < 100:
            main_content = ' '.join(response.css('body ::text').getall())
        
        # Clean up the content
        main_content = re.sub(r'\s+', ' ', main_content).strip()
        
        return main_content

def run_scrapy_crawler(
    start_url: str,
    search_terms: List[str] = None,
    depth: int = 1,
    max_pages: int = 5,
    respect_robots: bool = True
) -> List[Dict]:
    """
    Run a Scrapy crawler on the specified URL.
    
    Args:
        start_url: URL to start crawling from
        search_terms: List of terms to search for in content
        depth: Maximum crawl depth
        max_pages: Maximum number of pages to crawl
        respect_robots: Whether to respect robots.txt
        
    Returns:
        List[Dict]: List of dictionaries containing crawled data
    """
    # Create a temporary file to capture the output
    output_file = tempfile.NamedTemporaryFile(delete=False).name
    
    try:
        # Parse the domain from the start URL
        parsed_url = urlparse(start_url)
        domain = parsed_url.netloc
        
        # Normalize search terms
        if search_terms and isinstance(search_terms, str):
            search_terms = [term.strip() for term in search_terms.split(',')]
        
        # Configure the crawler process
        process = CrawlerProcess(settings={
            'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'ROBOTSTXT_OBEY': respect_robots,
            'LOG_LEVEL': 'INFO',
            'DOWNLOAD_DELAY': 1,  # Be nice to the servers
            'COOKIES_ENABLED': False,
            'RETRY_TIMES': 2,
            'FEED_FORMAT': 'json',
            'FEED_URI': f'file://{output_file}',
            'CONCURRENT_REQUESTS': 8,
            'DEPTH_LIMIT': depth,
            'CLOSESPIDER_PAGECOUNT': max_pages
        })
        
        # Create the spider with all necessary settings
        spider = WebCrawlerSpider(
            start_url=start_url,
            allowed_domains=[domain],
            search_terms=search_terms,
            max_pages=max_pages,
            max_depth=depth
        )
        
        # Start the crawling process
        process.crawl(spider)
        process.start()  # This will block until the crawling is finished
        
        # Get the crawl results directly from the spider
        crawl_results = spider.crawl_results
        
        # If no results were stored in the spider, try to read from the output file
        if not crawl_results:
            try:
                with open(output_file, 'r') as f:
                    crawl_results = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                logger.error("Failed to load crawl results from file")
                crawl_results = []
        
        return crawl_results
    
    except Exception as e:
        logger.error(f"Error running Scrapy crawler: {e}")
        return [{"url": start_url, "error": str(e)}]
    
    finally:
        # Clean up the temporary file
        try:
            os.unlink(output_file)
        except:
            pass

if __name__ == "__main__":
    # Test the crawler with a sample URL
    test_url = "https://en.wikipedia.org/wiki/Web_crawler"
    print(f"Testing Scrapy crawler with URL: {test_url}")
    
    results = run_scrapy_crawler(
        test_url,
        search_terms=["crawler", "spider", "bot"],
        depth=1,
        max_pages=2
    )
    
    print(f"Crawled {len(results)} pages")
    if results:
        first_result = results[0]
        print(f"First page URL: {first_result.get('url')}")
        print(f"Content length: {len(first_result.get('content', ''))}")
        print(f"Links found: {len(first_result.get('links', []))}")
        print(f"Search results: {len(first_result.get('search_results', []))}")
