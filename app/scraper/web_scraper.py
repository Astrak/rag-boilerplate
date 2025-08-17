import logging
import time
from typing import List, Set, Optional, Dict, Any
from urllib.parse import urlparse
import re
import requests
from bs4 import BeautifulSoup, Tag
from concurrent.futures import ThreadPoolExecutor, as_completed
from .config import ScraperConfig
from .models import Article, ScrapingResult, ScrapingProgress
from .utils import is_valid_url, is_same_domain, retry_on_failure
from .data_manager import DataManager

logger = logging.getLogger(__name__)

class WebScraper:
    """Handles web scraping operations including URL discovery and article extraction"""
    
    def __init__(self, config: ScraperConfig, data_manager: DataManager):
        self.config = config
        self.data_manager = data_manager
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config.user_agent
        })
        self.scraped_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()
        self.articles: List[Article] = []
    
    def is_url_excluded(self, url: str) -> bool:
        """Check if a URL should be excluded from scraping"""
        if not is_valid_url(url):
            return True
            
        parsed_url = urlparse(url)
        path = parsed_url.path
        
        # Add more sophisticated exclusion logic here
        excluded_patterns = [
            r'/wp-admin/',
            r'/wp-content/uploads/',
            r'/feed/',
            r'/tag/',
            r'/category/',
            r'/author/',
            r'/page/',
            r'\.(pdf|doc|docx|xls|xlsx|zip|rar)$'
        ]
        
        for pattern in excluded_patterns:
            if re.search(pattern, path, re.IGNORECASE):
                return True
        
        return False
    
    def discover_urls(self) -> Dict[str, List[str]]:
        """Discover URLs from the base URL using web crawling"""
        logger.info(f"Starting URL discovery from {self.config.base_url}")
        
        discovered_urls: Set[str] = set()
        to_visit: Set[str] = {self.config.base_url}
        to_revisit: Set[str] = set()
        visited: Set[str] = set()
        
        while to_visit:
            current_url = to_visit.pop()
            
            if current_url in visited:
                continue
                
            try:
                logger.debug(f"Crawling: {current_url}")
                response = self.session.get(current_url, timeout=self.config.timeout)
                response.raise_for_status()
                visited.add(current_url)
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract article links
                for selector in ['article div a[href]', 'h2 a[href]', '.entry-title a[href]']:
                    for link in soup.select(selector):
                        href = link.get('href')
                        if href and is_same_domain(href, self.config.base_url) and href not in discovered_urls:
                            discovered_urls.add(href)
                            logger.debug(f"Discovered article URL: {href}")
                
                # Extract navigation links for further crawling
                for link in soup.find_all('a', href=True):
                    href = link.get('href')
                    if href and is_same_domain(href, self.config.base_url):
                        to_visit.add(href)
                
                # Rate limiting
                time.sleep(self.config.delay)
                
            except Exception as e:
                logger.warning(f"Error crawling {current_url}: {e}")
                to_revisit.add(current_url)
        
        # Filter out excluded URLs
        filtered_urls = [url for url in discovered_urls if not self.is_url_excluded(url)]
        
        logger.info(f"URL discovery complete. Found {len(filtered_urls)} valid URLs, {len(to_revisit)} failed")
        
        return {
            "discovered": filtered_urls,
            "failed": list(to_revisit)
        }
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def scrape_article(self, url: str) -> Optional[Article]:
        """Scrape a single article with retry logic"""
        try:
            logger.debug(f"Scraping article: {url}")
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract article data with fallback selectors
            title = self._extract_title(soup, url)
            content = self._extract_content(soup, url)
            date = self._extract_date(soup, url)
            author = self._extract_author(soup, url)
            meta_description = self._extract_meta_description(soup, url)
            
            if not content:
                logger.warning(f"No content found for {url}")
                return None
            
            # Create Article object
            article = Article(
                url=url,
                title=title or "Untitled",
                content=content,
                date=date or "Unknown date",
                author=author or "Unknown author",
                meta_description=meta_description or "",
                word_count=len(content.split()),
                scraped_at=time.time()
            )
            
            logger.info(f"Successfully scraped: {url} ({article.word_count} words)")
            return article
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup, url: str) -> Optional[str]:
        """Extract article title with multiple fallback selectors"""
        selectors = [
            'h1.entry-title',
            'h1',
            '.entry-title',
            'title',
            '[property="og:title"]'
        ]
        
        for selector in selectors:
            try:
                element = soup.select_one(selector)
                if element:
                    title = element.get_text(strip=True)
                    if title and len(title) > 5:  # Basic validation
                        return title
            except Exception:
                continue
        
        logger.warning(f"Could not extract title from {url}")
        return None
    
    def _extract_content(self, soup: BeautifulSoup, url: str) -> Optional[str]:
        """Extract article content with multiple fallback selectors"""
        selectors = [
            '#contenu',
            '.entry-content',
            '.post-content',
            'article',
            '.content',
            'main'
        ]
        
        for selector in selectors:
            try:
                content_elem = soup.select_one(selector)
                if content_elem:
                    # Remove unwanted elements
                    for unwanted in content_elem(["script", "style", "nav", "footer", "iframe", "aside"]):
                        unwanted.decompose()
                    
                    content = content_elem.get_text(separator='\n', strip=True)
                    if content and len(content) > 100:  # Basic validation
                        return content
            except Exception:
                continue
        
        logger.warning(f"Could not extract content from {url}")
        return None
    
    def _extract_date(self, soup: BeautifulSoup, url: str) -> Optional[str]:
        """Extract article date with multiple fallback selectors"""
        selectors = [
            '.et_pb_title_container .published',
            '.entry-date',
            '.post-date',
            'time',
            '[property="article:published_time"]'
        ]
        
        for selector in selectors:
            try:
                element = soup.select_one(selector)
                if element:
                    date = element.get_text(strip=True)
                    if date:
                        return date
            except Exception:
                continue
        
        logger.warning(f"Could not extract date from {url}")
        return None
    
    def _extract_author(self, soup: BeautifulSoup, url: str) -> Optional[str]:
        """Extract article author with multiple fallback selectors"""
        selectors = [
            '.et_pb_title_container .author',
            '.entry-author',
            '.post-author',
            '.author',
            '[property="article:author"]'
        ]
        
        for selector in selectors:
            try:
                element = soup.select_one(selector)
                if element:
                    author = element.get_text(strip=True)
                    if author:
                        return author
            except Exception:
                continue
        
        logger.warning(f"Could not extract author from {url}")
        return None
    
    def _extract_meta_description(self, soup: BeautifulSoup, url: str) -> str:
        """Extract meta description"""
        try:
            meta_tag = soup.find('meta', attrs={'name': 'description'})
            if meta_tag:
                return meta_tag.get('content', '')
        except Exception:
            pass
        return ""
    
    def scrape_articles(self, urls: List[str]) -> ScrapingResult:
        """Scrape multiple articles using thread pool"""
        logger.info(f"Starting to scrape {len(urls)} articles...")
        
        self.articles = []
        self.scraped_urls.clear()
        self.failed_urls.clear()
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_url = {executor.submit(self.scrape_article, url): url for url in urls}
            
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                
                try:
                    article_data = future.result()
                    if article_data:
                        self.articles.append(article_data)
                        self.scraped_urls.add(url)
                    else:
                        self.failed_urls.add(url)
                except Exception as e:
                    logger.error(f"Error processing {url}: {e}")
                    self.failed_urls.add(url)
                
                # Rate limiting
                time.sleep(self.config.delay)
                
                # Log progress
                completed = len(self.scraped_urls) + len(self.failed_urls)
                progress = ScrapingProgress(
                    current=completed,
                    total=len(urls),
                    successful=len(self.scraped_urls),
                    failed=len(self.failed_urls)
                )
                
                logger.info(f"Progress: {progress.percentage:.1f}% ({completed}/{len(urls)}) - "
                          f"Success: {progress.successful}, Failed: {progress.failed}")
        
        # Save articles
        if self.articles:
            self.data_manager.save_articles(self.articles)
        
        # Create result
        result = ScrapingResult(
            successful_urls=list(self.scraped_urls),
            failed_urls=list(self.failed_urls),
            total_processed=len(urls),
            success_rate=len(self.scraped_urls) / len(urls) if urls else 0.0
        )
        
        logger.info(f"Scraping complete. Success rate: {result.success_rate:.1%}")
        return result
