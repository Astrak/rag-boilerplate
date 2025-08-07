from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup, Tag
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Set, Dict, Optional
import time
from typing import cast

delay = 0.05 # delay to not Ddos the server

class ArticleScraper:
    def __init__(self, base_url, excluded_paths):
        print('Initialize scraper')
        self.base_url = base_url
        self.excluded_paths = excluded_paths
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; ArticleScraper/1.0)'
        })
        self.scraped_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()
        self.articles: List[Dict] = []
    
    def is_url_excluded(self, url: str) -> bool:
        parsed_url = urlparse(url)
        path = parsed_url.path
        for excluded_path in self.excluded_paths:
            # Support both exact matches and regex patterns
            if excluded_path.startswith('/') and path.startswith(excluded_path):
                return True
            elif re.search(excluded_path, path):
                return True
        return False

    def discover_urls(self):
        urls = self._crawl_for_articles()
        return [url for url in urls if not self.is_url_excluded(url)]

    def create_vector_store(self) -> FAISS:
        print("Creating vector store...")
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=200,
        )
        for article in self.articles:
            text_chunks = text_splitter.split_text(article['content'])
            for i, chunk in enumerate(text_chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        'source': article['url'],
                        'title': article['title'],
                        'chunk_id': i,
                        'total_chunks': len(text_chunks),
                        'word_count': article['word_count'],
                        'meta_description': article['meta_description']
                    }
                )
                documents.append(doc)
        print(f"Created {len(documents)} document chunks from {len(self.articles)} articles")
        vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings(model="text-embedding-3-large"))
        vectorstore.save_local('./vectorstore')
        print(f"Vector store saved to ./vectorstore")
        return vectorstore

    def _crawl_for_articles(self) -> List[str]:
        discovered_urls = set()
        to_visit = {self.base_url}
        visited = set()
        while to_visit:
            current_url = to_visit.pop()
            if current_url in visited:
                continue
            try:
                response = self.session.get(current_url, timeout=10)
                response.raise_for_status()
                visited.add(current_url)
                soup = BeautifulSoup(response.content, 'html.parser')
                for selector in ['article div a[href]']:
                    for link in soup.select(selector):
                        href = cast(str, link.get('href'))
                        if href and self._is_same_domain(href) and not href in discovered_urls: 
                            discovered_urls.add(href)
                            print(f"{len(discovered_urls)} discovered urls")
                for link in soup.find_all('a', href=True):
                    href = cast(str, cast(Tag, link).get('href'))
                    if self._is_same_domain(href):
                        to_visit.add(href)
                time.sleep(delay) 
            except Exception as e:
                print(f"Error crawling {current_url}: {e}")
        return list(discovered_urls)
    
    def _is_same_domain(self, url: str) -> bool:
        return urlparse(url).netloc == urlparse(self.base_url).netloc

    def scrape_articles(self, urls: List[str]) -> None:
        print(f"Starting to scrape {len(urls)} articles...")
        with ThreadPoolExecutor(max_workers=3) as executor:
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
                    print(f"Error processing {url}: {e}")
                    self.failed_urls.add(url)
                time.sleep(delay)
                completed = len(self.scraped_urls) + len(self.failed_urls)
                if completed % 50 == 0:
                    print(f"Progress: {completed}/{len(urls)} articles processed")
    
    def scrape_article(self, url: str) -> Optional[Dict]:
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            title = cast(Tag, soup.select_one('h1.entry-title')).get_text(strip=True)
            content_selectors = [
                'article', 
                '.article-content', 
                '.post-content', 
                '.content',
                'main',
                '.entry-content'
            ]

            content = None
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    # Remove script and style elements
                    for script in content_elem(["script", "style", "nav", "footer"]):
                        script.decompose()
                    content = content_elem.get_text(separator='\n', strip=True)
                    break
            
            if not content:
                print(f"No content found for {url}")
                return None
            
            # Get metadata
            meta_description = ""
            meta_tag = soup.find('meta', attrs={'name': 'description'})
            if meta_tag:
                meta_description = meta_tag.get('content', '')
            
            article_data = {
                'url': url,
                'title': title or 'Untitled',
                'content': content,
                'meta_description': meta_description,
                'word_count': len(content.split()),
                'scraped_at': time.time()
            }
            
            print(f"Successfully scraped: {url} ({article_data['word_count']} words)")
            return article_data
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None