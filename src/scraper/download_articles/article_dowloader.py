import csv
import boto3
import os
import tempfile
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup, Tag
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import cast, Optional, Dict
import pickle
import gzip
from scraper.download_articles.pdf_extractor import PDFAdobeExtractor

DELAY = 0.05 # delay to not Ddos the server

class ArticleDownloader:
    def __init__(self, folder, article_selector, log_articles):
        self.folder = folder
        self.article_selector = article_selector
        self.log_articles = log_articles
        self.articles: list[dict] = []
        self.scraped_urls: set[str] = set()
        self.failed_urls: set[str] = set()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; ArticleScraper/1.0)'
        })
    
    def sync_urls_from_bucket(self):
        s3 = boto3.client('s3', region_name="eu-north-1")
        os.makedirs(self.folder, exist_ok=True)
        s3.download_file("rag-faiss-index-bucket", f"{self.folder[2:]}url-list.csv", f"{self.folder}url-list.csv")

    def scrape_articles(self) -> list[dict]:
        self.urls: list[str] = []
        with open(f'{self.folder}url-list.csv', 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                self.urls.append(row[0])
        print(f"Starting to scrape {len(self.urls)} articles...")
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_url = {executor.submit(self.scrape_article_or_pdf, url): url for url in self.urls}
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
                time.sleep(DELAY)
                completed = len(self.scraped_urls) + len(self.failed_urls)
                print(f"Progress: {completed}/{len(self.urls)} articles processed")
        with gzip.open(f"{self.folder}scraped_articles.pkl.gz", 'wb') as f:
            pickle.dump(self.articles, f)
        print(f"Saved {len(self.articles)} dictionaries to {self.folder}scraped-articles.pkl.gz (compressed)")
        return self.articles
    
    def scrape_article(self, url: str) -> Optional[dict]:
        print("Scrape article")
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            title = None
            title_elem = cast(Tag, soup.select_one('.article_title'))
            if title_elem:
                title = title_elem.get_text(strip=True)
            if not title:
                print(f"No title found for {url}")
            
            content = None
            content_elem = soup.select_one(self.article_selector)
            if content_elem:
                for script in content_elem(["script", "style", "nav", "footer", "iframe"]):
                    script.decompose()
                content = content_elem.get_text(separator='\n', strip=True)
                if self.log_articles == 'true':
                    print(content)
            if not content:
                print(f"No content found for {url} with selector: {self.article_selector}")
                return None
            
            date = None
            date_elem = cast(Tag, soup.select_one('.et_pb_title_container .published'))
            if date_elem:
                date = date_elem.get_text(strip=True)
            
            author = None
            author_element = cast(Tag, soup.select_one('.et_pb_title_container .author'))
            if author_element:
                author = author_element.get_text(strip=True)

            meta_description = ""
            meta_tag = soup.find('meta', attrs={'name': 'description'})
            if meta_tag:
                meta_description = cast(Tag, meta_tag).get('content', '')
                
            article_data = {
                'url': url,
                'title': title,
                'content': content,
                'date': date,
                'author': author,
                'meta_description': meta_description,
                'word_count': len(content.split()),
                'scraped_at': time.time()
            }
            print(f"Successfully scraped article: {url} ({article_data['word_count']} words)")
            return article_data
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None
        
    def scrape_pdf(self, url: str) -> Optional[Dict]:
        print('Scrape PDF')
        try:
            response = self.session.get(url, timeout=30, stream=True)
            response.raise_for_status()
            if "application/pdf" not in response.headers.get("Content-Type", "").lower():
                raise ValueError("Expected PDF but got different content type")
            suffix = os.path.splitext(urlparse(url).path)[1] or ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_file.write(chunk)
                tmp_path = tmp_file.name
            try:
                extractor = PDFAdobeExtractor()
                text = extractor.get_text_from_local_pdf(tmp_path)
                article_data = {
                    "url": url,
                    "title": "PDF",
                    "content": text,
                    "source_type": "pdf",
                    "metadata": {
                        "filename": os.path.basename(urlparse(url).path),
                    },
                }
                print(f"Successfully scraped PDF: {url}")
                return article_data
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        except Exception as e:
            print(f"Error scraping PDF {url}: {e}")

    def scrape_article_or_pdf(self, url: str) -> Optional[Dict]:
        try:
            head_response = self.session.head(url, allow_redirects=True, timeout=10)
            content_type = head_response.headers.get("Content-Type", "").lower()
            is_pdf = url.lower().endswith(".pdf") or "application/pdf" in content_type
            if is_pdf:
                return self.scrape_pdf(url)
            return self.scrape_article(url)
        except Exception as e:
            print(f"Error during initial check for {url}: {e}")
            try:
                return self.scrape_pdf(url)  # Might still be a PDF after redirects
            except Exception:
                try:
                    return self.scrape_article(url)
                except Exception as e2:
                    print(f"Failed to scrape {url} as both PDF and HTML: {e2}")
                    return None