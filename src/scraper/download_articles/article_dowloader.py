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
    def __init__(self, folder, log_articles):
        self.folder = folder
        self.selectors = {
            'article': '',
            'title': '',
            'date': '',
            'author': ''
        }
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
        print("Extracting contents for \033[93m" + self.folder + "\033[0m")
        full_path = f"./knowledge-sources/{self.folder}"
        self.urls: set[str] = set()
        if "." in self.folder and os.path.exists(f'{full_path}/selectors'):
            with open(f'{full_path}/selectors', 'r', encoding='utf-8') as f:
                lines = f.read().splitlines()
                for line in lines:
                    splits = line.split('=')
                    self.selectors[splits[0]] = splits[1]
                if not self.selectors['article']:
                    raise EnvironmentError('Missing article selector')
                print(f"Selectors are:\n", self.selectors)
        else:
            raise FileNotFoundError(f'No {full_path}/selectors found')
        if os.path.exists(f'{full_path}/url-list.csv'):
            with open(f'{full_path}/url-list.csv', 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    self.urls.add(row[0])
        if os.path.exists(f'{full_path}/local-knowledge'):
            entries = set(os.listdir(f'{full_path}/local-knowledge'))
            self.urls = self.urls | entries
        if os.path.exists(f"{full_path}/scraped_articles.pkl.gz"):
            with gzip.open(f"{full_path}/scraped_articles.pkl.gz", 'rb') as f:
                scraped_articles = pickle.load(f)
                for article in scraped_articles:
                    if article['url'] in self.urls:
                        self.articles.append(article)
                        self.urls.remove(article['url'])
                print(f"{len(self.articles)} already known articles")
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
                print(f"\033[92mProgress: {completed}/{len(self.urls)} articles processed\033[0m")
        with gzip.open(f"{full_path}/scraped_articles.pkl.gz", 'wb') as f:
            pickle.dump(self.articles, f)
        print(f"Saved {len(self.articles)} dictionaries to {full_path}/scraped-articles.pkl.gz (compressed)")
        if len(self.failed_urls):
            print(f"\033[91m{len(self.failed_urls)} failed lookups:\033[0m")
            for url in self.failed_urls:
                print(url)
        return self.articles
    
    def scrape_article(self, url: str) -> Optional[dict]:
        print("Scrape article")
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            title = None
            if self.selectors['title']:
                title_elem = cast(Tag, soup.select_one(self.selectors['title']))
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    if self.log_articles == 'true':
                        print("Title: ", title)
                if not title:
                    print(f"No title found for {url}")
            
            content = None
            content_elem = soup.select_one(self.selectors['article'])
            if content_elem:
                for script in content_elem(["script", "style", "nav", "footer", "iframe"]):
                    script.decompose()
                content = content_elem.get_text(separator='\n', strip=True)
                if self.log_articles == 'true':
                    print(content)
            if not content:
                print(f"\033[91mNo content found for {url} with selector: {self.selectors['article']}\033[0m")
                self.failed_urls.add(url)
                return None
            
            date = None
            if self.selectors['date']:
                date_elem = cast(Tag, soup.select_one(self.selectors['date']))
                if date_elem:
                    date = date_elem.get_text(strip=True)
                    if self.log_articles == 'true':
                        print("Date: ", date)
            if self.selectors['author']:
                author = None
                author_element = cast(Tag, soup.select_one(self.selectors['author']))
                if author_element:
                    author = author_element.get_text(strip=True)
                    if self.log_articles == 'true':
                        print("Author: ", author)

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
            print(f"Successfully scraped article: {url} \033[93m({article_data['word_count']} words)\033[0m")
            return article_data
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None
        
    def scrape_pdf(self, url: str) -> Optional[Dict]:
        print('Scrape PDF: ', url)
        try:
            path = ""
            is_tmp = False
            if not url.startswith("https://"):
                path = f"{self.folder}local-knowledge/{url}"
            else:
                response = self.session.get(url, timeout=30, stream=True)
                response.raise_for_status()
                if "application/pdf" not in response.headers.get("Content-Type", "").lower():
                    raise ValueError("Expected PDF but got different content type")
                suffix = os.path.splitext(urlparse(url).path)[1] or ".pdf"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            tmp_file.write(chunk)
                    path = tmp_file.name
                    is_tmp = True
            try:
                extractor = PDFAdobeExtractor()
                text = extractor.get_text_from_local_pdf(path, url)
                article_data = {
                    "url": url,
                    "title": "PDF",
                    "content": text,
                    "source_type": "pdf",
                    'word_count': len(text.split()),
                    'scraped_at': time.time(),
                    "metadata": {
                        "filename": os.path.basename(urlparse(url).path) if 'https://' in url else url,
                    },
                }
                if self.log_articles == 'true':
                    print(text)
                print(f"Successfully scraped PDF: {url} \033[93m({article_data['word_count']} words)\033[0m")
                return article_data
            except Exception as e:
                print(f"Error with extractor : {e}")
            finally:
                if os.path.exists(path) and is_tmp:
                    os.unlink(path)
        except Exception as e:
            print(f"Error scraping PDF {url}: {e}")

    def scrape_article_or_pdf(self, url: str) -> Optional[Dict]:
        if not url.startswith("https://"):
            if url.endswith(".pdf"):
                return self.scrape_pdf(url)
            else:
                raise ValueError("Entry is neither a URL nor a PDF")
        try:
            head_response = self.session.head(url, allow_redirects=True, timeout=10)
            content_type = head_response.headers.get("Content-Type", "")
            is_pdf = url.endswith(".pdf") or "application/pdf" in content_type
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