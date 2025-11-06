import csv
import boto3
import os
import requests
from bs4 import BeautifulSoup, Tag
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import cast, Optional
import pickle
import gzip

DELAY = 0.05 # delay to not Ddos the server

class ArticleDownloader:
    def __init__(self, folder, article_selector):
        self.folder = folder
        self.article_selector = article_selector
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
        s3.download_file("rag-faiss-index-bucket", f"{self.folder}url-list.csv", f"{self.folder}url-list.csv")
        self.urls: list[str] = []
        with open(f'{self.folder}url-list.csv', 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                self.urls.append(row[0])

    def scrape_articles(self) -> list[dict]:
        print(f"Starting to scrape {len(self.urls)} articles...")
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_url = {executor.submit(self.scrape_article, url): url for url in self.urls}
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
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            title = cast(Tag, soup.select_one('h1.entry-title')).get_text(strip=True)
            content_elem = soup.select_one(self.article_selector)
            content = None
            if content_elem:
                for script in content_elem(["script", "style", "nav", "footer", "iframe"]):
                    script.decompose()
                content = content_elem.get_text(separator='\n', strip=True)
            if not content:
                print(f"No content found for {url}")
                return None
            date = cast(Tag, soup.select_one('.et_pb_title_container .published')).get_text(strip=True)
            author = cast(Tag, soup.select_one('.et_pb_title_container .author')).get_text(strip=True)
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
            print(f"Successfully scraped: {url} ({article_data['word_count']} words)")
            return article_data
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None