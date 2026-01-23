from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup, Tag
from typing import cast
import time
import re
import csv
import sys
import os

DELAY = 0.05 # delay to not Ddos the server

class UrlDiscoverer:
    def __init__(self, folder, excluded_paths):
        self.folder = folder
        self.excluded_paths = excluded_paths
        self.known_urls = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; ArticleScraper/1.0)'
        })

    def discover_urls(self):
        retained_urls = []
        to_visit = set()
        to_revisit = set()
        visited = set()
        was_known = 0
        if "." in self.folder:
            self.base_url = f"https://{self.folder}"
            to_visit.add(self.base_url)
            if os.path.exists(f'./{self.folder}/url-list.csv'):
                with open(f'./{self.folder}/url-list.csv', 'r', encoding='utf-8') as file:
                    csv_reader = csv.reader(file)
                    for row in csv_reader:
                        visited.add(row[0])
                        was_known = len(visited)
                        retained_urls.append(row[0])
                    print(f"{len(visited)} pages already listed")
        print("\n\n\n\n")
        while to_visit:
            current_url = to_visit.pop()
            if current_url in visited:
                continue
            try:
                response = self.session.get(current_url, timeout=10)
                response.raise_for_status()
                visited.add(current_url)
                soup = BeautifulSoup(response.content, 'html.parser')
                for link in soup.select("a[href]"):
                    href = cast(str, link.get('href'))
                    href = self._ensure_url_is_absolute(href)
                    if href and self._is_same_domain(href) and not href in visited and not href in to_visit:
                        to_visit.add(href)
                        if not self._is_url_excluded(href):
                            retained_urls.append(href)
                time.sleep(DELAY) 
            except Exception as e:
                print(f"Error crawling {current_url}: {e}")
                to_revisit.add(current_url)
            sys.stdout.write(f"\033[F\033[F\033[F\033[F")
            sys.stdout.write(f"\rVisited: {len(visited) - was_known}\n") 
            sys.stdout.write(f"\rRemaining: {len(to_visit)}\n") 
            sys.stdout.write(f"\rFailed: {len(to_revisit)}\n") 
            sys.stdout.write(f"\rRecorded: {len(retained_urls) - was_known}\n") 
            sys.stdout.flush()
        retained_urls[:] = retained_urls[was_known:]
        with open(f'./{self.folder}/url-list.csv', "a", newline="") as f:
            writer = csv.writer(f)
            for item in retained_urls:
                writer.writerow([item])
        print("URL discovery complete")
    
    def _ensure_url_is_absolute(self, url: str) -> str:
        is_absolute = bool(urlparse(url).netloc)
        if not is_absolute:
            url = urljoin(self.base_url, url)
        return url
        
    def _is_same_domain(self, url: str) -> bool:
        return urlparse(url).netloc == urlparse(self.base_url).netloc

    def _is_url_excluded(self, url: str) -> bool:
        parsed_url = urlparse(url)
        path = parsed_url.path
        for excluded_path in self.excluded_paths:
            if excluded_path.startswith('/') and path.startswith(excluded_path):
                return True
            elif re.search(excluded_path, path):
                return True
        return False