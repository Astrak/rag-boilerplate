from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup, Tag
from typing import cast
import time
import re

DELAY = 0.05 # delay to not Ddos the server

class UrlDiscoverer:
    def __init__(self, base_url, excluded_paths):
        self.base_url = base_url
        self.excluded_paths = excluded_paths
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; ArticleScraper/1.0)'
        })

    def discover_urls(self) -> dict["str", list[str]]:
        discovered_urls = set()
        to_visit = {self.base_url}
        to_revisit = set()
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
                for selector in ['a[href]']:
                    for link in soup.select(selector):
                        href = cast(str, link.get('href'))
                        if href and self._is_same_domain(href) and not href in discovered_urls: 
                            discovered_urls.add(href)
                            print(f"{len(discovered_urls)} urls, added: {href}")
                for link in soup.find_all('a', href=True):
                    href = cast(str, cast(Tag, link).get('href'))
                    if self._is_same_domain(href):
                        to_visit.add(href)
                time.sleep(DELAY) 
            except Exception as e:
                print(f"Error crawling {current_url}: {e}")
                to_revisit.add(current_url)
        return {
            "discovered": list(url for url in discovered_urls if not self._is_url_excluded(url)), 
            "failed": list(to_revisit)
        }
        
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