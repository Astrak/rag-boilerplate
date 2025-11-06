from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup, Tag
from typing import cast
import time
import re

DELAY = 0.05 # delay to not Ddos the server
BASE_URL="https://www.ojim.fr/plan-du-site/"
EXCLUDED_PATHS = ["/page/", "/nous-contacter/", "/mentions-legales/"] # Polemia: ['/mot-clef/', '/page/', '/author/']

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (compatible; ArticleScraper/1.0)'
})

def discover_urls() -> dict["str", list[str]]:
    discovered_urls = set()
    to_visit = {BASE_URL}
    to_revisit = set()
    visited = set()
    while to_visit:
        current_url = to_visit.pop()
        if current_url in visited:
            continue
        try:
            response = session.get(current_url, timeout=10)
            response.raise_for_status()
            visited.add(current_url)
            soup = BeautifulSoup(response.content, 'html.parser')
            for selector in ['a[href]']:
                for link in soup.select(selector):
                    href = cast(str, link.get('href'))
                    if href and is_same_domain(href) and not href in discovered_urls: 
                        discovered_urls.add(href)
                        print(f"{len(discovered_urls)} urls, added: {href}")
            for link in soup.find_all('a', href=True):
                href = cast(str, cast(Tag, link).get('href'))
                if is_same_domain(href):
                    to_visit.add(href)
            time.sleep(DELAY) 
        except Exception as e:
            print(f"Error crawling {current_url}: {e}")
            to_revisit.add(current_url)
    return {
        "discovered": list(url for url in discovered_urls if not is_url_excluded(url)), 
        "failed": list(to_revisit)
    }
    
def is_same_domain(url: str) -> bool:
    return urlparse(url).netloc == urlparse(BASE_URL).netloc

def is_url_excluded(url: str) -> bool:
    parsed_url = urlparse(url)
    path = parsed_url.path
    for excluded_path in EXCLUDED_PATHS:
        if excluded_path.startswith('/') and path.startswith(excluded_path):
            return True
        elif re.search(excluded_path, path):
            return True
    return False

discover_urls()