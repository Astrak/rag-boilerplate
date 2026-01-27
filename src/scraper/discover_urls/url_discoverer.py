from urllib.parse import urlparse, urljoin, quote
import requests
from bs4 import BeautifulSoup, Tag
from typing import cast
import time
import re
import csv
import sys
import boto3
import os
from datetime import datetime, timedelta
from collections import Counter

DELAY = 0.05 # delay to not Ddos the server
IGNORE = {'.png', '.jpg', '.jpeg', '.xlsx', "/wp-login", "wp-admin", ".xml", ".docx", ".mp4", '.wma', '.mp3', '.doc'}
DONT_VISIT_BUT_RECORD = {".pdf"}
VISIT_BUT_DONT_RECORD = {"/page/", "?page="}

class UrlDiscoverer:
    def __init__(self, folder):
        self.folder = folder
        self.known_urls = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; ArticleScraper/1.0)'
        })
    
    def sync_urls_to_bucket(self):
        s3 = boto3.client('s3', region_name="eu-north-1")
        try:
            print("\033[KUploading url-list.csv to AWS backup...")
            s3.upload_file(
                Bucket="rag-faiss-index-bucket", 
                Filename=f"knowledge-sources/{self.folder}/url-list.csv", 
                Key=f"{self.folder}/url-list.csv"
            )
            print("\033[KSynced")
        except Exception as e:
            print("\033[KFailed to upload url-list.csv to AWS bucket")
        try:
            print("\033[KUploading blacklist.csv to AWS backup...")
            s3.upload_file(
                Bucket="rag-faiss-index-bucket", 
                Filename=f"knowledge-sources/{self.folder}/blacklist.csv", 
                Key=f"{self.folder}/blacklist.csv"
            )
            print("\033[KSynced")
        except Exception as e:
            print("\033[KFailed to upload blacklist.csv to AWS bucket")
        try:
            print("\033[KUploading ignore.csv to AWS backup...")
            s3.upload_file(
                Bucket="rag-faiss-index-bucket", 
                Filename=f"knowledge-sources/{self.folder}/ignore.csv", 
                Key=f"{self.folder}/ignore.csv"
            )
            print("\033[KSynced")
        except Exception as e:
            print("\033[KFailed to upload ignore.csv to AWS bucket")

    def discover_urls(self):
        if not "." in self.folder:
            print(f"{self.folder} is not a website, no URL will be explored, skipping.")
            return 
        print("Discovering URLs for \033[93m" + self.folder + "\033[0m")
        full_path = f"./knowledge-sources/{self.folder}"
        retained_urls = []
        to_visit = []
        to_revisit = set()
        visited = set()
        visited_in_session = []
        ignored = IGNORE
        visit_but_dont_record = VISIT_BUT_DONT_RECORD
        blacklisted = set()
        was_known = 0
        timer = datetime.utcnow()
        protocol = "https"
        try:
            response = self.session.get(f"{protocol}://{self.folder}", timeout=10)
            self.base_url = response.url 
        except Exception as e:
            if "SSLCertVerificationError" in str(e):
                protocol = "http"
                try:
                    response = self.session.get(f"{protocol}://{self.folder}", timeout=10)
                    self.base_url = response.url
                except Exception as ee:
                    raise TypeError("No protocol allowed connecting to this website")
        to_visit.append(self.base_url)
        if os.path.exists(f'{full_path}/url-list.csv'):
            with open(f'{full_path}/url-list.csv', 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    visited.add(row[0])
                    retained_urls.append(row[0])
                counts = Counter(retained_urls)
                duplicates = [item for item, count in counts.items() if count > 1]
                if len(duplicates):
                    print(f"\033[93mWarning: The existing url-list.csv contains {len(duplicates)} duplicates: \033[00m", duplicates)
                    was_known += len(duplicates)
                was_known = len(visited)
                print(f"{len(visited)} pages already listed")
        if os.path.exists(f'{full_path}/blacklist.csv'):
            with open(f'{full_path}/blacklist.csv', 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    blacklisted.add(row[0])
                print(f"{len(blacklisted)} blacklisted URLs")
        if os.path.exists(f'{full_path}/ignore.csv'):
            with open(f'{full_path}/ignore.csv', 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                counter = 0
                for row in csv_reader:
                    counter += 1
                    ignored.add(row[0])
            print(f"{counter} custom ignored keywords")                
        print("\n\n\n\n\n\n")
        while to_visit:
            current_url = cast(str, to_visit.pop())
            original_url = current_url
            if current_url in visited:
                continue
            try:
                response = self.session.get(current_url, timeout=10)
                response.raise_for_status()
                redirected_url = response.url
                visited_in_session.append(redirected_url)
                if redirected_url != current_url:
                    current_url = redirected_url
                    if current_url in to_visit:
                        to_visit.remove(current_url)
                    if current_url in visited:
                        continue
                visited.add(current_url)
                soup = BeautifulSoup(response.content, 'html.parser')
                links = {cast(str, href.get('href')) for href in soup.select("a[href]")}
                if "#" in links:
                    links.remove("#")
                new_article_found = False
                for href in links:
                    if any(sub in href for sub in ["mailto:", "javascript:", "tel:"]):
                        continue
                    href = self._ensure_url_is_absolute(href)
                    href = href.split('#')[0]
                    if all(sub not in href for sub in ["id=","page="]):
                        href = href.split('?')[0]
                    items = href.split('://')
                    items[1] = items[1].replace("//","/") 
                    href = protocol + "://" + items[1]
                    href = quote(href, safe=":/é?=") 
                    is_ignored = any(sub in href for sub in ignored)
                    dont_visit_but_record = any(sub in href for sub in DONT_VISIT_BUT_RECORD)
                    dont_record_but_visit = any(sub in href for sub in visit_but_dont_record) or href in blacklisted
                    if self._is_same_domain(href) and not is_ignored and href not in visited:
                        if href not in to_visit and not dont_visit_but_record:
                            to_visit.append(href)
                        if href not in retained_urls and not dont_record_but_visit:
                            retained_urls.append(href)
                            new_article_found = True
                if "/page/" in current_url and not new_article_found:
                    prefix = current_url.split("/page/")[0] + "/page/"
                    pattern = re.compile(re.escape(prefix) + r'(\d+)(?:/|$)')
                    to_visit.sort(
                        reverse=True,
                        key=lambda s: int(m.group(1)) if (m := pattern.match(s)) else float('inf')
                    )
                    page_number = int(re.search(r'/page/(\d+)(?:/|$)', current_url).group(1))
                    i = 0
                    while i < len(to_visit):
                        item = to_visit[i]
                        if "/page/" in item and item.startswith(prefix):
                            page = int(re.search(r'/page/(\d+)(?:/|$)', item).group(1))
                            if page < page_number: 
                                i += 1
                            else:
                                del to_visit[i]
                        else:
                            i += 1
                sys.stdout.write(f"\033[F\033[F\033[F\033[F\033[F\033[F")
                sys.stdout.write(f"\r\033[KVisiting: {current_url}\n") 
                sys.stdout.write(f"\r\033[KVisited: {len(visited) - was_known}\n") 
                sys.stdout.write(f"\r\033[KRemaining: {len(to_visit)}\n") 
                sys.stdout.write(f"\r\033[K\033[93mFailed: {len(to_revisit)}\033[0m\n") 
                sys.stdout.write(f"\r\033[K\033[92mRecorded: {len(retained_urls) - was_known}\033[0m\n") 
                sys.stdout.write(f"\r\033[KDuration: {timedelta(seconds=int((datetime.utcnow() - timer).total_seconds()))}\n") 
                sys.stdout.flush()
                time.sleep(DELAY) 
            except Exception as e:
                sys.stdout.write(f"\033[F\033[F\033[F\033[F\033[F\033[F\033[F")
                url = current_url if original_url == current_url else f"{original_url} (routed)"
                sys.stdout.write(f"\r\033[K\033[93mError: {url} : {e}\033[0m\n\033[K\n") 
                sys.stdout.write(f"\r\033[KVisiting: {current_url}\n") 
                sys.stdout.write(f"\r\033[KVisited: {len(visited) - was_known}\n") 
                sys.stdout.write(f"\r\033[KRemaining: {len(to_visit)}\n") 
                sys.stdout.write(f"\r\033[K\033[93mFailed: {len(to_revisit)}\033[0m\n") 
                sys.stdout.write(f"\r\033[K\033[92mRecorded: {len(retained_urls) - was_known}\033[0m\n") 
                sys.stdout.write(f"\r\033[KDuration: {timedelta(seconds=int((datetime.utcnow() - timer).total_seconds()))}\n") 
                sys.stdout.flush()
                to_revisit.add(current_url)
                retained_urls[:] = [url for url in retained_urls if not current_url == url]
        retained_urls[:] = retained_urls[was_known:]
        sys.stdout.write(f"\033[F\033[F\033[F\033[F\033[F\033[F\033[F")
        sys.stdout.write(f"\r\033[KVisited: {len(visited) - was_known}\n") 
        sys.stdout.write(f"\r\033[K\033[92mRecorded: {len(retained_urls)}\033[0m\n") 
        sys.stdout.write(f"\r\033[K\033[91mFailed: {len(to_revisit)}\033[0m\n") 
        sys.stdout.write(f"\r\033[KDuration: {timedelta(seconds=int((datetime.utcnow() - timer).total_seconds()))}\n") 
        sys.stdout.flush()
        if len(retained_urls):
            with open(f'{full_path}/url-list.csv', "a", newline="") as f:
                writer = csv.writer(f)
                for item in retained_urls:
                    writer.writerow([item])
            print(f"\033[92m{len(retained_urls)} new URLs discovered. Verify their relevance starting at line {was_known + 1}, in knowledge-sources/{self.folder}/url-list.csv\033[0m")
            self.sync_urls_to_bucket()
        else:
            print("\033[92mNo URLs recorded\033[0m")
        if len(to_revisit):
            print("\033[91mFailed URLs:\033[00m")
            for failed in to_revisit:
                print(failed)
        
    def _ensure_url_is_absolute(self, url: str) -> str:
        is_absolute = bool(urlparse(url).netloc)
        if not is_absolute:
            url = urljoin(self.base_url, url)
        return url
        
    def _is_same_domain(self, url: str) -> bool:
        domain = urlparse(self.base_url).netloc
        return urlparse(url).netloc in [domain, "www." + domain]