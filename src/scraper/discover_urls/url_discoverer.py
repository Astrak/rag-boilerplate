from urllib.parse import urlparse, urljoin
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

class UrlDiscoverer:
    def __init__(self, folder, excluded_paths):
        self.folder = folder
        self.excluded_paths = excluded_paths
        self.known_urls = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; ArticleScraper/1.0)'
        })
    
    def sync_urls_to_bucket(self):
        s3 = boto3.client('s3', region_name="eu-north-1")
        try:
            print("Uploading url-list.csv to AWS backup...")
            s3.upload_file(
                Bucket="rag-faiss-index-bucket", 
                Filename=f"knowledge-sources/{self.folder}/url-list.csv", 
                Key=f"{self.folder}/url-list.csv"
            )
            print("Synced")
        except Exception as e:
            print("Failed to upload url-list.csv to AWS bucket")
        try:
            print("Uploading blacklist.csv to AWS backup...")
            s3.upload_file(
                Bucket="rag-faiss-index-bucket", 
                Filename=f"knowledge-sources/{self.folder}/blacklist.csv", 
                Key=f"{self.folder}/blacklist.csv"
            )
            print("Synced")
        except Exception as e:
            print("Failed to upload blacklist.csv to AWS bucket")

    def discover_urls(self):
        if not "." in self.folder:
            print(f"{self.folder} is not a website, no URL will be explored, skipping.")
            return 
        print("Discovering URLs for " + self.folder)
        full_path = f"./knowledge-sources/{self.folder}"
        retained_urls = []
        to_visit = set()
        to_revisit = set()
        visited = set()
        blacklisted = set()
        was_known = 0
        timer = datetime.utcnow()
        response = self.session.get(f"https://{self.folder}", timeout=10)
        self.base_url = response.url # Use main domain that the folder domain redirects to (with or without www)
        to_visit.add(self.base_url)
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
        print("\n\n\n\n\n\n")
        while to_visit:
            current_url = cast(str, to_visit.pop())
            if current_url in visited:
                continue
            try:
                response = self.session.get(current_url, timeout=10)
                response.raise_for_status()
                visited.add(current_url)
                soup = BeautifulSoup(response.content, 'html.parser')
                page_links = set()
                links = [cast(str, href.get('href')) for href in soup.select("a[href]")]
                for href in links:
                    href = self._ensure_url_is_absolute(href)
                    href = href.split('#')[0] # Discard hashes
                    split = href.split('://')
                    split[1] = split[1].replace("//","/") # Ensure no malformed link is kept
                    href = "://".join(split)
                    if self._is_same_domain(href) and href not in blacklisted and href not in visited:
                        page_links.add(href)
                    if href and self._is_same_domain(href) and not href in visited and not href in to_visit:
                        exclude_visits = [".pdf", ".png", ".jpg", ".jpeg", ".xlsx", "/wp-admin", "/wp-login"]
                        if not any(sub in href for sub in exclude_visits): 
                            to_visit.add(href)
                        if not self._is_url_excluded(href) and not href in blacklisted and not href in retained_urls:
                            retained_urls.append(href)
                # if the only remaining links in a page are further pages of search results, 
                # typically of the form "page/5" in the URL, 
                # don't look further into similarly prefixed pages.
                # (supposes that no line of older articles in an existing url-list.csv are removed)
                # TODO: check if a number follows with regexp otherwise it may also just be a url like /page/my-article
                if re.search("/page/",current_url) and len(page_links) and all("/page/" in url for url in page_links):
                    prefix = current_url.split("/page/")[0]
                    to_visit = {url for url in to_visit if not url.startswith(prefix)}
                sys.stdout.write(f"\033[F\033[F\033[F\033[F\033[F\033[F")
                sys.stdout.write(f"\r\033[KVisiting: {current_url}\n") 
                sys.stdout.write(f"\r\033[KVisited: {len(visited) - was_known}\n") 
                sys.stdout.write(f"\r\033[KRemaining: {len(to_visit)}\n") 
                sys.stdout.write(f"\r\033[KFailed: {len(to_revisit)}\n") 
                sys.stdout.write(f"\r\033[KRecorded: {len(retained_urls) - was_known}\n") 
                sys.stdout.write(f"\r\033[KDuration: {timedelta(seconds=int((datetime.utcnow() - timer).total_seconds()))}\n") 
                sys.stdout.flush()
                time.sleep(DELAY) 
            except Exception as e:
                print(current_url, e)
                to_revisit.add(current_url)
                retained_urls[:] = [url for url in retained_urls if not current_url == url]
        retained_urls[:] = retained_urls[was_known:]
        sys.stdout.write(f"\033[F\033[F\033[F\033[F\033[F\033[F\033[F")
        sys.stdout.write(f"\r\033[KVisited: {len(visited) - was_known}\n") 
        sys.stdout.write(f"\r\033[KRecorded: {len(retained_urls)}\n") 
        sys.stdout.write(f"\r\033[KFailed: {len(to_revisit)}\n") 
        sys.stdout.write(f"\r\033[KDuration: {timedelta(seconds=int((datetime.utcnow() - timer).total_seconds()))}\n") 
        sys.stdout.flush()
        if len(retained_urls):
            with open(f'{full_path}/url-list.csv', "a", newline="") as f:
                writer = csv.writer(f)
                for item in retained_urls:
                    writer.writerow([item])
            print(f"URL discovery complete. Verify the relevance of the {len(retained_urls)} last new entries, starting at line {was_known}, in knowledge-sources/{self.folder}/url-list.csv")
            self.sync_urls_to_bucket()
        else:
            print("No URLs recorded")
        if len(to_revisit):
            print("\033[93mFailed URLs:\033[00m")
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

    def _is_url_excluded(self, url: str) -> bool:
        parsed_url = urlparse(url)
        path = parsed_url.path
        for excluded_path in self.excluded_paths:
            if excluded_path.startswith('/') and path.startswith(excluded_path):
                return True
            elif re.search(excluded_path, path):
                return True
        return False