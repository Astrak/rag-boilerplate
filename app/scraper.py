from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup, Tag
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import cast, Optional
import re
import tiktoken
import pickle
import gzip
import os
import numpy as np 
import faiss

DELAY = 0.05 # delay to not Ddos the server
MAX_TOKENS_PER_REQUEST = 260000
MODEL = "text-embedding-3-large"
CHECKPOINT_DIR = "polemia-embeddings"

class ArticleScraper:
    def __init__(self, base_url, excluded_paths = []):
        print('Initialize scraper')
        self.base_url = base_url
        self.excluded_paths = excluded_paths
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; ArticleScraper/1.0)'
        })
        self.scraped_urls: set[str] = set()
        self.failed_urls: set[str] = set()
        self.articles: list[dict] = []
    
    def is_url_excluded(self, url: str) -> bool:
        parsed_url = urlparse(url)
        path = parsed_url.path
        for excluded_path in self.excluded_paths:
            if excluded_path.startswith('/') and path.startswith(excluded_path):
                return True
            elif re.search(excluded_path, path):
                return True
        return False

    def prepare_articles_in_doc_batches_for_embeddings(self) -> list[list[Document]]:
        """Split articles in documents according to ideal token length for vectorization"""
        with gzip.open("./scraped_articles.pkl.gz", 'rb') as f:
            articles = pickle.load(f)
        documents: list[Document] = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2600,
            chunk_overlap=500,
        )
        for article in articles:
            text_chunks = text_splitter.split_text(article['content'])
            for i, chunk in enumerate(text_chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        'source': article['url'],
                        'title': article['title'],
                        'date': article['date'],
                        'author': article['author'],
                        'chunk_id': i,
                        'total_chunks': len(text_chunks),
                        'word_count': article['word_count'],
                        'meta_description': article['meta_description']
                    }
                )
                documents.append(doc)
        print(f"Created {len(documents)} document chunks from {len(articles)} articles")

        """Group all documents in subbatches inferior to OpenAI's token limit"""
        encoding = tiktoken.encoding_for_model(MODEL)
        batches: list[list[Document]] = []
        current_batch = []
        current_token_count = 0
        for doc in documents:
            text_tokens = len(encoding.encode(doc.page_content))
            if current_token_count + text_tokens > MAX_TOKENS_PER_REQUEST:
                batches.append(current_batch)
                current_batch = [doc]
                current_token_count = text_tokens
            else:
                current_batch.append(doc)
                current_token_count += text_tokens
        if current_batch:
            batches.append(current_batch)

        return batches

    def create_vector_store(self, batches: list[list[Document]]) -> FAISS:
        print("Creating vector store, make sure enough RAM is available...")
        
        """Create embeddings for the documents batches"""
        embeddings_model = OpenAIEmbeddings(model=MODEL)
        all_embeddings = []
        all_docs: list[Document] = []
        for i, batch in enumerate(batches):
            print(f"Processing batch {i+1}/{len(batches)}")
            try:
                batch_texts = [doc.page_content for doc in batch]
                batch_embeddings = embeddings_model.embed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)
                all_docs.extend(batch)
                time.sleep(0.1)
            except Exception as e:
                print(f"Error processing batch {i+1}: {e}")
        vector_store = FAISS.from_embeddings(
            embedding=embeddings_model,
            text_embeddings=[(doc.page_content, embedding) for doc, embedding in zip(all_docs, all_embeddings)],
            metadatas=[doc.metadata for doc in all_docs]  # Preserve metadata
        )
        vector_store.save_local('./vectorstore')
        print(f"Vector store saved to ./vectorstore")
        return vector_store

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
                for selector in ['article div a[href]']:
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
            "discovered": list(url for url in discovered_urls if not self.is_url_excluded(url)), 
            "failed": list(to_revisit)
        }
    
    def _is_same_domain(self, url: str) -> bool:
        return urlparse(url).netloc == urlparse(self.base_url).netloc

    def scrape_articles(self, urls: list[str]) -> list[dict]:
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
                time.sleep(DELAY)
                completed = len(self.scraped_urls) + len(self.failed_urls)
                print(f"Progress: {completed}/{len(urls)} articles processed")
        with gzip.open("./scraped_articles.pkl.gz", 'wb') as f:
            pickle.dump(self.articles, f)
        print(f"Saved {len(self.articles)} dictionaries to ./scraped-articles.pkl.gz (compressed)")
        return self.articles
    
    def scrape_article(self, url: str) -> Optional[dict]:
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            title = cast(Tag, soup.select_one('h1.entry-title')).get_text(strip=True)
            content_elem = soup.select_one('#contenu')
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
    
    def create_embeddings_with_checkpoint(self):
        batches = self.prepare_articles_in_doc_batches_for_embeddings()

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        progress_file = os.path.join(CHECKPOINT_DIR, "progress.pkl")
        if os.path.exists(progress_file):
            with open(progress_file, "rb") as f:
                completed_batches = pickle.load(f)
            print(f"Resuming from batch {len(completed_batches)}")
        else:
            completed_batches = []
        embeddings_model = OpenAIEmbeddings(model=MODEL)

        if len(completed_batches) >= len(batches):
            print("Embeddings complete")
            return
        
        for i, batch in enumerate(batches, start=len(completed_batches)):
            print(f"Processing batch {i+1}/{len(batches)}")
            try:
                batch_texts = [doc.page_content for doc in batch]
                batch_embeddings = embeddings_model.embed_documents(batch_texts)
                completed_batches.append(i)
                with open(progress_file, "wb") as f:
                    pickle.dump((completed_batches), f)
                batch_file = os.path.join(CHECKPOINT_DIR, f"batch_{i+1}.pkl")
                with open(batch_file, "wb") as f:
                    pickle.dump((batch, batch_embeddings), f)
                print(f"Batch {i+1} completed and saved")
            except Exception as e:
                print(f"Error in batch {i+1}: {e}")
                break
        print("Embeddings complete")

    def create_chunked_faiss_system(self):
        """Create multiple smaller FAISS indices"""
        embeddings = [f for f in os.listdir('./polemia-embeddings') if f.startswith('batch_') and f.endswith('.pkl')]
        n_embeddings = len(embeddings)
        chunk_size = 20 # memory ceiling is at around 24-26 of the given batches
        for i in range(0, n_embeddings, chunk_size):
            print(f"Processing batch chunk #{i}")
            embeddings_chunk: list[list[float]] = []
            textbatches_chunk: list[Document] = []
            for j in range(0,chunk_size):
                current_batch = i + j + 1
                if current_batch > n_embeddings:
                    break
                batch_file = f"./{CHECKPOINT_DIR}/batch_{current_batch}.pkl"
                print(f'Opening {batch_file}')
                with open(batch_file, 'rb') as f:
                    batch, embeddings_batch = pickle.load(f)
                    embeddings_chunk.extend(embeddings_batch)
                    textbatches_chunk.extend(batch)
            embeddings_array = np.array(embeddings_chunk, dtype=np.float32)
            dimension = embeddings_array.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_array)
            faiss.write_index(index, f"polemia-embeddings/faisschunk_{i//chunk_size}.index")
            with open(f"polemia-embeddings/textbatches_{i//chunk_size}.pkl", "wb") as f:
                pickle.dump(textbatches_chunk, f)
            print(f'Created vector index and batch text file for chunks {i}-{i//chunk_size}')