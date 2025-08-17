import os
from typing import List
from dataclasses import dataclass

@dataclass
class ScraperConfig:
    """Configuration class for the ArticleScraper"""
    
    # Rate limiting
    delay: float = float(os.getenv('SCRAPER_DELAY', '0.05'))
    max_workers: int = int(os.getenv('SCRAPER_MAX_WORKERS', '3'))
    timeout: int = int(os.getenv('SCRAPER_TIMEOUT', '15'))
    
    # OpenAI settings
    openai_model: str = os.getenv('OPENAI_MODEL', 'text-embedding-3-large')
    max_tokens_per_request: int = int(os.getenv('MAX_TOKENS_PER_REQUEST', '260000'))
    
    # File paths
    checkpoint_dir: str = os.getenv('CHECKPOINT_DIR', 'polemia-embeddings')
    scraped_articles_file: str = os.getenv('SCRAPED_ARTICLES_FILE', './scraped_articles.pkl.gz')
    vectorstore_dir: str = os.getenv('VECTORSTORE_DIR', './vectorstore')
    
    # Text processing
    chunk_size: int = int(os.getenv('CHUNK_SIZE', '2600'))
    chunk_overlap: int = int(os.getenv('CHUNK_OVERLAP', '500'))
    
    # FAISS settings
    faiss_chunk_size: int = int(os.getenv('FAISS_CHUNK_SIZE', '20'))
    
    # User agent
    user_agent: str = os.getenv('SCRAPER_USER_AGENT', 'Mozilla/5.0 (compatible; ArticleScraper/1.0)')
    
    # Retry settings
    max_retries: int = int(os.getenv('SCRAPER_MAX_RETRIES', '3'))
    retry_delay: float = float(os.getenv('SCRAPER_RETRY_DELAY', '1.0'))
