"""
RAG Boilerplate Scraper Package

This package provides a modular and robust web scraping solution for creating
vector databases from web content. It includes:

- ArticleScraper: Main orchestrator class
- WebScraper: Handles web crawling and article extraction
- TextProcessor: Manages text chunking and document preparation
- VectorStoreManager: Handles embeddings and FAISS operations
- DataManager: Manages data persistence and loading
- Configuration management and utilities
"""

from .scraper import ArticleScraper
from .web_scraper import WebScraper
from .text_processor import TextProcessor
from .vector_store import VectorStoreManager
from .data_manager import DataManager
from .config import ScraperConfig
from .models import Article, ScrapingResult, ScrapingProgress
from .utils import (
    retry_on_failure,
    is_valid_url,
    is_same_domain,
    sanitize_filename,
    create_directory_if_not_exists,
    get_file_size_mb
)

__version__ = "2.0.0"

__all__ = [
    "ArticleScraper",
    "WebScraper", 
    "TextProcessor",
    "VectorStoreManager",
    "DataManager",
    "ScraperConfig",
    "Article",
    "ScrapingResult",
    "ScrapingProgress",
    "retry_on_failure",
    "is_valid_url",
    "is_same_domain",
    "sanitize_filename",
    "create_directory_if_not_exists",
    "get_file_size_mb"
]
