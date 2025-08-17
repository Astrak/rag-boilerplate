import logging
import os
from typing import List, Optional, Dict, Any
from .config import ScraperConfig
from .models import Article, ScrapingResult
from .web_scraper import WebScraper
from .text_processor import TextProcessor
from .vector_store import VectorStoreManager
from .data_manager import DataManager
from .utils import create_directory_if_not_exists

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ArticleScraper:
    """
    Main scraper class that orchestrates web scraping, text processing, and vector store operations.
    
    This class provides a high-level interface for:
    - Discovering and scraping articles from websites
    - Processing and chunking text content
    - Creating embeddings and vector stores
    - Managing the entire scraping pipeline
    """
    
    def __init__(self, base_url: str, excluded_paths: List[str] = None, config: Optional[ScraperConfig] = None):
        """
        Initialize the ArticleScraper.
        
        Args:
            base_url: The base URL to start scraping from
            excluded_paths: List of URL patterns to exclude from scraping
            config: Configuration object (uses default if not provided)
        """
        logger.info('Initializing ArticleScraper')
        
        # Set up configuration
        if config is None:
            config = ScraperConfig()
        
        # Add base_url to config
        config.base_url = base_url
        config.excluded_paths = excluded_paths or []
        
        self.config = config
        self.excluded_paths = excluded_paths or []
        
        # Initialize components
        self.data_manager = DataManager(config)
        self.web_scraper = WebScraper(config, self.data_manager)
        self.text_processor = TextProcessor(config)
        self.vector_store_manager = VectorStoreManager(config)
        
        # Ensure directories exist
        create_directory_if_not_exists(self.config.checkpoint_dir)
        create_directory_if_not_exists(self.config.vectorstore_dir)
        
        logger.info(f'ArticleScraper initialized for {base_url}')
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete scraping pipeline: discover URLs, scrape articles, 
        process text, and create vector store.
        
        Returns:
            Dictionary containing results from each pipeline stage
        """
        logger.info("Starting full scraping pipeline")
        
        try:
            # Stage 1: Discover URLs
            logger.info("Stage 1: URL Discovery")
            discovery_result = self.web_scraper.discover_urls()
            discovered_urls = discovery_result["discovered"]
            
            if not discovered_urls:
                logger.warning("No URLs discovered. Pipeline cannot continue.")
                return {"error": "No URLs discovered"}
            
            # Stage 2: Scrape Articles
            logger.info("Stage 2: Article Scraping")
            scraping_result = self.web_scraper.scrape_articles(discovered_urls)
            
            if not scraping_result.successful_urls:
                logger.warning("No articles successfully scraped. Pipeline cannot continue.")
                return {"error": "No articles scraped", "scraping_result": scraping_result}
            
            # Stage 3: Load and Process Articles
            logger.info("Stage 3: Text Processing")
            articles = self.data_manager.load_articles()
            documents = self.text_processor.create_documents_from_articles(articles)
            batches = self.text_processor.create_batches_for_embeddings(documents)
            
            # Stage 4: Create Vector Store
            logger.info("Stage 4: Vector Store Creation")
            vector_store = self.vector_store_manager.create_vector_store(batches)
            
            # Stage 5: Create Chunked System (optional)
            logger.info("Stage 5: Creating Chunked FAISS System")
            self.vector_store_manager.create_chunked_faiss_system()
            
            pipeline_result = {
                "discovery": discovery_result,
                "scraping": scraping_result,
                "text_processing": {
                    "total_articles": len(articles),
                    "total_documents": len(documents),
                    "total_batches": len(batches)
                },
                "vector_store": {
                    "created": True,
                    "location": self.config.vectorstore_dir
                },
                "chunked_system": {
                    "created": True,
                    "location": self.config.checkpoint_dir
                }
            }
            
            logger.info("Full pipeline completed successfully")
            return pipeline_result
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def discover_and_scrape(self) -> ScrapingResult:
        """
        Discover URLs and scrape articles without creating vector store.
        
        Returns:
            ScrapingResult containing information about the scraping operation
        """
        logger.info("Starting URL discovery and article scraping")
        
        try:
            # Discover URLs
            discovery_result = self.web_scraper.discover_urls()
            discovered_urls = discovery_result["discovered"]
            
            if not discovered_urls:
                logger.warning("No URLs discovered")
                return ScrapingResult(
                    successful_urls=[],
                    failed_urls=[],
                    total_processed=0,
                    success_rate=0.0
                )
            
            # Scrape articles
            scraping_result = self.web_scraper.scrape_articles(discovered_urls)
            
            logger.info(f"Discovery and scraping completed. Success rate: {scraping_result.success_rate:.1%}")
            return scraping_result
            
        except Exception as e:
            logger.error(f"Discovery and scraping failed: {e}")
            raise
    
    def create_vector_store_from_existing(self) -> Any:
        """
        Create vector store from previously scraped articles.
        
        Returns:
            FAISS vector store object
        """
        logger.info("Creating vector store from existing articles")
        
        try:
            # Load articles
            articles = self.data_manager.load_articles()
            if not articles:
                logger.warning("No articles found. Please run scraping first.")
                return None
            
            # Process text
            documents = self.text_processor.create_documents_from_articles(articles)
            batches = self.text_processor.create_batches_for_embeddings(documents)
            
            # Create vector store
            vector_store = self.vector_store_manager.create_vector_store(batches)
            
            logger.info("Vector store created successfully from existing articles")
            return vector_store
            
        except Exception as e:
            logger.error(f"Failed to create vector store from existing articles: {e}")
            raise
    
    def create_embeddings_with_checkpoint(self) -> None:
        """
        Create embeddings with checkpointing for resumability.
        This is useful for large datasets that may take a long time to process.
        """
        logger.info("Starting checkpointed embeddings creation")
        
        try:
            # Load articles
            articles = self.data_manager.load_articles()
            if not articles:
                logger.warning("No articles found. Please run scraping first.")
                return
            
            # Process text
            documents = self.text_processor.create_documents_from_articles(articles)
            batches = self.text_processor.create_batches_for_embeddings(documents)
            
            # Create embeddings with checkpointing
            self.vector_store_manager.create_embeddings_with_checkpoint(batches)
            
            logger.info("Checkpointed embeddings creation completed")
            
        except Exception as e:
            logger.error(f"Checkpointed embeddings creation failed: {e}")
            raise
    
    def search_similar_documents(self, query: str, results: int = 20) -> List[Article]:
        """
        Search for similar documents using the chunked FAISS system.
        
        Args:
            query: The search query text
            results: Maximum number of results to return
            
        Returns:
            List of similar documents
        """
        logger.info(f"Searching for documents similar to: {query}")
        
        try:
            matching_documents = self.vector_store_manager.chunked_similarity_search(query)
            
            # Convert documents back to articles (simplified)
            articles = []
            for doc in matching_documents[:results]:
                # Create a simplified article from document metadata
                article = Article(
                    url=doc.metadata.get('source', ''),
                    title=doc.metadata.get('title', ''),
                    content=doc.page_content,
                    date=doc.metadata.get('date', ''),
                    author=doc.metadata.get('author', ''),
                    meta_description=doc.metadata.get('meta_description', ''),
                    word_count=len(doc.page_content.split()),
                    scraped_at=doc.metadata.get('scraped_at', 0)
                )
                articles.append(article)
            
            logger.info(f"Found {len(articles)} similar documents")
            return articles
            
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the current scraping state.
        
        Returns:
            Dictionary containing various statistics
        """
        try:
            articles_count = self.data_manager.get_articles_count()
            
            stats = {
                "articles_count": articles_count,
                "checkpoint_dir": self.config.checkpoint_dir,
                "vectorstore_dir": self.config.vectorstore_dir,
                "scraped_articles_file": self.config.scraped_articles_file,
                "articles_file_size_mb": None,
                "checkpoint_files_count": 0,
                "faiss_indices_count": 0
            }
            
            # Get file sizes and counts
            if os.path.exists(self.config.scraped_articles_file):
                from .utils import get_file_size_mb
                stats["articles_file_size_mb"] = get_file_size_mb(self.config.scraped_articles_file)
            
            if os.path.exists(self.config.checkpoint_dir):
                checkpoint_files = [f for f in os.listdir(self.config.checkpoint_dir) 
                                  if f.startswith('batch_') and f.endswith('.pkl')]
                faiss_files = [f for f in os.listdir(self.config.checkpoint_dir) 
                              if f.startswith('faisschunk_') and f.endswith('.index')]
                
                stats["checkpoint_files_count"] = len(checkpoint_files)
                stats["faiss_indices_count"] = len(faiss_files)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"error": str(e)}
    
    def cleanup_old_data(self, keep_recent_backups: int = 3) -> None:
        """
        Clean up old data files and backups.
        
        Args:
            keep_recent_backups: Number of recent backups to keep
        """
        logger.info("Starting cleanup of old data")
        
        try:
            # Clean up old backups
            self.data_manager.cleanup_old_backups(keep_recent_backups)
            
            # Clean up old checkpoint files (optional)
            # This could be extended to clean up old FAISS indices, etc.
            
            logger.info("Cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            raise
        