import logging
import os
import pickle
import gzip
from typing import List, Optional
from .models import Article
from .config import ScraperConfig
from .utils import create_directory_if_not_exists, get_file_size_mb

logger = logging.getLogger(__name__)

class DataManager:
    """Handles data persistence and loading operations for scraped articles"""
    
    def __init__(self, config: ScraperConfig):
        self.config = config
        self.scraped_articles_file = config.scraped_articles_file
        create_directory_if_not_exists(os.path.dirname(self.scraped_articles_file))
    
    def save_articles(self, articles: List[Article]) -> None:
        """Save articles to compressed pickle file"""
        try:
            # Convert Article objects to dictionaries for serialization
            article_dicts = [
                {
                    'url': article.url,
                    'title': article.title,
                    'content': article.content,
                    'date': article.date,
                    'author': article.author,
                    'meta_description': article.meta_description,
                    'word_count': article.word_count,
                    'scraped_at': article.scraped_at
                }
                for article in articles
            ]
            
            with gzip.open(self.scraped_articles_file, 'wb') as f:
                pickle.dump(article_dicts, f)
            
            file_size = get_file_size_mb(self.scraped_articles_file)
            logger.info(f"Saved {len(articles)} articles to {self.scraped_articles_file} (Size: {file_size:.2f} MB)")
            
        except Exception as e:
            logger.error(f"Error saving articles: {e}")
            raise
    
    def load_articles(self) -> List[Article]:
        """Load articles from compressed pickle file"""
        if not os.path.exists(self.scraped_articles_file):
            logger.warning(f"Articles file not found: {self.scraped_articles_file}")
            return []
        
        try:
            with gzip.open(self.scraped_articles_file, 'rb') as f:
                article_dicts = pickle.load(f)
            
            # Convert dictionaries back to Article objects
            articles = []
            for article_dict in article_dicts:
                try:
                    article = Article(
                        url=article_dict['url'],
                        title=article_dict['title'],
                        content=article_dict['content'],
                        date=article_dict['date'],
                        author=article_dict['author'],
                        meta_description=article_dict['meta_description'],
                        word_count=article_dict['word_count'],
                        scraped_at=article_dict['scraped_at']
                    )
                    articles.append(article)
                except Exception as e:
                    logger.warning(f"Error reconstructing article {article_dict.get('url', 'unknown')}: {e}")
                    continue
            
            file_size = get_file_size_mb(self.scraped_articles_file)
            logger.info(f"Loaded {len(articles)} articles from {self.scraped_articles_file} (Size: {file_size:.2f} MB)")
            return articles
            
        except Exception as e:
            logger.error(f"Error loading articles: {e}")
            raise
    
    def get_articles_count(self) -> int:
        """Get the total number of saved articles"""
        try:
            with gzip.open(self.scraped_articles_file, 'rb') as f:
                article_dicts = pickle.load(f)
            return len(article_dicts)
        except Exception:
            return 0
    
    def backup_articles(self, backup_suffix: str = None) -> str:
        """Create a backup of the articles file"""
        if not os.path.exists(self.scraped_articles_file):
            logger.warning("No articles file to backup")
            return ""
        
        if backup_suffix is None:
            import datetime
            backup_suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        backup_file = f"{self.scraped_articles_file}.backup_{backup_suffix}"
        
        try:
            import shutil
            shutil.copy2(self.scraped_articles_file, backup_file)
            logger.info(f"Created backup: {backup_file}")
            return backup_file
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            raise
    
    def cleanup_old_backups(self, keep_count: int = 5) -> None:
        """Remove old backup files, keeping only the most recent ones"""
        if not os.path.exists(self.scraped_articles_file):
            return
        
        backup_dir = os.path.dirname(self.scraped_articles_file)
        base_name = os.path.basename(self.scraped_articles_file)
        
        # Find all backup files
        backup_files = []
        for file in os.listdir(backup_dir):
            if file.startswith(base_name) and file.endswith('.backup_'):
                backup_files.append(os.path.join(backup_dir, file))
        
        if len(backup_files) <= keep_count:
            return
        
        # Sort by modification time and remove old ones
        backup_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        for old_backup in backup_files[keep_count:]:
            try:
                os.remove(old_backup)
                logger.info(f"Removed old backup: {old_backup}")
            except Exception as e:
                logger.warning(f"Could not remove old backup {old_backup}: {e}")
    
    def validate_articles_file(self) -> bool:
        """Validate that the articles file is not corrupted"""
        if not os.path.exists(self.scraped_articles_file):
            return False
        
        try:
            with gzip.open(self.scraped_articles_file, 'rb') as f:
                articles = pickle.load(f)
            
            if not isinstance(articles, list):
                logger.error("Articles file does not contain a list")
                return False
            
            for article in articles:
                if not isinstance(article, dict):
                    logger.error("Article is not a dictionary")
                    return False
                
                required_fields = ['url', 'title', 'content', 'date', 'author']
                for field in required_fields:
                    if field not in article:
                        logger.error(f"Article missing required field: {field}")
                        return False
            
            logger.info("Articles file validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Articles file validation failed: {e}")
            return False
