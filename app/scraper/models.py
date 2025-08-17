from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime

@dataclass
class Article:
    """Data class representing a scraped article"""
    url: str
    title: str
    content: str
    date: str
    author: str
    meta_description: str
    word_count: int
    scraped_at: float
    
    def __post_init__(self):
        """Validate article data after initialization"""
        if not self.url or not self.title or not self.content:
            raise ValueError("URL, title, and content are required")
        if self.word_count <= 0:
            raise ValueError("Word count must be positive")

@dataclass
class ScrapingResult:
    """Result of a scraping operation"""
    successful_urls: List[str]
    failed_urls: List[str]
    total_processed: int
    success_rate: float
    
    def __post_init__(self):
        """Calculate success rate"""
        if self.total_processed > 0:
            self.success_rate = len(self.successful_urls) / self.total_processed
        else:
            self.success_rate = 0.0

@dataclass
class ScrapingProgress:
    """Progress tracking for scraping operations"""
    current: int
    total: int
    successful: int
    failed: int
    
    @property
    def percentage(self) -> float:
        """Calculate completion percentage"""
        return (self.current / self.total * 100) if self.total > 0 else 0.0
    
    @property
    def remaining(self) -> int:
        """Calculate remaining items"""
        return max(0, self.total - self.current)
