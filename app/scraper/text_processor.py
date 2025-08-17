import logging
import tiktoken
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from .models import Article
from .config import ScraperConfig

logger = logging.getLogger(__name__)

class TextProcessor:
    """Handles text processing, splitting, and document preparation for embeddings"""
    
    def __init__(self, config: ScraperConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        self.encoding = tiktoken.encoding_for_model(config.openai_model)
    
    def split_article_text(self, article: Article) -> List[str]:
        """Split article text into chunks"""
        try:
            chunks = self.text_splitter.split_text(article.content)
            logger.info(f"Split article '{article.title}' into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error splitting text for article {article.url}: {e}")
            return [article.content]  # Fallback to original content
    
    def create_documents_from_article(self, article: Article) -> List[Document]:
        """Create Document objects from an article"""
        text_chunks = self.split_article_text(article)
        documents = []
        
        for i, chunk in enumerate(text_chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    'source': article.url,
                    'title': article.title,
                    'date': article.date,
                    'author': article.author,
                    'chunk_id': i,
                    'total_chunks': len(text_chunks),
                    'word_count': article.word_count,
                    'meta_description': article.meta_description,
                    'scraped_at': article.scraped_at
                }
            )
            documents.append(doc)
        
        return documents
    
    def create_documents_from_articles(self, articles: List[Article]) -> List[Document]:
        """Create Document objects from a list of articles"""
        all_documents = []
        
        for article in articles:
            try:
                article_docs = self.create_documents_from_article(article)
                all_documents.extend(article_docs)
            except Exception as e:
                logger.error(f"Error processing article {article.url}: {e}")
                continue
        
        logger.info(f"Created {len(all_documents)} document chunks from {len(articles)} articles")
        return all_documents
    
    def create_batches_for_embeddings(self, documents: List[Document]) -> List[List[Document]]:
        """Group documents into batches that fit within OpenAI's token limit"""
        batches: List[List[Document]] = []
        current_batch: List[Document] = []
        current_token_count = 0
        
        for doc in documents:
            try:
                text_tokens = len(self.encoding.encode(doc.page_content))
                
                if current_token_count + text_tokens > self.config.max_tokens_per_request:
                    if current_batch:  # Don't add empty batches
                        batches.append(current_batch)
                    current_batch = [doc]
                    current_token_count = text_tokens
                else:
                    current_batch.append(doc)
                    current_token_count += text_tokens
                    
            except Exception as e:
                logger.error(f"Error processing document: {e}")
                continue
        
        # Add the last batch if it's not empty
        if current_batch:
            batches.append(current_batch)
        
        logger.info(f"Created {len(batches)} batches for embeddings")
        return batches
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text string"""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.error(f"Error estimating tokens: {e}")
            return len(text.split()) * 1.3  # Rough estimate
    
    def validate_document_batch(self, batch: List[Document]) -> bool:
        """Validate that a batch of documents is within token limits"""
        total_tokens = sum(self.estimate_tokens(doc.page_content) for doc in batch)
        return total_tokens <= self.config.max_tokens_per_request
