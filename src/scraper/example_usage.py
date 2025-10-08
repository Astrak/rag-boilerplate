#!/usr/bin/env python3
"""
Example usage of the improved ArticleScraper

This script demonstrates how to use the new modular scraper architecture
with proper configuration, logging, and error handling.
"""

import os
import logging
from scraper import ArticleScraper, ScraperConfig

def main():
    """Main function demonstrating scraper usage"""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example configuration
    config = ScraperConfig(
        delay=0.1,  # Slower rate limiting for demo
        max_workers=2,  # Fewer workers for demo
        timeout=10,
        chunk_size=2000,  # Smaller chunks for demo
        chunk_overlap=200
    )
    
    # Initialize scraper
    base_url = "https://example.com"  # Replace with actual URL
    excluded_paths = ["/admin", "/login", "/private"]
    
    try:
        scraper = ArticleScraper(
            base_url=base_url,
            excluded_paths=excluded_paths,
            config=config
        )
        
        # Option 1: Run full pipeline (discover, scrape, process, vectorize)
        print("Running full pipeline...")
        result = scraper.run_full_pipeline()
        print(f"Pipeline result: {result}")
        
        # Option 2: Just discover and scrape (without vector store)
        # print("Discovering and scraping articles...")
        # scraping_result = scraper.discover_and_scrape()
        # print(f"Scraping completed with {scraping_result.success_rate:.1%} success rate")
        
        # Option 3: Create vector store from existing articles
        # print("Creating vector store from existing articles...")
        # vector_store = scraper.create_vector_store_from_existing()
        # print("Vector store created successfully")
        
        # Option 4: Search for similar documents
        # print("Searching for similar documents...")
        # query = "What is machine learning?"
        # similar_docs = scraper.search_similar_documents(query, results=5)
        # print(f"Found {len(similar_docs)} similar documents")
        
        # Get statistics
        stats = scraper.get_statistics()
        print(f"Scraper statistics: {stats}")
        
    except Exception as e:
        logging.error(f"Scraper failed: {e}")
        raise

def demo_step_by_step():
    """Demonstrate step-by-step usage"""
    
    config = ScraperConfig()
    scraper = ArticleScraper("https://example.com", config=config)
    
    # Step 1: Discover URLs
    print("Step 1: Discovering URLs...")
    discovery_result = scraper.web_scraper.discover_urls()
    print(f"Discovered {len(discovery_result['discovered'])} URLs")
    
    # Step 2: Scrape articles
    print("Step 2: Scraping articles...")
    if discovery_result['discovered']:
        scraping_result = scraper.web_scraper.scrape_articles(discovery_result['discovered'][:5])  # Limit to 5 for demo
        print(f"Scraped {len(scraping_result.successful_urls)} articles successfully")
    
    # Step 3: Process text
    print("Step 3: Processing text...")
    articles = scraper.data_manager.load_articles()
    if articles:
        documents = scraper.text_processor.create_documents_from_articles(articles)
        batches = scraper.text_processor.create_batches_for_embeddings(documents)
        print(f"Created {len(batches)} batches from {len(documents)} documents")
    
    # Step 4: Create vector store
    print("Step 4: Creating vector store...")
    if articles:
        vector_store = scraper.vector_store_manager.create_vector_store(batches)
        print("Vector store created successfully")

if __name__ == "__main__":
    print("ArticleScraper Example Usage")
    print("=" * 40)
    
    # Check if we have the required environment variables
    if not os.getenv('OPENAI_API_KEY'):
        print("Warning: OPENAI_API_KEY environment variable not set")
        print("Some features may not work without OpenAI API access")
        print()
    
    try:
        # Run the main demo
        main()
        
        print("\n" + "=" * 40)
        print("Demo completed successfully!")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed: {e}")
        logging.exception("Demo error details:")
