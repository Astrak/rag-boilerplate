from src.scraper.article_scraper import ArticleScraper
from vector_store import get_store
import csv
import boto3
import os
from env import fill_env

fill_env()

FOLDER = "./ojim-urls/"

# Create URL list

# Scrape articles
s3 = boto3.client('s3', region_name="eu-north-1")
os.makedirs(FOLDER, exist_ok=True)
s3.download_file("rag-faiss-index-bucket", f"{FOLDER}url-list.csv", f"{FOLDER}url-list.csv")
lines: list[str] = []
with open(f'{FOLDER}url-list.csv', 'r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        lines.append(row[0])

# EXCLUDED_PATHS = ['/mot-clef/', '/page/', '/author/']
scraper = ArticleScraper(base_url="https://www.ojim.fr/")
# articles = scraper.scrape_articles(lines)
# scraper.create_embeddings_with_checkpoint()
# scraper.create_chunked_faiss_system()
# store = scraper.create_vector_store()