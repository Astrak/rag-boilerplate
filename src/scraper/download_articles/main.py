from src.scraper.download_articles.article_dowloader import ArticleDownloader
import os

folder = os.getenv("FOLDER")
if not folder:
    raise EnvironmentError("FOLDER not found. Run with FOLDER='./my-folder/'")

article_selector = os.getenv("ARTICLE_SELECTOR")
if not article_selector:
    raise EnvironmentError("ARTICLE_SELECTOR not found. Run with ARTICLE_SELECTOR='#css-selector'")

print("Downloading articles")
print("Make sure to have checked the ArticleDownloader#scrape_article method and have vetted how metadata, author, date and title are selected.")

discoverer = ArticleDownloader(folder, article_selector)
discoverer.sync_urls_from_bucket()
discoverer.scrape_articles()
