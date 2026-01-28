from src.scraper.download_articles.article_dowloader import ArticleDownloader
import os

sources = os.getenv("SOURCES")
if not sources:
    print("No SOURCES variable specified, extracting from all sources in the knowledge-sources folder:")
    sources = os.listdir("./knowledge-sources/")
else:
    sources = sources.split(',')
print("Download or extract text from sources: ", sources)

log_articles = os.getenv("LOG_ARTICLES")
if not log_articles:
    raise EnvironmentError("LOG_ARTICLES not found. Run with LOG_ARTICLES='true' or 'false'")

def main():
    for source in sources:
        discoverer = ArticleDownloader(source, log_articles)
        discoverer.scrape_articles()
        discoverer.sync_gzip_to_bucket()

if __name__ == "__main__":
    main()
