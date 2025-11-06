from src.scraper.download_articles.article_dowloader import ArticleDownloader

FOLDER = "./ojim-urls/"

ARTICLE_SELECTOR = "#singleojimcontent" # "#contenu" pour Polémia

# Make sure to check the ArticleDownloader#scrape_article method and vet how metadata, author, date and title are selected.

discoverer = ArticleDownloader(folder=FOLDER, article_selector=ARTICLE_SELECTOR)
discoverer.sync_urls_from_bucket()
discoverer.scrape_articles()
