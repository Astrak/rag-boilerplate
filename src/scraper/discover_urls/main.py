from scraper.discover_urls.url_discoverer import UrlDiscoverer
import os

folder = os.getenv("FOLDER")
if not folder:
    raise EnvironmentError("FOLDER not found. Run with FOLDER=mywebsite.fr")

EXCLUDED_PATHS = [".png", ".jpg", ".jpeg", ".xlsx", "/wp-login", "/page", ".xml"]

def main():
    discoverer = UrlDiscoverer(folder, excluded_paths=EXCLUDED_PATHS)
    discoverer.discover_urls()

if __name__ == "__main__":
    main()