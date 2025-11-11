from scraper.discover_urls.url_discoverer import UrlDiscoverer
import os

base_url = os.getenv("BASE_URL")
if not base_url:
    raise EnvironmentError("BASE_URL not found. Run with BASE_URL=https://mywebsite.fr/")

EXCLUDED_PATHS = []

def main():
    discoverer = UrlDiscoverer(base_url, excluded_paths=EXCLUDED_PATHS)
    discoverer.discover_urls()

if __name__ == "__main__":
    main()