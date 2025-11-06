from scraper.discover_urls.url_discoverer import UrlDiscoverer

BASE_URL="https://observatoire-immigration.fr/"
EXCLUDED_PATHS = []

discoverer = UrlDiscoverer(base_url=BASE_URL, excluded_paths=EXCLUDED_PATHS)
discoverer.discover_urls()