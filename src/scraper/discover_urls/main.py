from scraper.discover_urls.url_discoverer import UrlDiscoverer

BASE_URL="https://www.ojim.fr/plan-du-site/"
EXCLUDED_PATHS = ["/page/", "/nous-contacter/", "/mentions-legales/", "/information-presse-medias/"] # Polemia: ['/mot-clef/', '/page/', '/author/']

discoverer = UrlDiscoverer(base_url=BASE_URL, excluded_paths=EXCLUDED_PATHS)
discoverer.discover_urls()