from scraper.discover_urls.url_discoverer import UrlDiscoverer
import os

sources = os.getenv("SOURCES")
if not sources:
    print("No SOURCES variable specified, updating all sources in the knowledge-sources folder:")
    sources = os.listdir("./knowledge-sources/")
    print(sources)

excluded_paths = [".png", ".jpg", ".jpeg", ".xlsx", "/wp-login", "/page", ".xml", "#"]

def main():
    for source in sources:
        discoverer = UrlDiscoverer(source, excluded_paths=excluded_paths)
        discoverer.discover_urls()
    print("Done. Don't forget to backup the data before proceeding.")

if __name__ == "__main__":
    main()