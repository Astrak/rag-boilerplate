from scraper.discover_urls.url_discoverer import UrlDiscoverer
import os

sources = os.getenv("SOURCES")
if not sources:
    print("No SOURCES variable specified, updating all sources in the knowledge-sources folder:")
    sources = os.listdir("./knowledge-sources/")
else:
    sources = sources.split(',')
print("Scrape sources: ", sources)

def main():
    # TODO: sync from AWS?
    for source in sources:
        try:
            discoverer = UrlDiscoverer(source)
            discoverer.discover_urls()
        except Exception as e:
            print(f'\033[91mFailed to discover URLs from {source}\033[0m')

if __name__ == "__main__":
    main()