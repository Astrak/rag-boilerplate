import os

from scraper.discover_urls.url_discoverer import UrlDiscoverer

sources_env = os.getenv("SOURCES")
if not sources_env:
    print("No SOURCES variable specified, updating all sources in the knowledge-sources folder:")
    sources: list[str] = os.listdir("./knowledge-sources/")
else:
    sources = sources_env.split(',')
print("Scrape sources: ", sources)

def main() -> None:
    # TODO: sync from AWS?
    for source in sources:
        try:
            discoverer = UrlDiscoverer(source)
            discoverer.discover_urls()
        except Exception:
            print(f'\033[91mFailed to discover URLs from {source}\033[0m')

if __name__ == "__main__":
    main()