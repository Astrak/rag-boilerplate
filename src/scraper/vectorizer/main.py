from src.scraper.vectorizer.vectorizer import Vectorizer
from src.scraper.vectorizer.env import fill_env
import os

sources = os.getenv("SOURCES")
if not sources:
    print("No SOURCES variable specified, vectorizing all sources in the knowledge-sources folder:")
    sources = os.listdir("./knowledge-sources/")
else:
    sources = sources.split(',')
print("Vectorize text from sources: ", sources)

fill_env()

def main():
    for source in sources:
        try:
            vectorizer = Vectorizer(source)
            vectorizer.create_embeddings_with_checkpoint()
            vectorizer.create_chunked_faiss_system()
            vectorizer.sync_embeddings_to_bucket()
        except Exception as e:
            print(f"Failed to vectorize {source}: ", e)

if __name__ == "__main__":
    main()