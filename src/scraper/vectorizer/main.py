from src.scraper.vectorizer.vectorizer import Vectorizer
from env import fill_env
import os

folder = os.getenv("FOLDER")
if not folder:
    raise EnvironmentError("FOLDER not found. Run with FOLDER='./my-folder/'")

fill_env()

CHECKPOINT_DIR = folder + 'embeddings'

vectorizer = Vectorizer(folder, checkpoint_dir=CHECKPOINT_DIR)
vectorizer.create_embeddings_with_checkpoint()
vectorizer.create_chunked_faiss_system()