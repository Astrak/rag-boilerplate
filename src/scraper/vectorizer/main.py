from src.scraper.vectorizer.vectorizer import Vectorizer
from env import fill_env

fill_env()

FOLDER = './ojim-urls/'
CHECKPOINT_DIR = FOLDER + 'embeddings'

vectorizer = Vectorizer(folder=FOLDER, checkpoint_dir=CHECKPOINT_DIR)
# vectorizer.create_embeddings_with_checkpoint()
vectorizer.create_chunked_faiss_system()