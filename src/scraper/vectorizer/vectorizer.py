from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import tiktoken
import pickle
import gzip
import os
import boto3
import numpy as np 
import faiss

DELAY = 0.05 # delay to not Ddos the server
MAX_TOKENS_PER_REQUEST = 260000
MODEL = "text-embedding-3-large"

class Vectorizer:
    def __init__(self, source):
        self.folder = f"./knowledge-sources/{source}"
        self.checkpoint_dir = f"{self.folder}/embeddings"

    def sync_embeddings_to_bucket(self):
        s3 = boto3.client('s3', region_name="eu-north-1")
        try:
            print("\033[KUploading embeddings to AWS backup...")
            print('Will upload embeddings:')
            for root, _, files in os.walk(self.checkpoint_dir):
                for filename in files:
                    print("Filename: ", f"{self.checkpoint_dir}/{filename}")
                    print("To: ", f"{self.folder.split('/')[2]}/embeddings/{filename}")
                    s3.upload_file(
                        Filename=f"{self.checkpoint_dir}/{filename}", 
                        Bucket="rag-faiss-index-bucket", 
                        Key=f"{self.folder.split('/')[2]}/embeddings/{filename}"
                    )
            print("\033[KSynced")
        except Exception as e:
            print("\033[KFailed to upload embeddings to AWS bucket")

    def create_embeddings_with_checkpoint(self):
        batches = self._prepare_articles_in_doc_batches_for_embeddings()
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        progress_file = os.path.join(self.checkpoint_dir, "progress.pkl")
        if os.path.exists(progress_file):
            with open(progress_file, "rb") as f:
                completed_batches = pickle.load(f)
            print(f"Resuming from batch {len(completed_batches)}")
        else:
            completed_batches = []
        embeddings_model = OpenAIEmbeddings(model=MODEL)
        if len(completed_batches) >= len(batches):
            print("Embeddings complete")
            return
        for i, batch in enumerate(batches, start=len(completed_batches)):
            print(f"Processing batch {i+1}/{len(batches)}")
            try:
                batch_texts = [doc.page_content for doc in batch]
                batch_embeddings = embeddings_model.embed_documents(batch_texts)
                completed_batches.append(i)
                with open(progress_file, "wb") as f:
                    pickle.dump((completed_batches), f)
                batch_file = os.path.join(self.checkpoint_dir, f"batch_{i+1}.pkl")
                with open(batch_file, "wb") as f:
                    pickle.dump((batch, batch_embeddings), f)
                print(f"Batch {i+1} completed and saved")
            except Exception as e:
                print(f"Error in batch {i+1}: {e}")
                break
        print("Embeddings complete")

    def create_chunked_faiss_system(self):
        """Create multiple smaller FAISS indices"""
        embeddings = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('batch_') and f.endswith('.pkl')]
        n_embeddings = len(embeddings)
        chunk_size = 20 # memory ceiling is at around 24-26 of the given batches
        for i in range(0, n_embeddings, chunk_size):
            print(f"Processing batch chunk #{i}")
            embeddings_chunk: list[list[float]] = []
            textbatches_chunk: list[Document] = []
            for j in range(0,chunk_size):
                current_batch = i + j + 1
                if current_batch > n_embeddings:
                    break
                batch_file = f"{self.checkpoint_dir}/batch_{current_batch}.pkl"
                print(f'Opening {batch_file}')
                with open(batch_file, 'rb') as f:
                    batch, embeddings_batch = pickle.load(f)
                    embeddings_chunk.extend(embeddings_batch)
                    textbatches_chunk.extend(batch)
            embeddings_array = np.array(embeddings_chunk, dtype=np.float32)
            dimension = embeddings_array.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_array)
            faiss.write_index(index, f"{self.checkpoint_dir}/faisschunk_{i//chunk_size}.index")
            with open(f"{self.checkpoint_dir}/textbatches_{i//chunk_size}.pkl", "wb") as f:
                pickle.dump(textbatches_chunk, f)
            print(f'Created vector index and batch text file for chunks {i}-{i//chunk_size}')
    
    def _prepare_articles_in_doc_batches_for_embeddings(self) -> list[list[Document]]:
        """Split articles in documents according to ideal token length for vectorization"""
        with gzip.open(f"{self.folder}/scraped_articles.pkl.gz", 'rb') as f:
            articles = pickle.load(f)
        documents: list[Document] = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2600,
            chunk_overlap=500,
        )
        for article in articles:
            text_chunks = text_splitter.split_text(article['content'])
            for i, chunk in enumerate(text_chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        'source': article.get('url'),
                        'title': article.get('title'),
                        'date': article.get('date'),
                        'author': article.get('author'),
                        'chunk_id': i,
                        'total_chunks': len(text_chunks),
                        'word_count': article.get('word_count'),
                        'meta_description': article.get('meta_description')
                    }
                )
                documents.append(doc)
        print(f"Created {len(documents)} document chunks from {len(articles)} articles")

        """Group all documents in subbatches inferior to OpenAI's token limit"""
        encoding = tiktoken.encoding_for_model(MODEL)
        batches: list[list[Document]] = []
        current_batch = []
        current_token_count = 0
        for doc in documents:
            text_tokens = len(encoding.encode(doc.page_content))
            if current_token_count + text_tokens > MAX_TOKENS_PER_REQUEST:
                batches.append(current_batch)
                current_batch = [doc]
                current_token_count = text_tokens
            else:
                current_batch.append(doc)
                current_token_count += text_tokens
        if current_batch:
            batches.append(current_batch)

        return batches