import logging
import os
import pickle
import gzip
import time
from typing import List, Optional, Tuple
import numpy as np
import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from .config import ScraperConfig
from .utils import create_directory_if_not_exists, retry_on_failure

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages vector store operations including embeddings and FAISS indices"""
    
    def __init__(self, config: ScraperConfig):
        self.config = config
        self.embeddings_model = OpenAIEmbeddings(model=config.openai_model)
        create_directory_if_not_exists(config.checkpoint_dir)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def create_embeddings_batch(self, batch: List[Document]) -> Tuple[List[Document], List[List[float]]]:
        """Create embeddings for a batch of documents with retry logic"""
        try:
            batch_texts = [doc.page_content for doc in batch]
            batch_embeddings = self.embeddings_model.embed_documents(batch_texts)
            logger.info(f"Successfully created embeddings for batch of {len(batch)} documents")
            return batch, batch_embeddings
        except Exception as e:
            logger.error(f"Error creating embeddings for batch: {e}")
            raise
    
    def create_vector_store(self, batches: List[List[Document]]) -> FAISS:
        """Create a FAISS vector store from document batches"""
        logger.info("Creating vector store, make sure enough RAM is available...")
        
        all_embeddings: List[List[float]] = []
        all_docs: List[Document] = []
        
        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)}")
            try:
                batch, batch_embeddings = self.create_embeddings_batch(batch)
                all_embeddings.extend(batch_embeddings)
                all_docs.extend(batch)
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                logger.error(f"Error processing batch {i+1}: {e}")
                continue
        
        if not all_docs:
            raise ValueError("No documents were successfully processed")
        
        try:
            vector_store = FAISS.from_embeddings(
                embedding=self.embeddings_model,
                text_embeddings=[(doc.page_content, embedding) for doc, embedding in zip(all_docs, all_embeddings)],
                metadatas=[doc.metadata for doc in all_docs]
            )
            
            # Save the vector store
            vector_store.save_local(self.config.vectorstore_dir)
            logger.info(f"Vector store saved to {self.config.vectorstore_dir}")
            
            return vector_store
            
        except Exception as e:
            logger.error(f"Error creating FAISS vector store: {e}")
            raise
    
    def create_embeddings_with_checkpoint(self, batches: List[List[Document]]) -> None:
        """Create embeddings with checkpointing for resumability"""
        progress_file = os.path.join(self.config.checkpoint_dir, "progress.pkl")
        
        # Load existing progress
        if os.path.exists(progress_file):
            with open(progress_file, "rb") as f:
                completed_batches = pickle.load(f)
            logger.info(f"Resuming from batch {len(completed_batches)}")
        else:
            completed_batches = []
        
        if len(completed_batches) >= len(batches):
            logger.info("Embeddings already complete")
            return
        
        # Process remaining batches
        for i, batch in enumerate(batches, start=len(completed_batches)):
            logger.info(f"Processing batch {i+1}/{len(batches)}")
            try:
                batch, batch_embeddings = self.create_embeddings_batch(batch)
                
                # Save batch
                batch_file = os.path.join(self.config.checkpoint_dir, f"batch_{i+1}.pkl")
                with open(batch_file, "wb") as f:
                    pickle.dump((batch, batch_embeddings), f)
                
                # Update progress
                completed_batches.append(i)
                with open(progress_file, "wb") as f:
                    pickle.dump(completed_batches, f)
                
                logger.info(f"Batch {i+1} completed and saved")
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error in batch {i+1}: {e}")
                break
        
        logger.info("Embeddings complete")
    
    def create_chunked_faiss_system(self) -> None:
        """Create multiple smaller FAISS indices for memory efficiency"""
        batch_files = [f for f in os.listdir(self.config.checkpoint_dir) 
                       if f.startswith('batch_') and f.endswith('.pkl')]
        n_embeddings = len(batch_files)
        
        if n_embeddings == 0:
            logger.warning("No batch files found for creating FAISS system")
            return
        
        logger.info(f"Creating chunked FAISS system from {n_embeddings} batches")
        
        for i in range(0, n_embeddings, self.config.faiss_chunk_size):
            logger.info(f"Processing batch chunk #{i}")
            
            embeddings_chunk: List[List[float]] = []
            textbatches_chunk: List[Document] = []
            
            for j in range(self.config.faiss_chunk_size):
                current_batch = i + j + 1
                if current_batch > n_embeddings:
                    break
                
                batch_file = os.path.join(self.config.checkpoint_dir, f"batch_{current_batch}.pkl")
                logger.debug(f'Opening {batch_file}')
                
                try:
                    with open(batch_file, 'rb') as f:
                        batch, embeddings_batch = pickle.load(f)
                        embeddings_chunk.extend(embeddings_batch)
                        textbatches_chunk.extend(batch)
                except Exception as e:
                    logger.error(f"Error loading batch {current_batch}: {e}")
                    continue
            
            if not embeddings_chunk:
                logger.warning(f"No embeddings found for chunk {i}")
                continue
            
            try:
                # Create FAISS index
                embeddings_array = np.array(embeddings_chunk, dtype=np.float32)
                dimension = embeddings_array.shape[1]
                index = faiss.IndexFlatL2(dimension)
                index.add(embeddings_array)
                
                # Save index and text batches
                chunk_id = i // self.config.faiss_chunk_size
                index_file = os.path.join(self.config.checkpoint_dir, f"faisschunk_{chunk_id}.index")
                text_file = os.path.join(self.config.checkpoint_dir, f"textbatches_{chunk_id}.pkl")
                
                faiss.write_index(index, index_file)
                with open(text_file, "wb") as f:
                    pickle.dump(textbatches_chunk, f)
                
                logger.info(f'Created vector index and batch text file for chunk {chunk_id}')
                
            except Exception as e:
                logger.error(f"Error creating FAISS chunk {i}: {e}")
                continue
    
    def search_chunked_system(self, query_embedding: List[float], results: int = 20) -> List[Document]:
        """Search across all chunks and merge results"""
        all_results: List[Document] = []
        
        index_files = [f for f in os.listdir(self.config.checkpoint_dir) 
                       if f.startswith('faisschunk_') and f.endswith('.index')]
        n_chunks = len(index_files)
        
        if n_chunks == 0:
            logger.warning("No FAISS index files found")
            return all_results
        
        results_per_chunk = max(1, results // n_chunks + 1)
        
        for chunk_id in range(n_chunks):
            try:
                index_file = os.path.join(self.config.checkpoint_dir, f"faisschunk_{chunk_id}.index")
                text_file = os.path.join(self.config.checkpoint_dir, f"textbatches_{chunk_id}.pkl")
                
                if not (os.path.exists(index_file) and os.path.exists(text_file)):
                    logger.warning(f"Missing files for chunk {chunk_id}")
                    continue
                
                # Load index and search
                index = faiss.read_index(index_file)
                scores, indices = index.search(np.array([query_embedding]), results_per_chunk)
                
                # Load corresponding text batches
                with open(text_file, "rb") as f:
                    chunk_texts: List[Document] = pickle.load(f)
                
                # Add results
                for score, idx in zip(scores[0], indices[0]):
                    if idx < len(chunk_texts):
                        all_results.append(chunk_texts[idx])
                
            except Exception as e:
                logger.error(f"Error searching chunk {chunk_id}: {e}")
                continue
        
        logger.info(f"Found {len(all_results)} matching documents across {n_chunks} chunks")
        return all_results
    
    def chunked_similarity_search(self, question_text: str) -> List[Document]:
        """Perform similarity search using chunked FAISS system"""
        logger.info(f'Received question: {question_text}')
        
        try:
            response = self.embeddings_model.embed_query(question_text)
            question_embeddings = response
            logger.info('Successfully generated embeddings for question')
            
            matching_documents = self.search_chunked_system(question_embeddings)
            logger.info('Found matching documents')
            
            return matching_documents
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
