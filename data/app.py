import faiss
import numpy as np

# Example: 128-dimensional vectors, 1000 vectors
dimension = 128
num_vectors = 1000
vectors = np.random.random((num_vectors, dimension)).astype('float32')

# Build a FAISS index
index = faiss.IndexFlatL2(dimension)  # Use IndexIVFFlat for large datasets
index.add(vectors)

print(f"Total vectors in index: {index.ntotal}")
