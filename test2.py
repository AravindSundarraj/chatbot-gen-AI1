import faiss
import numpy as np

# Example setup: creating an index with 100 vectors of dimension 128
d = 128  # dimension
nb = 100  # database size
nq = 10   # number of queries

# Generate random vectors
np.random.seed(1234)  # for reproducibility
db_vectors = np.random.random((nb, d)).astype('float32')
query_vectors = np.random.random((nq, d)).astype('float32')

# Initialize index
index = faiss.IndexFlatL2(d)  # L2 distance index
index.add(db_vectors)  # Add database vectors to the index

# Search: find the top 5 nearest neighbors
k = 5
distances, indices = index.search(query_vectors, k)

# Output
print("Distances:", distances)
print("Indices:", indices)