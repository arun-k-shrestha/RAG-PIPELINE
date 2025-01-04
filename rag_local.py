import faiss
import numpy as np

print("Hi")
embeddings = np.array([
    [0.1, 0.2, 0.3],   # sentence 1
    [0.2, 0.1, 0.0],   # sentence 2
    [0.9, 0.8, 0.7]    # sentence 3
]).astype('float32')

print("Embeddings:", embeddings)