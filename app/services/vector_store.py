import faiss
import numpy as np
from typing import List, Tuple

class VectorStore:
    def add_vectors(self, ids: List[str], vectors: List[List[float]], metadata: List[dict]):
        raise NotImplementedError

    def query(self, vector: List[float], top_k: int) -> List[Tuple[str, float, dict]]:
        raise NotImplementedError

class FaissVectorStore(VectorStore):
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner product if normalized
        self.id_map = []  # store ids
        self.metadatas = []

    def add_vectors(self, ids, vectors, metadata):
        vs = np.array(vectors).astype('float32')
        self.index.add(vs)
        self.id_map.extend(ids)
        self.metadatas.extend(metadata)

    def query(self, vector, top_k):
        v = np.array([vector]).astype('float32')
        D, I = self.index.search(v, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            results.append((self.id_map[idx], float(score), self.metadatas[idx]))
        return results
