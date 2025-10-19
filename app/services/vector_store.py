import faiss
import numpy as np
import os
import json
import pickle
import logging
from typing import List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class VectorStore:
    def add_vectors(self, ids: List[str], vectors: List[List[float]], metadata: List[dict]):
        raise NotImplementedError

    def query(self, vector: List[float], top_k: int) -> List[Tuple[str, float, dict]]:
        raise NotImplementedError

class FaissVectorStore(VectorStore):
    def __init__(self, dim: int, persist_path: str = "./data/vector_store"):
        """Initialize the FAISS vector store with optional persistence.
        
        Args:
            dim: Dimension of the vectors
            persist_path: Directory to save/load the vector store
        """
        self.dim = dim
        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize empty index and data structures
        self.index = faiss.IndexFlatIP(dim)
        self.id_map = []
        self.metadatas = []
        self.texts = []  # Store text content separately
        
        # Try to load existing data
        self._load()

    def add_vectors(self, ids, vectors, metadata, texts=None):
        """Add vectors to the index and save to disk.
        
        Args:
            ids: List of document chunk IDs
            vectors: List of vector embeddings
            metadata: List of metadata dictionaries
            texts: Optional list of text contents (will be stored in metadata if not None)
        """
        vs = np.array(vectors).astype('float32')
        self.index.add(vs)
        
        # Store text in metadata if provided
        if texts is not None:
            for i, text in enumerate(texts):
                if i < len(metadata):
                    metadata[i]['text'] = text
        
        self.id_map.extend(ids)
        self.metadatas.extend(metadata)
        self._save()
        
    def _save(self):
        """Save the index and metadata to disk."""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.persist_path / "index.faiss"))
            
            # Save id_map, metadatas, and texts
            with open(self.persist_path / "metadata.pkl", "wb") as f:
                pickle.dump({
                    'id_map': self.id_map,
                    'metadatas': self.metadatas,
                    'dim': self.dim,
                    'texts': self.texts if hasattr(self, 'texts') else []
                }, f)
            logger.info(f"Vector store saved to {self.persist_path}")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise
    
    def _load(self):
        """Load the index and metadata from disk if they exist."""
        index_path = self.persist_path / "index.faiss"
        meta_path = self.persist_path / "metadata.pkl"
        
        if index_path.exists() and meta_path.exists():
            try:
                # Load FAISS index
                self.index = faiss.read_index(str(index_path))
                
                # Load metadata
                with open(meta_path, "rb") as f:
                    data = pickle.load(f)
                    self.id_map = data['id_map']
                    self.metadatas = data['metadatas']
                    self.dim = data['dim']
                    self.texts = data.get('texts', [])
                
                logger.info(f"Loaded vector store with {len(self.id_map)} vectors from {self.persist_path}")
            except Exception as e:
                logger.error(f"Error loading vector store: {str(e)}")
                # Reset to empty index if loading fails
                self.index = faiss.IndexFlatIP(self.dim)
                self.id_map = []
                self.metadatas = []
                self.texts = []

    def query(self, vector, top_k):
        """Query the vector store for similar vectors.
        
        Returns:
            List of (id, score, metadata) tuples
        """
        v = np.array([vector]).astype('float32')
        D, I = self.index.search(v, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            if idx >= len(self.id_map) or idx >= len(self.metadatas):
                logger.warning(f"Index {idx} out of bounds for id_map/metadatas")
                continue
                
            # Ensure metadata is a dictionary
            metadata = self.metadatas[idx] if isinstance(self.metadatas[idx], dict) else {}
            
            # Add text to metadata if it's not already there
            if 'text' not in metadata and hasattr(self, 'texts') and idx < len(self.texts):
                metadata['text'] = self.texts[idx]
                
            results.append((self.id_map[idx], float(score), metadata))
            
        return results
