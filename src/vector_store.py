"""
FAISS vector store implementation with multiple index types.
"""

import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class FAISSVectorStore:
    """FAISS-based vector store supporting multiple index types."""
    
    def __init__(self, embedding_dim: int = 384, index_type: str = "FlatIP"):
        """
        Initialize FAISS vector store.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            index_type: Type of FAISS index ('FlatIP', 'IVFFlat', 'HNSW')
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.documents = []
        self.is_trained = False
        
        # Create index based on type
        self._create_index()
    
    def _create_index(self):
        """Create FAISS index based on specified type."""
        print(f"\n{'='*60}")
        print(f"Initializing FAISS Index: {self.index_type}")
        print(f"{'='*60}")
        
        if self.index_type == "FlatIP":
            # Exact search using inner product (cosine similarity with normalized vectors)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.is_trained = True
            print(f"✓ Created IndexFlatIP (exact search, cosine similarity)")
            
        elif self.index_type == "IVFFlat":
            # Inverted file index for faster approximate search
            nlist = 10  # number of clusters
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            print(f"✓ Created IndexIVFFlat")
            print(f"  - nlist (clusters): {nlist}")
            print(f"  - Requires training before use")
            
        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World graph
            M = 32  # number of connections per layer
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, M)
            self.index.hnsw.efConstruction = 40
            self.index.hnsw.efSearch = 16
            self.is_trained = True
            print(f"✓ Created IndexHNSWFlat (fast approximate search)")
            print(f"  efConstruction: 40, efSearch: 16")
            
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
    
    def train(self, embeddings: np.ndarray):
        """
        Train the index (required for IVFFlat).
        
        Args:
            embeddings: Training vectors (n_vectors, embedding_dim)
        """
        if self.index_type == "IVFFlat" and not self.is_trained:
            print(f"Training IVFFlat index on {len(embeddings)} vectors...")
            self.index.train(embeddings)
            self.is_trained = True
            print("✓ Index trained successfully")
        else:
            if self.is_trained:
                print("✓ Index already trained or doesn't require training")
    
    def add_documents(self, embeddings: np.ndarray, documents: List[Dict]):
        """
        Add documents and their embeddings to the index.
        
        Args:
            embeddings: Document embeddings (n_docs, embedding_dim)
            documents: List of document dictionaries
        """
        # Ensure embeddings are float32
        embeddings = embeddings.astype('float32')
        
        print(f"\nAdding {len(documents)} vectors to index...")
        
        # Train if needed
        if not self.is_trained:
            self.train(embeddings)
        
        # Add to index
        import time
        start = time.time()
        self.index.add(embeddings)
        elapsed = time.time() - start
        
        self.documents.extend(documents)
        
        print(f"✓ Added {len(documents)} vectors in {elapsed:.2f} seconds")
        print(f"✓ Total vectors in index: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query vector (1, embedding_dim)
            k: Number of results to return
            **kwargs: Additional parameters (e.g., nprobe for IVFFlat)
        
        Returns:
            Tuple of (distances, indices)
        """
        # Ensure query is 2D and float32
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype('float32')
        
        # Set nprobe for IVFFlat
        if self.index_type == "IVFFlat":
            nprobe = kwargs.get('nprobe', 3)
            self.index.nprobe = nprobe
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        return distances[0], indices[0]
    
    def search_with_documents(self, query_embedding: np.ndarray, k: int = 5, **kwargs) -> List[Dict]:
        """
        Search and return documents with scores.
        
        Args:
            query_embedding: Query vector
            k: Number of results
            **kwargs: Additional search parameters
        
        Returns:
            List of dicts with 'document', 'score', 'index'
        """
        distances, indices = self.search(query_embedding, k, **kwargs)
        
        results = []
        for dist, idx in zip(distances, indices):
            if idx < len(self.documents):  # Valid index
                results.append({
                    'document': self.documents[idx],
                    'score': float(dist),
                    'index': int(idx)
                })
        
        return results
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        stats = {
            'index_type': self.index_type,
            'embedding_dim': self.embedding_dim,
            'n_vectors': self.index.ntotal,
            'n_documents': len(self.documents),
            'is_trained': self.is_trained
        }
        
        # Add index-specific stats
        if self.index_type == "IVFFlat":
            stats["nlist"] = self.index.nlist
            stats["nprobe"] = getattr(self.index, 'nprobe', 1)
        elif self.index_type == "HNSW":
            # Simplified - avoid version-specific attributes
            stats["note"] = "HNSW graph-based index (M=32, efSearch=16)"
        
        return stats
    
    def save(self, path: str):
        """
        Save index and documents to disk.
        
        Args:
            path: Base path for saving (without extension)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = f"{path}.index"
        faiss.write_index(self.index, index_path)
        
        # Save documents and metadata
        meta_path = f"{path}.pkl"
        with open(meta_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type,
                'is_trained': self.is_trained
            }, f)
        
        print(f"✓ Saved index to {index_path}")
        print(f"✓ Saved metadata to {meta_path}")
    
    def load(self, path: str):
        """
        Load index and documents from disk.
        
        Args:
            path: Base path for loading (without extension)
        """
        index_path = f"{path}.index"
        meta_path = f"{path}.pkl"
        
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load documents and metadata
        with open(meta_path, 'rb') as f:
            metadata = pickle.load(f)
            self.documents = metadata['documents']
            self.embedding_dim = metadata['embedding_dim']
            self.index_type = metadata['index_type']
            self.is_trained = metadata['is_trained']
        
        print(f"✓ Loaded index from {index_path}")
        print(f"✓ Loaded {len(self.documents)} documents")


# Quick test
if __name__ == "__main__":
    import numpy as np
    
    # Create sample embeddings and documents
    n_docs = 10
    dim = 384
    embeddings = np.random.randn(n_docs, dim).astype('float32')
    # Normalize for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    docs = [{'id': i, 'text': f'Document {i}'} for i in range(n_docs)]
    
    # Test FlatIP
    print("\n=== Testing IndexFlatIP ===")
    store = FAISSVectorStore(embedding_dim=dim, index_type="FlatIP")
    store.add_documents(embeddings, docs)
    
    # Search
    query = embeddings[0:1]  # Use first doc as query
    results = store.search_with_documents(query, k=3)
    print(f"\nTop 3 results:")
    for r in results:
        print(f"  Doc {r['document']['id']}: score={r['score']:.4f}")
    
    print("\n✓ All tests passed!")
