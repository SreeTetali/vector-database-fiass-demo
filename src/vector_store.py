"""
FAISS vector store implementation with multiple index types.
"""

import faiss
import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time


class FAISSVectorStore:
    """
    FAISS-based vector store supporting multiple index types.
    Supports IndexFlatIP, IndexIVFFlat, and IndexHNSWFlat.
    """
    
    def __init__(
        self, 
        embedding_dim: int = 384,
        index_type: str = "FlatIP",
        index_dir: str = "data/indices"
    ):
        """
        Initialize FAISS vector store.
        
        Args:
            embedding_dim: Dimension of embeddings (default 384 for all-MiniLM-L6-v2)
            index_type: Type of FAISS index - "FlatIP", "IVFFlat", "HNSW"
            index_dir: Directory to save/load indices
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.index = None
        self.documents = []
        self.metadata = []
        
        print(f"✓ Initialized FAISSVectorStore with {index_type} index")
    
    def create_index(self, n_vectors: int = 0, nlist: int = 100, m: int = 32):
        """
        Create appropriate FAISS index based on index_type.
        
        Args:
            n_vectors: Number of vectors (used for IVF training)
            nlist: Number of clusters for IVF (default 100)
            m: Number of connections for HNSW (default 32)
        """
        if self.index_type == "FlatIP":
            # Exact search using Inner Product (cosine similarity with normalized vectors)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            print(f"✓ Created IndexFlatIP (exact search, cosine similarity)")
            
        elif self.index_type == "IVFFlat":
            # Inverted File Index - faster approximate search
            # Requires training
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            print(f"✓ Created IndexIVFFlat with {nlist} clusters (approximate search)")
            print(f"  Note: Requires training before adding vectors")
            
        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World - fast approximate search
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, m)
            self.index.hnsw.efConstruction = 40  # Quality of graph construction
            self.index.hnsw.efSearch = 16  # Search time/accuracy tradeoff
            print(f"✓ Created IndexHNSWFlat with M={m} (fast approximate search)")
            print(f"  efConstruction: {self.index.hnsw.efConstruction}, efSearch: {self.index.hnsw.efSearch}")
            
        else:
            raise ValueError(f"Unknown index type: {self.index_type}. Use 'FlatIP', 'IVFFlat', or 'HNSW'")
    
    def train(self, embeddings: np.ndarray):
        """
        Train index (required for IVFFlat).
        
        Args:
            embeddings: Training vectors (n_vectors, embedding_dim)
        """
        if self.index_type == "IVFFlat":
            if not self.index.is_trained:
                print(f"Training IVFFlat index with {embeddings.shape[0]} vectors...")
                start_time = time.time()
                self.index.train(embeddings)
                train_time = time.time() - start_time
                print(f"✓ Training completed in {train_time:.2f} seconds")
        else:
            print(f"⚠ {self.index_type} does not require training")
    
    def add_documents(
        self, 
        embeddings: np.ndarray, 
        documents: List[Dict],
        metadata: Optional[List[Dict]] = None
    ):
        """
        Add documents and their embeddings to the index.
        
        Args:
            embeddings: Document embeddings (n_docs, embedding_dim)
            documents: List of document dictionaries
            metadata: Optional metadata for each document
        """
        if self.index is None:
            self.create_index(n_vectors=embeddings.shape[0])
        
        # Train if needed (IVFFlat)
        if self.index_type == "IVFFlat" and not self.index.is_trained:
            self.train(embeddings)
        
        # Add vectors to index
        print(f"Adding {embeddings.shape[0]} vectors to index...")
        start_time = time.time()
        self.index.add(embeddings)
        add_time = time.time() - start_time
        
        # Store documents and metadata
        self.documents.extend(documents)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(documents))
        
        print(f"✓ Added {embeddings.shape[0]} vectors in {add_time:.2f} seconds")
        print(f"✓ Total vectors in index: {self.index.ntotal}")
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 5,
        nprobe: int = 10
    ) -> Tuple[List[float], List[int]]:
        """
        Search for k nearest neighbors.
        
        Args:
            query_embedding: Query vector (1, embedding_dim) or (embedding_dim,)
            k: Number of results to return
            nprobe: Number of clusters to search (IVFFlat only)
            
        Returns:
            Tuple of (distances, indices)
        """
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("Index is empty. Add documents first.")
        
        # Reshape query if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Set nprobe for IVF indices
        if self.index_type == "IVFFlat":
            self.index.nprobe = nprobe
        
        # Search
        start_time = time.time()
        distances, indices = self.index.search(query_embedding, k)
        search_time = time.time() - start_time
        
        # Convert to lists
        distances = distances[0].tolist()
        indices = indices[0].tolist()
        
        return distances, indices, search_time
    
    def search_with_documents(
        self, 
        query_embedding: np.ndarray, 
        k: int = 5,
        nprobe: int = 10,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search and return documents with scores.
        
        Args:
            query_embedding: Query vector
            k: Number of results
            nprobe: Number of clusters to search (IVFFlat)
            filter_metadata: Optional metadata filter (e.g., {"category": "Azure Functions"})
            
        Returns:
            List of result dictionaries with document, score, and metadata
        """
        distances, indices, search_time = self.search(query_embedding, k, nprobe)
        
        results = []
        for dist, idx in zip(distances, indices):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            doc = self.documents[idx]
            meta = self.metadata[idx]
            
            # Apply metadata filter if provided
            if filter_metadata:
                match = all(meta.get(key) == value for key, value in filter_metadata.items())
                if not match:
                    continue
            
            result = {
                "document": doc,
                "score": float(dist),
                "index": int(idx),
                "metadata": meta,
                "search_time_ms": search_time * 1000
            }
            results.append(result)
        
        return results[:k]  # Return top k after filtering
    
    def save(self, name: str = "vector_store"):
        """
        Save index, documents, and metadata to disk.
        
        Args:
            name: Base name for saved files
        """
        if self.index is None:
            raise ValueError("No index to save")
        
        # Save FAISS index
        index_file = self.index_dir / f"{name}_{self.index_type}.index"
        faiss.write_index(self.index, str(index_file))
        
        # Save documents and metadata
        data_file = self.index_dir / f"{name}_{self.index_type}_data.pkl"
        with open(data_file, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata,
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type
            }, f)
        
        print(f"✓ Saved index to: {index_file}")
        print(f"✓ Saved data to: {data_file}")
    
    def load(self, name: str = "vector_store"):
        """
        Load index, documents, and metadata from disk.
        
        Args:
            name: Base name of saved files
        """
        # Load FAISS index
        index_file = self.index_dir / f"{name}_{self.index_type}.index"
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")
        
        self.index = faiss.read_index(str(index_file))
        
        # Load documents and metadata
        data_file = self.index_dir / f"{name}_{self.index_type}_data.pkl"
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadata = data['metadata']
            self.embedding_dim = data['embedding_dim']
        
        print(f"✓ Loaded index from: {index_file}")
        print(f"✓ Loaded {len(self.documents)} documents")
        print(f"✓ Index contains {self.index.ntotal} vectors")
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        if self.index is None:
            return {"status": "No index created"}
        
        stats = {
            "index_type": self.index_type,
            "embedding_dim": self.embedding_dim,
            "n_vectors": self.index.ntotal,
            "n_documents": len(self.documents),
            "is_trained": getattr(self.index, 'is_trained', True)
        }
        
        if self.index_type == "IVFFlat":
            stats["nlist"] = self.index.nlist
            stats["nprobe"] = getattr(self.index, 'nprobe', 1)
        elif self.index_type == "HNSW":
            stats["M"] = self.index.hnsw.M
            stats["efSearch"] = self.index.hnsw.efSearch
        
        return stats


# Quick test
if __name__ == "__main__":
    # Test with dummy data
    print("Testing FAISSVectorStore...")
    
    # Create dummy embeddings
    n_docs = 100
    dim = 384
    embeddings = np.random.randn(n_docs, dim).astype('float32')
    faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
    
    # Create dummy documents
    documents = [{"id": i, "text": f"Document {i}"} for i in range(n_docs)]
    
    # Test FlatIP index
    print("\n--- Testing IndexFlatIP ---")
    store_flat = FAISSVectorStore(embedding_dim=dim, index_type="FlatIP")
    store_flat.add_documents(embeddings, documents)
    
    query = embeddings[0:1]  # Use first doc as query
    results = store_flat.search_with_documents(query, k=3)
    print(f"Top 3 results: {[r['document']['id'] for r in results]}")
    print(f"Search time: {results[0]['search_time_ms']:.2f}ms")
    
    print("\nStats:", store_flat.get_stats())
