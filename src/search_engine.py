"""
High-level search interface combining embedding and vector store.
"""

from typing import List, Dict, Optional
import numpy as np
from .embedding_engine import EmbeddingEngine
from .vector_store import FAISSVectorStore


class SemanticSearchEngine:
    """
    Complete semantic search engine combining embeddings and FAISS.
    """
    
    def __init__(
        self,
        index_type: str = "FlatIP",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize search engine.
        
        Args:
            index_type: FAISS index type ("FlatIP", "IVFFlat", "HNSW")
            model_name: Embedding model name
        """
        print(f"\n{'='*60}")
        print(f"Initializing Semantic Search Engine")
        print(f"Index Type: {index_type}")
        print(f"Embedding Model: {model_name}")
        print(f"{'='*60}\n")
        
        self.embedder = EmbeddingEngine(model_name=model_name)
        self.vector_store = FAISSVectorStore(
            embedding_dim=self.embedder.embedding_dim,
            index_type=index_type
        )
    
    def index_documents(self, documents: List[Dict], text_field: str = "text"):
        """
        Index documents for semantic search.
        
        Args:
            documents: List of document dictionaries
            text_field: Field containing text to index
        """
        print(f"\n--- Indexing {len(documents)} documents ---")
        
        # Generate embeddings
        embeddings = self.embedder.embed_documents(documents, text_field)
        
        # Extract metadata
        metadata = []
        for doc in documents:
            meta = {k: v for k, v in doc.items() if k != text_field}
            metadata.append(meta)
        
        # Add to vector store
        self.vector_store.add_documents(embeddings, documents, metadata)
        
        print(f"✓ Indexing complete\n")
    
    def search(
        self,
        query: str,
        k: int = 5,
        nprobe: int = 10,
        filter_category: Optional[str] = None
    ) -> List[Dict]:
        """
        Semantic search for query.
        
        Args:
            query: Search query text
            k: Number of results to return
            nprobe: Number of clusters to search (for IVFFlat)
            filter_category: Optional category filter
            
        Returns:
            List of search results with documents and scores
        """
        # Generate query embedding
        query_embedding = self.embedder.embed(query, show_progress=False)
        
        # Search
        filter_metadata = {"category": filter_category} if filter_category else None
        results = self.vector_store.search_with_documents(
            query_embedding,
            k=k,
            nprobe=nprobe,
            filter_metadata=filter_metadata
        )
        
        return results
    
    def save(self, name: str = "search_engine"):
        """Save the search engine (index and documents)."""
        self.vector_store.save(name)
    
    def load(self, name: str = "search_engine"):
        """Load a saved search engine."""
        self.vector_store.load(name)
    
    def get_stats(self) -> Dict:
        """Get search engine statistics."""
        return self.vector_store.get_stats()
    
    def print_results(self, results: List[Dict], max_text_len: int = 150):
        """
        Pretty print search results.
        
        Args:
            results: Search results from search()
            max_text_len: Maximum text length to display
        """
        print(f"\n{'='*80}")
        print(f"Search Results (Found {len(results)} documents)")
        print(f"{'='*80}\n")
        
        for i, result in enumerate(results, 1):
            doc = result['document']
            score = result['score']
            
            print(f"[{i}] Score: {score:.4f}")
            print(f"    Title: {doc.get('title', 'N/A')}")
            print(f"    Category: {doc.get('category', 'N/A')}")
            
            text = doc.get('text', '')
            if len(text) > max_text_len:
                text = text[:max_text_len] + "..."
            print(f"    Text: {text}")
            
            if 'tags' in doc:
                print(f"    Tags: {', '.join(doc['tags'])}")
            
            print()


# Quick test
if __name__ == "__main__":
    from .data_loader import DocumentationLoader
    
    # Load sample documents
    loader = DocumentationLoader()
    docs = loader.create_sample_docs()
    
    # Create search engine
    engine = SemanticSearchEngine(index_type="FlatIP")
    
    # Index documents
    engine.index_documents(docs)
    
    # Test search
    query = "How do I use Azure serverless functions?"
    results = engine.search(query, k=3)
    engine.print_results(results)
    
    # Show stats
    print("\nEngine Stats:")
    print(engine.get_stats())
