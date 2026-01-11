"""
Embedding generation using Sentence Transformers.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
from tqdm import tqdm


class EmbeddingEngine:
    """Generate embeddings for text using Sentence Transformers."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding model.
        
        Args:
            model_name: HuggingFace model name. Default is all-MiniLM-L6-v2
                       (384 dimensions, fast, good performance)
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✓ Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def embed(self, texts: Union[str, List[str]], show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text string or list of texts
            show_progress: Show progress bar for batch encoding
            
        Returns:
            numpy array of embeddings (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Encode with progress bar
        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        
        return embeddings.astype('float32')  # FAISS prefers float32
    
    def embed_documents(self, documents: List[dict], text_field: str = "text") -> np.ndarray:
        """
        Embed documents from list of dictionaries.
        
        Args:
            documents: List of document dicts
            text_field: Key containing text to embed
            
        Returns:
            numpy array of embeddings
        """
        texts = [doc[text_field] for doc in documents]
        print(f"Encoding {len(texts)} documents...")
        return self.embed(texts)


# Quick test
if __name__ == "__main__":
    # Test embedding generation
    engine = EmbeddingEngine()
    
    test_texts = [
        "Azure Functions is a serverless compute service",
        "Python virtual environments isolate dependencies",
        "Machine learning models require training data"
    ]
    
    embeddings = engine.embed(test_texts)
    print(f"\nGenerated embeddings shape: {embeddings.shape}")
    print(f"Sample embedding (first 10 dims): {embeddings[0][:10]}")
