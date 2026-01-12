"""
Quick test script to verify all components are working.
Run this after installation to ensure everything is set up correctly.
"""

import sys
import time

def test_imports():
    """Test all required imports."""
    print("\n" + "="*60)
    print("TEST 1: Checking imports...")
    print("="*60)
    
    try:
        import faiss
        print(f"✓ FAISS: {faiss.__version__}")
    except ImportError as e:
        print(f"✗ FAISS import failed: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✓ Sentence-Transformers")
    except ImportError as e:
        print(f"✗ Sentence-Transformers import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        from src.data_loader import DocumentationLoader
        from src.embedding_engine import EmbeddingEngine
        from src.vector_store import FAISSVectorStore
        from src.search_engine import SemanticSearchEngine
        print("✓ All project modules")
    except ImportError as e:
        print(f"✗ Project module import failed: {e}")
        return False
    
    print("\n✅ All imports successful!")
    return True


def test_data_loading():
    """Test data loading."""
    print("\n" + "="*60)
    print("TEST 2: Testing data loading...")
    print("="*60)
    
    try:
        from src.data_loader import DocumentationLoader
        loader = DocumentationLoader()
        docs = loader.create_sample_docs()
        
        if len(docs) > 0:
            print(f"✓ Loaded {len(docs)} documents")
            print(f"✓ Sample doc: {docs[0]['title']}")
            print("\n✅ Data loading successful!")
            return True, docs
        else:
            print("✗ No documents loaded")
            return False, []
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False, []


def test_embeddings(docs):
    """Test embedding generation."""
    print("\n" + "="*60)
    print("TEST 3: Testing embedding generation...")
    print("="*60)
    
    try:
        from src.embedding_engine import EmbeddingEngine
        
        print("Loading embedding model (may take a moment on first run)...")
        embedder = EmbeddingEngine()
        
        # Test single text
        test_text = "This is a test sentence"
        embedding = embedder.embed(test_text, show_progress=False)
        
        print(f"✓ Embedding shape: {embedding.shape}")
        print(f"✓ Embedding dimension: {embedding.shape[1]}")
        
        # Test document batch
        embeddings = embedder.embed_documents(docs[:5], text_field="text")
        print(f"✓ Batch embeddings shape: {embeddings.shape}")
        
        print("\n✅ Embedding generation successful!")
        return True, embeddings
    except Exception as e:
        print(f"✗ Embedding generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_vector_store(embeddings, docs):
    """Test FAISS vector store."""
    print("\n" + "="*60)
    print("TEST 4: Testing FAISS vector store...")
    print("="*60)
    
    try:
        from src.vector_store import FAISSVectorStore
        import numpy as np
        
        # Test FlatIP index
        store = FAISSVectorStore(embedding_dim=384, index_type="FlatIP")
        store.add_documents(embeddings, docs[:5])
        
        print(f"✓ Created FlatIP index")
        print(f"✓ Index stats: {store.get_stats()}")
        
        # Test search
        query_emb = embeddings[0:1]
        results = store.search_with_documents(query_emb, k=3)
        
        print(f"✓ Search returned {len(results)} results")
        print(f"✓ Top result: {results[0]['document']['title']}")
        
        print("\n✅ Vector store successful!")
        return True
    except Exception as e:
        print(f"✗ Vector store failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_search_engine():
    """Test complete search engine."""
    print("\n" + "="*60)
    print("TEST 5: Testing complete search engine...")
    print("="*60)
    
    try:
        from src.search_engine import SemanticSearchEngine
        from src.data_loader import DocumentationLoader
        
        # Load data
        loader = DocumentationLoader()
        docs = loader.load_documents()
        
        # Create engine
        engine = SemanticSearchEngine(index_type="HNSW")
        engine.index_documents(docs)
        
        # Test search
        query = "How do I use Azure Functions?"
        results = engine.search(query, k=3)
        
        print(f"✓ Query: '{query}'")
        print(f"✓ Found {len(results)} results")
        print(f"✓ Top result: {results[0]['document']['title']}")
        print(f"✓ Score: {results[0]['score']:.4f}")
        
        print("\n✅ Search engine successful!")
        return True
    except Exception as e:
        print(f"✗ Search engine failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("FAISS VECTOR DATABASE DEMO - SETUP TEST")
    print("="*60)
    
    start_time = time.time()
    
    # Run tests
    all_passed = True
    
    if not test_imports():
        all_passed = False
        print("\n❌ Import test failed. Please check installation.")
        return
    
    success, docs = test_data_loading()
    if not success:
        all_passed = False
        print("\n❌ Data loading test failed.")
        return
    
    success, embeddings = test_embeddings(docs)
    if not success:
        all_passed = False
        print("\n❌ Embedding test failed.")
        return
    
    if not test_vector_store(embeddings, docs):
        all_passed = False
        print("\n❌ Vector store test failed.")
    
    if not test_search_engine():
        all_passed = False
        print("\n❌ Search engine test failed.")
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*60)
        print(f"\nTotal time: {elapsed:.2f} seconds")
        print("\n✅ Setup is complete and working correctly!")
        print("\nNext steps:")
        print("  1. Run: jupyter notebook")
        print("  2. Open: notebooks/01_data_preparation.ipynb")
        print("  3. Follow notebooks in sequence")
    else:
        print("❌ SOME TESTS FAILED")
        print("="*60)
        print("\nPlease review error messages above and check:")
        print("  - Virtual environment is activated")
        print("  - All dependencies installed: pip install -r requirements.txt")
        print("  - Python version is 3.9+")
    
    print()


if __name__ == "__main__":
    main()
