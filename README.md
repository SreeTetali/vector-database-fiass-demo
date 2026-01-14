# FAISS Vector Database Implementation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FAISS](https://img.shields.io/badge/FAISS-1.9.0+-green.svg)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Production-grade vector similarity search implementation using Facebook AI Similarity Search (FAISS) for semantic search over Python/Azure technical documentation. The project demonstrates multiple FAISS index types, semantic search, and performance benchmarking suitable for enterprise AI/ML use cases.

---

## 🎯 Project Overview

This project showcases an end-to-end **vector database** implementation using FAISS over a curated set of Python and Azure documentation snippets. It is designed as a portfolio-ready example to demonstrate:

- Understanding of **vector databases**
- Practical use of **FAISS index types**
- Building a **semantic search** system
- Reasoning about **latency, accuracy, and memory tradeoffs**

### Key Features

- ⚡ **Multiple FAISS Index Types**
  - `IndexFlatIP` (exact search)
  - `IndexIVFFlat` (clustered approximate search)
  - `IndexHNSWFlat` (graph-based approximate search)
- 🔍 **Semantic Search**
  - Natural language queries over Python/Azure docs
- 📊 **Performance Benchmarks**
  - Build time
  - Search latency (mean, median, P95)
  - Approximate memory usage
- 💾 **Persistence**
  - Save/load FAISS indices and associated metadata
- 🏷️ **Metadata Filtering**
  - Filter results by fields like `category`
- 🧱 **Modular Architecture**
  - Clear separation between data loading, embedding, indexing, and search API

---

## 📊 Performance Summary (Demo Scale)

With a small demo corpus of Python/Azure documentation snippets (easily extendable to 10k+ documents), typical behavior is:

| Index Type     | Build Time (demo) | Avg Search Latency (demo) | P95 Latency (demo) | Recommended Use Case                    |
|----------------|-------------------|----------------------------|--------------------|-----------------------------------------|
| **IndexFlatIP** | Very low          | Low                        | Low                | Small datasets, 100% accuracy           |
| **IndexIVFFlat** | Low              | Very low                   | Very low           | Medium–large datasets, speed/accuracy   |
| **IndexHNSWFlat** | Low             | Very low                   | Very low           | Production systems, sub-ms latency      |

On larger corpora, the relative behavior is:

- `IndexFlatIP`: linear in number of vectors
- `IndexIVFFlat`: significantly faster with tunable recall
- `IndexHNSWFlat`: near-logarithmic search with high recall

These patterns match common FAISS usage recommendations from production guides.

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **FAISS (faiss-cpu)** – core vector index implementation
- **Sentence-Transformers** – `all-MiniLM-L6-v2` embedding model
- **NumPy / Pandas** – numerical and tabular utilities
- **Jupyter** – interactive exploration and demos
- **Matplotlib / Seaborn** – basic visualization for benchmarks

---

## 📁 Project Structure

```text
vector-database-faiss-demo/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── setup_instructions.md              # Detailed local setup steps
├── data/
│   ├── raw/                           # (Optional) Raw scraped docs (gitignored)
│   ├── processed/                     # Preprocessed documents
│   │   └── sample_docs.json           # Python/Azure doc snippets
│   └── indices/                       # Saved FAISS indices (gitignored)
├── notebooks/
│   ├── 01_data_preparation.ipynb      # Load + explore documentation
│   ├── 02_faiss_indexing.ipynb        # Build and compare FAISS indices
│   └── 03_performance_benchmarks.ipynb# Detailed latency/memory benchmarks
├── src/
│   ├── __init__.py                    # Package init
│   ├── data_loader.py                 # Sample Python/Azure doc loader
│   ├── embedding_engine.py            # Sentence-Transformers wrapper
│   ├── vector_store.py                # FAISS index abstraction
│   └── search_engine.py               # High-level semantic search API
├── optional_chromadb/
│   └── chromadb_comparison.ipynb      # (Optional) ChromaDB comparison
├── tests/
│   └── test_vector_store.py           # Unit tests for vector store
└── test_setup.py                      # Sanity check script for full stack
```

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/SreeTetali/vector-database-faiss-demo.git
cd vector-database-faiss-demo
```

### 2. Create and Activate Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS / Linux)
source venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
# Quick dependency sanity check
python test_setup.py
```

You should see all tests passing: imports, data loading, embeddings, FAISS store, and full search engine.

---

## 💡 Usage Examples

### Example 1 – Basic Semantic Search

```python
from src.search_engine import SemanticSearchEngine
from src.data_loader import DocumentationLoader

# Load sample Python/Azure docs
loader = DocumentationLoader()
docs = loader.load_documents()

# Create search engine with HNSW index (fast approximate)
engine = SemanticSearchEngine(index_type="HNSW")

# Index documents
engine.index_documents(docs)

# Run a semantic query
query = "How do I deploy serverless applications on Azure?"
results = engine.search(query, k=3)

# Pretty print results
engine.print_results(results)
```

### Example 2 – Working Directly with FAISSVectorStore

```python
from src.data_loader import DocumentationLoader
from src.embedding_engine import EmbeddingEngine
from src.vector_store import FAISSVectorStore

# Load documents and embeddings
loader = DocumentationLoader()
docs = loader.load_documents()

embedder = EmbeddingEngine()
embeddings = embedder.embed_documents(docs)

# Create and populate an IVFFlat index
store = FAISSVectorStore(embedding_dim=embeddings.shape[1], index_type="IVFFlat")
# Index is created automatically - training happens in add_documents()
store.add_documents(embeddings, docs)

# Search
query_emb = embedder.embed(["Azure machine learning services"], show_progress=False)
results = store.search_with_documents(query_emb, k=3, nprobe=3)

for r in results:
    print(r["score"], "-", r["document"]["title"])
```

### Example 3 – Saving and Loading Indices

```python
# After building and populating the index
store.save("data/indices/python_azure_docs")

# Later or in another process
new_store = FAISSVectorStore(embedding_dim=384, index_type="IVFFlat")
new_store.load("data/indices/python_azure_docs")

# Now new_store is ready for search
```

---

## 🔍 Index Types Explained

### 1. IndexFlatIP – Exact Inner Product (Cosine)

- Uses inner product on L2-normalized embeddings (equivalent to cosine similarity)
- No training phase
- Complexity: O(N) per query (linear scan)
- **Best for:**
  - Small datasets (e.g., < 10k vectors)
  - Situations where recall must be 100%

### 2. IndexIVFFlat – Inverted File (Clustered)

- Clusters vectors into `nlist` centroids and only searches `nprobe` closest clusters
- Needs training on representative data
- Complexity: O(N/nlist × nprobe)
- **Tunable tradeoff:**
  - Higher `nprobe` → Better recall, higher latency
  - Lower `nprobe` → Lower recall, lower latency
- **Best for:**
  - Medium to large datasets (10k–1M+ vectors)
  - Balanced speed/accuracy requirements

### 3. IndexHNSWFlat – Graph-Based ANN

- Builds a Hierarchical Navigable Small World graph of vectors
- No training step; index construction is part of `add`
- **Parameters:**
  - `M` – number of neighbors in the graph
  - `efSearch` – controls search breadth (accuracy vs speed)
- Excellent latency/recall tradeoff, suited for production
- **Best for:**
  - Production systems requiring sub-millisecond latency
  - Large datasets with high accuracy requirements

---

## 📈 Benchmarking Approach

The notebook `03_performance_benchmarks.ipynb` runs:

1. **Build Time Benchmark**
   - Measure wall-clock time to construct each index type

2. **Search Latency Benchmark**
   - Generate multiple natural-language queries
   - Measure and summarize latency: mean, median, P95

3. **Approximate Memory Benchmark**
   - Inspect Python object sizes for the index and stored docs
   - Provide rough MB estimates per index type

The results are visualized with box plots and bar charts for quick comparison.

---

## 🧪 Testing

A convenience script is included:

```bash
python test_setup.py
```

It runs five checks:

1. Imports (FAISS, Sentence-Transformers, NumPy, project modules)
2. Data loading
3. Embedding generation
4. Vector store behavior
5. Full semantic search engine

You can also add unit tests under `tests/` and run:

```bash
pytest
```

---

## 🧱 Scalability & Production Notes

For real-world deployment:

- **Larger Corpora (100k+ documents)**
  - Prefer IndexIVFFlat or IndexHNSWFlat
  - Consider GPU acceleration (`faiss-gpu`)

- **Memory Optimization**
  - Use FAISS Product Quantization (IndexIVFPQ) for compression

- **Hybrid Search**
  - Combine vector similarity with keyword or metadata filters

- **MLOps**
  - Version embeddings and indices
  - Monitor latency and recall over time
  - Periodically retrain IVF centroids if data distribution changes

---

## 🧭 Roadmap

Planned or natural extensions:

- [ ] Expand demo corpus to 10k+ documents
- [ ] Add ChromaDB demo for side-by-side vector DB comparison
- [ ] Add REST API (FastAPI) for querying the vector index
- [ ] Add Dockerfile and deployment instructions
- [ ] Integrate with an LLM to form a minimal RAG system
- [ ] Add evaluation scripts for recall@k vs index configuration

---

## 🤝 Contributing

Although this is primarily a portfolio project, improvements are welcome:

1. Fork the repository
2. Create a feature branch:
   ```bash
   git checkout -b feature/my-improvement
   ```
3. Commit your changes:
   ```bash
   git commit -am "Add my improvement"
   ```
4. Push the branch:
   ```bash
   git push origin feature/my-improvement
   ```
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## 👤 Author

**Sree Tetali**

- GitHub: [@SreeTetali](https://github.com/SreeTetali)
- LinkedIn: [linkedin.com/in/sree-tetali](https://www.linkedin.com/in/sree-tetali)
- Email: sree.tetali@gmail.com

---

**Built to demonstrate practical vector database and FAISS expertise for managing AI/LLM-powered systems.**
