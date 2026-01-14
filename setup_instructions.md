# Detailed Setup Instructions

Complete guide for setting up the FAISS Vector Database Demo project.

---

## System Requirements

- **Operating System:** Windows 10/11, macOS 10.14+, or Linux
- **Python:** 3.9, 3.10, or 3.11
- **Memory:** Minimum 4GB RAM (8GB+ recommended)
- **Disk Space:** ~500MB for dependencies and models

---

## Step-by-Step Setup

### 1. Install Python

#### Windows
- Download from [python.org](https://www.python.org/downloads/)
- Or use Windows Store
- Or use Chocolatey: `choco install python --version=3.11.0`

#### macOS
```bash
# Using Homebrew
brew install python@3.11
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

---

### 2. Clone Repository

```bash
git clone https://github.com/SreeTetali/vector-database-faiss-demo.git
cd vector-database-faiss-demo
```

---

### 3. Create Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate (Windows - Command Prompt)
venv\Scripts\activate.bat

# Activate (Windows - PowerShell)
venv\Scripts\Activate.ps1

# Activate (macOS/Linux)
source venv/bin/activate
```

**Note:** You should see `(venv)` in your terminal prompt when activated.

---

### 4. Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

This will install:
- `faiss-cpu` (vector search)
- `sentence-transformers` (embeddings)
- `numpy`, `pandas` (data processing)
- `jupyter` (notebooks)
- `matplotlib`, `seaborn` (visualization)

---

### 5. Verify Installation

```bash
# Test FAISS
python -c "import faiss; print(f'FAISS: {faiss.__version__}')"

# Test Sentence-Transformers
python -c "from sentence_transformers import SentenceTransformer; print('Sentence-Transformers: OK')"

# Test all imports
python -c "from src.data_loader import DocumentationLoader; print('All imports: OK')"
```

**Expected output:**
```text
FAISS: 1.9.0
Sentence-Transformers: OK
All imports: OK
```

---

### 6. Run Complete Setup Test

```bash
# Run comprehensive test suite
python test_setup.py
```

This validates:
- All package imports
- Data loading functionality
- Embedding generation
- FAISS vector store operations
- Complete search engine pipeline

You should see: **🎉 ALL TESTS PASSED! 🎉**

---

### 7. Download Embedding Model (Automatic on First Run)

The embedding model (~80MB) will download automatically on first use:

```bash
python src/embedding_engine.py
```

This downloads `all-MiniLM-L6-v2` from HuggingFace and caches it locally in `~/.cache/huggingface/`.

---

### 8. Create Sample Data

```bash
python src/data_loader.py
```

This creates `data/processed/sample_docs.json` with 15 Python/Azure documentation samples.

---

### 9. Launch Jupyter Notebooks

```bash
jupyter notebook
```

Browser should open automatically at `http://localhost:8888`. Navigate to `notebooks/` and run in sequence:

1. `01_data_preparation.ipynb` - Load and explore documentation
2. `02_faiss_indexing.ipynb` - Build and compare FAISS indices
3. `03_performance_benchmarks.ipynb` - Performance analysis and metrics

---

## Troubleshooting

### Issue: `faiss-cpu` installation fails

**Solution 1:** Update pip and retry
```bash
python -m pip install --upgrade pip setuptools wheel
pip install faiss-cpu
```

**Solution 2:** Try conda (if you have Anaconda/Miniconda)
```bash
conda install -c pytorch faiss-cpu
```

**Solution 3:** Check Python version
```bash
python --version  # Should be 3.9, 3.10, or 3.11
```

---

### Issue: Jupyter kernel not found

**Solution:** Install IPython kernel for your venv
```bash
python -m ipykernel install --user --name=venv --display-name="Python (venv)"
```

Then in Jupyter: `Kernel` → `Change Kernel` → Select `Python (venv)`

---

### Issue: Import errors in notebooks

**Solution:** Notebooks include path fix, but verify:
```python
import sys
sys.path.append('..')  # Add parent directory to path
```

---

### Issue: Sentence-Transformers download slow/fails

**Solution 1:** Manual download with progress
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("Model loaded successfully!")
```

**Solution 2:** Pre-download using HuggingFace CLI
```bash
pip install huggingface-hub
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2
```

---

### Issue: Permission denied on Windows

**Solution:** Change PowerShell execution policy
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Or run PowerShell/Command Prompt as Administrator.

---

### Issue: `ModuleNotFoundError` after installation

**Solution:** Ensure virtual environment is activated
```bash
# Check if (venv) appears in prompt
# If not, activate:
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```

---

## GPU Support (Optional)

For faster embedding generation and indexing on large datasets (requires NVIDIA GPU with CUDA):

### Install FAISS-GPU

```bash
# Requires CUDA 11.x or 12.x
pip uninstall faiss-cpu
pip install faiss-gpu
```

### Install PyTorch with CUDA

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Note:** GPU acceleration provides significant speedup for large corpora (10k+ documents).

---

## Development Setup (Optional)

For contributing or extending the project:

```bash
# Install development dependencies
pip install pytest pytest-cov black flake8 mypy isort

# Install project in editable mode
pip install -e .

# Run tests
pytest tests/ -v

# Format code
black src/

# Sort imports
isort src/

# Type checking
mypy src/

# Linting
flake8 src/
```

---

## VS Code Setup (Recommended)

### Required Extensions

Install these VS Code extensions:
- **Python** (Microsoft)
- **Jupyter** (Microsoft)
- **Pylance** (Microsoft)

### Workspace Settings

Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./venv/Scripts/python.exe",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "jupyter.notebookFileRoot": "${workspaceFolder}"
}
```

For macOS/Linux, change interpreter path to:
```json
"python.defaultInterpreterPath": "./venv/bin/python"
```

---

## Quick Start Commands (Copy/Paste)

### Windows
```bash
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python test_setup.py
jupyter notebook
```

### macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python test_setup.py
jupyter notebook
```

---

## Next Steps

After successful setup:

1. ✅ **Run test suite** → `python test_setup.py`
2. ✅ **Explore notebooks** → Start with `01_data_preparation.ipynb`
3. ✅ **Experiment with queries** → Try different search terms
4. ✅ **Compare index types** → Test FlatIP, IVFFlat, HNSW
5. ✅ **Review benchmarks** → Analyze performance tradeoffs
6. ✅ **Extend with your data** → Replace sample docs with your corpus

---

## Getting Help

- **Issues:** [Open GitHub Issue](https://github.com/SreeTetali/vector-database-faiss-demo/issues)
- **FAISS Documentation:** [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- **Sentence-Transformers:** [Official Docs](https://www.sbert.net/)

---

## Common Setup Verification

Run these commands to verify everything works:

```bash
# 1. Check Python version
python --version

# 2. Verify virtual environment
which python  # macOS/Linux
where python  # Windows

# 3. Test imports
python -c "import faiss, sentence_transformers, numpy, pandas; print('✓ All imports OK')"

# 4. Run full test suite
python test_setup.py

# 5. Launch Jupyter
jupyter notebook
```

---

**Setup complete! Happy coding! 🚀**
