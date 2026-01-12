# Detailed Setup Instructions

Complete guide for setting up the FAISS Vector Database Demo project.

## System Requirements

- **Operating System:** Windows 10/11, macOS 10.14+, or Linux
- **Python:** 3.9, 3.10, or 3.11
- **Memory:** Minimum 4GB RAM (8GB+ recommended)
- **Disk Space:** ~500MB for dependencies and models

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

Linux (Ubuntu/Debian)

bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip

2. Clone Repository

bash
git clone https://github.com/SreeTetali/vector-database-fiass-demo.git
cd vector-database-fiass-demo

3. Create Virtual Environment

bash
# Create venv
python -m venv venv

# Activate (Windows - Command Prompt)
venv\Scripts\activate.bat

# Activate (Windows - PowerShell)
venv\Scripts\Activate.ps1

# Activate (macOS/Linux)
source venv/bin/activate

Note: You should see (venv) in your terminal prompt when activated.
4. Install Dependencies

bash
# Upgrade pip
python -m pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# This will install:
# - faiss-cpu (vector search)
# - sentence-transformers (embeddings)
# - numpy, pandas (data processing)
# - jupyter (notebooks)
# - matplotlib, seaborn (visualization)

5. Verify Installation

bash
# Test FAISS
python -c "import faiss; print(f'FAISS: {faiss.__version__}')"

# Test Sentence-Transformers
python -c "from sentence_transformers import SentenceTransformer; print('Sentence-Transformers: OK')"

# Test all imports
python -c "from src.data_loader import DocumentationLoader; print('All imports: OK')"

Expected output:

text
FAISS: 1.9.0
Sentence-Transformers: OK
All imports: OK

6. Download Embedding Model (First Run)

The embedding model (~80MB) will download automatically on first use:

bash
python src/embedding_engine.py

This downloads all-MiniLM-L6-v2 from HuggingFace and caches it locally.
7. Create Sample Data

bash
python src/data_loader.py

This creates data/processed/sample_docs.json with 15 Python/Azure documentation samples.
8. Run Quick Test

bash
# Test complete search engine
python src/search_engine.py

Expected output showing search results for "How do I use Azure serverless functions?"
9. Launch Jupyter Notebooks

bash
jupyter notebook

Browser should open automatically. Navigate to notebooks/ and run:

    01_data_preparation.ipynb

    02_faiss_indexing.ipynb

    03_performance_benchmarks.ipynb

Troubleshooting
Issue: faiss-cpu installation fails

Solution 1: Update pip

bash
python -m pip install --upgrade pip setuptools wheel
pip install faiss-cpu

Solution 2: Try conda

bash
conda install -c pytorch faiss-cpu

Issue: Jupyter kernel not found

Solution:

bash
python -m ipykernel install --user --name=venv

Issue: Import errors in notebooks

Solution: Add parent directory to path (already in notebooks):

python
import sys
sys.path.append('..')

Issue: Sentence-Transformers download slow/fails

Solution: Download model manually:

python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

Issue: Permission denied on Windows

Solution: Run as Administrator or change execution policy:

powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

GPU Support (Optional)

For faster embedding generation and indexing on large datasets:
Install FAISS-GPU

bash
# Requires CUDA 11.x or 12.x
pip uninstall faiss-cpu
pip install faiss-gpu

Install PyTorch with CUDA

bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Development Setup (Optional)

For contributing or extending the project:

bash
# Install development dependencies
pip install pytest pytest-cov black flake8 mypy

# Install in editable mode
pip install -e .

# Run tests
pytest tests/

# Format code
black src/

# Type checking
mypy src/

VS Code Setup (Recommended)
Extensions

    Python (Microsoft)

    Jupyter (Microsoft)

    Pylance (Microsoft)

Settings (.vscode/settings.json)

json
{
    "python.defaultInterpreterPath": "./venv/Scripts/python.exe",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true
}

Next Steps

    ✅ Complete setup

    ✅ Run notebooks in sequence

    ✅ Experiment with different queries

    ✅ Try different index types

    ✅ Review performance benchmarks

    ➡️ Extend with your own data!

Getting Help

    Issues: Open GitHub issue

    Questions: Check FAISS documentation

    Community: FAISS Google Group

Setup complete! Happy coding! 🚀