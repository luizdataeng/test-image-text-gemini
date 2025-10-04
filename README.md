## Multimodal RAG (Vertex AI) – project_test_multimodal_RAG.ipynb

This repository contains a hands-on notebook for building a DIY multimodal Retrieval Augmented Generation (RAG) pipeline using Google Cloud Vertex AI. The notebook demonstrates how to extract text and images from PDFs, generate text and multimodal embeddings, perform similarity search, and answer questions grounded in both text and image context.

> The notebook adapts Google Cloud sample materials and uses the Vertex AI Embeddings and Gemini APIs. Running it against Vertex AI incurs costs.

### What you will build
- **Document search over PDFs**: search across text chunks and extracted images
- **Embeddings**: `text-embedding-005` and `multimodalembedding@001`
- **Similarity search**: cosine similarity for text and images
- **Multimodal Q&A**: Gemini with retrieved text and image context

### Repository contents
- `project_test_multimodal_RAG.ipynb` – Primary tutorial notebook
- `updated_building_DIY_multimodal_qa_system_with_mRAG.ipynb` – Related reference
- `context_caching.ipynb`, `controlled_generation.ipynb`, `Async_Concurrency&Batching_Feature&Vector_Engineering.ipynb` – Additional experiments
- `pyproject.toml` – Dependencies for local execution

### Requirements
- **Python**: ≥ 3.13 (matches `pyproject.toml`)
- **Google Cloud project** with billing enabled
- **Vertex AI API** enabled
- **Credentials** with permission to use Vertex AI (see Authentication below)

### Install locally
You can run the notebook locally with Jupyter or VS Code.

1) Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

2) Install the minimal dependencies used by the notebook
```bash
pip install google-cloud-aiplatform pymupdf rich pillow pandas numpy scikit-learn jupyter ipykernel
```

3) Launch Jupyter
```bash
python -m ipykernel install --user --name=multimodal-rag
jupyter notebook
```
Open `project_test_multimodal_RAG.ipynb` and select the `multimodal-rag` kernel.

### Google Cloud setup
1) Enable required APIs
```bash
gcloud services enable aiplatform.googleapis.com
```

2) Set your project and region
```bash
export PROJECT_ID="YOUR_GCP_PROJECT_ID"
export LOCATION="us-central1"  # or your preferred Vertex AI region
gcloud config set project "$PROJECT_ID"
```

3) Authenticate
- For a user account (good for local dev):
```bash
gcloud auth application-default login
```
- Or use a service account key and point `GOOGLE_APPLICATION_CREDENTIALS` to it:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

The notebook will call:
```python
import vertexai
vertexai.init(project=PROJECT_ID, location=LOCATION)
```
Make sure `PROJECT_ID` and `LOCATION` are set accordingly (either via environment or directly in the notebook cell).

### Data and assets
The notebook can download example PDFs and images from public GCS links. It also supports processing local PDFs.

- **Default folder** for PDFs: `map/`
- **Generated** images and intermediates: `images/`

If running locally with your own data:
- Place one or more PDFs in a folder (e.g., `map/`)
- Update the `pdf_folder_path` variable in the notebook to point to your folder

### Running the notebook
1) Open `project_test_multimodal_RAG.ipynb`
2) Run the install cell (or skip if you installed packages via pip already)
3) Verify authentication (see Google Cloud setup)
4) Set `PROJECT_ID` and `LOCATION`
5) Choose between:
   - Loading pre-computed metadata (fast), or
   - Extracting metadata and generating embeddings from PDFs (takes minutes for larger docs)
6) Explore similarity search over text and images
7) Run the multimodal RAG examples that use Gemini with retrieved context

### Costs
This workflow uses billable Vertex AI endpoints. See Vertex AI pricing and monitor usage in the Cloud Console. Consider using smaller test files while iterating.

### Troubleshooting
- **Kernel restart snippet error (NameError: IPython is not defined)**:
  - If the notebook includes a restart cell, ensure `import IPython` is on its own line, or simply restart the kernel via the UI after installing packages.
- **Permission or quota errors from Vertex AI**:
  - Confirm the Vertex AI API is enabled, credentials are set, and the region matches your resources.
- **Authentication issues locally**:
  - Prefer `gcloud auth application-default login` or set `GOOGLE_APPLICATION_CREDENTIALS` to a service account with Vertex AI permissions.
- **Python version / dependency conflicts**:
  - The project declares Python ≥ 3.13 in `pyproject.toml`. If your environment differs, align versions or install only the notebook’s minimal dependencies as shown above.

### Notes
- The notebook contains sections sourced from Google Cloud samples and may show deprecation warnings for some SDK features over time; follow linked deprecation notes in outputs if encountered.
- Running end-to-end extraction on multiple large PDFs can take several minutes; start with pre-computed metadata if available.
