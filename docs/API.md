# API Reference

This document lists all public functions, classes, and their usage examples for this repository.

- Project: test-image-text-gemini
- Language: Python 3.13+

## Modules
- `multimodal_qa_with_rag_utils`
- `main`

---

## Module: `multimodal_qa_with_rag_utils`

### set_global_variable(variable_name: str, value: any) -> None
Sets the value of a global variable by name.

- **Args**:
  - `variable_name`: Name of the global variable to set
  - `value`: Value to assign
- **Returns**: `None`

Example:
```python
set_global_variable("TEXT_EMBEDDINGS", [0.1, 0.2, 0.3])
```

---

### get_text_embedding_from_text_embedding_model(text: str, return_array: Optional[bool] = False) -> list
Generates a 768-d text embedding using a configured text embedding model.

- **Args**:
  - `text`: Input text
  - `return_array`: Return `numpy.ndarray` if True, else list
- **Returns**: `list | numpy.ndarray`
- **Notes**: Requires a global `text_embedding_model` to be initialized externally.

Example:
```python
vec = get_text_embedding_from_text_embedding_model("hello world")
```

---

### get_image_embedding_from_multimodal_embedding_model(image_uri: str, embedding_size: int = 512, text: Optional[str] = None, return_array: Optional[bool] = False) -> list
Extracts an image embedding from multimodal embedding model; optionally uses contextual text.

- **Args**:
  - `image_uri`: Local path or URI of image
  - `embedding_size`: One of {128, 256, 512, 1408}
  - `text`: Optional contextual text
  - `return_array`: Return `numpy.ndarray` if True
- **Returns**: `list | numpy.ndarray`
- **Notes**: Requires a global `multimodal_embedding_model` to be initialized externally.

Example:
```python
img_vec = get_image_embedding_from_multimodal_embedding_model("images/B2_room.jpeg", embedding_size=512)
```

---

### load_image_bytes(image_path) -> bytes
Loads image bytes from URL or local path.

- **Args**:
  - `image_path`: URL or local path
- **Returns**: `bytes`
- **Raises**: `ValueError` when `image_path` missing

Example:
```python
img_bytes = load_image_bytes("images/B2_room.jpeg")
```

---

### get_pdf_doc_object(pdf_path: str) -> tuple[fitz.Document, int]
Opens a PDF and returns document object and page count.

- **Args**:
  - `pdf_path`: Path to PDF
- **Returns**: `(doc, num_pages)`
- **Raises**: `FileNotFoundError` if invalid path

Example:
```python
doc, num_pages = get_pdf_doc_object("data/gemma_technical_paper.pdf")
```

---

### class Color
ANSI color codes used for formatted console printing.

- **Attributes**: `PURPLE, CYAN, DARKCYAN, BLUE, GREEN, YELLOW, RED, BOLD, UNDERLINE, END`

Example:
```python
c = Color()
print(c.RED + "Error" + c.END)
```

---

### get_text_overlapping_chunk(text: str, character_limit: int = 1000, overlap: int = 100) -> dict
Chunks text with overlap.

- **Args**:
  - `text`: Input text
  - `character_limit`: Max chars per chunk
  - `overlap`: Overlapping chars
- **Returns**: `{chunk_number: str}`
- **Raises**: `ValueError` if `overlap > character_limit`

Example:
```python
chunks = get_text_overlapping_chunk(long_text, 1000, 100)
```

---

### get_page_text_embedding(text_data: Union[dict, str]) -> dict
Embeds either full text or each chunk.

- **Args**:
  - `text_data`: Dict of chunks or raw string
- **Returns**: `{chunk_number|"text_embedding": embedding}`

Example:
```python
embeds = get_page_text_embedding({1: "hello", 2: "world"})
```

---

### get_chunk_text_metadata(page: fitz.Page, character_limit: int = 1000, overlap: int = 100, embedding_size: int = 128) -> tuple[str, dict, dict, dict]
Extracts text from a PDF page, chunks it, and computes embeddings.

- **Args**:
  - `page`: `fitz.Page`
  - `character_limit`, `overlap`, `embedding_size`
- **Returns**: `(text, page_text_embeddings, chunked_text, chunk_embeddings)`

Example:
```python
text, page_emb, chunks, chunk_emb = get_chunk_text_metadata(page)
```

---

### get_image_for_gemini(doc: fitz.Document, image: tuple, image_no: int, image_save_dir: str, file_name: str, page_num: int) -> Tuple[Image, str]
Extracts, converts to JPEG, saves, and returns as Gemini Image plus path.

- **Args**: `doc, image, image_no, image_save_dir, file_name, page_num`
- **Returns**: `(vertexai.generative_models.Image, image_path)`

Example:
```python
img_obj, path = get_image_for_gemini(doc, image, 0, "images/out", "paper", 1)
```

---

### get_gemini_response(generative_multimodal_model, model_input: List[str], stream: bool = True, generation_config: Optional[GenerationConfig] = ..., safety_settings: Optional[dict] = ..., print_exception: bool = False) -> str
Streams a response from Gemini given mixed text/image inputs.

- **Args**:
  - `generative_multimodal_model`: A configured Vertex AI Gemini model
  - `model_input`: List including instruction, strings, and `Image` objects
  - `stream`: Stream responses if True
  - `generation_config`, `safety_settings`, `print_exception`
- **Returns**: `str`

Example:
```python
resp = get_gemini_response(model, ["Hello", image_obj], stream=True)
```

---

### get_text_metadata_df(filename: str, text_metadata: Dict[Union[int, str], Dict]) -> pd.DataFrame
Builds a DataFrame from text metadata.

- **Args**: `filename`, `text_metadata`
- **Returns**: `pd.DataFrame`

Example:
```python
df_text = get_text_metadata_df("paper.pdf", text_meta)
```

---

### get_image_metadata_df(filename: str, image_metadata: Dict[Union[int, str], Dict]) -> pd.DataFrame
Builds a DataFrame from image metadata.

- **Args**: `filename`, `image_metadata`
- **Returns**: `pd.DataFrame`

Example:
```python
df_img = get_image_metadata_df("paper.pdf", image_meta)
```

---

### get_document_metadata(generative_multimodal_model, pdf_folder_path: str, image_save_dir: str, image_description_prompt: str, embedding_size: int = 128, ..., add_sleep_after_page: bool = False, sleep_time_after_page: int = 2, add_sleep_after_document: bool = False, sleep_time_after_document: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]
Processes PDFs in a folder to extract text and images; computes metadata dataframes.

- **Args**: `generative_multimodal_model, pdf_folder_path, image_save_dir, image_description_prompt, embedding_size, generation_config, safety_settings, add_sleep_after_page, sleep_time_after_page, add_sleep_after_document, sleep_time_after_document`
- **Returns**: `(text_metadata_df, image_metadata_df)`

Example:
```python
text_df, image_df = get_document_metadata(model, "data", "images/out", "Describe this image", 128)
```

---

### get_user_query_text_embeddings(user_query: str) -> np.ndarray
Returns text embedding for a user query.

- **Args**: `user_query`
- **Returns**: `np.ndarray`

Example:
```python
q_vec = get_user_query_text_embeddings("What is the contribution?")
```

---

### get_user_query_image_embeddings(image_query_path: str, embedding_size: int) -> np.ndarray
Returns image embedding for a user query image.

- **Args**: `image_query_path`, `embedding_size`
- **Returns**: `np.ndarray`

Example:
```python
q_img_vec = get_user_query_image_embeddings("images/B2_room.jpeg", 512)
```

---

### get_cosine_score(dataframe: pd.DataFrame, column_name: str, input_text_embd: np.ndarray) -> float
Computes dot-product similarity between an embedding column and the input vector.

- **Args**: `dataframe`, `column_name`, `input_text_embd`
- **Returns**: `float`

Example:
```python
score = get_cosine_score(df.iloc[0], "text_embedding_chunk", q_vec)
```

---

### print_text_to_image_citation(final_images: Dict[int, Dict[str, Any]], print_top: bool = True) -> None
Prints formatted citations for matched images.

- **Args**: `final_images`, `print_top`
- **Returns**: `None`

Example:
```python
print_text_to_image_citation(results_images)
```

---

### print_text_to_text_citation(final_text: Dict[int, Dict[str, Any]], print_top: bool = True, chunk_text: bool = True) -> None
Prints formatted citations for matched text.

- **Args**: `final_text`, `print_top`, `chunk_text`
- **Returns**: `None`

Example:
```python
print_text_to_text_citation(results_text)
```

---

### get_similar_image_from_query(text_metadata_df: pd.DataFrame, image_metadata_df: pd.DataFrame, query: str = "", image_query_path: str = "", column_name: str = "", image_emb: bool = True, top_n: int = 3, embedding_size: int = 128) -> Dict[int, Dict[str, Any]]
Finds top-N similar images using either image or text similarity.

- **Args**: `text_metadata_df, image_metadata_df, query, image_query_path, column_name, image_emb, top_n, embedding_size`
- **Returns**: `Dict[int, Dict[str, Any]]`

Example:
```python
images = get_similar_image_from_query(text_df, image_df, query="diagram of", column_name="mm_embedding_from_img_only", image_emb=True, top_n=3, embedding_size=128)
```

---

### get_similar_text_from_query(query: str, text_metadata_df: pd.DataFrame, column_name: str = "", top_n: int = 3, chunk_text: bool = True, print_citation: bool = False) -> Dict[int, Dict[str, Any]]
Finds top-N similar text chunks from text metadata.

- **Args**: `query, text_metadata_df, column_name, top_n, chunk_text, print_citation`
- **Returns**: `Dict[int, Dict[str, Any]]`

Example:
```python
texts = get_similar_text_from_query("introduction", text_df, column_name="text_embedding_chunk", top_n=5)
```

---

### display_images(images: Iterable[Union[str, PIL.Image.Image]], resize_ratio: float = 0.5) -> None
Displays paths or PIL images with resizing.

- **Args**: `images`, `resize_ratio`
- **Returns**: `None`

Example:
```python
display_images(["images/B2_room.jpeg"], 0.5)
```

---

### get_answer_from_qa_system(query: str, text_metadata_df, image_metadata_df, top_n_text: int = 10, top_n_image: int = 5, instruction: Optional[str] = None, model=None, generation_config: Optional[GenerationConfig] = ..., safety_settings: Optional[dict] = ...) -> Union[str, None]
Combines top text and images into a Gemini prompt to generate an answer.

- **Args**: `query, text_metadata_df, image_metadata_df, top_n_text, top_n_image, instruction, model, generation_config, safety_settings`
- **Returns**: `(response, matching_results_chunks_data, matching_results_image_fromdescription_data)`

Example:
```python
response, text_hits, image_hits = get_answer_from_qa_system(
    query="What is the dataset size?",
    text_metadata_df=text_df,
    image_metadata_df=image_df,
    model=model,
)
```

---

## Module: `main`

### main() -> None
Entry point that prints a hello message. Useful as a sanity check.

Example:
```python
from main import main
main()
```

---

## Quickstart

- **Prepare Vertex AI models**: Initialize `text_embedding_model`, `multimodal_embedding_model`, and a `generative_multimodal_model` before calling API functions that depend on them.
- **Process PDFs**: Call `get_document_metadata(model, pdf_folder_path, image_save_dir, prompt, embedding_size)` to build metadata frames.
- **Search and cite**: Use `get_similar_text_from_query` and `get_similar_image_from_query`, then `print_text_to_*_citation`.
- **Ask a question**: Use `get_answer_from_qa_system` with the metadata and model.

## Requirements
See `pyproject.toml` for dependencies. Ensure environment access to Vertex AI and Google Cloud credentials.
