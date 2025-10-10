# PDN Website Crawler

This project aims to crawl data across the PDN website. In the initial phase, it focuses on the homepage SmartSlider at `https://pdn.gov.my/v2/index.php`: downloading slider images, analyzing them with Groq's multimodal model `meta-llama/llama-4-scout-17b-16e-instruct`, and writing RAG-ready artifacts that you can ingest into a vector database. More sources and sections of the site will be added in subsequent phases.

## Scope

- **Homepage**: Crawl the home page slider, download images, analyze with Groq vision, and generate RAG-ready outputs.

## Features

- Async crawling with [Crawl4AI](https://pypi.org/project/Crawl4AI/)
- SmartSlider DOM parsing with robust selectors and fallbacks
- Image downloading with browser-like headers
- Groq multimodal analysis for compact, structured JSON
- RAG outputs in JSONL and text chunks
- Daily-rotating logs to `logs/app.log`

## Project Structure

```
.
├── main.py                  # Entry point: runs the PDN crawler (slider phase)
├── config.py                # BASE_URL, GROQ_MODEL, output directories
├── utils
│   ├── logger.py            # Daily-rotating logger (console + file)
│   ├── scraper_utils.py     # Browser config for Crawl4AI
│   └── homepage.py        # Slider parse → download → Groq analyze → RAG outputs
├── requirements.txt         # Python dependencies
├── .gitignore               # Excludes .env, logs, data, rag, etc.
└── README.MD                # This file
```

## How the Homepage Crawler Works

The homepage SmartSlider pipeline is implemented in `utils/homepage.py` and orchestrated by `main.py`.

- **`fetch_index_html()`**: Loads the PDN homepage using Crawl4AI with a Chromium browser. It returns the raw HTML to preserve SmartSlider markup.
- **`parse_slider_images()`**: Parses SmartSlider DOM to extract candidate image entries with fields: `src`, `abs_url`, `alt`, `title`, and optional `public_id`. Uses robust selectors with fallbacks.
- **`download_image()`**: Downloads each image to `data/images/` with browser-like headers. Skips non-http sources. Saves using `md5(url)+ext` to avoid filename collisions.
- **`groq_analyze_image()`**: Sends each image to Groq multimodal model (`GROQ_MODEL`) and requests a compact JSON with fields such as `title`, `summary`, `language`, `ocr_text`, `labels`, `entities`, `date`, `call_to_action`, `urls`. Falls back to a `{"raw": "..."}` payload if the response isn’t valid JSON.
- **`process_homepage()`**: Ties it together: fetch → parse → download → analyze → write outputs. Idempotent by default: if a chunk exists for an image URL, it skips re-analysis and JSONL append (unless forced). For each image it writes one JSON line to `data/rag/slider_images.jsonl`, a text chunk to `data/rag/chunks/<id>.txt`, and a canonical record JSON to `data/rag/records/<id>.json`. Adds `image_sha256` for content-dedup signals.

`main.py` initializes the crawler via `crawl_homepage()` and calls the homepage processing function with your configured `GROQ_API_KEY` and `GROQ_MODEL`.

### Output Artifacts & Schema

Upon completion you will have:

- **Images**: `data/images/`
- **JSONL**: `data/rag/slider_images.jsonl` (one JSON object per line)
- **Chunks**: `data/rag/chunks/<id>.txt` (plain text used for embeddings)
- **Records**: `data/rag/records/<id>.json` (canonical per image record; latest analysis overwrites)

Each JSONL record has this shape:

```json
{
  "id": "<md5-of-source-url>",
  "source_url": "https://.../banner.jpg",
  "local_path": "data/images/banner.jpg",
  "alt": "...",
  "title": "...",
  "public_id": "<optional-smartslider-public-id>",
  "llm_model": "meta-llama/llama-4-scout-17b-16e-instruct",
  "image_sha256": "<sha256-of-downloaded-image-bytes>",
  "analysis": {
    "title": "...",
    "summary": "Concise RAG-friendly synopsis (<= ~80 words)",
    "language": "ms|en|...",
    "ocr_text": "...",
    "labels": ["health", "announcement"],
    "entities": ["PDN", "Kementerian Kesihatan"],
    "date": "2025-01-31",
    "call_to_action": "Register now",
    "urls": ["https://pdn.gov.my/..."]
    /* or { "raw": "<model text>" } if parsing failed */
  },
  "chunk": "<preferred text used for embedding; summary → ocr_text → fallback>"
}
```

`chunk` is selected in this priority: `analysis.summary` → `analysis.ocr_text` → truncated stringified `analysis`. This guarantees every record has a non-empty chunk for embedding.

## How This Is Relevant for an LLM Chatbot (RAG)

You can ingest the outputs into any vector database (FAISS, Chroma, PgVector, etc.) to ground a chatbot with timely homepage information.

- **What to embed**: Use the `.txt` files in `data/rag/chunks/` or the `chunk` field from JSONL.
- **Metadata to store**: At minimum keep `id`, `source_url`, `alt`, `title`, `public_id`, `llm_model`. Optionally include full `analysis` for filtering and citations.
- **Retrieval strategy**:
  - Query the vector DB with the user’s question.
  - Retrieve top-k chunks and pass both chunk text and metadata to the LLM.
  - Encourage the model to cite `source_url` or `public_id` in its answer.
- **Multi-lingual handling**: Use `analysis.language` to filter results or to bias answer language.
- **Freshness**: Re-run the crawler on a schedule; re-index changed records based on `id` (md5 of image URL) or by comparing JSONL contents.

Example ingestion approach:

1. Read `slider_images.jsonl` line by line, parse JSON, pick `record["chunk"]` for embeddings.
2. Store `record["source_url"]`, `record["title"]`, and `record["analysis"]` as metadata.
3. Generate embeddings with your preferred model and upsert into your vector DB.

During answering, show citations and, when useful, render the `source_url` for user verification.

## Roadmap

- **Documents**: images/PDFs with OCR and summarization for richer RAG
- **Indexing**: script to ingest chunks + metadata into a vector DB
- **Scheduling**: periodic recrawls and change detection

## Installation

1. **Create and activate a virtual environment**

   ```bash
   uv venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**

   ```bash
   uv pip install -r requirements.txt
   ```

3. **Set up your environment variables**

   Create a `.env` file in the root directory with content similar to:

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   # Optional: set to 1/true to reprocess even if chunks already exist
   FORCE_REPROCESS=0
   ```

## Usage

### Run the crawler

It will:

- Load PDN homepage
- Extract SmartSlider image URLs
- Download images into `data/images/`
- Analyze each image with Groq vision
- Save records to `data/rag/slider_images.jsonl` and chunks to `data/rag/chunks/`

```bash
uv run main.py
```

Outputs will be printed with total images processed.

Rerunning the script without changes is idempotent: previously processed images are skipped. To force redownload and reanalysis:

```bash
FORCE_REPROCESS=1 uv run main.py
```

## Configuration

The `config.py` file contains key constants used throughout the project:

- **BASE_URL**: The PDN homepage to crawl.
- **GROQ_MODEL**: Groq multimodal model for image analysis (defaults to `meta-llama/llama-4-scout-17b-16e-instruct`).
- **IMAGES_DIR**, **RAG_DIR**: Output directories for images and RAG outputs.

You can modify these values as needed.

## Additional Notes

- **Logging:** Logs are written to console and `logs/app.log` with daily rotation (14 backups).
- **Improvements:** The code is modular for maintainability and extension.
- **Dependencies:** Ensure that the package versions specified in `requirements.txt` are installed.
- **RAG ingestion:** The `.txt` chunks can be ingested into your preferred vector DB (e.g., FAISS, Chroma, PgVector). Use the JSONL file for richer metadata.
