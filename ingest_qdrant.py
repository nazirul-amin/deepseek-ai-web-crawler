import os
import json
import argparse
from typing import List, Dict, Iterable

import requests
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from utils.logger import get_logger
from dotenv import load_dotenv

logger = get_logger("ingest_qdrant")
load_dotenv()

def iter_records(rag_dir: str) -> Iterable[Dict]:
    """Iterate canonical records under rag_dir/records (JSON files)."""
    records_dir = os.path.join(rag_dir, "records")
    yield from read_records_dir(records_dir)


def read_records_dir(dir_path: str) -> Iterable[Dict]:
    if not os.path.isdir(dir_path):
        logger.error(f"Records directory not found: {dir_path}")
        return []
    for name in os.listdir(dir_path):
        if not name.endswith(".json"):
            continue
        p = os.path.join(dir_path, name)
        try:
            with open(p, "r", encoding="utf-8") as f:
                yield json.load(f)
        except Exception as e:
            logger.error(f"Failed to parse {p}: {e}")


essential_meta = [
    "id",
    "source_url",
    "title",
    "alt",
    "public_id",
    "llm_model",
    "image_sha256",
]

def build_payload(rec: Dict) -> Dict:
    payload = {k: rec.get(k) for k in essential_meta if k in rec}
    analysis = rec.get("analysis") or {}
    if isinstance(analysis, dict):
        payload["analysis"] = analysis
    return payload


def embed_texts_ollama(ollama_url: str, ollama_model: str, texts: List[str]):
    """Embed texts using an Ollama server via /api/embed. Returns (vectors, dim)."""
    if ollama_url.endswith("/"):
        base = ollama_url[:-1]
    else:
        base = ollama_url

    vectors: List[List[float]] = []
    dim: int = 0
    for t in texts:
        # Use /api/embed with {model, input} only
        try:
            r = requests.post(
                f"{base}/api/embed",
                json={"model": ollama_model, "input": t},
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to reach Ollama at {base}: {e}") from e

        # Accept common shapes
        embedding = None
        if isinstance(data, dict):
            if isinstance(data.get("embedding"), list):
                embedding = data["embedding"]
            elif isinstance(data.get("data"), list) and data["data"]:
                maybe = data["data"][0]
                if isinstance(maybe, dict) and isinstance(maybe.get("embedding"), list):
                    embedding = maybe["embedding"]
            elif isinstance(data.get("embeddings"), list) and data["embeddings"]:
                first = data["embeddings"][0]
                if isinstance(first, list):
                    embedding = first
        if not isinstance(embedding, list):
            raise RuntimeError(f"Unexpected Ollama /api/embed response: {data}")
        if not dim:
            dim = len(embedding)
        vectors.append(embedding)

    return vectors, dim


def ensure_collection(client: QdrantClient, name: str, vector_size: int) -> None:
    try:
        info = client.get_collection(name)
        # If collection exists but dimensions mismatch, warn and exit
        existing = info.config.params.vectors
        existing_size = None
        if hasattr(existing, "size"):
            existing_size = existing.size
        elif isinstance(existing, dict):
            # Named vectors (not used here)
            # Pick any size to compare, but encourage manual recreate if mismatch
            existing_size = list(existing.values())[0].size  # type: ignore
        if existing_size and existing_size != vector_size:
            logger.error(
                f"Collection '{name}' exists with vector size {existing_size},"
                f" but model dimension is {vector_size}. Please recreate collection."
            )
            raise SystemExit(1)
    except Exception:
        logger.info(f"Creating collection '{name}' with dim={vector_size}")
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def batched(iterable: List[int], n: int) -> Iterable[List[int]]:
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def main():
    parser = argparse.ArgumentParser(description="Ingest PDN slider chunks into Qdrant")
    parser.add_argument(
        "--rag-dir",
        default=os.environ.get("RAG_DIR", "data/rag"),
        help="Path to rag output directory",
    )
    parser.add_argument(
        "--collection",
        default=os.environ.get("QDRANT_COLLECTION", "pdn_chatbot"),
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant URL",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        help="Base URL for Ollama server (when using ollama backend)",
    )
    parser.add_argument(
        "--ollama-model",
        default=os.environ.get("OLLAMA_MODEL", "nomic-embed-text"),
        help="Ollama embedding model name (when using ollama backend)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for embedding and upsert",
    )
    args = parser.parse_args()

    client = QdrantClient(url=args.qdrant_url, api_key=os.getenv("QDRANT_API_KEY") or None)

    # Load data
    records: List[Dict] = []
    seen_ids = set()
    for rec in iter_records(args.rag_dir):
        rid = str(rec.get("id") or rec.get("source_url"))
        if not rid or rid in seen_ids:
            continue
        text = rec.get("chunk") or ""
        if not text.strip():
            continue
        records.append(rec)
        seen_ids.add(rid)

    if not records:
        logger.warning("No records to ingest.")
        return

    logger.info(
        f"Embedding and upserting {len(records)} records into '{args.collection}' using Ollama embeddings"
    )

    texts = [r.get("chunk", "") for r in records]
    ids = [str(r.get("id")) for r in records]
    payloads = [build_payload(r) for r in records]

    # Probe dimension with the first text (or a dummy string if empty)
    probe_text = texts[0] or "dim probe"
    probe_vecs, dim = embed_texts_ollama(args.ollama_url, args.ollama_model, [probe_text])
    if not probe_vecs or not isinstance(probe_vecs[0], list):
        raise RuntimeError("Failed to obtain embedding from Ollama for dimension probing")
    ensure_collection(client, args.collection, dim)

    for batch_ids in batched(list(range(len(texts))), args.batch_size):
        batch_texts = [texts[i] for i in batch_ids]
        batch_vecs, _ = embed_texts_ollama(args.ollama_url, args.ollama_model, batch_texts)
        points = [
            PointStruct(
                id=str(ids[i]),  # upsert by stable id
                vector=batch_vecs[j],
                payload=payloads[i],
            )
            for j, i in enumerate(batch_ids)
        ]
        client.upsert(collection_name=args.collection, points=points)

    logger.info("Ingestion completed.")


if __name__ == "__main__":
    main()
