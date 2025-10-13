import os
import json
import base64
import hashlib
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from utils.logger import get_logger

logger = get_logger("pdn_homepage")


def _ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def _b64_data_url(local_path: str) -> str:
    ext = os.path.splitext(local_path)[1].lower().lstrip(".") or "png"
    with open(local_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{ext};base64,{b64}"


def _safe_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _file_sha256(local_path: str) -> str:
    """Compute SHA256 hash of a file's contents."""
    h = hashlib.sha256()
    with open(local_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


async def fetch_index_html(crawler: AsyncWebCrawler, url: str, session_id: str) -> Optional[str]:
    result = await crawler.arun(
        url=url,
        config=CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            session_id=session_id,
            process_iframes=False,
            remove_overlay_elements=True,
            excluded_tags=["form", "header", "nav"],
        ),
    )
    if not result.success:
        logger.error(f"Failed to fetch {url}: {result.error_message}")
        return None

    html = getattr(result, "html", None) or getattr(result, "cleaned_html", None)
    return html


def parse_slider_images(html: str, base_url: str) -> List[Dict]:
    """
    Extract image entries from SmartSlider markup.
    Returns list of dicts: {src, abs_url, alt, title, public_id}
    """
    soup = BeautifulSoup(html, "html.parser")

    root = (
        soup.select_one("#tmSlider")
        or soup.select_one(".n2-section-smartslider")
        or soup
    )

    images: List[Dict] = []
    seen = set()

    selectors = [
        ".n2-ss-slide-background-image img",
        ".n2-ss-slide-backgrounds img",
        "#n2-ss-2 img",
        "picture.skip-lazy img",
    ]

    def add_img(img):
        src = img.get("src") or ""
        if not src:
            return
        if src.startswith("data:") or src.startswith("blob:"):
            return
        abs_url = urljoin(base_url, src)
        if abs_url in seen:
            return
        seen.add(abs_url)

        public_id = None
        parent_bg = img.find_parent(class_="n2-ss-slide-background")
        if parent_bg and parent_bg.has_attr("data-public-id"):
            public_id = parent_bg["data-public-id"]

        images.append({
            "src": src,
            "abs_url": abs_url,
            "alt": img.get("alt", ""),
            "title": img.get("title", ""),
            "public_id": public_id,
        })

    for sel in selectors:
        for img in root.select(sel):
            add_img(img)

    if not images:
        for img in root.select("img"):
            src = img.get("src") or ""
            if "/v2/images/SmartSlider3/" in src or "/v2/images/it_pdn/" in src:
                add_img(img)

    return images


def download_image(url: str, dest_dir: str) -> Optional[str]:
    """
    Download image to a deterministic path using md5(url) + extension to avoid filename collisions.
    Returns the local file path or None on failure.
    """
    _ensure_dirs(dest_dir)
    if url.startswith("data:") or url.startswith("blob:"):
        logger.debug(f"Skipping non-http image source: {url[:64]}...")
        return None

    parsed_path = urlparse(url).path
    ext = os.path.splitext(parsed_path)[1].lower() or ".png"
    name = f"{_safe_id(url)}{ext}"
    local_path = os.path.join(dest_dir, name)

    if os.path.exists(local_path):
        logger.info(f"Image already exists, skipping download: {local_path}")
        return local_path

    try:
        logger.info(f"Downloading image: {url}")
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        }
        resp = requests.get(url, timeout=30, headers=headers, allow_redirects=True)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(resp.content)
        logger.info(f"Downloaded image to: {local_path}")
        return local_path
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return None


def groq_analyze_image(local_path: str, api_key: str, model: str) -> Dict:
    try:
        from groq import Groq
    except Exception as e:
        raise RuntimeError(
            "Groq SDK is not installed. Install dependencies (e.g., `uv pip install -r requirements.txt`) "
            "or `pip install groq` and retry."
        ) from e

    client = Groq(api_key=api_key)

    data_url = _b64_data_url(local_path)

    system_prompt = (
        "You analyze banner images and return structured JSON. "
        "Extract: title, summary, language, ocr_text, labels (list), entities (list), "
        "date (if any), call_to_action (if any), urls (list of any URLs found). "
        "Respond with ONLY valid JSON."
    )

    user_text = (
        "Analyze this image and extract the requested fields. "
        "Keep summary concise for RAG (<= 80 words)."
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        temperature=0.2,
        max_tokens=1200,
    )

    text = completion.choices[0].message.content or "{}"
    try:
        return json.loads(text)
    except Exception:
        return {"raw": text}


async def process_homepage(
    crawler: AsyncWebCrawler,
    page_url: str,
    images_dir: str,
    rag_dir: str,
    groq_api_key: str,
    groq_model: str,
    *,
    force: bool = False,
) -> Dict:
    """
    Process the PDN homepage slider end-to-end.
    Idempotent by default: if a chunk already exists for an image URL, it skips re-analysis.
    Set force=True to reprocess.
    """
    _ensure_dirs(images_dir, rag_dir, os.path.join(rag_dir, "chunks"), os.path.join(rag_dir, "records"))

    html = await fetch_index_html(crawler, page_url, session_id="pdn_homepage_session")
    if not html:
        return {"count": 0, "items": []}

    items = parse_slider_images(html, base_url=page_url)
    logger.info(f"Found {len(items)} slider images")
    records_dir = os.path.join(rag_dir, "records")
    processed = []

    for item in items:
        src_url = item["abs_url"]
        rec_id = _safe_id(src_url)
        chunk_path = os.path.join(rag_dir, "chunks", f"{rec_id}.txt")
        record_path = os.path.join(records_dir, f"{rec_id}.json")

        # Skip if already processed and not forcing re-analysis
        if os.path.exists(chunk_path) and not force:
            logger.info(f"Skipping already processed slider image (no force): {src_url}")
            continue

        logger.info(f"Downloading slider image: {src_url}")
        local = download_image(src_url, images_dir)
        if not local:
            logger.warning(f"Slider image download failed: {src_url}")
            continue

        logger.info(f"Invoking Groq Vision for slider image: {src_url}")
        analysis = groq_analyze_image(local, api_key=groq_api_key, model=groq_model)
        try:
            logger.info(
                f"Groq response (slider image) keys: {list(analysis.keys()) if isinstance(analysis, dict) else 'n/a'}"
            )
        except Exception:
            pass
        
        try:
            if isinstance(analysis, dict):
                if "raw" in analysis:
                    logger.info("Groq full response (slider image, raw): %s", analysis.get("raw", ""))
                else:
                    logger.info(
                        "Groq full response (slider image, json): %s",
                        json.dumps(analysis, ensure_ascii=False, indent=2),
                    )
        except Exception:
            pass
        chunk_text = analysis.get("summary") or analysis.get("ocr_text") or json.dumps(analysis)[:1000]

        record = {
            "id": rec_id,
            "source_url": src_url,
            "local_path": local,
            "alt": item.get("alt", ""),
            "title": item.get("title", ""),
            "public_id": item.get("public_id"),
            "llm_model": groq_model,
            "image_sha256": _file_sha256(local),
            "analysis": analysis,
            "chunk": chunk_text,
        }

        # Write chunk file for easy ingestion
        with open(chunk_path, "w", encoding="utf-8") as f:
            f.write(chunk_text)

        # Write canonical per-record JSON (overwrites to keep latest)
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        processed.append(record)
        logger.info(
            f"Wrote slider record id={rec_id}, chunk_len={len(chunk_text)}, local_image={local} -> {record_path}"
        )

    return {"count": len(processed), "items": processed}
