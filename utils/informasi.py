import os
import json
from typing import List, Dict, Optional, Tuple, Set
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig

from utils.logger import get_logger
from utils.maklumat_korporat import (
    _safe_id,
    _ensure_dirs,
    analyze_text,
    normalize_record,
)

logger = get_logger("pdn_informasi")


async def fetch_html(crawler: AsyncWebCrawler, url: str) -> Optional[str]:
    result = await crawler.arun(
        url=url,
        config=CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            session_id="pdn_informasi_session",
            process_iframes=False,
            remove_overlay_elements=True,
            excluded_tags=["form"],
        ),
    )
    if not result.success:
        logger.error(f"Failed to fetch {url}: {result.error_message}")
        return None
    return getattr(result, "html", None) or getattr(result, "cleaned_html", None)


def extract_tm_content(html: str) -> Tuple[str, List[Dict], List[str]]:
    soup = BeautifulSoup(html, "html.parser")
    container = soup.select_one("main.tm-content")
    if not container:
        return "", [], []

    # Remove noise
    for sel in [
        ".uk-breadcrumb",
        "#system-message-container",
        ".itemContentFooter",
        ".itemBackToTop",
        ".clr",
        "script",
        "style",
    ]:
        for el in container.select(sel):
            el.decompose()

    text = (container.get_text("\n", strip=True) or "").strip()
    filtered_lines: List[str] = []
    for line in text.splitlines():
        lt = (line or "").strip()
        if not lt:
            continue
        if lt.startswith("Warning") or "count(): Parameter must be an array" in lt:
            continue
        if lt.startswith("Read ") and " times" in lt:
            continue
        if lt.startswith("Last modified on"):
            continue
        if lt.lower() in ("back to top",):
            continue
        filtered_lines.append(lt)
    text = "\n".join(filtered_lines)

    # Images
    imgs: List[Dict] = []
    for img in container.select("img"):
        src = img.get("src") or ""
        if not src:
            continue
        alt = img.get("alt", "")
        imgs.append({"src": src, "alt": alt})

    # Links within tm-content
    links: List[str] = []
    for a in container.select("a[href]"):
        href = a.get("href") or ""
        if href and not href.startswith("#"):
            links.append(href)

    return text, imgs, links


async def process_informasi_page(
    crawler: AsyncWebCrawler,
    page_url: str,
    base_url: str,
    images_dir: str,
    rag_dir: str,
    api_key: Optional[str],
    llm_model: Optional[str],
    *,
    force: bool = False,
    visited: Optional[Set[str]] = None,
    max_depth: int = 1,
    depth: int = 0,
) -> Optional[Dict]:
    if visited is None:
        visited = set()
    if page_url in visited:
        return None
    visited.add(page_url)

    logger.info(f"Visiting informasi page: {page_url}")
    html = await fetch_html(crawler, page_url)
    if not html:
        return None

    content_text, imgs, page_links = extract_tm_content(html)
    logger.info(
        f"Extracted informasi content: text_len={len(content_text)}, images={len(imgs)}, links={len(page_links)}"
    )

    rec_id = _safe_id(page_url)
    records_dir = os.path.join(rag_dir, "records")
    chunks_dir = os.path.join(rag_dir, "chunks")
    _ensure_dirs(records_dir, chunks_dir)

    chunk_path = os.path.join(chunks_dir, f"{rec_id}.txt")
    record_path = os.path.join(records_dir, f"{rec_id}.json")
    if os.path.exists(chunk_path) and not force:
        logger.info(f"Skipping already processed informasi page (no force): {page_url}")
        return None

    # We do not download/analyze images for Informasi pages; keep only links for potential future use
    image_results: List[Dict] = []

    # Page text analysis (provider-aware)
    page_analysis = None
    if content_text:
        try:
            page_analysis = analyze_text(content_text, api_key=api_key, llm_model=llm_model)
        except Exception as e:
            logger.warning(f"Informasi text analysis failed for {page_url}: {e}")

    # Build record and chunk
    record: Dict = {
        "id": rec_id,
        "type": "page",
        "source_url": page_url,
        "title": None,
        "content": content_text,
        "images": image_results,
        "analysis": page_analysis,
    }

    # Chunk: use page text only
    record["chunk"] = (content_text or "").strip()

    # Normalize and write
    record = normalize_record(record)
    with open(chunk_path, "w", encoding="utf-8") as f:
        f.write(record.get("chunk", ""))
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    logger.info(
        f"Wrote informasi record id={rec_id}, chunk_len={len(record.get('chunk',''))} -> {record_path}"
    )

    # Follow links within .tm-content (one level by default)
    if depth < max_depth and page_links:
        for href in page_links:
            abs_href = urljoin(base_url, href)
            parsed = urlparse(abs_href)
            if parsed.scheme in ("http", "https") and ("pdn.gov.my" in (parsed.netloc or "")):
                try:
                    await process_informasi_page(
                        crawler=crawler,
                        page_url=abs_href,
                        base_url=base_url,
                        images_dir=images_dir,
                        rag_dir=rag_dir,
                        api_key=api_key,
                        llm_model=llm_model,
                        force=force,
                        visited=visited,
                        max_depth=max_depth,
                        depth=depth + 1,
                    )
                except Exception as e:
                    logger.warning(f"Failed following informasi link {abs_href}: {e}")

    return record


async def crawl_informasi(
    crawler: AsyncWebCrawler,
    start_url: str,
    images_dir: str,
    rag_dir: str,
    *,
    api_key: Optional[str] = None,
    llm_model: Optional[str] = None,
    force: bool = False,
) -> Dict:
    _ensure_dirs(images_dir, rag_dir, os.path.join(rag_dir, "records"), os.path.join(rag_dir, "chunks"))
    visited: Set[str] = set()
    base_url = start_url  # for urljoin
    rec = await process_informasi_page(
        crawler=crawler,
        page_url=start_url,
        base_url=start_url,
        images_dir=images_dir,
        rag_dir=rag_dir,
        api_key=api_key,
        llm_model=llm_model,
        force=force,
        visited=visited,
        max_depth=1,
        depth=0,
    )
    return {"count": 1 if rec else 0}
