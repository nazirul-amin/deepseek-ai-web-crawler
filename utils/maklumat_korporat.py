import os
import json
import hashlib
from typing import List, Dict, Optional, Tuple, Set
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig

from utils.logger import get_logger
from utils.homepage import download_image, groq_analyze_image  # reuse

logger = get_logger("pdn_maklumat_korporat")


def _ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def _safe_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


async def fetch_html(crawler: AsyncWebCrawler, url: str) -> Optional[str]:
    result = await crawler.arun(
        url=url,
        config=CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            session_id="pdn_maklumat_korporat_session",
            process_iframes=True,
            remove_overlay_elements=True,
            excluded_tags=["form", "header", "nav"],
        ),
    )
    if not result.success:
        logger.error(f"Failed to fetch {url}: {result.error_message}")
        return None
    html = getattr(result, "html", None) or getattr(result, "cleaned_html", None)
    return html


def parse_nav_links(nav_html: str, base_url: str) -> List[str]:
    """Extract ONLY 'Maklumat Korporat' navigation links.

    We select anchors from the main nav and dropdowns, then filter to internal links
    whose path starts with '/v2/index.php/maklumat-korporat'.
    """
    soup = BeautifulSoup(nav_html, "html.parser")
    links: Set[str] = set()
    for a in soup.select(
        ".tm-navigation-wrapper .uk-navbar-nav a[href], .uk-dropdown a[href], .uk-nav-sub a[href]"
    ):
        href = a.get("href") or ""
        if not href or href.startswith("#"):
            continue
        abs_url = urljoin(base_url, href)
        parsed = urlparse(abs_url)
        # internal only
        if "pdn.gov.my" not in (parsed.netloc or ""):
            continue
        if parsed.scheme not in ("http", "https"):
            continue
        # keep only Maklumat Korporat section
        if not (parsed.path or "").startswith("/v2/index.php/maklumat-korporat"):
            continue
        links.add(abs_url)
    return sorted(links)


def extract_main_body(html: str) -> Tuple[str, List[Dict], List[str]]:
    """Return text, image tags (src, alt), and in-body links from .mainbody-wrapper."""
    soup = BeautifulSoup(html, "html.parser")
    container = soup.select_one(".mainbody-wrapper")
    if not container:
        return "", [], []

    # Collect text from headings, paragraphs, list items
    text_parts: List[str] = []
    for sel in ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]:
        for el in container.select(sel):
            t = (el.get_text(" ", strip=True) or "").strip()
            if t:
                text_parts.append(t)
    content_text = "\n".join(text_parts)

    # Collect images
    imgs: List[Dict] = []
    for img in container.select("img"):
        src = img.get("src") or ""
        if not src:
            continue
        alt = img.get("alt", "")
        imgs.append({"src": src, "alt": alt})

    # Collect in-body links
    links: List[str] = []
    for a in container.select("a[href]"):
        href = a.get("href") or ""
        if href and not href.startswith("#"):
            links.append(href)

    return content_text, imgs, links


async def process_page(
    crawler: AsyncWebCrawler,
    page_url: str,
    base_url: str,
    images_dir: str,
    rag_dir: str,
    groq_api_key: Optional[str],
    groq_model: Optional[str],
    *,
    force: bool = False,
    visited: Optional[Set[str]] = None,
    max_depth: int = 0,
    depth: int = 0,
) -> Optional[Dict]:
    # init visited set and guard
    if visited is None:
        visited = set()
    if page_url in visited:
        return None
    visited.add(page_url)

    html = await fetch_html(crawler, page_url)
    if not html:
        return None

    content_text, imgs, page_links = extract_main_body(html)

    rec_id = _safe_id(page_url)
    records_dir = os.path.join(rag_dir, "records")
    chunks_dir = os.path.join(rag_dir, "chunks")
    _ensure_dirs(records_dir, chunks_dir)

    chunk_path = os.path.join(chunks_dir, f"{rec_id}.txt")
    record_path = os.path.join(records_dir, f"{rec_id}.json")

    if os.path.exists(chunk_path) and not force:
        logger.debug(f"Skipping already processed page: {page_url}")
        return None

    image_results: List[Dict] = []
    for it in imgs:
        abs_src = urljoin(base_url, it["src"]) if it["src"] else None
        if not abs_src:
            continue
        local_path = download_image(abs_src, images_dir)
        analysis = None
        if local_path and groq_api_key and groq_model:
            try:
                analysis = groq_analyze_image(local_path, groq_api_key, groq_model)
            except Exception as e:
                logger.warning(f"Groq analysis failed for {abs_src}: {e}")
        image_results.append({
            "source_url": abs_src,
            "alt": it.get("alt", ""),
            "local_path": local_path,
            "analysis": analysis,
        })

    # Build record
    record: Dict = {
        "id": rec_id,
        "type": "page",
        "source_url": page_url,
        "title": None,  # can be improved by extracting <h1> if present
        "content": content_text,
        "images": image_results,
    }

    # Chunk: prefer page text, fallback to image analyses summaries
    chunk_text = content_text.strip()
    if not chunk_text and image_results:
        parts: List[str] = []
        for ir in image_results:
            a = ir.get("analysis") or {}
            if isinstance(a, dict):
                summary = a.get("summary") or a.get("ocr_text")
                if summary:
                    parts.append(str(summary))
        chunk_text = "\n".join(parts)
    if not chunk_text:
        chunk_text = ""
    if len(chunk_text) > 6000:
        chunk_text = chunk_text[:6000]

    record["chunk"] = chunk_text

    # Append to pages.jsonl (separate log file for site pages)
    pages_jsonl = os.path.join(rag_dir, "pages.jsonl")
    with open(pages_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Write chunk and record
    with open(chunk_path, "w", encoding="utf-8") as f:
        f.write(chunk_text)
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    # Follow in-body links one level deep if allowed
    if depth < max_depth and page_links:
        for href in page_links:
            abs_href = urljoin(base_url, href)
            parsed = urlparse(abs_href)
            # internal only
            if parsed.scheme in ("http", "https") and ("pdn.gov.my" in (parsed.netloc or "")):
                try:
                    await process_page(
                        crawler=crawler,
                        page_url=abs_href,
                        base_url=base_url,
                        images_dir=images_dir,
                        rag_dir=rag_dir,
                        groq_api_key=groq_api_key,
                        groq_model=groq_model,
                        force=force,
                        visited=visited,
                        max_depth=max_depth,
                        depth=depth + 1,
                    )
                except Exception as e:
                    logger.warning(f"Failed following link {abs_href}: {e}")

    return record


async def crawl_maklumat_korporate(
    crawler: AsyncWebCrawler,
    base_url: str,
    images_dir: str,
    rag_dir: str,
    *,
    groq_api_key: Optional[str] = None,
    groq_model: Optional[str] = None,
    force: bool = False,
) -> Dict:
    """
    Crawl PDN site navigation and extract main body content and images for RAG.
    Writes RAG artifacts under rag_dir (records/, chunks/, pages.jsonl).
    """
    _ensure_dirs(images_dir, rag_dir, os.path.join(rag_dir, "records"), os.path.join(rag_dir, "chunks"))

    # Fetch the base page that contains the navigation
    nav_html = await fetch_html(crawler, base_url)
    if not nav_html:
        return {"count": 0, "pages": []}

    # Parse navigation links
    links = parse_nav_links(nav_html, base_url)
    if not links:
        logger.warning("No navigation links found.")
        return {"count": 0, "pages": []}

    # Process each page
    processed: List[Dict] = []
    visited: Set[str] = set()
    for url in links:
        try:
            rec = await process_page(
                crawler=crawler,
                page_url=url,
                base_url=base_url,
                images_dir=images_dir,
                rag_dir=rag_dir,
                groq_api_key=groq_api_key,
                groq_model=groq_model,
                force=force,
                visited=visited,
                max_depth=1,
                depth=0,
            )
            if rec:
                processed.append(rec)
        except Exception as e:
            logger.error(f"Failed processing {url}: {e}")

    logger.info(f"Processed {len(processed)} site pages from navigation.")
    return {"count": len(processed), "pages": processed}
