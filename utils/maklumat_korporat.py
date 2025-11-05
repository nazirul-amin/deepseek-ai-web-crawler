import os
import json
import hashlib
from typing import List, Dict, Optional, Tuple, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from pypdf import PdfReader

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
            process_iframes=False,
            remove_overlay_elements=True,
            excluded_tags=["form"],
        ),
    )
    if not result.success:
        logger.error(f"Failed to fetch {url}: {result.error_message}")
        return None
    html = getattr(result, "html", None) or getattr(result, "cleaned_html", None)
    return html


def download_file(url: str, dest_dir: str) -> Optional[str]:
    """Download a file (e.g., PDF) to dest_dir using md5(url)+ext. Returns local path or None."""
    os.makedirs(dest_dir, exist_ok=True)
    parsed_path = urlparse(url).path
    ext = os.path.splitext(parsed_path)[1].lower() or ""
    name = f"{_safe_id(url)}{ext}"
    local_path = os.path.join(dest_dir, name)
    if os.path.exists(local_path):
        return local_path
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        }
        resp = requests.get(url, timeout=60, headers=headers, allow_redirects=True)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(resp.content)
        return local_path
    except Exception as e:
        logger.warning(f"Failed to download file {url}: {e}")
        return None


def extract_pdf_text(local_path: str, max_chars: int = 20000) -> str:
    """Extract text from a PDF using pypdf. Truncate to max_chars to keep prompt size reasonable."""
    try:
        reader = PdfReader(local_path)
        parts: List[str] = []
        for page in reader.pages:
            t = page.extract_text() or ""
            if t:
                parts.append(t)
        text = "\n".join(parts)
        if len(text) > max_chars:
            text = text[:max_chars]
        return text
    except Exception as e:
        logger.warning(f"Failed to extract PDF text from {local_path}: {e}")
        return ""


def groq_analyze_text(doc_text: str, api_key: Optional[str], model: Optional[str]) -> Optional[Dict]:
    """Analyze plain text with Groq and return structured JSON fields similar to image analysis."""
    if not api_key or not model:
        return None
    try:
        from groq import Groq
    except Exception as e:
        logger.warning("Groq SDK not available for text analysis; skipping.")
        return None
    client = Groq(api_key=api_key)

    system_prompt = (
        "You analyze document text and return structured JSON. "
        "Extract: title, summary (<= 120 words), language, labels (list), entities (list), "
        "date (if any), urls (list). Respond with ONLY valid JSON."
    )
    user_text = (
        "Analyze the following document text and extract the requested fields. "
        "If no explicit title, infer a concise one."
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{user_text}\n\nTEXT:\n{doc_text}"},
        ],
        temperature=0.2,
        max_tokens=1200,
    )
    text = completion.choices[0].message.content or "{}"
    try:
        return json.loads(text)
    except Exception:
        return {"raw": text}


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
    """Return text, image tags (src, alt), and in body links from .mainbody-wrapper."""
    soup = BeautifulSoup(html, "html.parser")
    container = soup.select_one(".mainbody-wrapper main.tm-content") or soup.select_one(".mainbody-wrapper")
    if not container:
        return "", [], []

    # Remove unwanted sections
    for sel in [
        ".uk-breadcrumb",
        ".itemContentFooter",
        ".itemBackToTop",
        ".clr",
        "script",
        "style",
    ]:
        for el in container.select(sel):
            el.decompose()

    # Collect text from the remaining main body.
    # Use newlines to better separate sections; capture table cell contents too.
    content_text = (container.get_text("\n", strip=True) or "").strip()
    # Filter noisy lines like breadcrumbs, warnings, footer stats, back-to-top, plugin boilerplate
    filtered_lines: List[str] = []
    for line in content_text.splitlines():
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
        if lt.startswith("Plugins:") or lt.startswith("K2 Plugins:") or lt.startswith("JoomlaWorks"):
            continue
        filtered_lines.append(lt)
    content_text = "\n".join(filtered_lines)

    # Collect images
    imgs: List[Dict] = []
    for img in container.select("img"):
        src = img.get("src") or ""
        if not src:
            continue
        alt = img.get("alt", "")
        imgs.append({"src": src, "alt": alt})

    # Collect in body links
    links: List[str] = []
    for a in container.select("a[href]"):
        href = a.get("href") or ""
        if href and not href.startswith("#"):
            links.append(href)

    return content_text, imgs, links


def extract_title(html: str) -> Optional[str]:
    """Extract a page title from the main content or breadcrumb.
    Priority: first H1 in .mainbody-wrapper → breadcrumb active span → <title> tag.
    """
    soup = BeautifulSoup(html, "html.parser")
    container = soup.select_one(".mainbody-wrapper")
    if container:
        h1 = container.select_one("h1")
        if h1:
            t = (h1.get_text(" ", strip=True) or "").strip()
            if t:
                return t
        # breadcrumb active
        bc_active = container.select_one(".uk-breadcrumb .uk-active span, .uk-breadcrumb li.uk-active span")
        if bc_active:
            t = (bc_active.get_text(" ", strip=True) or "").strip()
            if t:
                return t
    # fallback to document title
    doc_title = soup.select_one("title")
    if doc_title:
        t = (doc_title.get_text(" ", strip=True) or "").strip()
        if t:
            return t
    return None


def _strip_code_fences(text: str) -> str:
    try:
        t = text.strip()
        if t.startswith("```"):
            # remove the first fence line (e.g., ```json) including any language tag
            t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.endswith("```"):
            t = t[: t.rfind("```")]
        return t.strip()
    except Exception:
        return text


def _parse_analysis_obj(a: Optional[Dict]) -> Optional[Dict]:
    if not isinstance(a, dict):
        return None
    out: Dict = {
        "title": a.get("title"),
        "summary": a.get("summary"),
        "language": a.get("language"),
        "labels": a.get("labels") if isinstance(a.get("labels"), list) else [],
        "entities": a.get("entities") if isinstance(a.get("entities"), list) else [],
        "date": a.get("date"),
        "urls": a.get("urls") if isinstance(a.get("urls"), list) else [],
        "raw": a.get("raw") if isinstance(a.get("raw"), str) else None,
    }
    raw_txt = out.get("raw")
    if raw_txt and not out.get("summary"):
        try:
            stripped = _strip_code_fences(raw_txt)
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                out["title"] = out.get("title") or parsed.get("title")
                out["summary"] = out.get("summary") or parsed.get("summary")
                out["language"] = out.get("language") or parsed.get("language")
                if not out.get("labels") and isinstance(parsed.get("labels"), list):
                    out["labels"] = parsed.get("labels")
                if not out.get("entities") and isinstance(parsed.get("entities"), list):
                    out["entities"] = parsed.get("entities")
                out["date"] = out.get("date") or parsed.get("date")
                if not out.get("urls") and isinstance(parsed.get("urls"), list):
                    out["urls"] = parsed.get("urls")
        except Exception:
            pass
    return out


def _derive_text_from_filename(url: Optional[str]) -> str:
    try:
        if not url:
            return ""
        name = os.path.basename(urlparse(url).path)
        name = os.path.splitext(name)[0]
        return name.replace("_", " ").replace("-", " ")
    except Exception:
        return ""


def normalize_record(record: Dict) -> Dict:
    rec = dict(record)
    rec.setdefault("document_path", rec.get("document_path") or None)
    rec.setdefault("images", rec.get("images") or [])

    # Normalize per-image analyses
    imgs = []
    for it in rec.get("images", []) or []:
        img = dict(it)
        img["analysis"] = _parse_analysis_obj(img.get("analysis"))
        imgs.append(img)
    rec["images"] = imgs

    # Root analysis
    root_analysis = _parse_analysis_obj(rec.get("analysis"))
    if not root_analysis:
        # Prefer first image analysis if present
        if rec.get("type") == "image" and rec.get("images"):
            root_analysis = rec["images"][0].get("analysis")
        elif rec.get("type") == "page" and rec.get("images"):
            # Aggregate best available
            for im in rec["images"]:
                cand = im.get("analysis")
                if cand and (cand.get("summary") or cand.get("ocr_text")):
                    root_analysis = cand
                    break
        # If still None, synthesize minimal
        if not root_analysis:
            root_analysis = {
                "title": rec.get("title"),
                "summary": None,
                "language": None,
                "labels": [],
                "entities": [],
                "date": None,
                "urls": [],
                "raw": None,
            }
    rec["analysis"] = root_analysis

    # Title fallback from analysis
    if not rec.get("title") and isinstance(rec.get("analysis"), dict):
        rec["title"] = rec["analysis"].get("title")

    # Build chunk if empty
    chunk = (rec.get("chunk") or "").strip()
    if not chunk:
        # 1) analysis.summary
        if isinstance(rec.get("analysis"), dict) and rec["analysis"].get("summary"):
            chunk = str(rec["analysis"]["summary"]).strip()
        # 2) content (pdf/page)
        if not chunk:
            content = (rec.get("content") or "").strip()
            if content:
                chunk = content[:6000]
        # 3) image analyses
        if not chunk and rec.get("images"):
            parts: List[str] = []
            for im in rec["images"]:
                a = im.get("analysis") or {}
                if isinstance(a, dict):
                    txt = a.get("summary") or a.get("ocr_text")
                    if txt:
                        parts.append(str(txt))
            chunk = "\n".join(parts).strip()
        # 4) image alt text
        if not chunk and rec.get("images"):
            alts = [im.get("alt") for im in rec["images"] if im.get("alt")]
            if alts:
                chunk = "\n".join(alts)
        # 5) derive from filename
        if not chunk:
            chunk = _derive_text_from_filename(rec.get("source_url"))
    rec["chunk"] = chunk or ""

    return rec


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
    logger.info(f"Visiting page: {page_url}")

    # Determine resource type by extension and handle images/PDFs accordingly
    parsed_url = urlparse(page_url)
    path_lower = (parsed_url.path or "").lower()
    image_exts = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")
    pdf_exts = (".pdf",)
    binary_skip_exts = (".doc", ".docx", ".xls", ".xlsx", ".zip")

    if path_lower.endswith(image_exts):
        logger.info(f"Resource type detected: image ({path_lower})")
        # Direct image resource: download and analyze with Groq vision
        rec_id = _safe_id(page_url)
        records_dir = os.path.join(rag_dir, "records")
        chunks_dir = os.path.join(rag_dir, "chunks")
        _ensure_dirs(records_dir, chunks_dir)

        chunk_path = os.path.join(chunks_dir, f"{rec_id}.txt")
        record_path = os.path.join(records_dir, f"{rec_id}.json")
        if os.path.exists(chunk_path) and not force:
            logger.debug(f"Skipping already processed image resource: {page_url}")
            return None

        logger.info(f"Downloading image resource: {page_url}")
        local_path = download_image(page_url, images_dir)
        if local_path:
            logger.info(f"Downloaded image to: {local_path}")
        else:
            logger.warning(f"Image download failed: {page_url}")
        analysis = None
        if local_path and groq_api_key and groq_model:
            try:
                logger.info(f"Invoking Groq Vision for image: {page_url}")
                analysis = groq_analyze_image(local_path, groq_api_key, groq_model)
                try:
                    logger.info(f"Groq response (image) keys: {list(analysis.keys()) if isinstance(analysis, dict) else 'n/a'}")
                except Exception:
                    pass
                # Log full Groq response for visibility
                try:
                    if isinstance(analysis, dict):
                        if "raw" in analysis:
                            logger.info("Groq full response (image, raw): %s", analysis.get("raw", ""))
                        else:
                            logger.info(
                                "Groq full response (image, json): %s",
                                json.dumps(analysis, ensure_ascii=False, indent=2),
                            )
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"Groq image analysis failed for {page_url}: {e}")
        # Build minimal content from analysis
        chunk_text = ""
        if isinstance(analysis, dict):
            chunk_text = (analysis.get("summary") or analysis.get("ocr_text") or "").strip()

        record: Dict = {
            "id": rec_id,
            "type": "image",
            "source_url": page_url,
            "title": (analysis.get("title") if isinstance(analysis, dict) else None),
            "content": "",
            "images": [
                {
                    "source_url": page_url,
                    "alt": "",
                    "local_path": local_path,
                    "analysis": analysis,
                }
            ],
            "chunk": chunk_text,
        }

        record = normalize_record(record)
        with open(chunk_path, "w", encoding="utf-8") as f:
            f.write(record.get("chunk", ""))
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        logger.info(f"Wrote image record id={rec_id}, chunk_len={len(chunk_text)} -> {record_path}")
        return record

    logger.info(f"Visiting page: {page_url}")
    if path_lower.endswith(pdf_exts):
        logger.info(f"Resource type detected: pdf ({path_lower})")
        # Direct PDF resource: download, extract text, analyze with Groq text
        rec_id = _safe_id(page_url)
        records_dir = os.path.join(rag_dir, "records")
        chunks_dir = os.path.join(rag_dir, "chunks")
        pdfs_dir = os.path.join(os.path.dirname(images_dir), "pdfs")
        _ensure_dirs(records_dir, chunks_dir, pdfs_dir)

        chunk_path = os.path.join(chunks_dir, f"{rec_id}.txt")
        record_path = os.path.join(records_dir, f"{rec_id}.json")
        if os.path.exists(chunk_path) and not force:
            logger.info(f"Skipping already processed PDF resource: {page_url}")
            return None

        logger.info(f"Downloading PDF resource: {page_url}")
        local_path = download_file(page_url, pdfs_dir)
        if local_path:
            logger.info(f"Downloaded PDF to: {local_path}")
        else:
            logger.warning(f"PDF download failed: {page_url}")
        doc_text = extract_pdf_text(local_path) if local_path else ""
        logger.info(f"Extracted PDF text length: {len(doc_text)}")
        analysis = groq_analyze_text(doc_text, groq_api_key, groq_model) if doc_text else None
        if isinstance(analysis, dict):
            try:
                logger.info(f"Groq response (pdf) keys: {list(analysis.keys())}")
            except Exception:
                pass
            # Log full Groq response for visibility
            try:
                if "raw" in analysis:
                    logger.info("Groq full response (pdf, raw): %s", analysis.get("raw", ""))
                else:
                    logger.info(
                        "Groq full response (pdf, json): %s",
                        json.dumps(analysis, ensure_ascii=False, indent=2),
                    )
            except Exception:
                pass
        chunk_text = ""
        if isinstance(analysis, dict):
            chunk_text = (analysis.get("summary") or "").strip()
        if not chunk_text:
            chunk_text = (doc_text or "")

        record: Dict = {
            "id": rec_id,
            "type": "document",
            "source_url": page_url,
            "title": (analysis.get("title") if isinstance(analysis, dict) else None),
            "content": doc_text,
            "images": [],
            "document_path": local_path,
            "analysis": analysis,
            "chunk": chunk_text,
        }

        record = normalize_record(record)
        with open(chunk_path, "w", encoding="utf-8") as f:
            f.write(record.get("chunk", ""))
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        return record

    if path_lower.endswith(binary_skip_exts):
        logger.debug(f"Skipping unsupported binary resource: {page_url}")
        return None

    logger.info(f"Resource type detected: html (page)")
    html = await fetch_html(crawler, page_url)
    if not html:
        return None

    content_text, imgs, page_links = extract_main_body(html)
    logger.info(f"Extracted main content: text_len={len(content_text)}, images={len(imgs)}, links={len(page_links)}")
    title_text = extract_title(html)

    # Build page-level analysis using Groq text model (same model as images), if available
    page_analysis = None
    if content_text and groq_api_key and groq_model:
        try:
            page_analysis = groq_analyze_text(content_text, groq_api_key, groq_model)
            if isinstance(page_analysis, dict):
                try:
                    logger.info(f"Groq response (page text) keys: {list(page_analysis.keys())}")
                except Exception:
                    pass
                try:
                    if "raw" in page_analysis:
                        logger.info("Groq full response (page text, raw): %s", page_analysis.get("raw", ""))
                    else:
                        logger.info(
                            "Groq full response (page text, json): %s",
                            json.dumps(page_analysis, ensure_ascii=False, indent=2),
                        )
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Groq page text analysis failed for {page_url}: {e}")

    rec_id = _safe_id(page_url)
    records_dir = os.path.join(rag_dir, "records")
    chunks_dir = os.path.join(rag_dir, "chunks")
    _ensure_dirs(records_dir, chunks_dir)

    chunk_path = os.path.join(chunks_dir, f"{rec_id}.txt")
    record_path = os.path.join(records_dir, f"{rec_id}.json")

    if os.path.exists(chunk_path) and not force:
        logger.info(f"Skipping already processed page (no force): {page_url}")
        return None

    image_results: List[Dict] = []
    for it in imgs:
        abs_src = urljoin(base_url, it["src"]) if it["src"] else None
        if not abs_src:
            continue
        logger.info(f"Downloading page image: {abs_src}")
        local_path = download_image(abs_src, images_dir)
        if local_path:
            logger.info(f"Downloaded page image to: {local_path}")
        else:
            logger.warning(f"Page image download failed: {abs_src}")
        analysis = None
        if local_path and groq_api_key and groq_model:
            try:
                logger.info(f"Invoking Groq Vision for page image: {abs_src}")
                analysis = groq_analyze_image(local_path, groq_api_key, groq_model)
                try:
                    logger.info(f"Groq response (page image) keys: {list(analysis.keys()) if isinstance(analysis, dict) else 'n/a'}")
                except Exception:
                    pass
                # Log full Groq response for visibility
                try:
                    if isinstance(analysis, dict):
                        if "raw" in analysis:
                            logger.info("Groq full response (page image, raw): %s", analysis.get("raw", ""))
                        else:
                            logger.info(
                                "Groq full response (page image, json): %s",
                                json.dumps(analysis, ensure_ascii=False, indent=2),
                            )
                except Exception:
                    pass
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
        "title": title_text,
        "content": content_text,
        "images": image_results,
        "analysis": page_analysis,
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

    record["chunk"] = chunk_text

    record = normalize_record(record)

    # Write chunk and record
    with open(chunk_path, "w", encoding="utf-8") as f:
        f.write(record.get("chunk", ""))
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    logger.info(f"Wrote page record id={rec_id}, title={title_text!r}, chunk_len={len(chunk_text)} -> {record_path}")

    # Follow in body links one level deep if allowed
    if depth < max_depth and page_links:
        for href in page_links:
            abs_href = urljoin(base_url, href)
            parsed = urlparse(abs_href)
            # internal only; traverse content pages, plus direct image/PDF resources
            if parsed.scheme in ("http", "https") and ("pdn.gov.my" in (parsed.netloc or "")):
                path_l = (parsed.path or "").lower()
                if path_l.endswith(image_exts) or path_l.endswith(pdf_exts) or path_l.startswith("/v2/index.php/"):
                    try:
                        logger.info(f"Following in body link (depth {depth+1}): {abs_href}")
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
    Writes RAG artifacts under rag_dir (records/, chunks/).
    """
    _ensure_dirs(images_dir, rag_dir, os.path.join(rag_dir, "records"), os.path.join(rag_dir, "chunks"))

    # Fetch the base page that contains the navigation (keep header/nav)
    nav_html = await fetch_html(crawler, base_url)
    if not nav_html:
        return {"count": 0, "pages": []}

    # Parse navigation links
    links = parse_nav_links(nav_html, base_url)
    logger.info(f"Found {len(links)} navigation links for Maklumat Korporat")
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
