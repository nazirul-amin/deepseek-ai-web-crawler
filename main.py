import asyncio

from crawl4ai import AsyncWebCrawler
from dotenv import load_dotenv

from config import BASE_URL, GROQ_MODEL, IMAGES_DIR, RAG_DIR
from utils.browser_config import get_browser_config
from utils.homepage import process_homepage
from utils.maklumat_korporat import crawl_maklumat_korporate
import os
from utils.logger import get_logger

logger = get_logger("main")

load_dotenv()


async def crawl_homepage():
    """
    Crawl PDN homepage slider, download images, analyze with selected provider, and save RAG outputs.
    Provider is chosen via ANALYSIS_PROVIDER env (groq [default] or openai).
    """
    # Provider selection and key validation are handled inside utils.* (provider-aware)

    browser_config = get_browser_config()

    force_reprocess = os.getenv("FORCE_REPROCESS", "").lower() in ("1", "true", "yes", "y")

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await process_homepage(
            crawler=crawler,
            page_url=BASE_URL,
            images_dir=IMAGES_DIR,
            rag_dir=RAG_DIR,
            api_key=os.getenv("GROQ_API_KEY"),
            llm_model=GROQ_MODEL,
            force=force_reprocess,
        )
        logger.info(f"Processed {result['count']} slider images. Outputs saved under '{RAG_DIR}'.")


async def main():
    """
    Entry point of the script.
    """
    await crawl_homepage()
    # Run site-wide crawler over navigation; provider selection and key validation are handled in utils
    async with AsyncWebCrawler(config=get_browser_config()) as crawler:
        site_result = await crawl_maklumat_korporate(
            crawler=crawler,
            base_url=BASE_URL,
            images_dir=IMAGES_DIR,
            rag_dir=RAG_DIR,
            api_key=os.getenv("GROQ_API_KEY"),
            llm_model=GROQ_MODEL,
            force=os.getenv("FORCE_REPROCESS", "").lower() in ("1", "true", "yes", "y"),
        )
        logger.info(f"Processed {site_result['count']} site pages from navigation. Outputs saved under '{RAG_DIR}'.")


if __name__ == "__main__":
    asyncio.run(main())
