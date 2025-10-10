from crawl4ai import BrowserConfig


def get_browser_config() -> BrowserConfig:
    """
    Returns the browser configuration for the crawler.
    """
    return BrowserConfig(
        browser_type="chromium",
        headless=False,
        verbose=True,
    )
