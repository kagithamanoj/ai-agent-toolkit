"""
Web Scraping Agent — Autonomous agent that crawls websites and extracts structured data.

Inspired by: crawl4ai patterns
Uses: LangChain + BeautifulSoup for intelligent scraping

Usage:
    from agents.web_scraping_agent import WebScrapingAgent
    agent = WebScrapingAgent()
    result = agent.scrape("https://example.com", extract=["title", "links", "headings"])
"""

import os
import re
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from dotenv import load_dotenv

load_dotenv()


class WebScrapingAgent:
    """Autonomous web scraping agent with LLM-powered extraction."""

    def __init__(self, user_agent: str = "AI-Agent-Toolkit/1.0"):
        self.user_agent = user_agent
        self.session_cache: dict[str, str] = {}

    def fetch(self, url: str, timeout: int = 15) -> str:
        """Fetch raw HTML from a URL."""
        if url in self.session_cache:
            return self.session_cache[url]

        req = Request(url)
        req.add_header("User-Agent", self.user_agent)

        try:
            with urlopen(req, timeout=timeout) as response:
                html = response.read().decode("utf-8", errors="ignore")
                self.session_cache[url] = html
                return html
        except HTTPError as e:
            raise RuntimeError(f"HTTP {e.code} fetching {url}: {e.reason}")

    def extract_text(self, html: str) -> str:
        """Extract clean text from HTML (no external deps)."""
        # Remove script and style elements
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Clean whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Decode common HTML entities
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        text = text.replace("&nbsp;", " ").replace("&quot;", '"')
        return text

    def extract_links(self, html: str, base_url: str = "") -> list[dict]:
        """Extract all links from HTML."""
        links = []
        for match in re.finditer(r'<a\s[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html, re.DOTALL):
            href, text = match.group(1), match.group(2)
            text = re.sub(r"<[^>]+>", "", text).strip()
            if href.startswith("/") and base_url:
                href = base_url.rstrip("/") + href
            if href and not href.startswith(("#", "javascript:", "mailto:")):
                links.append({"url": href, "text": text[:100]})
        return links

    def extract_headings(self, html: str) -> list[dict]:
        """Extract all headings (h1-h6) from HTML."""
        headings = []
        for match in re.finditer(r"<h([1-6])[^>]*>(.*?)</h\1>", html, re.DOTALL):
            level = int(match.group(1))
            text = re.sub(r"<[^>]+>", "", match.group(2)).strip()
            if text:
                headings.append({"level": level, "text": text})
        return headings

    def extract_metadata(self, html: str) -> dict:
        """Extract page metadata (title, description, og tags)."""
        meta = {}

        # Title
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.DOTALL)
        meta["title"] = title_match.group(1).strip() if title_match else ""

        # Meta tags
        for match in re.finditer(r'<meta\s[^>]*(?:name|property)=["\']([^"\']+)["\'][^>]*content=["\']([^"\']+)["\']', html):
            meta[match.group(1)] = match.group(2)

        # Also match reversed attribute order
        for match in re.finditer(r'<meta\s[^>]*content=["\']([^"\']+)["\'][^>]*(?:name|property)=["\']([^"\']+)["\']', html):
            meta[match.group(2)] = match.group(1)

        return meta

    def extract_tables(self, html: str) -> list[list[list[str]]]:
        """Extract tables from HTML as lists of rows."""
        tables = []
        for table_match in re.finditer(r"<table[^>]*>(.*?)</table>", html, re.DOTALL):
            table_html = table_match.group(1)
            rows = []
            for row_match in re.finditer(r"<tr[^>]*>(.*?)</tr>", table_html, re.DOTALL):
                cells = []
                for cell_match in re.finditer(r"<t[dh][^>]*>(.*?)</t[dh]>", row_match.group(1), re.DOTALL):
                    cell_text = re.sub(r"<[^>]+>", "", cell_match.group(1)).strip()
                    cells.append(cell_text)
                if cells:
                    rows.append(cells)
            if rows:
                tables.append(rows)
        return tables

    def scrape(self, url: str, extract: list[str] = None) -> dict:
        """
        Scrape a URL and extract structured data.

        Args:
            url: The URL to scrape
            extract: List of what to extract. Options:
                     ["text", "links", "headings", "metadata", "tables"]
                     Default: all

        Returns:
            dict with extracted data
        """
        extract = extract or ["text", "links", "headings", "metadata", "tables"]

        html = self.fetch(url)
        result = {"url": url, "html_size": len(html)}

        if "text" in extract:
            result["text"] = self.extract_text(html)
            result["text_size"] = len(result["text"])

        if "links" in extract:
            result["links"] = self.extract_links(html, url.split("/")[0] + "//" + url.split("/")[2])
            result["link_count"] = len(result["links"])

        if "headings" in extract:
            result["headings"] = self.extract_headings(html)

        if "metadata" in extract:
            result["metadata"] = self.extract_metadata(html)

        if "tables" in extract:
            result["tables"] = self.extract_tables(html)
            result["table_count"] = len(result["tables"])

        return result

    def crawl(self, start_url: str, max_pages: int = 5, same_domain: bool = True) -> list[dict]:
        """
        Crawl multiple pages starting from a URL.

        Args:
            start_url: Starting URL
            max_pages: Maximum pages to crawl
            same_domain: Only follow links on the same domain
        """
        from urllib.parse import urlparse

        domain = urlparse(start_url).netloc
        visited = set()
        queue = [start_url]
        results = []

        while queue and len(results) < max_pages:
            url = queue.pop(0)
            if url in visited:
                continue

            visited.add(url)
            try:
                result = self.scrape(url, extract=["text", "links", "metadata"])
                results.append(result)
                print(f"  ✅ [{len(results)}/{max_pages}] {url} ({result.get('text_size', 0)} chars)")

                # Add new links to queue
                for link in result.get("links", []):
                    link_url = link["url"]
                    if same_domain:
                        try:
                            if urlparse(link_url).netloc == domain and link_url not in visited:
                                queue.append(link_url)
                        except Exception:
                            pass
            except Exception as e:
                print(f"  ❌ {url}: {e}")

        return results


if __name__ == "__main__":
    agent = WebScrapingAgent()

    print("🕷️ Web Scraping Agent — Demo")
    print("=" * 50)

    # Single page scrape
    result = agent.scrape("https://example.com")
    print(f"\n📄 Title: {result['metadata'].get('title', 'N/A')}")
    print(f"📝 Text: {result['text'][:200]}...")
    print(f"🔗 Links: {result['link_count']}")
    print(f"📑 Headings: {len(result['headings'])}")
    print(f"📊 Tables: {result['table_count']}")

    print("\n✅ Web Scraping Agent ready!")
