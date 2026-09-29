import io
from urllib.error import HTTPError

import pytest

from agents import web_scraping_agent as wsa
from agents.web_scraping_agent import WebScrapingAgent

HTML = """<html><head><title> Demo </title>
<meta name="description" content="A demo page">
<meta content="Card" property="og:title">
<style>p{color:red}</style><script>var x=1;</script></head>
<body><h1>Main</h1><h2>Sub <b>bold</b></h2>
<p>Fish &amp; chips</p>
<a href="/about">About</a> <a href="https://other.com/x">Other</a>
<a href="#top">Top</a> <a href="mailto:a@b.c">Mail</a>
<table><tr><th>k</th><th>v</th></tr><tr><td>a</td><td>1</td></tr></table>
</body></html>"""


@pytest.fixture
def agent():
    return WebScrapingAgent()


def test_extract_text_strips_scripts_styles_and_decodes(agent):
    text = agent.extract_text(HTML)
    assert "var x" not in text and "color:red" not in text
    assert "Fish & chips" in text


def test_extract_links_resolves_relative_and_drops_junk(agent):
    links = agent.extract_links(HTML, "https://site.com")
    assert [x["url"] for x in links] == ["https://site.com/about", "https://other.com/x"]


def test_extract_headings(agent):
    assert agent.extract_headings(HTML) == [
        {"level": 1, "text": "Main"}, {"level": 2, "text": "Sub bold"}]


def test_extract_metadata_handles_both_attribute_orders(agent):
    meta = agent.extract_metadata(HTML)
    assert meta["title"] == "Demo"
    assert meta["description"] == "A demo page"
    assert meta["og:title"] == "Card"


def test_extract_tables(agent):
    assert agent.extract_tables(HTML) == [[["k", "v"], ["a", "1"]]]


def _fake_urlopen(pages, calls=None):
    def opener(req, timeout=None):
        url = req.full_url
        if calls is not None:
            calls.append(url)
        if url not in pages:
            raise HTTPError(url, 404, "Not Found", {}, None)
        return io.BytesIO(pages[url].encode())
    return opener


def test_fetch_caches_and_sets_user_agent(agent, monkeypatch):
    calls = []
    monkeypatch.setattr(wsa, "urlopen", _fake_urlopen({"https://a.com/": "<p>x</p>"}, calls))
    agent.fetch("https://a.com/")
    agent.fetch("https://a.com/")
    assert calls == ["https://a.com/"]


def test_fetch_wraps_http_errors(agent, monkeypatch):
    monkeypatch.setattr(wsa, "urlopen", _fake_urlopen({}))
    with pytest.raises(RuntimeError, match="HTTP 404"):
        agent.fetch("https://a.com/missing")


def test_scrape_selects_requested_fields(agent, monkeypatch):
    monkeypatch.setattr(wsa, "urlopen", _fake_urlopen({"https://a.com/p": HTML}))
    out = agent.scrape("https://a.com/p", extract=["headings", "tables"])
    assert "headings" in out and out["table_count"] == 1
    assert "text" not in out and "links" not in out


def test_crawl_stays_on_domain_and_respects_max_pages(agent, monkeypatch):
    pages = {
        "https://a.com/": '<a href="https://a.com/1">1</a><a href="https://b.com/">off</a>',
        "https://a.com/1": '<a href="https://a.com/">home</a>',
        "https://b.com/": "<p>should never be fetched</p>",
    }
    calls = []
    monkeypatch.setattr(wsa, "urlopen", _fake_urlopen(pages, calls))
    results = agent.crawl("https://a.com/", max_pages=5)
    assert [r["url"] for r in results] == ["https://a.com/", "https://a.com/1"]
    assert "https://b.com/" not in calls
