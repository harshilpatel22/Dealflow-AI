
from __future__ import annotations
from typing import Iterable
import re
import feedparser
from bs4 import BeautifulSoup
from core import Row, now_iso

def _clean_html_to_text(html: str) -> str:
    return BeautifulSoup(html or "", "html.parser").get_text(" ", strip=True)

def _get_entry_summary(e) -> str:
    """Prefer summary_detail.value if present, else summary, else description."""
    try:
        sd = getattr(e, "summary_detail", None)
        if sd and isinstance(sd, dict):
            val = sd.get("value", "") or ""
            if val:
                return val
    except Exception:
        pass
    return e.get("summary", "") or e.get("description", "") or ""

def _extract_company_and_amount(title: str) -> tuple[str, str]:
    """
    Heuristic: extract 'Company raises $X ...' from common headline patterns.
    Returns (company_name, amount_string) where amount_string may be '' if not found.
    """
    t = title or ""
    m = re.search(r"^\s*([A-Za-z0-9&\-\.,'! ]{2,80}?)\s+(raises|secures|lands|snags|bags)\s+([$\€£]\s?[0-9.,]+[MB]?(?:\s*million|\s*billion)?)", t, flags=re.I)
    if m:
        comp = m.group(1).strip(" -:–")
        amt = m.group(3).replace(" ", "")
        return comp, amt
    comp = re.split(r"\s[-–:]\s", t, 1)[0].strip()
    return comp, ""

class CrunchbaseNewsScraper:
    """https://news.crunchbase.com/feed/"""
    def __init__(self):
        self.feed_url = "https://news.crunchbase.com/feed/"

    def run(self) -> Iterable[Row]:
        feed = feedparser.parse(self.feed_url)
        for e in feed.entries[:40]:
            title = e.get("title", "")
            summary = _clean_html_to_text(_get_entry_summary(e))
            name, amt = _extract_company_and_amount(title)
            desc = (f"{summary} (Amount: {amt})" if amt and amt not in summary else summary) or title
            link = e.get("link", "")
            yield Row(
                name=name or title,
                description=desc[:600],
                website=link,
                source="CrunchbaseNews",
                source_url=link,
                discovered_at=now_iso()
            )

class VentureBeatStartupsScraper:
    """https://venturebeat.com/category/startups/feed/"""
    def __init__(self):
        self.feed_url = "https://venturebeat.com/category/startups/feed/"

    def run(self) -> Iterable[Row]:
        feed = feedparser.parse(self.feed_url)
        for e in feed.entries[:40]:
            title = e.get("title", "")
            summary = _clean_html_to_text(_get_entry_summary(e))
            name, amt = _extract_company_and_amount(title)
            desc = (f"{summary} (Amount: {amt})" if amt and amt not in summary else summary) or title
            link = e.get("link", "")
            yield Row(
                name=name or title,
                description=desc[:600],
                website=link,
                source="VentureBeat",
                source_url=link,
                discovered_at=now_iso()
            )

class EUStartupsScraper:
    """https://www.eu-startups.com/feed/"""
    def __init__(self):
        self.feed_url = "https://www.eu-startups.com/feed/"

    def run(self) -> Iterable[Row]:
        feed = feedparser.parse(self.feed_url)
        for e in feed.entries[:50]:
            title = e.get("title", "")
            summary = _clean_html_to_text(_get_entry_summary(e))
            name, amt = _extract_company_and_amount(title)
            desc = (f"{summary} (Amount: {amt})" if amt and amt not in summary else summary) or title
            link = e.get("link", "")
            yield Row(
                name=name or title,
                description=desc[:600],
                website=link,
                source="EU-Startups",
                source_url=link,
                discovered_at=now_iso()
            )

class FinSMEscraper:
    """https://www.finsmes.com/feed"""
    def __init__(self):
        self.feed_url = "https://www.finsmes.com/feed"

    def run(self) -> Iterable[Row]:
        feed = feedparser.parse(self.feed_url)
        for e in feed.entries[:60]:
            title = e.get("title", "")
            summary = _clean_html_to_text(_get_entry_summary(e))
            name, amt = _extract_company_and_amount(title)
            desc = (f"{summary} (Amount: {amt})" if amt and amt not in summary else summary) or title
            link = e.get("link", "")
            yield Row(
                name=name or title,
                description=desc[:600],
                website=link,
                source="FinSMEs",
                source_url=link,
                discovered_at=now_iso()
            )
