
from __future__ import annotations
from typing import Iterable
from datetime import datetime, timedelta
import os, re
import feedparser
from bs4 import BeautifulSoup
from core import Http, Row, now_iso

# — Product Hunt (RSS) —
class ProductHuntScraper:
    def __init__(self):
        self.feed_url = "https://www.producthunt.com/feed"

    def run(self) -> Iterable[Row]:
        feed = feedparser.parse(self.feed_url)
        for e in feed.entries[:25]:
            desc = BeautifulSoup(e.get("summary",""), "html.parser").get_text(" ", strip=True)
            link = e.get("link","")
            yield Row(
                name=e.get("title",""),
                description=desc,
                website=link,          # PH discussion link (safe default)
                source="ProductHunt",
                source_url=link,
                discovered_at=now_iso()
            )

# — Hacker News: Show HN —
class HackerNewsScraper:
    def __init__(self):
        self.http = Http(headers={"Accept": "application/json"})
        self.base = "https://hacker-news.firebaseio.com/v0"

    def run(self) -> Iterable[Row]:
        ids = self.http.get(f"{self.base}/showstories.json").json()[:40]
        for sid in ids:
            item = self.http.get(f"{self.base}/item/{sid}.json").json()
            if not item:
                continue
            title = item.get("title","")
            if not title.lower().startswith("show hn:"):
                continue
            url = item.get("url") or f"https://news.ycombinator.com/item?id={sid}"
            text = item.get("text") or title
            soup = BeautifulSoup(text, "html.parser")
            desc = soup.get_text(" ", strip=True)
            yield Row(
                name=re.sub(r"^show hn:\s*", "", title, flags=re.I)[:120],
                description=desc or title,
                website=url,
                source="HackerNews",
                source_url=f"https://news.ycombinator.com/item?id={sid}",
                discovered_at=now_iso()
            )

# — TechCrunch: Startups feed —
class TechCrunchScraper:
    def __init__(self):
        self.http = Http()
        self.feed_url = "https://techcrunch.com/category/startups/feed/"

    def _company_site_from_article(self, article_url: str) -> str:
        try:
            html = self.http.get(article_url).text
            doc = BeautifulSoup(html, "html.parser")
            for a in doc.select("article a[href]"):
                href = a.get("href","")
                if href.startswith("https://techcrunch.com"):
                    continue
                if href.startswith("http"):
                    return href
            og = doc.select_one("meta[property='og:url']")
            return og.get("content","") if og else ""
        except Exception:
            return ""

    def run(self) -> Iterable[Row]:
        feed = feedparser.parse(self.feed_url)
        for e in feed.entries[:25]:
            title = e.get("title","")
            summary = BeautifulSoup(e.get("summary",""), "html.parser").get_text(" ", strip=True)
            text = f"{title} {summary}".lower()
            if not any(k in text for k in ("raises","funding","launches","announces","seed","series a","series b","series c")):
                continue
            m = re.match(r"([^:–\-]+)\s+(raises|launches|announces)\b", title, flags=re.I)
            name = (m.group(1).strip() if m else title.split(" – ")[0].split(":")[0]).strip()
            website = self._company_site_from_article(e.get("link",""))
            yield Row(
                name=name,
                description=summary[:500],
                website=website or e.get("link",""),
                source="TechCrunch",
                source_url=e.get("link",""),
                discovered_at=now_iso()
            )

# — GitHub: recent popular repos —
class GitHubScraper:
    def __init__(self):
        token = os.getenv("GITHUB_TOKEN","")
        headers = {"Accept": "application/vnd.github+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.http = Http(headers=headers, timeout=25)
        self.api = "https://api.github.com"

    def run(self) -> Iterable[Row]:
        date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
        params = {
            "q": f"created:>{date} stars:>50",
            "sort": "stars",
            "order": "desc",
            "per_page": 50,
        }
        data = self.http.get(f"{self.api}/search/repositories", params=params).json()
        for repo in data.get("items", []):
            desc = (repo.get("description") or "").strip()
            homepage = (repo.get("homepage") or "").strip()
            if not homepage or len(desc) < 20:
                continue
            yield Row(
                name=repo["name"].replace("-"," ").title(),
                description=desc,
                website=homepage,
                source="GitHub",
                source_url=repo.get("html_url",""),
                discovered_at=now_iso()
            )

class DevToScraper:
    def _init_(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def get_startups(self) -> List[Dict]:
        startups = []
        
        try:
            url = "https://dev.to/api/articles?tag=startup&per_page=15"
            print(f"Fetching live data from: {url}")
            
            response = requests.get(url, headers=self.headers, timeout=10)
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                articles = response.json()
                print(f"Found {len(articles)} startup articles")
                
                for article in articles:
                    try:
                        title = article.get('title', '')
                        description = article.get('description', '') or title
                        url_link = article.get('url', '')
                        
                        name = self._extract_company_from_title(title)
                        
                        if len(name) > 2:
                            startup = {
                                "name": name,
                                "description": description[:400],
                                "website": url_link,
                                "source": "Dev.to",
                                "source_url": url_link,
                                "discovered_at": datetime.now()
                            }
                            startups.append(startup)
                            print(f" Scraped: {name}")
                        
                    except Exception as e:
                        print(f" Error parsing article: {e}")
                        continue
        
        except Exception as e:
            print(f" Error scraping Dev.to: {e}")
        
        return startups
    
    def _extract_company_from_title(self, title):
        stop_words = ['how', 'i', 'built', 'my', 'our', 'the', 'a', 'an', 'why', 'what', 'building']
        words = [w for w in title.lower().split() if w not in stop_words]
        return ' '.join(words[:3]).title() if words else title.split()[0]