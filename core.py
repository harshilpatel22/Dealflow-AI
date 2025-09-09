
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from datetime import datetime
from urllib.parse import urlparse
import time, random, logging, requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def norm_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""

def jitter(base: float = 0.6, spread: float = 0.8) -> float:
    return base + random.random() * spread

class Http:
    """Simple HTTP client with retries/backoff and a polite default User-Agent."""
    def __init__(self, headers: Optional[Dict[str, str]] = None, timeout: int = 20, retries: int = 3):
        self.s = requests.Session()
        self.s.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; StartupScout/1.0; +mailto:contact@example.com)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            **(headers or {})
        })
        self.timeout = timeout
        self.retries = retries

    def get(self, url: str, **kw) -> requests.Response:
        last_exc = None
        for i in range(self.retries):
            try:
                r = self.s.get(url, timeout=self.timeout, **kw)
                if r.status_code in (429, 500, 502, 503, 504):
                    sleep_s = 2**i + jitter()
                    logging.warning(f"{r.status_code} for {url}; backing off {sleep_s:.1f}s")
                    time.sleep(sleep_s)
                    continue
                r.raise_for_status()
                return r
            except requests.RequestException as e:
                last_exc = e
                sleep_s = 1.5**i + jitter()
                logging.warning(f"net err {e} on {url}; retry in {sleep_s:.1f}s")
                time.sleep(sleep_s)
        raise last_exc

@dataclass
class Row:
    name: str
    description: str
    website: str
    source: str
    source_url: str
    discovered_at: str

    def serialize(self) -> Dict[str, Any]:
        d = asdict(self)
        d["name"] = (d["name"] or "").strip()[:200]
        d["description"] = (d["description"] or "").strip()[:2000]
        d["website"] = (d["website"] or "").strip()
        d["source_url"] = (d["source_url"] or "").strip()
        return d
