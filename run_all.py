
from typing import List, Dict
from core import norm_domain
from writers import write_csv, write_jsonl
from scrapers import ProductHuntScraper, HackerNewsScraper, TechCrunchScraper, GitHubScraper, DevToScraper
from funding_scrapers import CrunchbaseNewsScraper, VentureBeatStartupsScraper, EUStartupsScraper, FinSMEscraper

def dedupe(rows: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for r in rows:
        key = (norm_domain(r.get("website","")) or norm_domain(r.get("source_url","")), r.get("name","").lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out

def main():
    rows = []
    sources = [
        ProductHuntScraper(),
        HackerNewsScraper(),
        TechCrunchScraper(),
        GitHubScraper(),
        CrunchbaseNewsScraper(),
        VentureBeatStartupsScraper(),
        EUStartupsScraper(),
        FinSMEscraper(),
        DevToScraper()
    ]
    for src in sources:
        for row in src.run():
            rows.append(row.serialize())

    rows = dedupe(rows)
    write_csv("out/startups.csv", rows)
    write_jsonl("out/startups.jsonl", rows)
    print(f"total rows: {len(rows)}")

if __name__ == "__main__":
    main()
