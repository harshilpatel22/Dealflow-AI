
from typing import List, Dict
import csv, json, pathlib, logging

def write_csv(path: str, rows: List[Dict]):
    p = pathlib.Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    write_header = not p.exists()
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if write_header:
            w.writeheader()
        w.writerows(rows)
    logging.info(f"Wrote {len(rows)} rows → {p}")

def write_jsonl(path: str, rows: List[Dict]):
    p = pathlib.Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
