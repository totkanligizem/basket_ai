from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://www.trendyol.com"
OUT_PATH = Path("data/generated/category_trees/trendyol_category_tree.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

SLEEP = 1.5

# Trendyol bazen /sr gibi endpoint'leri bloklar. Daha "site ana sayfa" üzerinden kategori linkleri almak daha stabil.
START_URL = "https://www.trendyol.com/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Referer": "https://www.google.com/",
}

@dataclass
class Node:
    name: str
    url: str

def fetch(session: requests.Session, url: str) -> str:
    r = session.get(url, headers=HEADERS, timeout=30)
    # Bazı durumlarda 403 yerine 200 dönüp bot sayfası verir; yine de kontrol edeceğiz.
    if r.status_code == 403:
        raise requests.HTTPError(f"403 Forbidden for {url}", response=r)
    r.raise_for_status()
    return r.text

def parse_category_links(html: str) -> list[Node]:
    soup = BeautifulSoup(html, "lxml")
    nodes: list[Node] = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = (a.get_text() or "").strip()
        if not text:
            continue

        # Trendyol kategori link pattern: /<slug>-x-c<id>
        if "-x-c" in href and href.startswith("/"):
            url = BASE + href.split("?")[0]
            nodes.append(Node(name=text, url=url))

    # unique
    uniq = {}
    for n in nodes:
        uniq[(n.name, n.url)] = n
    return list(uniq.values())

def main() -> None:
    session = requests.Session()

    print("[info] fetching homepage")
    html = fetch(session, START_URL)

    cats = parse_category_links(html)
    if not cats:
        raise RuntimeError("No category links found on homepage (markup may have changed or bot page served).")

    # limit: çok büyütmeyelim, pipeline stabil olsun
    cats = cats[:80]

    rows = []
    for c in cats:
        rows.append({
            "level1": c.name,
            "level2": None,
            "level3": None,
            "url_level1": c.url,
            "url_level2": None,
            "url_level3": None
        })
        time.sleep(SLEEP)

    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"[done] wrote: {OUT_PATH} rows={len(rows)}")

if __name__ == "__main__":
    main()
