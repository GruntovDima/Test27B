"""One-time bundle authoring; the delivered checker never downloads texts."""
import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parent / "sources"
root.mkdir(parents=True, exist_ok=True)


def fetch(number):
    url = f"https://www.gutenberg.org/cache/epub/{number}/pg{number}.txt"
    with urllib.request.urlopen(url, timeout=60) as response:
        raw = response.read()
    text = raw.decode("utf-8-sig").replace("\r\n", "\n")
    (root / f"pg{number}.txt").write_text(text, encoding="utf-8")
    return {"id": number, "url": url, "sha256": hashlib.sha256(raw).hexdigest(),
            "normalized_utf8_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "characters": len(text), "catalog": f"https://www.gutenberg.org/ebooks/{number}"}


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    rows = list(pool.map(fetch, [11, 55, 120, 1661]))
(root / "provenance.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
for row in rows:
    print(row["id"], row["characters"], row["sha256"])
