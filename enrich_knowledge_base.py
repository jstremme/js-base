#!/usr/bin/env python3
"""
Enrich Joel Stremmel's knowledge base with paper metadata.

- Arxiv papers: fetches abstract, date, authors via arxiv API
- ACL Anthology papers: scrapes abstract from HTML, extracts year from URL
- Other papers: extracts year from URL heuristics
- Non-paper items: sets summary/date/authors to null
- Idempotent: skips items that already have a non-null summary
- Final step: regenerates the HTML page
"""

import argparse
import http.server
import json
import re
import socketserver
import ssl
import time
import threading
import urllib.request
import urllib.parse
import webbrowser
from pathlib import Path
from xml.etree import ElementTree as ET

BASE_DIR = Path(__file__).resolve().parent
JSON_PATH = BASE_DIR / "joel_stremmel_knowledge_base.json"
HTML_PATH = BASE_DIR / "joel_stremmel_knowledge_base.html"

ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
ARXIV_API = "http://export.arxiv.org/api/query"
BATCH_SIZE = 50
API_DELAY = 3  # seconds between arxiv batches


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_arxiv_id(url: str) -> str | None:
    """Extract arxiv paper ID from a URL like arxiv.org/abs/2402.12329 or /pdf/2402.12329."""
    m = re.search(r'arxiv\.org/(?:abs|pdf|html)/(\d+\.\d+)(?:v\d+)?', url)
    return m.group(1) if m else None


def truncate_to_sentences(text: str, max_sentences: int = 3) -> str:
    """Truncate text to approximately max_sentences sentences."""
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= max_sentences:
        return text
    return ' '.join(sentences[:max_sentences])


def extract_year_from_url(url: str) -> str | None:
    """Try to extract a 4-digit year from a URL."""
    # ACL patterns like 2024.emnlp-main.557 or D15-1013
    m = re.search(r'/(\d{4})\.\w+', url)
    if m:
        return m.group(1)
    # Old ACL pattern like D15-1013 -> 2015
    m = re.search(r'/[A-Z](\d{2})-\d+', url)
    if m:
        yr = int(m.group(1))
        return str(2000 + yr) if yr < 50 else str(1900 + yr)
    # Generic 4-digit year in URL
    m = re.search(r'[/\-_.](\d{4})[/\-_.]', url)
    if m:
        yr = int(m.group(1))
        if 1990 <= yr <= 2030:
            return str(yr)
    # Year at end of path
    m = re.search(r'/(\d{4})/?$', url)
    if m:
        yr = int(m.group(1))
        if 1990 <= yr <= 2030:
            return str(yr)
    return None


# ---------------------------------------------------------------------------
# Arxiv enrichment
# ---------------------------------------------------------------------------

def fetch_arxiv_batch(ids: list[str]) -> dict[str, dict]:
    """Fetch metadata for a batch of arxiv IDs. Returns {id: {summary, date, authors}}."""
    id_list = ",".join(ids)
    params = urllib.parse.urlencode({
        "id_list": id_list,
        "max_results": len(ids),
    })
    url = f"{ARXIV_API}?{params}"

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
        xml_data = resp.read()

    root = ET.fromstring(xml_data)
    results = {}

    for entry in root.findall("atom:entry", ARXIV_NS):
        entry_id_el = entry.find("atom:id", ARXIV_NS)
        if entry_id_el is None:
            continue
        entry_id = entry_id_el.text.strip()
        # Extract just the ID part
        m = re.search(r'(\d+\.\d+)', entry_id)
        if not m:
            continue
        arxiv_id = m.group(1)

        summary_el = entry.find("atom:summary", ARXIV_NS)
        published_el = entry.find("atom:published", ARXIV_NS)
        authors = [
            a.find("atom:name", ARXIV_NS).text.strip()
            for a in entry.findall("atom:author", ARXIV_NS)
            if a.find("atom:name", ARXIV_NS) is not None
        ]

        summary = truncate_to_sentences(summary_el.text.strip()) if summary_el is not None else None
        date = published_el.text[:10] if published_el is not None else None

        results[arxiv_id] = {
            "summary": summary,
            "date": date,
            "authors": authors if authors else None,
        }

    return results


def enrich_arxiv(items_by_id: dict[str, list]) -> int:
    """Enrich all arxiv items. items_by_id maps arxiv_id -> [item references].
    Returns count of enriched items."""
    all_ids = list(items_by_id.keys())
    enriched = 0

    for i in range(0, len(all_ids), BATCH_SIZE):
        batch = all_ids[i:i + BATCH_SIZE]
        print(f"  Fetching arxiv batch {i // BATCH_SIZE + 1} ({len(batch)} papers)...")
        try:
            results = fetch_arxiv_batch(batch)
        except Exception as e:
            print(f"    Error fetching batch: {e}")
            continue

        for arxiv_id in batch:
            if arxiv_id in results:
                meta = results[arxiv_id]
                for item in items_by_id[arxiv_id]:
                    item["summary"] = meta["summary"]
                    item["date"] = meta["date"]
                    item["authors"] = meta["authors"]
                    enriched += 1
            else:
                # Not found in API response — set nulls
                for item in items_by_id[arxiv_id]:
                    item.setdefault("summary", None)
                    item.setdefault("date", None)
                    item.setdefault("authors", None)

        if i + BATCH_SIZE < len(all_ids):
            print(f"    Waiting {API_DELAY}s before next batch...")
            time.sleep(API_DELAY)

    return enriched


# ---------------------------------------------------------------------------
# ACL Anthology enrichment
# ---------------------------------------------------------------------------

def fetch_acl_abstract(url: str) -> str | None:
    """Fetch the abstract from an ACL Anthology page."""
    # Disable SSL verification (ACL Anthology sometimes has cert issues)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"    Failed to fetch {url}: {e}")
        return None

    # Look for abstract in the HTML
    # ACL uses <div class="acl-abstract"> or <span class="d-block">
    patterns = [
        r'<div[^>]*class="acl-abstract"[^>]*>.*?<span[^>]*>(.*?)</span>',
        r'<div[^>]*class="card-body acl-abstract"[^>]*>.*?<span[^>]*>(.*?)</span>',
        r'<abstract[^>]*>(.*?)</abstract>',
        r'<h\d[^>]*>Abstract</h\d>\s*<p>(.*?)</p>',
    ]
    for pat in patterns:
        m = re.search(pat, html, re.DOTALL | re.IGNORECASE)
        if m:
            abstract = re.sub(r'<[^>]+>', '', m.group(1)).strip()
            return truncate_to_sentences(abstract)

    return None


def extract_acl_year(url: str) -> str | None:
    """Extract year from ACL Anthology URL patterns."""
    # Pattern: /2024.emnlp-main.557/
    m = re.search(r'/(\d{4})\.\w+', url)
    if m:
        return m.group(1)
    # Pattern: /D15-1013/ -> 2015
    m = re.search(r'/([A-Z])(\d{2})-\d+', url)
    if m:
        yr = int(m.group(2))
        return str(2000 + yr) if yr < 50 else str(1900 + yr)
    return None


def extract_acl_authors(html: str) -> list[str] | None:
    """Try to extract author names from ACL page HTML."""
    # Look for author links
    authors = re.findall(r'<a[^>]*href="/people/[^"]*"[^>]*>([^<]+)</a>', html)
    if authors:
        return [a.strip() for a in authors if a.strip()]
    return None


def enrich_acl(items: list[dict]) -> int:
    """Enrich ACL Anthology items. Returns count enriched."""
    enriched = 0
    for item in items:
        url = item["url"]
        print(f"  Fetching ACL: {url}")

        year = extract_acl_year(url)
        item["date"] = f"{year}-01-01" if year else None

        abstract = fetch_acl_abstract(url)
        if abstract:
            item["summary"] = abstract
            enriched += 1
        else:
            item.setdefault("summary", None)

        # Try to get authors from page
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
                html = resp.read().decode("utf-8", errors="replace")
            authors = extract_acl_authors(html)
            item["authors"] = authors
        except Exception:
            item.setdefault("authors", None)

        time.sleep(1)  # be polite

    return enriched


# ---------------------------------------------------------------------------
# Other papers + non-papers
# ---------------------------------------------------------------------------

def enrich_other_papers(items: list[dict]) -> int:
    """For non-arxiv, non-ACL papers: extract year from URL."""
    count = 0
    for item in items:
        year = extract_year_from_url(item["url"])
        item["date"] = f"{year}-01-01" if year else None
        item.setdefault("summary", None)
        item.setdefault("authors", None)
        count += 1
    return count


def set_null_fields(items: list[dict]):
    """Set summary/date/authors to null for non-paper items."""
    for item in items:
        item.setdefault("summary", None)
        item.setdefault("date", None)
        item.setdefault("authors", None)


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def generate_html(data: dict) -> str:
    """Generate the complete HTML page with enriched metadata display."""
    json_data = json.dumps(data, ensure_ascii=False)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Joel Stremmel — Knowledge Base</title>
<style>
  :root {{
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #232733;
    --border: #2e3343;
    --text: #e1e4ed;
    --text2: #8b90a0;
    --accent: #6c8cff;
    --accent2: #4a6adf;
    --paper: #e8a44a;
    --repo: #7ee787;
    --blog: #d2a8ff;
    --video: #ff7b72;
    --tool: #79c0ff;
    --pod: #ffa657;
    --doc: #56d4dd;
    --news: #f778ba;
    --other: #8b949e;
    --pending: #ffd700;
  }}

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
    min-height: 100vh;
  }}

  .header {{
    position: sticky;
    top: 0;
    z-index: 100;
    background: var(--bg);
    border-bottom: 1px solid var(--border);
    padding: 16px 24px;
  }}

  .header-inner {{
    max-width: 960px;
    margin: 0 auto;
  }}

  .header h1 {{
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 4px;
    color: var(--text);
  }}
  .header h1 span {{ color: var(--text2); font-weight: 400; }}

  .instructions {{
    font-size: 12px;
    color: var(--text2);
    margin-bottom: 12px;
    line-height: 1.4;
  }}
  .instructions code {{
    background: var(--surface2);
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 11px;
  }}

  .controls {{
    display: flex;
    gap: 10px;
    align-items: center;
    flex-wrap: wrap;
  }}

  #search {{
    flex: 1;
    min-width: 200px;
    padding: 8px 12px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    font-size: 14px;
    outline: none;
    transition: border-color 0.15s;
  }}
  #search:focus {{ border-color: var(--accent); }}
  #search::placeholder {{ color: var(--text2); }}

  .filter-bar {{
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
  }}

  .filter-btn {{
    padding: 4px 10px;
    font-size: 12px;
    border-radius: 12px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--text2);
    cursor: pointer;
    transition: all 0.15s;
  }}
  .filter-btn:hover {{ border-color: var(--accent); color: var(--text); }}
  .filter-btn.active {{ background: var(--accent2); border-color: var(--accent); color: #fff; }}

  .stats {{
    font-size: 12px;
    color: var(--text2);
    padding: 2px 0;
    white-space: nowrap;
  }}

  .legend {{
    display: flex;
    gap: 16px;
    align-items: center;
    font-size: 11px;
    color: var(--text2);
    margin-top: 8px;
  }}
  .legend-item {{ display: flex; align-items: center; gap: 4px; }}
  .legend-dot {{
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }}
  .legend-dot.filled {{ background: var(--accent); }}
  .legend-dot.empty {{ border: 1.5px solid var(--text2); }}
  .legend-dot.pending-dot {{ background: var(--pending); }}

  /* --- Add Bookmark Panel --- */
  .bookmark-toggle {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 10px;
    font-size: 12px;
    border-radius: 12px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--text2);
    cursor: pointer;
    transition: all 0.15s;
  }}
  .bookmark-toggle:hover {{ border-color: var(--accent); color: var(--text); }}
  .bookmark-toggle.active {{ background: var(--accent2); border-color: var(--accent); color: #fff; }}

  .bookmark-panel {{
    display: none;
    max-width: 960px;
    margin: 0 auto;
    padding: 16px 24px;
    border-bottom: 1px solid var(--border);
  }}
  .bookmark-panel.open {{ display: block; }}

  .bookmark-panel .form-row {{
    display: flex;
    gap: 10px;
    margin-bottom: 8px;
    align-items: center;
    flex-wrap: wrap;
  }}

  .bookmark-panel input[type="text"],
  .bookmark-panel textarea,
  .bookmark-panel select {{
    padding: 6px 10px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    font-size: 13px;
    outline: none;
    font-family: inherit;
  }}
  .bookmark-panel input[type="text"]:focus,
  .bookmark-panel textarea:focus,
  .bookmark-panel select:focus {{ border-color: var(--accent); }}

  .bookmark-panel input[type="text"] {{ flex: 1; min-width: 200px; }}
  .bookmark-panel textarea {{ flex: 1; min-width: 200px; height: 60px; resize: vertical; }}
  .bookmark-panel select {{ min-width: 140px; }}

  .bookmark-panel label {{
    font-size: 12px;
    color: var(--text2);
    min-width: 70px;
  }}

  .btn {{
    padding: 6px 14px;
    font-size: 12px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--text);
    cursor: pointer;
    transition: all 0.15s;
    font-family: inherit;
  }}
  .btn:hover {{ border-color: var(--accent); color: var(--accent); }}
  .btn-primary {{
    background: var(--accent2);
    border-color: var(--accent);
    color: #fff;
  }}
  .btn-primary:hover {{ background: var(--accent); }}
  .bookmark-actions {{
    display: flex;
    gap: 8px;
    margin-top: 4px;
  }}

  /* --- Remove button --- */
  .remove-btn {{
    opacity: 0;
    font-size: 11px;
    color: var(--text2);
    background: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 2px 6px;
    cursor: pointer;
    transition: all 0.15s;
    flex-shrink: 0;
  }}
  .item-row:hover .remove-btn {{ opacity: 1; }}
  .remove-btn:hover {{
    color: #ff6b6b;
    border-color: #ff6b6b;
    background: rgba(255,107,107,0.1);
  }}

  .item-row.removed {{
    opacity: 0.4;
    text-decoration: line-through;
  }}
  .item-row.removed .remove-btn {{
    opacity: 1;
    color: var(--accent);
    border-color: var(--accent);
  }}

  .removed-badge {{
    font-size: 9px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 1px 5px;
    border-radius: 3px;
    background: rgba(255,107,107,0.15);
    color: #ff6b6b;
    flex-shrink: 0;
  }}

  /* --- Main content --- */
  .container {{
    max-width: 960px;
    margin: 0 auto;
    padding: 16px 24px 80px;
  }}

  .category {{
    margin-bottom: 8px;
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }}
  .category.hidden {{ display: none; }}

  .cat-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 16px;
    background: var(--surface);
    cursor: pointer;
    user-select: none;
    transition: background 0.15s;
  }}
  .cat-header:hover {{ background: var(--surface2); }}

  .cat-title {{
    font-size: 14px;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 8px;
  }}

  .cat-count {{
    font-size: 11px;
    color: var(--text2);
    background: var(--surface2);
    padding: 2px 8px;
    border-radius: 10px;
  }}

  .chevron {{
    font-size: 12px;
    color: var(--text2);
    transition: transform 0.2s;
  }}
  .category.open .chevron {{ transform: rotate(90deg); }}

  .cat-items {{
    display: none;
    border-top: 1px solid var(--border);
  }}
  .category.open .cat-items {{ display: block; }}

  .item-row {{
    border-bottom: 1px solid var(--border);
    transition: background 0.1s;
  }}
  .item-row:last-child {{ border-bottom: none; }}
  .item-row:hover {{ background: var(--surface); }}
  .item-row.hidden {{ display: none; }}

  .item {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 16px;
    cursor: pointer;
  }}

  .enrichment-dot {{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
  }}
  .enrichment-dot.filled {{ background: var(--accent); }}
  .enrichment-dot.empty {{ border: 1.5px solid var(--text2); }}

  .type-badge {{
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 1px 6px;
    border-radius: 3px;
    flex-shrink: 0;
    width: 52px;
    text-align: center;
  }}

  .type-paper {{ background: rgba(232,164,74,0.15); color: var(--paper); }}
  .type-repo {{ background: rgba(126,231,135,0.15); color: var(--repo); }}
  .type-blog {{ background: rgba(210,168,255,0.15); color: var(--blog); }}
  .type-video {{ background: rgba(255,123,114,0.15); color: var(--video); }}
  .type-tool {{ background: rgba(121,192,255,0.15); color: var(--tool); }}
  .type-pod {{ background: rgba(255,166,87,0.15); color: var(--pod); }}
  .type-docs {{ background: rgba(86,212,221,0.15); color: var(--doc); }}
  .type-news {{ background: rgba(247,120,186,0.15); color: var(--news); }}
  .type-other {{ background: rgba(139,148,158,0.15); color: var(--other); }}

  .item-link {{
    color: var(--text);
    text-decoration: none;
    font-size: 13px;
    line-height: 1.4;
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .item-link:hover {{ color: var(--accent); }}

  .date-badge {{
    font-size: 10px;
    color: var(--text2);
    background: var(--surface2);
    padding: 1px 6px;
    border-radius: 3px;
    flex-shrink: 0;
    white-space: nowrap;
  }}

  .authors-inline {{
    font-size: 11px;
    color: var(--text2);
    flex-shrink: 0;
    max-width: 180px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}

  .item-domain {{
    font-size: 11px;
    color: var(--text2);
    flex-shrink: 0;
    max-width: 120px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}

  .pending-badge {{
    font-size: 9px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 1px 5px;
    border-radius: 3px;
    background: rgba(255,215,0,0.15);
    color: var(--pending);
    flex-shrink: 0;
  }}

  .item-detail {{
    display: none;
    padding: 6px 16px 10px 38px;
    font-size: 12px;
    color: var(--text2);
    line-height: 1.5;
    border-top: 1px dashed var(--border);
  }}
  .item-row.expanded .item-detail {{ display: block; }}
  .item-detail .detail-summary {{ margin-bottom: 4px; }}
  .item-detail .detail-authors {{ font-style: italic; }}

  .no-results {{
    text-align: center;
    padding: 60px 20px;
    color: var(--text2);
    font-size: 14px;
    display: none;
  }}

  .keyboard-hint {{
    font-size: 11px;
    color: var(--text2);
    padding: 2px 0;
  }}
  kbd {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 0 4px;
    font-size: 11px;
    font-family: inherit;
  }}

  @media (max-width: 640px) {{
    .header {{ padding: 12px 16px; }}
    .container {{ padding: 12px 16px 60px; }}
    .item {{ padding: 6px 12px; }}
    .item-domain, .authors-inline {{ display: none; }}
    .controls {{ flex-direction: column; align-items: stretch; }}
    .bookmark-panel {{ padding: 12px 16px; }}
    .bookmark-panel .form-row {{ flex-direction: column; }}
  }}
</style>
</head>
<body>

<div class="header">
  <div class="header-inner">
    <h1>Joel Stremmel <span>— Knowledge Base</span></h1>
    <p class="instructions">Click "+ Add" to bookmark new links, or hover any item and click "Remove" to delete it. Run <code>python enrich_knowledge_base.py --serve</code> for live editing, or use "Export JSON" to save manually.</p>
    <div class="controls">
      <input type="text" id="search" placeholder="Search titles, summaries, authors, URLs..." autofocus>
      <div class="filter-bar" id="filters"></div>
      <button class="bookmark-toggle" id="bookmark-toggle">+ Add</button>
      <button class="btn" id="bm-export">Export JSON</button>
      <button class="btn" id="bm-clear-removals">Clear Removals</button>
      <button class="btn btn-primary" id="bm-save-enrich" style="display:none">Save &amp; Enrich</button>
      <span class="stats" id="stats"></span>
      <span id="bm-pending-count" style="font-size:12px;color:var(--text2)"></span>
      <span id="server-status" style="font-size:11px;color:var(--text2)"></span>
    </div>
    <div class="keyboard-hint" style="margin-top:6px">
      <kbd>/</kbd> focus search &nbsp; <kbd>Esc</kbd> clear &nbsp; <kbd>E</kbd> expand all &nbsp; <kbd>C</kbd> collapse all
    </div>
  </div>
</div>

<div class="bookmark-panel" id="bookmark-panel">
  <div class="form-row">
    <label>URL</label>
    <input type="text" id="bm-url" placeholder="https://...">
  </div>
  <div class="form-row">
    <label>Title</label>
    <input type="text" id="bm-title" placeholder="Title (auto-populated if possible)">
  </div>
  <div class="form-row">
    <label>Category</label>
    <select id="bm-category"></select>
    <label>Type</label>
    <select id="bm-type">
      <option value="paper">paper</option>
      <option value="repo">repo</option>
      <option value="blog">blog</option>
      <option value="video">video</option>
      <option value="tool">tool</option>
      <option value="pod">pod</option>
      <option value="docs">docs</option>
      <option value="news">news</option>
      <option value="other" selected>other</option>
    </select>
  </div>
  <div class="form-row">
    <label>Summary</label>
    <textarea id="bm-summary" placeholder="Optional summary (2-3 sentences)"></textarea>
  </div>
  <div class="bookmark-actions">
    <button class="btn btn-primary" id="bm-add">Add Bookmark</button>
  </div>
</div>

<div class="container" id="container">
  <div class="no-results" id="no-results">No matching items.</div>
</div>

<script>
const DATA = ''' + json_data + ''';

const TYPES = ["paper","repo","blog","video","tool","pod","docs","news","other"];
const activeTypes = new Set();

// --- Pending additions from localStorage ---
function getPending() {
  try { return JSON.parse(localStorage.getItem("kb_pending_additions") || "[]"); }
  catch { return []; }
}
function savePending(arr) {
  localStorage.setItem("kb_pending_additions", JSON.stringify(arr));
}

// --- Pending removals from localStorage ---
function getRemovals() {
  try { return JSON.parse(localStorage.getItem("kb_pending_removals") || "[]"); }
  catch { return []; }
}
function saveRemovals(arr) {
  localStorage.setItem("kb_pending_removals", JSON.stringify(arr));
}
function toggleRemoval(url) {
  const removals = getRemovals();
  const idx = removals.indexOf(url);
  if (idx >= 0) {
    removals.splice(idx, 1);
  } else {
    removals.push(url);
  }
  saveRemovals(removals);
  render();
  applyFilters();
}
function getMergedData(includeRemovals = true) {
  const pending = getPending();
  const removals = getRemovals();
  const removedSet = new Set(removals);
  const merged = JSON.parse(JSON.stringify(DATA));

  // Add pending items
  pending.forEach(p => {
    const cat = merged.categories.find(c => c.name === p._category);
    if (cat) {
      const item = {...p};
      delete item._category;
      cat.items.push(item);
    }
  });

  // Remove items marked for removal (only when exporting)
  if (!includeRemovals) {
    merged.categories.forEach(cat => {
      cat.items = cat.items.filter(item => !removedSet.has(item.url));
    });
  }

  merged.metadata.total_items = 0;
  merged.categories.forEach(c => merged.metadata.total_items += c.items.length);
  return merged;
}

function getDomain(url) {
  try { return new URL(url).hostname.replace("www.", ""); } catch { return ""; }
}

function formatAuthors(authors) {
  if (!authors || authors.length === 0) return "";
  if (authors.length <= 3) return authors.join(", ");
  return authors[0] + " et al.";
}

function isEnriched(item) {
  return item.summary != null && item.summary !== "";
}

function formatYear(date) {
  if (!date) return null;
  return date.substring(0, 4);
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function render() {
  const container = document.getElementById("container");
  const noResults = document.getElementById("no-results");
  container.innerHTML = "";
  container.appendChild(noResults);

  const merged = getMergedData(true);
  const pending = getPending();
  const pendingUrls = new Set(pending.map(p => p.url));
  const removals = getRemovals();
  const removedUrls = new Set(removals);

  // Populate category dropdown
  const catSelect = document.getElementById("bm-category");
  catSelect.innerHTML = "";
  merged.categories.forEach(cat => {
    const opt = document.createElement("option");
    opt.value = cat.name;
    opt.textContent = cat.name;
    catSelect.appendChild(opt);
  });

  merged.categories.forEach((cat, ci) => {
    const div = document.createElement("div");
    div.className = "category open";
    div.dataset.index = ci;

    const header = document.createElement("div");
    header.className = "cat-header";
    header.innerHTML = `
      <span class="cat-title">${escapeHtml(cat.name)} <span class="cat-count">${cat.items.length}</span></span>
      <span class="chevron">&#9654;</span>
    `;
    header.addEventListener("click", () => div.classList.toggle("open"));

    const itemsDiv = document.createElement("div");
    itemsDiv.className = "cat-items";

    cat.items.forEach((item, ii) => {
      const row = document.createElement("div");
      row.className = "item-row";
      row.dataset.cat = ci;
      row.dataset.item = ii;
      row.dataset.type = item.type;

      const searchParts = [item.title, item.url, cat.name];
      if (item.summary) searchParts.push(item.summary);
      if (item.authors) searchParts.push(item.authors.join(" "));
      row.dataset.search = searchParts.join(" ").toLowerCase();

      const enriched = isEnriched(item);
      const domain = getDomain(item.url);
      const year = formatYear(item.date);
      const authorsStr = formatAuthors(item.authors);
      const isPending = pendingUrls.has(item.url);
      const isRemoved = removedUrls.has(item.url);

      if (isRemoved) {
        row.classList.add("removed");
      }

      // Main row
      const mainDiv = document.createElement("div");
      mainDiv.className = "item";

      let dotClass = enriched ? "filled" : "empty";
      let dotHtml = `<span class="enrichment-dot ${dotClass}"></span>`;
      let badgeHtml = `<span class="type-badge type-${item.type}">${escapeHtml(item.type)}</span>`;
      let yearHtml = year ? `<span class="date-badge">${year}</span>` : "";
      let linkHtml = `<a class="item-link" href="${escapeHtml(item.url)}" target="_blank" rel="noopener" onclick="event.stopPropagation()">${escapeHtml(item.title)}</a>`;
      let authorsHtml = authorsStr ? `<span class="authors-inline">${escapeHtml(authorsStr)}</span>` : "";
      let pendingHtml = isPending ? `<span class="pending-badge">pending</span>` : "";
      let removedHtml = isRemoved ? `<span class="removed-badge">removing</span>` : "";
      let domainHtml = `<span class="item-domain">${escapeHtml(domain)}</span>`;
      let removeBtnHtml = `<button class="remove-btn" data-url="${encodeURIComponent(item.url)}">${isRemoved ? "Undo" : "Remove"}</button>`;

      mainDiv.innerHTML = dotHtml + badgeHtml + linkHtml + yearHtml + authorsHtml + pendingHtml + removedHtml + domainHtml + removeBtnHtml;

      // Click row to expand detail (not the link)
      mainDiv.addEventListener("click", (e) => {
        if (e.target.tagName === "A") return;
        row.classList.toggle("expanded");
      });

      // Detail panel
      const detailDiv = document.createElement("div");
      detailDiv.className = "item-detail";
      if (item.summary) {
        detailDiv.innerHTML += `<div class="detail-summary">${escapeHtml(item.summary)}</div>`;
      }
      if (item.authors && item.authors.length > 0) {
        detailDiv.innerHTML += `<div class="detail-authors">${escapeHtml(item.authors.join(", "))}</div>`;
      }
      if (item.date) {
        detailDiv.innerHTML += `<div style="margin-top:2px">Published: ${escapeHtml(item.date)}</div>`;
      }
      if (!item.summary && !item.authors && !item.date) {
        detailDiv.innerHTML = `<div style="color:var(--text2);font-style:italic">No enriched metadata available.</div>`;
      }

      row.appendChild(mainDiv);
      row.appendChild(detailDiv);
      itemsDiv.appendChild(row);
    });

    div.appendChild(header);
    div.appendChild(itemsDiv);
    container.appendChild(div);
  });

  updatePendingCount();
  updateStats();
}

function applyFilters() {
  const q = document.getElementById("search").value.toLowerCase().trim();
  const tokens = q.split(/\\s+/).filter(Boolean);
  const filterByType = activeTypes.size > 0;

  let totalVisible = 0;

  document.querySelectorAll(".category").forEach(cat => {
    const items = cat.querySelectorAll(".item-row");
    let catVisible = 0;

    items.forEach(item => {
      const matchType = !filterByType || activeTypes.has(item.dataset.type);
      const searchStr = item.dataset.search;
      const matchSearch = tokens.length === 0 || tokens.every(t => searchStr.includes(t));

      if (matchType && matchSearch) {
        item.classList.remove("hidden");
        catVisible++;
      } else {
        item.classList.add("hidden");
      }
    });

    if (catVisible === 0) {
      cat.classList.add("hidden");
    } else {
      cat.classList.remove("hidden");
      cat.classList.add("open");
      cat.querySelector(".cat-count").textContent = catVisible;
    }
    totalVisible += catVisible;
  });

  document.getElementById("no-results").style.display = totalVisible === 0 ? "block" : "none";
  updateStats(totalVisible);
}

function updateStats(visible) {
  const total = getMergedData().metadata.total_items;
  if (visible === undefined || visible === total) {
    document.getElementById("stats").textContent = total + " items";
  } else {
    document.getElementById("stats").textContent = visible + " / " + total + " items";
  }
}

function updatePendingCount() {
  const pending = getPending();
  const removals = getRemovals();
  const el = document.getElementById("bm-pending-count");
  const parts = [];
  if (pending.length > 0) parts.push(pending.length + " to add");
  if (removals.length > 0) parts.push(removals.length + " to remove");
  el.textContent = parts.join(", ");
}

function renderFilters() {
  const bar = document.getElementById("filters");
  TYPES.forEach(type => {
    const btn = document.createElement("button");
    btn.className = "filter-btn";
    btn.textContent = type;
    btn.addEventListener("click", () => {
      if (activeTypes.has(type)) {
        activeTypes.delete(type);
        btn.classList.remove("active");
      } else {
        activeTypes.add(type);
        btn.classList.add("active");
      }
      applyFilters();
    });
    bar.appendChild(btn);
  });
}

// --- Bookmark panel ---
document.getElementById("bookmark-toggle").addEventListener("click", () => {
  const panel = document.getElementById("bookmark-panel");
  const toggle = document.getElementById("bookmark-toggle");
  panel.classList.toggle("open");
  toggle.classList.toggle("active");
});

document.getElementById("bm-add").addEventListener("click", () => {
  const url = document.getElementById("bm-url").value.trim();
  const title = document.getElementById("bm-title").value.trim();
  const category = document.getElementById("bm-category").value;
  const type = document.getElementById("bm-type").value;
  const summary = document.getElementById("bm-summary").value.trim() || null;

  if (!url) { alert("URL is required."); return; }
  if (!title) { alert("Title is required."); return; }

  const pending = getPending();
  pending.push({
    title: title,
    url: url,
    type: type,
    source: "user",
    summary: summary,
    date: null,
    authors: null,
    _category: category
  });
  savePending(pending);

  // Clear form
  document.getElementById("bm-url").value = "";
  document.getElementById("bm-title").value = "";
  document.getElementById("bm-summary").value = "";

  render();
  applyFilters();
});

// --- Export: save directly to repo file via File System Access API ---
let savedFileHandle = null;

document.getElementById("bm-export").addEventListener("click", async () => {
  const merged = getMergedData(false); // Exclude removed items
  const jsonStr = JSON.stringify(merged, null, 2);

  const pendingCount = getPending().length;
  const removalCount = getRemovals().length;

  // Try File System Access API (Chrome/Edge) to write in-place
  if (window.showSaveFilePicker) {
    try {
      if (!savedFileHandle) {
        savedFileHandle = await window.showSaveFilePicker({
          suggestedName: "joel_stremmel_knowledge_base.json",
          types: [{ description: "JSON", accept: { "application/json": [".json"] } }],
        });
      }
      const writable = await savedFileHandle.createWritable();
      await writable.write(jsonStr);
      await writable.close();
      // Clear pending additions and removals since they're now applied
      savePending([]);
      saveRemovals([]);
      render();
      applyFilters();
      alert(`Saved! ${pendingCount} additions, ${removalCount} removals applied. Run enrich script to update HTML.`);
      return;
    } catch (e) {
      if (e.name === "AbortError") return; // user cancelled picker
      // Fall through to download
    }
  }

  // Fallback: regular download
  const blob = new Blob([jsonStr], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "joel_stremmel_knowledge_base.json";
  a.click();
  URL.revokeObjectURL(a.href);
});

// --- Clear pending removals ---
document.getElementById("bm-clear-removals").addEventListener("click", () => {
  const removals = getRemovals();
  if (removals.length === 0) {
    alert("No pending removals to clear.");
    return;
  }
  if (confirm(`Clear ${removals.length} pending removal(s)?`)) {
    saveRemovals([]);
    render();
    applyFilters();
  }
});

// --- Auto-populate title from URL ---
async function fetchArxivTitle(arxivId) {
  try {
    const resp = await fetch(`https://export.arxiv.org/api/query?id_list=${arxivId}&max_results=1`);
    const text = await resp.text();
    const match = text.match(/<title>([^<]+)<\\/title>/g);
    if (match && match.length > 1) {
      // First title is "ArXiv Query", second is the paper title
      let title = match[1].replace(/<\\/?title>/g, "").trim();
      // Clean up arxiv ID prefix if present
      title = title.replace(/^\\[\\d+\\.\\d+\\]\\s*/, "");
      return title;
    }
  } catch (e) {
    console.log("Failed to fetch arxiv title:", e);
  }
  return null;
}

function extractArxivId(url) {
  const m = url.match(/arxiv\\.org\\/(?:abs|pdf|html)\\/(\\d+\\.\\d+)/);
  return m ? m[1] : null;
}

document.getElementById("bm-url").addEventListener("blur", async () => {
  const url = document.getElementById("bm-url").value.trim();
  const titleField = document.getElementById("bm-title");
  const typeField = document.getElementById("bm-type");

  if (!url || titleField.value.trim()) return;

  // Check for arxiv URL
  const arxivId = extractArxivId(url);
  if (arxivId) {
    titleField.value = "Loading...";
    typeField.value = "paper";
    const title = await fetchArxivTitle(arxivId);
    if (title) {
      titleField.value = title;
    } else {
      titleField.value = "";
    }
    return;
  }

  // Fallback: guess title from URL path
  try {
    const u = new URL(url);
    const pathParts = u.pathname.split("/").filter(Boolean);
    if (pathParts.length > 0) {
      const last = decodeURIComponent(pathParts[pathParts.length - 1])
        .replace(/[-_]/g, " ")
        .replace(/\\.[^.]+$/, "");
      titleField.value = last;
    }
  } catch {}
});

// --- Keyboard shortcuts ---
document.getElementById("search").addEventListener("input", applyFilters);

document.addEventListener("keydown", e => {
  if (e.key === "/" && document.activeElement.tagName !== "INPUT" && document.activeElement.tagName !== "TEXTAREA") {
    e.preventDefault();
    document.getElementById("search").focus();
  }
  if (e.key === "Escape") {
    const search = document.getElementById("search");
    search.value = "";
    search.blur();
    activeTypes.clear();
    document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
    applyFilters();
  }
  if ((e.key === "e" || e.key === "E") && document.activeElement.tagName !== "INPUT" && document.activeElement.tagName !== "TEXTAREA") {
    document.querySelectorAll(".category").forEach(c => c.classList.add("open"));
  }
  if ((e.key === "c" || e.key === "C") && document.activeElement.tagName !== "INPUT" && document.activeElement.tagName !== "TEXTAREA") {
    document.querySelectorAll(".category").forEach(c => c.classList.remove("open"));
  }
});

// --- Event delegation for remove buttons ---
document.getElementById("container").addEventListener("click", (e) => {
  if (e.target.classList.contains("remove-btn")) {
    e.stopPropagation();
    const url = decodeURIComponent(e.target.dataset.url);
    toggleRemoval(url);
  }
});

render();
renderFilters();

// --- Server integration (when running with --serve) ---
const SERVER_BASE = "http://localhost:8765";
let serverAvailable = false;

async function checkServer() {
  try {
    const resp = await fetch(SERVER_BASE + "/joel_stremmel_knowledge_base.json", { method: "HEAD" });
    serverAvailable = resp.ok;
  } catch {
    serverAvailable = false;
  }
  updateServerUI();
}

function updateServerUI() {
  const saveEnrichBtn = document.getElementById("bm-save-enrich");
  const exportBtn = document.getElementById("bm-export");
  const statusEl = document.getElementById("server-status");

  if (serverAvailable) {
    saveEnrichBtn.style.display = "inline-block";
    exportBtn.style.display = "none";
    statusEl.textContent = "server connected";
    statusEl.style.color = "var(--repo)";
  } else {
    saveEnrichBtn.style.display = "none";
    exportBtn.style.display = "inline-block";
    statusEl.textContent = "";
  }
}

document.getElementById("bm-save-enrich").addEventListener("click", async () => {
  const btn = document.getElementById("bm-save-enrich");
  const originalText = btn.textContent;

  try {
    btn.textContent = "Saving...";
    btn.disabled = true;

    // Get merged data (excluding removed items)
    const merged = getMergedData(false);

    // Save to server
    const saveResp = await fetch(SERVER_BASE + "/api/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(merged),
    });
    const saveResult = await saveResp.json();
    if (!saveResult.success) throw new Error(saveResult.error || "Save failed");

    btn.textContent = "Enriching...";

    // Run enrichment
    const enrichResp = await fetch(SERVER_BASE + "/api/enrich", { method: "POST" });
    const enrichResult = await enrichResp.json();
    if (!enrichResult.success) throw new Error(enrichResult.error || "Enrich failed");

    // Clear pending lists
    savePending([]);
    saveRemovals([]);

    // Show success and reload
    const msg = `Saved! ${enrichResult.papers_processed} papers processed, ${enrichResult.items_with_summary} items enriched. Reloading...`;
    btn.textContent = "Done!";
    setTimeout(() => location.reload(), 500);

  } catch (err) {
    alert("Error: " + err.message);
    btn.textContent = originalText;
    btn.disabled = false;
  }
});

// Check server on load
checkServer();
</script>
</body>
</html>'''


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_enrichment(verbose=True):
    """Run the enrichment pipeline. Returns a summary dict."""
    log = print if verbose else lambda *a, **k: None

    log("Loading knowledge base...")
    with open(JSON_PATH) as f:
        data = json.load(f)

    # Classify items
    arxiv_items = {}  # arxiv_id -> [item references]
    acl_items = []
    other_paper_items = []
    non_paper_items = []

    for cat in data["categories"]:
        for item in cat["items"]:
            # Skip already enriched items (idempotent)
            if item.get("summary") is not None:
                # Ensure all fields exist
                item.setdefault("date", None)
                item.setdefault("authors", None)
                continue

            if item.get("type") == "paper":
                url = item.get("url", "")
                arxiv_id = extract_arxiv_id(url)
                if arxiv_id:
                    arxiv_items.setdefault(arxiv_id, []).append(item)
                elif "aclanthology.org" in url:
                    acl_items.append(item)
                else:
                    other_paper_items.append(item)
            else:
                non_paper_items.append(item)

    # Enrich arxiv papers
    arxiv_count = 0
    if arxiv_items:
        log(f"\nEnriching {len(arxiv_items)} arxiv papers...")
        arxiv_count = enrich_arxiv(arxiv_items)
        log(f"  Enriched {arxiv_count} arxiv papers.")
    else:
        log("\nNo new arxiv papers to enrich.")

    # Enrich ACL papers
    acl_count = 0
    if acl_items:
        log(f"\nEnriching {len(acl_items)} ACL Anthology papers...")
        acl_count = enrich_acl(acl_items)
        log(f"  Enriched {acl_count} ACL papers with abstracts.")
    else:
        log("\nNo new ACL papers to enrich.")

    # Other papers
    if other_paper_items:
        log(f"\nProcessing {len(other_paper_items)} other papers (URL heuristics)...")
        enrich_other_papers(other_paper_items)

    # Non-paper items
    if non_paper_items:
        log(f"\nSetting null fields for {len(non_paper_items)} non-paper items...")
        set_null_fields(non_paper_items)

    # Write enriched JSON
    log(f"\nWriting enriched JSON to {JSON_PATH}...")
    with open(JSON_PATH, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Generate HTML
    log(f"Generating HTML to {HTML_PATH}...")
    html = generate_html(data)
    with open(HTML_PATH, "w") as f:
        f.write(html)

    # Summary
    total_papers = len(arxiv_items) + len(acl_items) + len(other_paper_items)
    enriched_summary = sum(
        1 for cat in data["categories"]
        for item in cat["items"]
        if item.get("summary") is not None
    )

    summary = {
        "papers_processed": total_papers,
        "arxiv_enriched": arxiv_count,
        "acl_enriched": acl_count,
        "items_with_summary": enriched_summary,
        "total_items": data["metadata"]["total_items"],
    }

    log(f"\nDone!")
    log(f"  Total papers processed: {total_papers}")
    log(f"  Items with summary: {enriched_summary}")
    log(f"  Total items: {data['metadata']['total_items']}")

    return summary


# ---------------------------------------------------------------------------
# HTTP Server for live editing
# ---------------------------------------------------------------------------

SERVER_PORT = 8765

class KnowledgeBaseHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with API endpoints for the knowledge base."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def do_GET(self):
        # Serve index as the knowledge base HTML
        if self.path == "/" or self.path == "":
            self.path = "/joel_stremmel_knowledge_base.html"
        # Ignore favicon requests
        if self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/save":
            self._handle_save()
        elif self.path == "/api/enrich":
            self._handle_enrich()
        else:
            self.send_error(404, "Not Found")

    def _handle_save(self):
        """Save the JSON data from the client."""
        try:
            content_length = int(self.headers["Content-Length"])
            body = self.rfile.read(content_length)
            data = json.loads(body)

            with open(JSON_PATH, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self._send_json({"success": True, "message": "JSON saved"})
        except Exception as e:
            self._send_json({"success": False, "error": str(e)}, status=500)

    def _handle_enrich(self):
        """Run the enrichment pipeline and regenerate HTML."""
        try:
            summary = run_enrichment(verbose=False)
            self._send_json({"success": True, **summary})
        except Exception as e:
            self._send_json({"success": False, "error": str(e)}, status=500)

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        # Quieter logging - only show API calls
        try:
            msg = str(args[0]) if args else ""
            if "/api/" in msg:
                print(f"[API] {msg}")
        except:
            pass


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def run_server(port=SERVER_PORT, open_browser=True):
    """Run the local development server."""
    with ReusableTCPServer(("", port), KnowledgeBaseHandler) as httpd:
        url = f"http://localhost:{port}"
        print(f"\n{'='*60}")
        print(f"  Knowledge Base Server running at: {url}")
        print(f"  Press Ctrl+C to stop")
        print(f"{'='*60}\n")

        if open_browser:
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")


def main():
    parser = argparse.ArgumentParser(description="Knowledge Base Enrichment Tool")
    parser.add_argument("--serve", "-s", action="store_true",
                        help="Start local server for live editing")
    parser.add_argument("--port", "-p", type=int, default=SERVER_PORT,
                        help=f"Server port (default: {SERVER_PORT})")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't open browser automatically")
    args = parser.parse_args()

    if args.serve:
        run_server(port=args.port, open_browser=not args.no_browser)
    else:
        run_enrichment()


if __name__ == "__main__":
    main()
