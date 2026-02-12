# Knowledge Base

A curated collection of 370+ bookmarks organized across 22 categories — papers, repos, blogs, tools, videos, and more. Includes an enrichment pipeline that pulls metadata (abstracts, publication dates, authors) from arxiv and ACL Anthology, and a single-file HTML viewer with search, filtering, and bookmark management.

## Files

| File | Description |
|------|-------------|
| `joel_stremmel_knowledge_base.json` | The knowledge base data (enriched with metadata) |
| `joel_stremmel_knowledge_base.html` | Self-contained HTML viewer (open in any browser) |
| `enrich_knowledge_base.py` | Enrichment script and local server for live editing |
| `Knowledge Base.app` | macOS dock app to launch the server |

## Quick Start

**Live editing mode** (recommended):
```bash
python3 enrich_knowledge_base.py --serve
```
This starts a local server at `http://localhost:8765`, opens your browser, and enables one-click save & enrich.

**One-time enrichment**:
```bash
python3 enrich_knowledge_base.py
```

No external dependencies — uses only the Python standard library.

## macOS Dock App

Drag `Knowledge Base.app` to your dock for one-click server launch. It opens Terminal with the server running and your browser pointed to the knowledge base.

## HTML Viewer Features

- **Search** across titles, URLs, categories, summaries, and author names
- **Type filters** (paper, repo, blog, video, tool, pod, docs, news, other)
- **Enrichment indicators**: filled dot for enriched papers, empty dot for catalogued-only items
- **Expandable detail panels**: click any row to see the full abstract, author list, and date
- **Date badges** and **abbreviated author names** inline
- **Keyboard shortcuts**: `/` focus search, `Esc` clear, `E` expand all, `C` collapse all

## Adding & Removing Bookmarks

### With Live Server (recommended)
1. Start the server: `python3 enrich_knowledge_base.py --serve`
2. Click **+ Add** to add bookmarks (arxiv URLs auto-fetch paper titles)
3. Hover any item and click **Remove** to mark it for deletion
4. Click **Save & Enrich** — saves changes, enriches new papers, regenerates HTML, and reloads

### Without Server
1. Click **+ Add** to add bookmarks, **Remove** to mark deletions
2. Click **Export JSON** to download the merged data
3. Replace `joel_stremmel_knowledge_base.json` with the export
4. Run `python3 enrich_knowledge_base.py` to enrich and regenerate HTML

## Enrichment

The script queries external APIs to add `summary`, `date`, and `authors` fields to paper entries:

- **Arxiv papers**: abstract, publication date, and full author list via the arxiv API
- **ACL Anthology papers**: abstract scraped from the paper page, year from URL
- **Other papers**: year extracted from URL heuristics
- **Non-paper items**: fields set to `null` — displayed as "catalogued only"

The script is **idempotent** — it skips items that already have a non-null `summary`, so re-running it only processes new or previously failed items.

## Item Schema

```json
{
  "title": "Paper Title",
  "url": "https://arxiv.org/abs/2402.12329",
  "type": "paper",
  "source": "curated",
  "summary": "2-3 sentence description from abstract.",
  "date": "2024-02-19",
  "authors": ["First Author", "Second Author"]
}
```

Non-paper items and un-enriched papers have `summary`, `date`, and `authors` set to `null`.
