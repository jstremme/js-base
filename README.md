# Knowledge Base

A curated collection of 370+ bookmarks organized across 22 categories — papers, repos, blogs, tools, videos, and more. Includes an enrichment pipeline that pulls metadata (abstracts, publication dates, authors) from arxiv and ACL Anthology, and a single-file HTML viewer with search, filtering, and a bookmark-adding workflow.

## Files

| File | Description |
|------|-------------|
| `joel_stremmel_knowledge_base.json` | The knowledge base data (enriched with metadata) |
| `joel_stremmel_knowledge_base.html` | Self-contained HTML viewer (open in any browser) |
| `enrich_knowledge_base.py` | Enrichment script — fetches paper metadata and regenerates the HTML |

## Enrichment

The script queries external APIs to add `summary`, `date`, and `authors` fields to paper entries:

- **Arxiv papers** (111): abstract, publication date, and full author list via the arxiv API
- **ACL Anthology papers** (21): abstract scraped from the paper page, year from URL
- **Other papers** (9): year extracted from URL heuristics
- **Non-paper items** (229): fields set to `null` — displayed as "catalogued only"

The script is **idempotent** — it skips items that already have a non-null `summary`, so re-running it only processes new or previously failed items.

```bash
python3 enrich_knowledge_base.py
```

No external dependencies — uses only the Python standard library (`urllib`, `xml.etree`, `json`, `ssl`, `re`).

## HTML Viewer Features

- **Search** across titles, URLs, categories, summaries, and author names
- **Type filters** (paper, repo, blog, video, tool, podcast, documentation, news, other)
- **Enrichment indicators**: filled dot for enriched papers, empty dot for catalogued-only items
- **Expandable detail panels**: click any row to see the full abstract, author list, and date
- **Date badges** and **abbreviated author names** inline
- **Keyboard shortcuts**: `/` focus search, `Esc` clear, `E` expand all, `C` collapse all

### Adding New Bookmarks

1. Click **+ Add Bookmark** in the header
2. Fill in URL, title, category, and type
3. The item appears immediately with a "pending" badge (stored in `localStorage`)
4. Click **Export JSON** to download the merged data file
5. Replace `joel_stremmel_knowledge_base.json` with the export
6. Re-run `python3 enrich_knowledge_base.py` — new arxiv/ACL papers get auto-enriched and the HTML is regenerated

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
