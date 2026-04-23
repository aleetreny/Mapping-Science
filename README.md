# OpenAlex Statistics and Probability Corpus

This repo now includes a Python workflow to:

1. query OpenAlex for a Statistics and Probability corpus
2. store the metadata in a local SQLite database
3. optionally download PDFs through OpenAlex's content API

The default corpus profile is tuned for downloadable papers:

- `primary_topic.subfield.id:2613`
- `has_abstract:true`
- `type:article`
- `language:en`
- `has_references:true`
- `is_retracted:false`
- `is_paratext:false`
- `is_oa:true`
- `has_pdf_url:true`
- `has_fulltext:true`
- `has_content.pdf:true`

If you want the broader metadata-only corpus later, switch to `--profile core`.

## Files

- `openalex_corpus.py`: downloader and SQLite ingester
- `example_queries.sql`: starter SQL queries for exploring the database

## PowerShell usage

Set your key for the current shell:

```powershell
$env:OPENALEX_API_KEY="YOUR_KEY_HERE"
```

Build the default downloadable corpus metadata and download 10 PDFs:

```powershell
python .\openalex_corpus.py --download-pdfs
```

Build the full downloadable corpus metadata without downloading PDFs:

```powershell
python .\openalex_corpus.py --no-download-pdfs
```

Build the broader core corpus using any topic in the subfield:

```powershell
python .\openalex_corpus.py --profile core --subfield-scope any --no-download-pdfs
```

Download more PDFs later without changing the metadata profile:

```powershell
python .\openalex_corpus.py --download-pdfs --max-pdfs 50
```

## Output

By default the script writes to:

- database: `data/openalex_stats_probability/corpus.sqlite`
- PDFs: `data/openalex_stats_probability/pdfs/`

The SQLite database contains:

- `works`
- `authors`
- `work_authorships`
- `topics`
- `work_topics`
- `pdf_downloads`
- `sync_jobs`
- view: `download_queue`

## Notes

- OpenAlex API docs: <https://developers.openalex.org/>
- Works reference: <https://developers.openalex.org/api-reference/works/list-works>
- Works filters: <https://developers.openalex.org/api-reference/works>
- Content downloads: <https://developers.openalex.org/how-to-use-the-api/get-content>
- Subfield taxonomy: <https://developers.openalex.org/how-to-use-the-api/api-overview>

- OpenAlex's free API budget is limited. Metadata requests are cheap; PDF content downloads are much more expensive.
- OpenAlex can serve PDFs, but the original copyright and licensing of each PDF still apply. The script stores the reported OA/license metadata so you can audit later.
