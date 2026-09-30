# Data collection

This directory contains the scripts for collecting and processing PolitiFact and Snopes articles. Run the scripts in this order for each source:

1. **RawScrape** downloads article HTML and saves it, together with download metadata, in a Parquet data file.
2. **ScrapeData** reads the saved HTML from that Parquet file and extracts a structured dataset, including claims, verdicts, authors, dates, and article text.

```text
Website → RawScrape.py → *_HTML.parquet → ScrapeData.py → *_DATA.parquet
```

The raw files store HTML strings in an `html` column; the scripts do not create separate `.html` files. Once the raw dataset exists, the parsing stage can run offline without downloading the articles again.

## Files

| File | Purpose |
| --- | --- |
| `PolitiFact_RawScrape.py` | Crawls PolitiFact listing pages and downloads the full HTML of linked fact-check articles. |
| `PolitiFact_ScrapeData.py` | Parses saved PolitiFact HTML into claims, verdicts, speaker information, summaries, analysis, rulings, sources, and metadata. |
| `Snopes_RawScrape.py` | Crawls a selected Snopes category and downloads the full HTML of its articles. |
| `Snopes_ScrapeData.py` | Parses saved Snopes HTML into claims, verdicts, summaries, analysis, sources, and metadata. |

## Setup

Use Python 3 and install the required packages:

```bash
python3 -m pip install requests pandas beautifulsoup4 pyarrow
```

`pyarrow` provides Parquet support. Run the examples below from the **repository root**; there is no need to change into `Code/Data_Collection`.

All relative input and output paths are resolved from the repository root, regardless of your current working directory. By default, all four scripts use the repository's `Data/` directory. Absolute paths are also supported. To launch a script from outside the repository, use its absolute script path.

## PolitiFact usage

### 1. Download the raw HTML

This example collects up to five listing pages:

```bash
python3 Code/Data_Collection/PolitiFact_RawScrape.py \
  --category latest \
  --limit 5 \
  --output-dir Data \
  --output-name PolitiFact_HTML.parquet
```

### 2. Extract the structured data

```bash
python3 Code/Data_Collection/PolitiFact_ScrapeData.py \
  --input-path Data/PolitiFact_HTML.parquet \
  --output-path Data/PolitiFact_DATA.parquet
```

The parser accepts a single input Parquet file and writes a single output Parquet file. Output directories are created automatically.

## Snopes usage

### 1. Download the raw HTML

```bash
python3 Code/Data_Collection/Snopes_RawScrape.py \
  --category fact-check \
  --start 1 \
  --limit 5 \
  --output-dir Data \
  --output-name Snopes_HTML.parquet
```

### 2. Extract the structured data

```bash
python3 Code/Data_Collection/Snopes_ScrapeData.py \
  --input-path Data/Snopes_HTML.parquet \
  --output-path Data/Snopes_DATA.parquet
```

No source-code edits are needed to select input or output files. Output directories are created automatically.

The parser writes the processed Parquet file and prints its shape, non-null counts for key fields, and verdict counts.

## RawScrape options

| Option | PolitiFact default | Snopes default | Meaning |
| --- | --- | --- | --- |
| `--output-dir` | `Data` | `Data` | Directory for the raw Parquet file, relative to the repository root or absolute; created if needed. |
| `--output-name` | `PolitiFact_HTML.parquet` | `Snopes_HTML.parquet` | Raw output filename. |
| `--category` | `latest` | `fact-check` | Listing category to crawl. |
| `--time-stamp` | No date filter | No date filter | Date cutoff in `YYYYMMDD` format, e.g. `20260101`. Stops when an article with a known date older than the cutoff is encountered; the cutoff date is included. |
| `--limit` | `200` | `300` | Last listing page to crawl, inclusive; `-1` removes the page limit. |
| `--start` | Not exposed on the CLI | `1` | First listing page to crawl (Snopes only). |

PolitiFact accepts only `latest` through its CLI. Snopes category keys are `main`, `fact-check`, `trending`, `latest`, `politics`, and `entertainment`; `ALL` is not supported. Snopes currently maps `latest` and `trending` to the same URL.

For Snopes, `--start 501 --limit 1000` requests pages 501 through 1000, rather than 1,000 pages. Use `--output-name Snopes_HTML_501_1000.parquet`, then pass `--input-path Data/Snopes_HTML_501_1000.parquet` to the parser. Crawling can stop earlier if no articles are found, the date cutoff is reached, or a listing-page request fails.

## Parser options and custom directories

Both `ScrapeData` scripts accept:

| Option | Default | Meaning |
| --- | --- | --- |
| `--input-path` | `Data/PolitiFact_HTML.parquet` or `Data/Snopes_HTML.parquet` | Single raw input file, relative to the repository root or absolute. |
| `--output-dir` | `Data` | Output directory; the filename is `PolitiFact_DATA.parquet` or `Snopes_DATA.parquet`. |
| `--output-path` | Unset | Explicit output filename and path; overrides `--output-dir` and is resolved from the repository root if relative. |

For example, save both stages under the repository's `Data/custom/` directory:

```bash
python3 Code/Data_Collection/Snopes_RawScrape.py \
  --limit 5 \
  --output-dir Data/custom

python3 Code/Data_Collection/Snopes_ScrapeData.py \
  --input-path Data/custom/Snopes_HTML.parquet \
  --output-dir Data/custom
```

Changing `--output-dir` does not change a parser's input path; pass `--input-path` when reading a raw file outside the default location. Use `--output-name` on a raw scraper or `--output-path` on a parser to name individual batches.

## Output data and reruns

Both raw datasets contain `date`, `article_url`, `retrieved_at`, `http_status`, `download_error`, and `html`. Check the download fields when HTML is missing.

Both processed datasets extract fields such as `article_url`, `title`, `claim`, `verdict`, `author`, `publication_date`, `keywords`, `short_summary`, `analysis_text`, `sources_text`, `source_urls`, and `og_image`. Their schemas differ:

- **PolitiFact** additionally extracts speaker and statement details, a separate `ruling_text`, and a combined `context` containing the summary, analysis, and ruling. It prints parsing failures and excludes those failed records from the output.
- **Snopes** retains download metadata on successfully parsed records and sets `context` to `None`. Records with missing HTML return an `article_url` and `parse_error`. Publication dates are converted to UTC.

Fields can be missing when the saved HTML lacks the expected content. An empty crawl or a parser run with no successfully parsed records now raises an explicit error instead of reporting success without a usable output. Existing output files are not replaced in that case. Inspect the error before proceeding to the next stage.

Output files are overwritten when the same path is reused; runs do not append or automatically merge batches. Use distinct filenames for separate batches and keep each parser's input path aligned with its raw output file. Re-run only `ScrapeData` when you want to extract data again from existing HTML.
