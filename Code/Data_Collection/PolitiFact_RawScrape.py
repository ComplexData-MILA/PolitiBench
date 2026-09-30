import requests, json, argparse, time, re
import pandas as pd
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parents[2]

FACTCHECK_PATH_RE = re.compile(
    r"^/factchecks/\d{4}/[a-z]{3}/\d{1,2}/"
)

MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

def date_from_article_url(url):
    """
    Example:
    /factchecks/2026/aug/13/alexander-vindman/...
    """

    parts = urlparse(url).path.strip("/").split("/")

    # [
    #   "factchecks",
    #   "2026",
    #   "aug",
    #   "13",
    #   ...
    # ]

    if len(parts) < 4 or parts[0] != "factchecks":
        return None

    try:
        year = int(parts[1])
        month = MONTHS[parts[2].lower()]
        day = int(parts[3])

        return datetime(
            year,
            month,
            day
        )

    except (ValueError, KeyError):
        return None

#============================================== HELPERS ==============================================

URLs = {
    "main": "https://www.politifact.com",
    "latest": "https://www.politifact.com/factchecks/list/",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0.0.0 Safari/537.36"
    )
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

#============================================== Function ==============================================

def parse_date(date):

    cleaned_date = (
        date
        .split("(")[0]
        .replace("Sept.", "Sep")
        .replace(".", "")
        .strip()
    )

    formats = [
        "%b %d, %Y",
        "%B %d, %Y"
    ]

    for fmt in formats:

        try:
            return datetime.strptime(
                cleaned_date,
                fmt
            )

        except ValueError:
            continue

    raise ValueError(
        f"Unknown date format: {date}"
    )

def get_article_html(article_url, max_retries=3):
    if not article_url:
        return None, None

    for attempt in range(max_retries):

        response = SESSION.get(
            article_url,
            timeout=(5, 30)
        )

        if response.status_code == 429:

            retry_after = response.headers.get("Retry-After")

            if retry_after:
                try:
                    wait_time = float(retry_after)
                except ValueError:
                    wait_time = 10 * (2 ** attempt)
            else:
                wait_time = 10 * (2 ** attempt)

            print(
                f"429 RATE LIMITED: {article_url}\n"
                f"Waiting {wait_time:.0f}s before retry "
                f"({attempt + 1}/{max_retries})"
            )

            time.sleep(wait_time)
            continue

        response.raise_for_status()

        return response.text, response.status_code

    return None, 429

def getArticles(category='latest', timestamp=None, limit=-1, start=1):

    done = False
    articles = []

    page = start

    timestamp_datetime = None

    if timestamp is not None:

        timestamp_datetime = datetime.strptime(
            str(timestamp),
            "%Y%m%d"
        )


    while True:
        if page == 1:
            page_url = URLs[category]
        else:
            page_url = f"{URLs[category]}page/{page}/"

        if page % 25 == 0:
            if limit == -1:
                print(f"Extracted {page} pages out of the entire web.")
            else:
                print(f"Extracted {page} pages out of {limit} pages.")

        try:
            response = SESSION.get(
                page_url,
                timeout=(5, 30)
            )
            response.raise_for_status()

        except requests.RequestException as e:
            print(f"FAILED TO DOWNLOAD LISTING PAGE {page_url}: {e}")
            break

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        article_links = soup.find_all(
            "a",
            href=FACTCHECK_PATH_RE
        )

        # Deduplicate because the same article URL may appear
        # more than once in the page HTML.
        article_urls = []
        seen = set()

        for link in soup.find_all("a", href=True):

            href = link["href"].strip()

            # Handles:
            # /factchecks/...
            # https://politifact.com/factchecks/...
            # https://www.politifact.com/factchecks/...
            absolute_url = urljoin(
                response.url,
                href
            )

            parsed = urlparse(absolute_url)

            # Only PolitiFact links
            if parsed.netloc.lower() not in {
                "politifact.com",
                "www.politifact.com",
            }:
                continue

            # Only actual dated fact-check articles
            if not FACTCHECK_PATH_RE.match(parsed.path):
                continue

            # Canonicalize so www/non-www versions don't duplicate
            article_url = (
                "https://www.politifact.com"
                + parsed.path
            )

            if article_url not in seen:
                seen.add(article_url)
                article_urls.append(article_url)


        if not article_urls:
            print(
                f"No fact-check URLs found on page {page}."
            )
            break


        print(
            f"Page {page}: found "
            f"{len(article_urls)} fact-check URLs"
        )


        for article_url in article_urls:
            try:

                # =================================
                # DATE FROM URL
                # =================================

                article_datetime = date_from_article_url(
                    article_url
                )

                date = (
                    article_datetime.strftime("%b %d, %Y")
                    if article_datetime
                    else None
                )

                # =================================
                # FILTER BY TIMESTAMP
                # =================================

                if (
                    timestamp_datetime is not None
                    and article_datetime is not None
                ):
                    if article_datetime < timestamp_datetime:
                        done = True
                        break

                # =================================
                # DOWNLOAD FULL ARTICLE HTML
                # =================================

                html = None
                status = None
                download_error = None

                try:

                    html, status = get_article_html(
                        article_url
                    )

                    time.sleep(1)

                except requests.RequestException as e:

                    download_error = str(e)

                    if getattr(e, "response", None) is not None:
                        status = e.response.status_code

                    print(
                        f"FAILED TO DOWNLOAD ARTICLE: "
                        f"{article_url}: {e}"
                    )

                # =================================
                # ARTICLE OBJECT
                # =================================

                article = {
                    "date": date,
                    "article_url": article_url,
                    "retrieved_at": (
                        datetime.now()
                        .astimezone()
                        .isoformat()
                    ),
                    "http_status": status,
                    "download_error": download_error,
                    "html": html,
                }

                articles.append(article)

            except Exception as e:
                print("FAILED:", article_url, e)

        page += 1

        if (limit != -1 and page > limit) or done:
            break

        time.sleep(1)

    return articles

def getData(outputDirectory, category, timeStamp, limit, dataset_name):
    outputDirectory = REPO_ROOT / Path(outputDirectory).expanduser()
    outputDirectory.mkdir(
        parents=True,
        exist_ok=True
    )

    articles = getArticles(
        category=category,
        timestamp=timeStamp,
        limit=limit
    )

    if not articles:
        raise ValueError("No articles collected. Check listing access, category, and date/page filters; no dataset was written.")

    outputFile = outputDirectory / dataset_name

    print(
        f"Finished collecting articles. "
        f"Collected {len(articles)} articles."
    )

    df = pd.DataFrame(articles)

    df.to_parquet(
        outputFile,
        index=False,
        compression="zstd"
    )

    print(f"Saved raw HTML dataset to {outputFile}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the articles from the Snopes.com web"
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Data"),
        help="Output directory, relative to the repository root or absolute (default: Data).",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="PolitiFact_HTML.parquet",
        help="Filename for the raw HTML Parquet dataset.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="latest",
        choices=["latest"],
        help="Category of articles (ALL, latest, trending, politics, entertainment).",
    )
    parser.add_argument(
        "--time-stamp",
        type=int,
        default=None,
        help="Format : (YYYYMMDD), Leave None if do not want to filter by time"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum number of pages to crawl (-1 : no limit)",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    output_root = args.output_dir
    category_name = args.category
    time_stamp = args.time_stamp
    limit = args.limit
    output_name = args.output_name

    getData(
        outputDirectory=output_root, 
        category=category_name, 
        timeStamp=time_stamp, 
        limit=limit, 
        dataset_name=output_name
        )

if __name__ == "__main__":
    main()
