import requests, json, argparse, time
import pandas as pd
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parents[2]

#============================================== HELPERS ==============================================

URLs = {
    "main": "https://www.snopes.com/",
    "fact-check": "https://www.snopes.com/fact-check/",
    "trending": "https://www.snopes.com/top/",
    "latest": "https://www.snopes.com/top/",
    "politics": "https://www.snopes.com/category/politics/",
    "entertainment": "https://www.snopes.com/category/Entertainment/"
}

HEADERS = {
    "User-Agent": "Mozilla/5.0"
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

def get_article_html(article_url):
    if not article_url:
        return None, None

    response = SESSION.get(
        article_url,
        timeout=(5, 30)
    )

    status = response.status_code
    response.raise_for_status()

    return response.text, status

def getArticles(category='fact-check', timestamp=None, limit=-1, start=1):

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
            page_url = f"{URLs[category]}?pagenum={page}"

        if page % 25 == 0:
            if limit == -1:
                print(f"Extracted {page} pages out of the entire web.")
            else:
                processed = page - start
                print(
                    f"Processed {processed} pages "
                    f"(currently page {page}, ending at page {limit})."
                )

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

        article_wrappers = soup.find_all(
            "div",
            class_="article_wrapper"
        )

        if not article_wrappers:
            print("No more articles found.")
            break

        for card in article_wrappers:
            try:

                # =================================
                # URL
                # =================================

                a_tag = card.find(
                    "a",
                    class_="outer_article_link_wrapper"
                )

                article_url = urljoin(response.url, a_tag["href"])
                article_url = article_url.replace(".com//", ".com/")

                # =================================
                # DATE
                # =================================

                date_tag = card.find(
                    "span",
                    class_="article_date"
                )

                date = (
                    date_tag.get_text(strip=True)
                    if date_tag else None
                )
                
                article_datetime = None

                if date:
                    article_datetime = parse_date(date)

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

                if article_url:
                    try:
                        html, status = get_article_html(article_url)
                        time.sleep(0.5)

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
                    "retrieved_at": datetime.now()
                        .astimezone()
                        .isoformat(),
                    "http_status": status,
                    "download_error": download_error,
                    "html": html,
                }

                articles.append(article)

            except Exception as e:

                print("FAILED:", e)

        page += 1

        if (limit != -1 and page > limit) or done:
            break

        time.sleep(1)

    return articles

def getData(outputDirectory, category, timeStamp, limit, dataset_name, start):
    outputDirectory = REPO_ROOT / Path(outputDirectory).expanduser()
    outputDirectory.mkdir(
        parents=True,
        exist_ok=True
    )

    articles = getArticles(
        category=category,
        timestamp=timeStamp,
        limit=limit,
        start=start
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
        default="Snopes_HTML.parquet",
        help="Filename for the raw HTML Parquet dataset.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="fact-check",
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
        default=300,
        help="Maximum number of pages to crawl (-1 : no limit)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Start page (Starting page is 1).",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    output_root = args.output_dir
    category_name = args.category
    time_stamp = args.time_stamp
    limit = args.limit
    output_name = args.output_name
    start=args.start

    getData(
        outputDirectory=output_root, 
        category=category_name, 
        timeStamp=time_stamp, 
        limit=limit, 
        dataset_name=output_name,
        start=start
    )

if __name__ == "__main__":
    main()
