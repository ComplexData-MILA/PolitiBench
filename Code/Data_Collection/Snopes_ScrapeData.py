import argparse
import json
from pathlib import Path
import re
import html as html_lib
import pandas as pd
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def clean_text(x):
    if x is None:
        return None

    x = html_lib.unescape(str(x))
    x = re.sub(r"\s+", " ", x).strip()

    return x or None


def get_meta(soup, **attrs):
    tag = soup.find("meta", attrs=attrs)

    if tag and tag.get("content"):
        return clean_text(tag["content"])

    return None


def get_jsonld(soup):
    """
    Extract all JSON-LD objects from the page.
    """
    objects = []

    for tag in soup.find_all("script", type="application/ld+json"):
        raw = tag.string or tag.get_text()

        if not raw or not raw.strip():
            continue

        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue

        if isinstance(obj, list):
            objects.extend(
                x for x in obj
                if isinstance(x, dict)
            )

        elif isinstance(obj, dict) and isinstance(obj.get("@graph"), list):
            objects.extend(
                x for x in obj["@graph"]
                if isinstance(x, dict)
            )

        elif isinstance(obj, dict):
            objects.append(obj)

    return objects


def get_jsonld_type(objects, type_name):
    for obj in objects:
        obj_type = obj.get("@type")

        if obj_type == type_name:
            return obj

        if isinstance(obj_type, list) and type_name in obj_type:
            return obj

    return {}


# ---------------------------------------------------------
# Source URL extraction
# ---------------------------------------------------------

URL_RE = re.compile(
    r"""(?ix)
    \b(
        https?://[^\s<>"']+
        |
        www\.[^\s<>"']+
        |
        (?:[a-z0-9-]+\.)+
        (?:com|org|net|gov|edu|io|co|us|uk|ca|de|fr|au|info)
        /[^\s<>"']+
    )
    """
)


def normalize_url(url):
    if not url:
        return None

    url = html_lib.unescape(url.strip())

    # Remove citation punctuation
    url = url.rstrip(".,;:!?)]}\"'")

    if url.startswith("www."):
        url = "https://" + url

    elif not re.match(r"^https?://", url, re.I):
        url = "https://" + url

    return url


def extract_sources(soup):

    source_rows = soup.select("#sources_rows p")

    sources_text = []
    source_urls = []

    seen = set()

    for p in source_rows:

        text = clean_text(
            p.get_text(" ", strip=True)
        )

        if text:
            sources_text.append(text)

        # In case Snopes uses actual hyperlinks
        for a in p.find_all("a", href=True):

            href = a["href"].strip()

            if href.startswith(
                ("http://", "https://", "www.")
            ):
                url = normalize_url(href)

                if url and url not in seen:
                    seen.add(url)
                    source_urls.append(url)

        # Snopes currently often stores URLs as plain text
        if text:
            for match in URL_RE.finditer(text):

                url = normalize_url(match.group(1))

                if url and url not in seen:
                    seen.add(url)
                    source_urls.append(url)

    return sources_text, source_urls


# ---------------------------------------------------------
# Article body
# ---------------------------------------------------------

def extract_analysis_text(soup):

    article = soup.select_one(
        "article#article-content"
    )

    if article is None:
        return None

    # Create a copy so removing things doesn't affect soup
    body = BeautifulSoup(
        str(article),
        "html.parser"
    )

    # Remove claim/rating box and non-content HTML
    for tag in body.select(
        """
        #fact_check_rating_container,
        script,
        style,
        noscript,
        input,
        iframe
        """
    ):
        tag.decompose()

    blocks = []

    # Keep actual written content
    for tag in body.find_all(
        ["h2", "h3", "p", "li"]
    ):

        text = clean_text(
            tag.get_text(" ", strip=True)
        )

        if not text:
            continue

        # prevent consecutive duplicates
        if blocks and blocks[-1] == text:
            continue

        blocks.append(text)

    # Some blockquotes contain raw text rather than <p>
    for tag in body.find_all("blockquote"):

        if tag.find(["p", "li"]):
            continue

        text = clean_text(
            tag.get_text(" ", strip=True)
        )

        if text and text not in blocks:
            blocks.append(text)

    if not blocks:
        return None

    return "\n\n".join(blocks)


# ---------------------------------------------------------
# Author
# ---------------------------------------------------------

def extract_author(author_obj):

    if isinstance(author_obj, dict):
        return clean_text(
            author_obj.get("name")
        )

    if isinstance(author_obj, list):

        names = []

        for author in author_obj:

            if isinstance(author, dict):
                name = clean_text(
                    author.get("name")
                )
            else:
                name = clean_text(author)

            if name:
                names.append(name)

        if names:
            return ", ".join(names)

    return clean_text(author_obj)


# ---------------------------------------------------------
# Parse one Snopes article
# ---------------------------------------------------------

def parse_snopes(record):

    raw_html = record.get("html")

    if not isinstance(raw_html, str) or not raw_html.strip():
        return {
            "article_url": record.get("article_url"),
            "parse_error":
                record.get("download_error")
                or "missing_html"
        }

    soup = BeautifulSoup(
        raw_html,
        "html.parser"
    )

    # ------------------------------------------
    # Structured metadata
    # ------------------------------------------

    jsonld = get_jsonld(soup)

    article_data = get_jsonld_type(
        jsonld,
        "Article"
    )

    claim_data = get_jsonld_type(
        jsonld,
        "ClaimReview"
    )

    # ------------------------------------------
    # Claim
    # ------------------------------------------

    claim = clean_text(
        claim_data.get("claimReviewed")
    )

    # fallback
    if not claim:

        el = soup.select_one(
            "#fact_check_rating_container .claim_cont"
        )

        if el:
            claim = clean_text(
                el.get_text(" ", strip=True)
            )

    # ------------------------------------------
    # Verdict
    # ------------------------------------------

    rating = (
        claim_data.get("reviewRating")
        or {}
    )

    verdict = clean_text(
        rating.get("alternateName")
    )

    # fallback
    if not verdict:

        el = soup.select_one(
            "#main_rating .rating_title_wrap, "
            "#main_rating"
        )

        if el:
            verdict = clean_text(
                el.get_text(" ", strip=True)
            )

    # ------------------------------------------
    # Author
    # ------------------------------------------

    author = extract_author(
        article_data.get("author")
    )

    if not author:
        author = get_meta(
            soup,
            property="article:author"
        )

    # ------------------------------------------
    # Date
    # ------------------------------------------

    publication_date = clean_text(
        article_data.get("datePublished")
    )

    if not publication_date:
        publication_date = get_meta(
            soup,
            property="article:published_time"
        )

    # ------------------------------------------
    # Title
    # ------------------------------------------

    title = clean_text(
        article_data.get("headline")
    )

    if not title:
        title = get_meta(
            soup,
            property="og:title"
        )

    # ------------------------------------------
    # Summary / subtitle
    # ------------------------------------------

    short_summary = clean_text(
        article_data.get(
            "alternativeHeadline"
        )
    )

    if not short_summary:
        short_summary = get_meta(
            soup,
            property="og:description"
        )

    if not short_summary:
        short_summary = get_meta(
            soup,
            name="description"
        )

    # ------------------------------------------
    # Keywords
    # ------------------------------------------

    keywords = article_data.get(
        "keywords"
    )

    if isinstance(keywords, str):

        keywords = [
            clean_text(k)
            for k in keywords.split(",")
            if clean_text(k)
        ]

    elif isinstance(keywords, list):

        keywords = [
            clean_text(k)
            for k in keywords
            if clean_text(k)
        ]

    else:

        keywords = [
            clean_text(tag.get("content"))
            for tag in soup.find_all(
                "meta",
                property="article:tag"
            )
            if tag.get("content")
        ]

    # ------------------------------------------
    # Image
    # ------------------------------------------

    image = clean_text(
        article_data.get("thumbnailUrl")
    )

    if not image:

        images = article_data.get("image")

        if isinstance(images, list) and images:
            image = clean_text(images[0])

        elif isinstance(images, str):
            image = clean_text(images)

    if not image:
        image = get_meta(
            soup,
            property="og:image"
        )

    # ------------------------------------------
    # Sources
    # ------------------------------------------

    sources_text, source_urls = (
        extract_sources(soup)
    )

    # ------------------------------------------
    # Article analysis
    # ------------------------------------------

    analysis_text = (
        extract_analysis_text(soup)
    )

    # ------------------------------------------
    # Result
    # ------------------------------------------

    return {
        "article_url":
            record.get("article_url"),

        "title":
            title,

        "claim":
            claim,

        "verdict":
            verdict,

        "author":
            author,

        "publication_date":
            publication_date,

        "keywords":
            keywords,

        # Snopes doesn't have an equivalent
        # PolitiFact statement-context field
        "context":
            None,

        "sources_text":
            sources_text,

        "source_urls":
            source_urls,

        "og_image":
            image,

        "analysis_text":
            analysis_text,

        "short_summary":
            short_summary,

        # Keep download metadata too
        "retrieved_at":
            record.get("retrieved_at"),

        "http_status":
            record.get("http_status"),

        "download_error":
            record.get("download_error"),
    }


# ---------------------------------------------------------
# Process downloaded file
# ---------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract structured data from saved Snopes HTML."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("Data/Snopes_HTML.parquet"),
        help="Raw Parquet file, relative to the repository root or absolute.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Data"),
        help="Output directory, relative to the repository root or absolute (default: Data).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output Parquet file, relative to the repository root or absolute; overrides --output-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_file = REPO_ROOT / args.input_path.expanduser()
    output_file = args.output_path or args.output_dir / "Snopes_DATA.parquet"
    output_file = REPO_ROOT / output_file.expanduser()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    downloaded = pd.read_parquet(input_file)

    if "html" not in downloaded.columns:
        raise ValueError("Raw Snopes dataset must contain an html column.")
    results = []
    for record in downloaded.to_dict(orient="records"):
        try:
            results.append(parse_snopes(record))
        except Exception as error:
            results.append({
                "article_url": record.get("article_url"),
                "parse_error": f"{type(error).__name__}: {error}",
            })
    if not any("parse_error" not in record for record in results):
        raise ValueError(f"No records were parsed from {input_file}; check missing HTML or download errors.")

    df = pd.DataFrame(results)

    # Convert date properly
    df["publication_date"] = pd.to_datetime(
        df["publication_date"],
        errors="coerce",
        utc=True
    )

    # Save
    df.to_parquet(
        output_file,
        index=False
    )

    print(df.shape)

    print(
        df[
            [
                "claim",
                "verdict",
                "author",
                "publication_date",
                "analysis_text"
            ]
        ]
        .notna()
        .sum()
    )

    print(df["verdict"].value_counts())

if __name__ == "__main__":
    main()
