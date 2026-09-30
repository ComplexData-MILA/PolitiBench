import argparse
import re
from html import escape
from pathlib import Path
from urllib.parse import urljoin
import json
import pandas as pd
from bs4 import BeautifulSoup, NavigableString

REPO_ROOT = Path(__file__).resolve().parents[2]

VERDICT_CLASS_MAP = {
    "m-statement--true": "True",
    "m-statement--mostly-true": "Mostly True",
    "m-statement--half-true": "Half True",
    "m-statement--barely-true": "Mostly False",
    "m-statement--mostly-false": "Mostly False",
    "m-statement--false": "False",
    "m-statement--fire": "Pants on Fire",
    "m-statement--pants-fire": "Pants on Fire",
}

VERDICT_TEXT_MAP = {
    "true": "True",
    "mostly true": "Mostly True",
    "half true": "Half True",
    "half-true": "Half True",
    "mostly false": "Mostly False",
    "false": "False",
    "pants on fire": "Pants on Fire",
    "pants on fire!": "Pants on Fire",
}

def parse_statement_new(soup):

    result = {
        "speaker": None,
        "statement_description": None,
        "claim": None,
        "verdict": None,
        "tags": [],
    }

    # Find:
    # "stated on June 17, 2026 in an advertisement:"
    desc_string = soup.find(
        string=lambda text:
            isinstance(text, str)
            and re.match(
                r"^\s*stated on .+? in .+?:?\s*$",
                text,
                flags=re.IGNORECASE,
            )
    )

    if desc_string is None:
        return result

    desc_tag = desc_string.parent

    result["statement_description"] = (
        desc_string.strip()
    )

    # Speaker normally appears immediately before description
    speaker_tag = desc_tag.find_previous(
        "a",
        href=True,
    )

    if speaker_tag:
        result["speaker"] = clean_text(
            speaker_tag
        )

    # Find rating image after statement description
    verdict_img = None

    for element in desc_tag.next_elements:

        if getattr(element, "name", None) == "img":

            alt = element.get("alt", "").strip()

            normalized = alt.lower()

            if normalized in VERDICT_TEXT_MAP:
                verdict_img = element
                result["verdict"] = (
                    VERDICT_TEXT_MAP[normalized]
                )
                break

    # Everything textual between the description
    # and rating image should contain the claim.
    claim_parts = []

    if verdict_img is not None:

        for element in desc_tag.next_elements:

            if element is verdict_img:
                break

            if not isinstance(
                element,
                NavigableString,
            ):
                continue

            text = str(element).strip()

            if not text:
                continue

            if re.match(
                r"^stated on ",
                text,
                flags=re.IGNORECASE,
            ):
                continue

            if text not in claim_parts:
                claim_parts.append(text)

    if claim_parts:
        result["claim"] = " ".join(
            claim_parts
        )

    return result

def split_article_text(soup, base_url):

    short_summary = soup.select_one(
        ".short-on-time"
    )

    sources_section = soup.select_one(
        "section#sources"
    )

    if short_summary is None:
        return None, None, None

    full_parts = []
    analysis_parts = []
    ruling_parts = []

    inside_ruling = False

    node = short_summary.find_next(
        ["h2", "h3", "h4", "p"]
    )

    while node is not None:

        # Skip anything actually inside the
        # short-summary box itself
        if short_summary in node.parents:
            node = node.find_next(
                ["h2", "h3", "h4", "p"]
            )
            continue

        # Stop once we enter sources
        if (
            sources_section is not None
            and (
                node is sources_section
                or sources_section in node.parents
            )
        ):
            break

        plain_text = clean_text(node)

        if not plain_text:
            node = node.find_next(
                ["h2", "h3", "h4", "p"]
            )
            continue

        normalized = re.sub(
            r"[^a-z]+",
            " ",
            plain_text.lower(),
        ).strip()

        if (
            node.name in {"h2", "h3", "h4"}
            and normalized in {
                "our ruling",
                "our rating",
                "our conclusion",

                # Spanish
                "nuestro veredicto",
                "nuestra calificacion",
                "nuestra conclusion",
            }
        ):
            inside_ruling = True

            full_parts.append(
                plain_text
            )

            node = node.find_next(
                ["h2", "h3", "h4", "p"]
            )

            continue

        is_ruling_sentence = (
            re.match(
                r"^(we rate|we rated) "
                r"(this|the) "
                r"(claim|statement)",
                plain_text,
                flags=re.IGNORECASE,
            )
            is not None
            or
            re.match(
                r"^calificamos "
                r"(esta|la) "
                r"(declaraci[oó]n|afirmaci[oó]n)",
                plain_text,
                flags=re.IGNORECASE,
            )
            is not None
        )

        if is_ruling_sentence:
            inside_ruling = True

            text_with_links = clean_text_with_links(
                node,
                base_url,
            )

            if text_with_links:
                full_parts.append(text_with_links)
                ruling_parts.append(text_with_links)

            node = node.find_next(
                ["h2", "h3", "h4", "p"]
            )
            continue

        text_with_links = (
            clean_text_with_links(
                node,
                base_url,
            )
        )

        if text_with_links:

            full_parts.append(
                text_with_links
            )

            if inside_ruling:
                ruling_parts.append(
                    text_with_links
                )
            else:
                analysis_parts.append(
                    text_with_links
                )

        node = node.find_next(
            ["h2", "h3", "h4", "p"]
        )

    return (
        "\n\n".join(full_parts)
        if full_parts else None,

        "\n\n".join(analysis_parts)
        if analysis_parts else None,

        "\n\n".join(ruling_parts)
        if ruling_parts else None,
    )

def clean_text(tag):
    if tag is None:
        return None

    text = tag.get_text(
        " ",
        strip=True,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text or None

def clean_text_with_links(tag, base_url):
    if tag is None:
        return None

    fragment = BeautifulSoup(
        str(tag),
        "html.parser",
    )

    root = fragment.find()

    if root is None:
        return None

    for anchor in root.select("a[href]"):
        href = anchor.get("href")

        if not href:
            continue

        absolute_url = urljoin(
            base_url,
            href,
        )

        anchor_text = anchor.get_text(
            " ",
            strip=True,
        )

        safe_url = escape(
            absolute_url,
            quote=True,
        )

        safe_anchor_text = escape(
            anchor_text,
            quote=False,
        )

        if safe_anchor_text:
            replacement = (
                f'<a href="{safe_url}">'
                f"{safe_anchor_text}"
                f"</a>"
            )
        else:
            replacement = (
                f'<a href="{safe_url}">'
                f"{safe_url}"
                f"</a>"
            )

        anchor.replace_with(
            NavigableString(replacement)
        )

    text = root.get_text(
        " ",
        strip=True,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text or None

def get_meta_content(soup, *, name=None, property_name=None):
    if name is not None:
        tag = soup.find(
            "meta",
            attrs={"name": name},
        )
    else:
        tag = soup.find(
            "meta",
            attrs={"property": property_name},
        )

    if tag is None:
        return None

    value = tag.get("content")

    return value.strip() if value else None

def parse_verdict(statement_tag, soup):
    if statement_tag is not None:
        classes = statement_tag.get(
            "class",
            [],
        )

        for css_class, verdict in VERDICT_CLASS_MAP.items():
            if css_class in classes:
                return verdict

    image_url = get_meta_content(
        soup,
        property_name="og:image",
    )

    if not image_url:
        return None

    image_url = image_url.lower()

    fallback_map = {
        "meter-mostly-true": "Mostly True",
        "meter-half-true": "Half True",
        "meter-mostly-false": "Mostly False",
        "meter-barely-true": "Mostly False",
        "meter-pants-fire": "Pants on Fire",
        "meter-false": "False",
        "meter-true": "True",
    }

    for pattern, verdict in fallback_map.items():
        if pattern in image_url:
            return verdict

    return None

def build_context(short_summary, analysis_text, ruling_text):
    sections = []

    if short_summary:
        sections.append(
            "SHORT SUMMARY\n"
            f"{short_summary}"
        )

    if analysis_text:
        sections.append(
            "ARTICLE\n"
            f"{analysis_text}"
        )

    if ruling_text:
        sections.append(
            "OUR RULING\n"
            f"{ruling_text}"
        )

    return (
        "\n\n".join(sections)
        if sections
        else None
    )

def parse_statement_description(description):
    result = {
        "statement_date": None,
        "statement_context": None,
    }

    if not description:
        return result

    match = re.match(
        r"^stated on (.+?) in (.+?):?$",
        description,
        flags=re.IGNORECASE,
    )

    if not match:
        return result

    result["statement_date"] = (
        match.group(1).strip()
    )

    result["statement_context"] = (
        match.group(2).strip()
    )

    return result

def parse_sources(soup, base_url):
    sources_section = soup.select_one(
        "section#sources"
    )

    if sources_section is None:
        return None, []

    sources_text = sources_section.get_text(
        "\n",
        strip=True,
    )

    source_urls = []

    for anchor in sources_section.select(
        "a[href]"
    ):
        href = anchor.get("href")

        if href:
            source_urls.append(
                urljoin(
                    base_url,
                    href,
                )
            )

    source_urls = list(
        dict.fromkeys(source_urls)
    )

    return (
        sources_text or None,
        source_urls,
    )

def parse_politifact_html(html, source_url=None):

    if (
        not isinstance(html, str)
        or not html.strip()
    ):
        raise ValueError(
            "HTML is empty or missing"
        )

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    canonical_tag = soup.find(
        "link",
        rel="canonical",
    )

    canonical_url = (
        canonical_tag.get("href")
        if canonical_tag
        else None
    )

    base_url = (
        canonical_url
        or source_url
        or "https://www.politifact.com"
    )

    statement = parse_statement_new(soup)

    speaker = statement["speaker"]
    statement_description = (
        statement["statement_description"]
    )
    claim = statement["claim"]
    verdict = statement["verdict"]
    tags = statement["tags"]

    tags = [
        tag
        for tag in tags
        if tag
    ]

    title = (
        get_meta_content(
            soup,
            property_name="og:title",
        )
        or clean_text(soup.find("h1"))
    )

    author = get_meta_content(
        soup,
        name="author",
    )

    if author is None:
        author = clean_text(
            soup.select_one(
                ".m-author__content a"
            )
        )

    publication_date = (
        get_meta_content(
            soup,
            property_name="article:published_time",
        )
    )

    if publication_date is None:

        time_tag = soup.find(
            "time",
            attrs={"datetime": True},
        )

        if time_tag:
            publication_date = time_tag.get(
                "datetime"
            )

    description = get_meta_content(
        soup,
        name="description",
    )

    keywords_raw = get_meta_content(
        soup,
        name="keywords",
    )

    keywords = (
        [
            keyword.strip()
            for keyword in keywords_raw.split(",")
            if keyword.strip()
        ]
        if keywords_raw
        else []
    )

    short_summary_tag = soup.select_one(
        ".short-on-time"
    )

    short_summary = clean_text_with_links(
        tag=short_summary_tag,
        base_url=base_url,
    )

    (
        article_text,
        analysis_text,
        ruling_text,
    ) = split_article_text(
        soup=soup,
        base_url=base_url,
    )

    context = build_context(
        short_summary=short_summary,
        analysis_text=analysis_text,
        ruling_text=ruling_text,
    )

    (
        sources_text,
        source_urls,
    ) = parse_sources(
        soup=soup,
        base_url=base_url,
    )

    statement_parts = (
        parse_statement_description(
            statement_description
        )
    )

    return {
        "canonical_url": canonical_url,
        "title": title,
        "description": description,
        "keywords": keywords,
        "author": author,
        "publication_date": publication_date,
        "speaker": speaker,
        "statement_description": statement_description,
        "statement_date": statement_parts[
            "statement_date"
        ],
        "statement_context": statement_parts[
            "statement_context"
        ],
        "claim": claim,
        "verdict": verdict,
        "tags": tags,
        "context": context,
        "short_summary": short_summary,
        "analysis_text": analysis_text,
        "ruling_text": ruling_text,
        "sources_text": sources_text,
        "source_urls": source_urls,
        "og_image": get_meta_content(
            soup,
            property_name="og:image",
        ),
    }

def get_html_column(raw_df):
    possible_columns = [
        "raw_html",
        "html_file",
        "html",
    ]

    for column in possible_columns:
        if column in raw_df.columns:
            return column

    raise ValueError(
        "Could not find an HTML column. "
        "Expected one of: "
        "'raw_html', 'html_file', or 'html'."
    )

def parse_dataframe(raw_df):
    html_column = get_html_column(
        raw_df
    )

    records = []
    failed_records = []

    for row in raw_df.to_dict(
        orient="records"
    ):
        article_url = row.get(
            "article_url"
        )

        html = row.get(
            html_column
        )

        try:
            parsed_record = (
                parse_politifact_html(
                    html=html,
                    source_url=article_url,
                )
            )

            record = {
                "article_url": article_url,
                **parsed_record,
            }

            records.append(
                record
            )

        except Exception as error:
            failed_records.append({
                "article_url": article_url,
                "error": str(error),
            })

            print(
                f"FAILED TO PARSE "
                f"{article_url}: {error}"
            )

    processed_df = pd.DataFrame(
        records
    )

    for date_column in [
        "publication_date",
        "statement_date",
    ]:
        if date_column in processed_df.columns:
            processed_df[date_column] = (
                pd.to_datetime(
                    processed_df[date_column],
                    errors="coerce",
                )
            )

    return (
        processed_df,
        failed_records,
    )

def process_parquet_file(
    input_file = "Data/PolitiFact_HTML.parquet",
    output_file = "Data/PolitiFact_DATA.parquet",
):
    input_file = REPO_ROOT / Path(input_file).expanduser()
    output_file = REPO_ROOT / Path(output_file).expanduser()
    print(
        f"Reading {input_file}"
    )

    raw_df = pd.read_parquet(
        input_file
    )

    (
        processed_df,
        failed_records,
    ) = parse_dataframe(
        raw_df
    )

    output_file = Path(output_file)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if processed_df.empty:
        raise ValueError(f"No records were parsed from {input_file}; no output dataset was written.")

    processed_df.to_parquet(
        output_file,
        index=False,
        compression="zstd",
    )

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract flat structured metadata "
            "and linked article text from "
            "raw PolitiFact HTML."
        )
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path(
            "Data/PolitiFact_HTML.parquet"
        ),
        help=(
            "Raw Parquet file, relative to the repository root or absolute."
        ),
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

def main():
    args = parse_args()

    process_parquet_file(
        input_file=args.input_path,
        output_file=args.output_path or args.output_dir / "PolitiFact_DATA.parquet",
    )

if __name__ == "__main__":
    main()
