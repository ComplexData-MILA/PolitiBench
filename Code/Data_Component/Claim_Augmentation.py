"""Prefix claims with metadata, then decontextualize their propositions."""

import argparse
import re
import time
from collections import Counter
from pathlib import Path
import sys

import pandas as pd
from bs4 import BeautifulSoup

from pipeline_utils import (
    DEFAULT_MODEL, DEFAULT_OUTPUT_DIR, add_file_arguments, create_client,
    data_size, nonnegative_int, read_dataset, save_outputs,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Prompt.claim_augmentation import build_aug_prompt


GENERIC_SPEAKERS = {
    "facebook posts",
    "threads posts",
    "instagram posts",
    "x posts",
    "tiktok posts",
    "viral image",
    "bloggers",
    "chain email",
}

def clean_meta_value(x):
    if x is None or pd.isna(x):
        return None

    x = str(x).strip()

    if not x or x.lower() == "nan":
        return None

    return x.rstrip(":").strip()

def build_metadata_prefix(metadata: dict):
    speaker = clean_meta_value(metadata.get("speaker"))
    desc = clean_meta_value(metadata.get("statement_description"))
    context = clean_meta_value(metadata.get("statement_context"))

    speaker_lower = speaker.lower() if speaker else ""

    # Named speaker/source
    if speaker and speaker_lower not in GENERIC_SPEAKERS:
        if desc:
            return f"{speaker}, {desc}"
        return speaker

    # Generic social/source claims
    if speaker and desc:
        return f"{speaker}, {desc}"

    if context:
        return f"In {context}"

    return ""

def clean_claim_for_prefix(claim: str):
    claim = str(claim).strip()

    patterns = [
        r"^Says\s+that\s+",
        r"^Says\s+",
        r"^Claims\s+that\s+",
        r"^Claims\s+",
    ]

    for pattern in patterns:
        claim = re.sub(pattern, "", claim, flags=re.IGNORECASE).strip()

    return claim

def build_augmented_claim(claim: str, metadata: dict):
    if claim is None or pd.isna(claim):
        return None
    claim = str(claim).strip()
    if not claim:
        return None
    claim_body = clean_claim_for_prefix(claim)

    prefix = build_metadata_prefix(metadata)

    if not prefix:
        return claim

    return f"{prefix}: {claim_body}"

def build_claim_metadata(row):
    return {
        "speaker": row.get("speaker"),
        "statement_description": row.get("statement_description"),
        "statement_context": row.get("statement_context"),
    }

def augment_dataframe(df):
    df = df.copy()
    rows = df.to_dict(orient="records")
    df["metadata"] = [build_claim_metadata(row) for row in rows]
    df["augmented_claim_cat"] = [
        build_augmented_claim(row["claim"], metadata)
        for row, metadata in zip(rows, df["metadata"])
    ]
    return df


CLAIM1_COL = "claim"
CLAIM2_COL = "augmented_claim_cat"
ARTICLE_COL = "analysis_text"


def load_data(input_path, prefix_only=False):
    required = [CLAIM1_COL] if prefix_only else [CLAIM1_COL, ARTICLE_COL]
    return read_dataset(input_path, required)


OUTPUT_LABEL_PATTERN = re.compile(
    r"^(?:decontextualized|augmented|rewritten)\s+claim\s*:\s*",
    flags=re.IGNORECASE,
)

def normalize_text(text) -> str:
    if text is None or pd.isna(text):
        return ""

    return re.sub(r"\s+", " ", str(text)).strip()

def clean_context(text) -> str:
    if text is None or pd.isna(text):
        return ""

    text = BeautifulSoup(
        str(text),
        "html.parser",
    ).get_text(" ", strip=True)

    return normalize_text(text)

def parse_augmentation_output(raw_output: str) -> str:
    """
    Parse a plain-text decontextualized claim.

    Handles:
    - accidental Markdown code fences;
    - accidental output labels;
    - surrounding quotation marks;
    - unnecessary whitespace.
    """
    if raw_output is None:
        raise ValueError("Model returned None")

    text = str(raw_output).strip()

    if not text:
        raise ValueError("Model returned an empty response")

    # Remove optional Markdown fences.
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```(?:text)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    text = text.strip()

    # Remove labels despite the prompt requesting no label.
    text = OUTPUT_LABEL_PATTERN.sub("", text).strip()

    # Remove quotation marks only when they wrap the entire output.
    quote_pairs = [
        ('"', '"'),
        ("“", "”"),
        ("'", "'"),
    ]

    for opening, closing in quote_pairs:
        if (
            text.startswith(opening)
            and text.endswith(closing)
            and len(text) > 1
        ):
            text = text[1:-1].strip()
            break

    text = normalize_text(text)

    if not text:
        raise ValueError("No claim remained after parsing")

    # Catch common instruction-following failures.
    invalid_starts = (
        "here is",
        "here's",
        "the decontextualized claim is",
        "explanation:",
        "analysis:",
        "i cannot",
        "i can't",
    )

    if text.lower().startswith(invalid_starts):
        raise ValueError(
            f"Unexpected model response format: {text[:100]}"
        )

    return text

VERDICT_LEAKAGE_PATTERNS = [
    r"\bmostly true\b",
    r"\bhalf true\b",
    r"\bmostly false\b",
    r"\bpants on fire\b",
    r"\bwe rate\b",
    r"\bour ruling\b",
    r"\bfact[- ]checker(?:'s)? conclusion\b",
    r"\bpolitifact rated\b",
]

def has_verdict_leakage(text: str) -> bool:
    text = normalize_text(text)

    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in VERDICT_LEAKAGE_PATTERNS
    )

def aug_claim(statement: str, context: str, *, client, model=DEFAULT_MODEL,
              max_output_tokens: int = 1000, max_retries: int = 5) -> dict:
    from openai import APIConnectionError, APIError, RateLimitError

    messages = build_aug_prompt(statement, context)

    system_prompt = messages[0]["content"]
    user_prompt = messages[1]["content"]

    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=model,
                instructions=system_prompt,
                input=user_prompt,
                max_output_tokens=max_output_tokens,
                reasoning={"effort": "medium"},
                text={"verbosity": "low"},
                service_tier="default",
            )

            if response.status != "completed":
                raise ValueError(
                    f"Incomplete response: {response.incomplete_details}"
                )

            if response.usage.output_tokens >= max_output_tokens:
                raise ValueError(
                    "Response reached the output-token limit"
                )

            output = response.output_text

            if not output or not output.strip():
                raise ValueError(
                    "The model returned an empty response."
                )

            return {
                "text": output.strip(),
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        except (RateLimitError, APIConnectionError, APIError) as exc:
            if attempt == max_retries - 1:
                raise

            wait_seconds = 2 ** attempt

            print(
                f"API request failed: {type(exc).__name__}. "
                f"Retrying in {wait_seconds} seconds..."
            )

            time.sleep(wait_seconds)

    raise RuntimeError("GPT generation failed")

def run_augmentation(df, datasize=-1, output_dir=DEFAULT_OUTPUT_DIR,
                     start_row=0, output_name=None,
                     model=DEFAULT_MODEL, client=None, prefix_only=False):
    if start_row < 0 or (datasize != -1 and datasize <= 0):
        raise ValueError("start_row must be nonnegative; datasize must be -1 or positive.")
    stop_row = None if datasize == -1 else start_row + datasize
    selected_df = df.iloc[start_row:stop_row].copy()
    if selected_df.empty:
        raise ValueError("No rows selected. Check --start-row, --data-size, and the input dataset.")
    selected_df = augment_dataframe(selected_df)
    output_name = output_name or ("Claims_Metadata" if prefix_only else "Claims_Decontextualized")
    if prefix_only:
        save_outputs(selected_df, output_dir, output_name)
        print(f"Saved {len(selected_df)} claims with concatenated metadata.")
        return selected_df
    if ARTICLE_COL not in selected_df.columns:
        raise ValueError(f"Missing required column: {ARTICLE_COL}")
    if client is None:
        client = create_client()

    records = []
    start_time = time.time()

    for position, (_, row) in enumerate(selected_df.iterrows(), start=1):
        original_claim = normalize_text(row[CLAIM1_COL])
        prefixed_claim = normalize_text(row[CLAIM2_COL])
        context = clean_context(row[ARTICLE_COL])

        raw_output = None
        parsed_claim = prefixed_claim
        status = "fallback"
        error = None

        input_tokens = 0
        output_tokens = 0
        total_tokens = 0

        try:
            if not prefixed_claim:
                raise ValueError("Input claim is empty")

            if not context:
                raise ValueError("Article context is empty")

            generation = aug_claim(
                statement=prefixed_claim, context=context, client=client, model=model
            )

            raw_output = generation["text"]
            input_tokens = generation["input_tokens"]
            output_tokens = generation["output_tokens"]
            total_tokens = generation["total_tokens"]

            parsed_claim = parse_augmentation_output(raw_output)

            prefix, separator, _ = prefixed_claim.partition(":")
            if separator and not parsed_claim.startswith(prefix + ":"):
                raise ValueError("Model changed the metadata prefix")

            if has_verdict_leakage(parsed_claim):
                raise ValueError(
                    "Potential verdict leakage detected"
                )

            status = "unchanged" if (
                normalize_text(parsed_claim)
                == normalize_text(prefixed_claim)
            ) else "augmented"

        except Exception as exc:
            # Safely fall back to the prefixed claim.
            parsed_claim = prefixed_claim
            error = f"{type(exc).__name__}: {exc}"

        record = row.to_dict()

        record.update(
            {
                "original_claim": original_claim,
                "input_prefixed_claim": prefixed_claim,
                "decontextualized_claim": parsed_claim,
                "raw_decontextualization_output": raw_output,
                "decontextualization_status": status,
                "used_decontextualization": status == "augmented",
                "decontextualization_error": error,
                "decontextualization_model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }
        )

        records.append(record)

        if position % 500 == 0 or position == len(selected_df):
            elapsed = time.time() - start_time

            counts = Counter(
                item["decontextualization_status"]
                for item in records
            )

            print(
                f"[{position}/{len(selected_df)}] "
                f"augmented={counts['augmented']} | "
                f"unchanged={counts['unchanged']} | "
                f"fallback={counts['fallback']} | "
                f"elapsed={elapsed:.1f}s"
            )

        # Checkpoints can be read directly by downstream stages.
        if position % 25 == 0:
            save_outputs(pd.DataFrame(records), output_dir, output_name)

    output_df = pd.DataFrame(records)
    save_outputs(output_df, output_dir, output_name)
    print(f"Saved {len(output_df)} decontextualized claims to {output_dir}.")
    print(f"Input tokens: {int(output_df['input_tokens'].sum()):,}")
    print(f"Output tokens: {int(output_df['output_tokens'].sum()):,}")
    return output_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_file_arguments(parser, "Data/PolitiFact_DATA.parquet", None)
    parser.add_argument("--prefix-only", action="store_true",
                        help="Only concatenate metadata; skip the LLM and write Claims_Metadata by default.")
    parser.add_argument("--data-size", "--datasize", type=data_size, default=-1,
                        help="Maximum rows to process; -1 processes all remaining rows.")
    parser.add_argument("--start-row", type=nonnegative_int, default=0,
                        help="Zero-based first input row (default: 0).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name.")
    args = parser.parse_args()
    args.output_name = args.output_name or ("Claims_Metadata" if args.prefix_only else "Claims_Decontextualized")
    return args


def main():
    args = parse_args()
    df = load_data(args.input_path, prefix_only=args.prefix_only)
    augmented_df = run_augmentation(
        df=df, datasize=args.data_size, output_dir=args.output_dir,
        start_row=args.start_row, output_name=args.output_name, model=args.model,
        prefix_only=args.prefix_only,
    )
    display_columns = [CLAIM1_COL, CLAIM2_COL]
    if not args.prefix_only:
        display_columns += ["decontextualized_claim", "decontextualization_status",
                            "decontextualization_error"]
    print(augmented_df[display_columns].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
