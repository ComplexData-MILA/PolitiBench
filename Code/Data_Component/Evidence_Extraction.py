"""Extract and format supporting, weakening, and contextual evidence."""

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
import sys
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from pipeline_utils import (
    DEFAULT_MODEL, DEFAULT_OUTPUT_DIR, add_file_arguments, create_client,
    data_size, read_dataset, repo_path, save_dataset, save_outputs,
)


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Prompt.evidence_extraction import build_extraction_messages


def load_data(input_path, datasize=-1):
    df = read_dataset(input_path, ["decontextualized_claim", "analysis_text"])
    valid = (
        df["decontextualized_claim"].fillna("").astype(str).str.strip().ne("")
        & df["analysis_text"].fillna("").astype(str).str.strip().ne("")
    )
    df = df.loc[valid].copy()
    if datasize > 0:
        if "verdict" not in df.columns:
            raise ValueError("Balanced sampling requires a verdict column.")
        if datasize % 6 != 0:
            raise ValueError("--data-size must be divisible by 6 for balanced sampling.")
        if df["verdict"].nunique() != 6:
            raise ValueError("Balanced sampling requires exactly six verdict classes; use --data-size -1 for all rows.")
        if df.groupby("verdict").size().min() < datasize // 6:
            raise ValueError("Not enough rows in each verdict class for the requested sample.")
        df = (
            df.groupby("verdict", group_keys=False)
              .sample(n=datasize // 6, replace=False, random_state=42)
              .sample(frac=1, random_state=42)
        )
    return df.reset_index(drop=True)


def select_items(items, limit=2):
    if not isinstance(items, list):
        return []

    items = [
        item for item in items
        if isinstance(item, dict) and item.get("fact")
    ]

    items = sorted(
        items,
        key=lambda item: item.get("importance", 0),
        reverse=True,
    )

    selected = []
    seen = set()

    for item in items:
        fact = re.sub(
            r"\s+",
            " ",
            str(item["fact"]),
        ).strip()

        normalized = re.sub(
            r"\W+",
            " ",
            fact.lower(),
        ).strip()

        if not normalized or normalized in seen:
            continue

        seen.add(normalized)

        selected.append({
            **item,
            "fact": fact.rstrip(".") + ".",
        })

        if len(selected) >= limit:
            break

    return selected

def format_evidence(extracted_json, limit=2):
    # Select at most k supporting facts in total before separating.
    support_items = select_items(
        extracted_json.get("supporting_facts", []),
        limit=limit,
    )

    full_support = [
        item["fact"]
        for item in support_items
        if item.get("support_type") == "full"
    ]

    partial_support = [
        item["fact"]
        for item in support_items
        if item.get("support_type") == "partial"
    ]

    weakening_items = select_items(
        extracted_json.get("weakening_facts", []),
        limit=limit,
    )

    context_items = select_items(
        extracted_json.get("missing_context", []),
        limit=limit,
    )

    weakening = [
        item["fact"]
        for item in weakening_items
    ]

    context = [
        item["fact"]
        for item in context_items
    ]

    full_text = (
        " ".join(full_support)
        if full_support
        else "None extracted."
    )

    partial_text = (
        " ".join(partial_support)
        if partial_support
        else "None extracted."
    )

    weakening_text = (
        " ".join(weakening)
        if weakening
        else "None extracted."
    )

    context_text = (
        " ".join(context)
        if context
        else "None extracted."
    )

    return (
        f"Full support: {full_text}\n"
        f"Partial support: {partial_text}\n"
        f"Contradiction: {weakening_text}\n"
        f"Context: {context_text}"
    )

# ====================================================================================
# HELPER FUNCTIONS
# ====================================================================================

def empty_json():
    return {
        "supporting_facts": [],
        "weakening_facts": [],
        "missing_context": [],
    }

class EvidenceFact(BaseModel):
    fact: str
    importance: Literal[1, 2, 3, 4, 5]

class SupportingFact(EvidenceFact):
    support_type: Literal["full", "partial"]

class ExtractedEvidence(BaseModel):
    supporting_facts: list[SupportingFact] = Field(default_factory=list)
    weakening_facts: list[EvidenceFact] = Field(default_factory=list)
    missing_context: list[EvidenceFact] = Field(default_factory=list)

# ====================================================================================
# STEPS
# ====================================================================================

# EXTRACTION
def generate_extraction(statement: str, article: str, k: int = 2,
                        max_retries: int = 5, *, client, model=DEFAULT_MODEL) -> tuple[dict, dict]:
    from openai import APIConnectionError, APITimeoutError, RateLimitError

    if isinstance(article, list):
        article = "\n\n".join(map(str, article))

    statement = str(statement).strip()

    if "stated on" in statement.lower():
        _, separator, proposition = statement.partition(":")
        if separator and proposition.strip():
            statement = proposition.strip()

    if not article:
        return empty_json(), {
            "resolved_model": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "attempts": 0,
            "status": "skipped_empty_article",
        }

    messages = build_extraction_messages(
        statement,
        article,
        k=k,
    )

    system_prompt = messages[0]["content"]
    input_messages = messages[1:]

    for attempt in range(max_retries):
        try:
            response = client.responses.parse(
                model=model,
                instructions=system_prompt,
                input=input_messages,
                reasoning={"effort": "medium"},
                service_tier="default",
                text_format=ExtractedEvidence,
            )

            if response.status != "completed":
                raise RuntimeError(
                    f"Incomplete response: {response.incomplete_details}"
                )

            parsed = response.output_parsed

            if parsed is None:
                raise ValueError(
                    "The API returned no parsed extraction. "
                    f"Raw output: {response.output_text!r}"
                )

            evidence = parsed.model_dump()

            # Enforce k even if the model returns too many items.
            for category in [
                "supporting_facts",
                "weakening_facts",
                "missing_context",
            ]:
                evidence[category] = sorted(
                    evidence[category],
                    key=lambda item: item["importance"],
                    reverse=True,
                )[:k]

            usage = {
                "resolved_model": response.model,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
                "attempts": attempt + 1,
            }

            return evidence, usage

        except (
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
        ) as exc:
            if attempt == max_retries - 1:
                raise

            wait_seconds = min(2 ** attempt, 30)

            print(
                f"API error: {exc}. "
                f"Retrying in {wait_seconds}s...",
                flush=True,
            )

            time.sleep(wait_seconds)

    raise RuntimeError("Unreachable")

def experiment(df, k=2, output_dir=DEFAULT_OUTPUT_DIR,
               output_name="Evidence", model=DEFAULT_MODEL, resume=False, client=None):
    if df.empty:
        raise ValueError("No rows with a nonempty claim and analysis_text to process.")
    output_dir = repo_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{output_name}.json"
    manifest_file = output_dir / f"{output_name}.run.json"
    run_config = {
        "input_sha256": hashlib.sha256(
            df.to_json(orient="records", date_format="iso").encode("utf-8")
        ).hexdigest(),
        "model": model,
        "k": k,
    }

    completed_rows = []
    if resume:
        if not output_file.exists() or not manifest_file.exists():
            raise ValueError("Resume requires an existing output JSON and matching .run.json file.")
        if json.loads(manifest_file.read_text()) != run_config:
            raise ValueError("Checkpoint input, model, or k differs from this run. Use a new output name.")
        completed_rows = json.loads(output_file.read_text())
        if not isinstance(completed_rows, list) or len(completed_rows) > len(df):
            raise ValueError("Invalid checkpoint row count.")
        expected = {
            "Extracted_Evidence", "Formatted_Evidence", "extraction_model",
            "extraction_input_tokens", "extraction_output_tokens", "extraction_total_tokens",
        }
        if any(not isinstance(row, dict) or not expected.issubset(row) for row in completed_rows):
            raise ValueError("Invalid evidence checkpoint records.")
        # JSON converts dates and arrays. Restore input values from the verified
        # source so resumed rows retain the same types as newly processed rows.
        completed_rows = [
            {**source, **{key: saved[key] for key in expected}}
            for source, saved in zip(
                df.iloc[:len(completed_rows)].to_dict(orient="records"), completed_rows
            )
        ]
        print(f"Resuming from row {len(completed_rows)}/{len(df)}", flush=True)

    if len(completed_rows) < len(df) and client is None:
        client = create_client()

    if not resume:
        # Reset the old checkpoint before associating it with this run's configuration.
        save_dataset(pd.DataFrame(), output_file)
        temporary_manifest = manifest_file.with_suffix(".json.tmp")
        temporary_manifest.write_text(json.dumps(run_config, indent=2))
        os.replace(temporary_manifest, manifest_file)

    start = time.perf_counter()
    new_calls = 0
    try:
        for _, row in df.iloc[len(completed_rows):].iterrows():
            evidence, usage = generate_extraction(
                statement=row["decontextualized_claim"], article=row["analysis_text"],
                k=k, client=client, model=model,
            )
            result_row = row.to_dict()
            result_row.update({
                "Extracted_Evidence": evidence,
                "Formatted_Evidence": format_evidence(evidence, limit=k),
                "extraction_model": usage["resolved_model"],
                "extraction_input_tokens": usage["input_tokens"],
                "extraction_output_tokens": usage["output_tokens"],
                "extraction_total_tokens": usage["total_tokens"],
            })
            completed_rows.append(result_row)
            new_calls += 1
            if len(completed_rows) % 50 == 0:
                save_dataset(pd.DataFrame(completed_rows), output_file)
                print(f"Checkpoint saved: {len(completed_rows)}/{len(df)}", flush=True)
    finally:
        # Keep completed work when a request fails or the user interrupts a run.
        save_dataset(pd.DataFrame(completed_rows), output_file)

    result = pd.DataFrame(completed_rows)
    save_outputs(result, output_dir, output_name)
    print(f"New API calls: {new_calls}")
    print(f"Extraction time this run: {time.perf_counter() - start:.2f}s")
    print(f"Saved evidence to {output_file} and {output_file.with_suffix('.parquet')}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and format evidence for decontextualized claims.")
    add_file_arguments(parser, "Data/Components/Claims_Decontextualized.parquet", "Evidence")
    parser.add_argument("--data-size", "--datasize", type=data_size, default=-1,
                        help="-1 uses all valid rows; a positive multiple of 6 samples six verdict classes equally.")
    parser.add_argument("--k", type=int, default=2, choices=[2, 3, 4],
                        help="Maximum evidence facts per category.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name.")
    parser.add_argument("--resume", action="store_true", help="Resume a matching evidence checkpoint.")
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_data(args.input_path, datasize=args.data_size)
    experiment(df=df, k=args.k, output_dir=args.output_dir,
               output_name=args.output_name, model=args.model, resume=args.resume)


if __name__ == "__main__":
    main()
