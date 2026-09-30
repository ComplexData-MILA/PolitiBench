"""Rate whether a claim contains enough information to be fact-checked."""

import argparse
import json
from pathlib import Path
import re
import time
import sys

import pandas as pd

from evaluation_utils import (
    DEFAULT_OUTPUT_DIR, filename_stem, fingerprint, has_value, load_dataframe,
    positive_int, repo_path, save_dataframe, write_json,
)
from local_model import LocalTextModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Prompt.feasibility import build_messages


DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_INPUT = Path("Data/Components/Claims_Decontextualized.parquet")

def parse_response(response: str) -> dict:
    response = response.strip()

    # Parse the final "| 0", "| 1", or "| 2".
    match = re.search(r"\|\s*([012])\s*$", response)

    if not match:
        return {
            "feasibility_explanation": None,
            "feasibility_rating": None,
            "feasibility_parse_success": False,
            "feasibility_error": "Expected a final `| 0`, `| 1`, or `| 2`.",
        }

    rating = int(match.group(1))
    explanation = response[:match.start()].strip()

    if not explanation:
        return {
            "feasibility_explanation": None,
            "feasibility_rating": rating,
            "feasibility_parse_success": False,
            "feasibility_error": "The explanation was empty.",
        }

    return {
        "feasibility_explanation": explanation,
        "feasibility_rating": rating,
        "feasibility_parse_success": True,
        "feasibility_error": None,
    }

def run_feasibility(df, text_column, output_path, model_name=DEFAULT_MODEL,
                    device="auto", max_new_tokens=160, checkpoint_every=25,
                    resume=False, generator=None):
    if text_column not in df.columns:
        raise ValueError(f"Missing text column: {text_column}. Available columns: {df.columns.tolist()}")
    if df.empty:
        raise ValueError("No rows selected for feasibility evaluation.")
    df = df.copy().reset_index(drop=True)
    output_path = repo_path(output_path)
    if output_path.suffix.lower() not in {".parquet", ".json", ".csv", ".jsonl", ".ndjson"}:
        raise ValueError("Output must be Parquet, JSON, CSV, JSONL, or NDJSON.")
    manifest_path = output_path.with_suffix(output_path.suffix + ".run.json")
    config = {
        "input_sha256": fingerprint(df), "text_column": text_column,
        "model": model_name, "device": device, "max_new_tokens": max_new_tokens,
    }
    fields = {
        "feasibility_raw_output": None, "feasibility_explanation": None,
        "feasibility_rating": None, "feasibility_parse_success": False,
        "feasibility_error": None, "feasibility_model": model_name,
        "feasibility_text_column": text_column,
    }
    checkpoint = None
    if resume:
        if not output_path.exists() or not manifest_path.exists():
            raise ValueError("Resume requires the output dataset and its .run.json manifest.")
        if json.loads(manifest_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Feasibility checkpoint input or settings differ from this run.")
        checkpoint = load_dataframe(output_path)
        if len(checkpoint) != len(df) or any(column not in checkpoint for column in fields):
            raise ValueError("Invalid feasibility checkpoint schema or row count.")

    for column, default in fields.items():
        values = checkpoint[column].tolist() if checkpoint is not None else [default] * len(df)
        df[column] = pd.Series(values, dtype=object)
    # A rating without a successfully parsed explanation must be retried.
    completed = (
        df["feasibility_parse_success"].eq(True)
        & df["feasibility_rating"].isin([0, 1, 2])
        & df["feasibility_explanation"].map(has_value)
    )
    df["feasibility_parse_success"] = completed.astype(bool)
    pending = ~completed & df[text_column].map(has_value)
    if pending.any() and generator is None:
        generator = LocalTextModel(model_name, device=device, max_new_tokens=max_new_tokens)
        generator._load()

    save_dataframe(df, output_path)
    write_json(config, manifest_path)
    started = time.perf_counter()
    try:
        for position in df.index:
            if completed.iloc[position]:
                continue
            statement = df.at[position, text_column]
            if not has_value(statement):
                df.at[position, "feasibility_error"] = "Missing statement."
            else:
                try:
                    raw_output, _ = generator.generate(build_messages(str(statement)))
                    parsed = parse_response(raw_output)
                    df.at[position, "feasibility_raw_output"] = raw_output
                    for column, value in parsed.items():
                        df.at[position, column] = value
                except Exception as error:
                    df.at[position, "feasibility_raw_output"] = None
                    df.at[position, "feasibility_explanation"] = None
                    df.at[position, "feasibility_rating"] = None
                    df.at[position, "feasibility_parse_success"] = False
                    df.at[position, "feasibility_error"] = f"{type(error).__name__}: {error}"
            if (position + 1) % checkpoint_every == 0:
                save_dataframe(df, output_path)
                print(f"Processed {position + 1}/{len(df)} rows.")
    finally:
        save_dataframe(df, output_path)
    ratings = pd.to_numeric(df.loc[df["feasibility_parse_success"], "feasibility_rating"])
    print(f"Saved feasibility results to {output_path} in {time.perf_counter() - started:.1f}s")
    print(ratings.value_counts().sort_index().to_string())
    return df


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", "--input", dest="input_path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", type=filename_stem, default="Feasibility",
                        help="Filename stem for the default Parquet output.")
    parser.add_argument("--output-path", "--output", dest="output_path", type=Path,
                        help="Explicit output file; overrides --output-dir and --output-name.")
    parser.add_argument("--text-column", default="claim")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--max-new-tokens", type=positive_int, default=160)
    parser.add_argument("--checkpoint-every", type=positive_int, default=25)
    parser.add_argument("--limit", type=positive_int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume this output with matching input and settings.")
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_dataframe(args.input_path)
    if args.limit is not None:
        df = df.head(args.limit)
    output_path = args.output_path or args.output_dir / f"{args.output_name}.parquet"
    run_feasibility(df, args.text_column, output_path, model_name=args.model,
                    device=args.device, max_new_tokens=args.max_new_tokens,
                    checkpoint_every=args.checkpoint_every, resume=args.resume)


if __name__ == "__main__":
    main()
