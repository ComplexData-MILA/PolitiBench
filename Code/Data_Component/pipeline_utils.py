"""Shared paths, dataset I/O, and runtime configuration for preprocessing."""

import argparse
import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path("Data/Components")
DEFAULT_MODEL = "gpt-5.6-luna"


def repo_path(path):
    """Resolve relative paths from the repository, preserving absolute paths."""
    return REPO_ROOT / Path(path).expanduser()


def read_dataset(path, required_columns=()):
    path = repo_path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".json":
        df = pd.read_json(path, orient="records", convert_dates=False)
    else:
        raise ValueError("Input must be a .parquet or records-oriented .json file.")
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return df


def save_dataset(df, output_path):
    """Atomically write a single dataset file."""
    output_path = repo_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if output_path.suffix == ".parquet":
        df.to_parquet(temporary_path, index=False)
    elif output_path.suffix == ".json":
        df.to_json(temporary_path, orient="records", indent=2,
                   force_ascii=False, date_format="iso")
    else:
        raise ValueError("Output must be a .parquet or .json file.")
    os.replace(temporary_path, output_path)


def save_outputs(df, output_dir, output_name):
    output_dir = repo_path(output_dir)
    for suffix in (".parquet", ".json"):
        save_dataset(df, output_dir / f"{output_name}{suffix}")


def output_name(value):
    if not value or value in {".", ".."} or Path(value).name != value or "/" in value or "\\" in value:
        raise argparse.ArgumentTypeError("Use a filename stem without a directory.")
    if value.endswith((".parquet", ".json")):
        raise argparse.ArgumentTypeError("Omit the .parquet or .json extension.")
    return value


def data_size(value):
    value = int(value)
    if value != -1 and value <= 0:
        raise argparse.ArgumentTypeError("Use -1 for all rows or a positive row count.")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("Value must be zero or greater.")
    return value


def add_file_arguments(parser, input_path, default_name):
    parser.add_argument("--input-path", type=Path, default=Path(input_path),
                        help="Input Parquet or JSON file, relative to the repository root or absolute.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory relative to the repository root or absolute (default: Data/Components).")
    parser.add_argument("--output-name", type=output_name, default=default_name,
                        help="Output filename stem; both .parquet and .json are written.")


def create_client():
    """Load credentials only when an API-backed stage actually runs."""
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(REPO_ROOT / ".env", override=False)
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY in the environment or the repository-root .env file.")
    return OpenAI()
