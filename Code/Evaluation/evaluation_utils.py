"""Shared paths, dataset handling, metrics, and checkpoints for evaluation."""

import argparse
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = Path("Data/Components/Evidence.parquet")
DEFAULT_OUTPUT_DIR = Path("Results/Evaluation")
LABELS = ["True", "Mostly true", "Half true", "Mostly false", "False", "Pants on fire"]
CANONICAL_LABELS = {label.lower(): label for label in LABELS}
CANONICAL_LABELS["pants on fire!"] = "Pants on fire"


def repo_path(path):
    return REPO_ROOT / Path(path).expanduser()


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than zero.")
    return value


def filename_stem(value):
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise argparse.ArgumentTypeError("Use a filename stem without a directory.")
    return value


def load_environment():
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=False)


def create_client():
    from openai import OpenAI

    load_environment()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY in the environment or repository-root .env file.")
    return OpenAI()


def make_json_serializable(value):
    if isinstance(value, np.ndarray):
        return [make_json_serializable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return make_json_serializable(value.item())
    if isinstance(value, dict):
        return {str(key): make_json_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_serializable(item) for item in value]
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    return value


def write_json(value, path):
    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(make_json_serializable(value), ensure_ascii=False,
                                    indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def load_dataframe(path):
    path = repo_path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True, convert_dates=False)
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            for key in ("data", "statements", "claims", "results"):
                if isinstance(data.get(key), list):
                    return pd.DataFrame(data[key])
            return pd.DataFrame([data])
    raise ValueError(f"Unsupported input format: {suffix}")


def save_dataframe(df, path):
    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".json":
        write_json(df.to_dict(orient="records"), path)
        return
    temporary = path.with_suffix(suffix + ".tmp")
    if suffix == ".parquet":
        df.to_parquet(temporary, index=False)
    elif suffix == ".csv":
        df.to_csv(temporary, index=False)
    elif suffix in {".jsonl", ".ndjson"}:
        df.to_json(temporary, orient="records", lines=True, force_ascii=False, date_format="iso")
    else:
        raise ValueError(f"Unsupported output format: {suffix}")
    os.replace(temporary, path)


def has_value(value):
    value = make_json_serializable(value)
    if value is None:
        return False
    if isinstance(value, (str, list, dict)):
        return bool(value.strip() if isinstance(value, str) else value)
    return True


def normalize_label(label):
    return CANONICAL_LABELS.get(str(label).strip().lower(), "UNKNOWN")


def format_evidence_for_prompt(evidence):
    evidence = make_json_serializable(evidence)
    if isinstance(evidence, (dict, list)):
        return json.dumps(evidence, ensure_ascii=False, indent=2)
    return "" if evidence is None else str(evidence)


def load_data(input_file, required_columns, limit=None):
    df = load_dataframe(input_file)
    required_columns = list(dict.fromkeys(["verdict", *required_columns]))
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {df.columns.tolist()}")
    valid_rows = pd.Series(True, index=df.index)
    for column in required_columns:
        valid_rows &= df[column].map(has_value)
    print(f"Loaded rows: {len(df)}; paired valid rows: {int(valid_rows.sum())}; excluded: {int((~valid_rows).sum())}")
    df = df.loc[valid_rows].copy()
    normalized = df["verdict"].map(normalize_label)
    if normalized.eq("UNKNOWN").any():
        invalid = df.loc[normalized.eq("UNKNOWN"), "verdict"].value_counts().to_dict()
        raise ValueError(f"Unsupported gold verdict labels: {invalid}. Expected the six PolitiFact labels.")
    df["verdict"] = normalized
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive.")
        df = df.head(limit)
    if df.empty:
        raise ValueError("No valid rows remain for evaluation.")
    return df.reset_index(drop=True)


def fingerprint(df):
    payload = json.dumps(make_json_serializable(df.to_dict(orient="records")),
                         sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_results(results, output_file, run_config):
    metrics = {"accuracy_6_class": None, "macro_f1_6_class": None}
    if results:
        gold = [row["gold"] for row in results]
        predictions = [row["prediction"] for row in results]
        metrics = {
            "accuracy_6_class": accuracy_score(gold, predictions),
            "macro_f1_6_class": f1_score(gold, predictions, labels=LABELS,
                                          average="macro", zero_division=0),
        }
    write_json({"run_config": run_config, "metrics": metrics,
                "num_examples": len(results), "results": results}, output_file)


def run_verdict_evaluation(eval_df, claim_col, evidence_col, classify, output_file,
                           settings, overwrite=False, checkpoint_every=50):
    if eval_df.empty:
        raise ValueError("No rows to evaluate.")
    output_file = repo_path(output_file)
    run_config = {**settings, "claim_column": claim_col, "evidence_column": evidence_col,
                  "input_sha256": fingerprint(eval_df)}
    results = []
    if output_file.exists() and not overwrite:
        checkpoint = json.loads(output_file.read_text(encoding="utf-8"))
        if not isinstance(checkpoint, dict) or checkpoint.get("run_config") != run_config:
            raise ValueError("Checkpoint input or settings do not match, or the checkpoint is from an older script. Use a new output prefix or --overwrite.")
        results = checkpoint.get("results")
        if not isinstance(results, list) or len(results) > len(eval_df):
            raise ValueError("Invalid checkpoint row count.")
        for position, item in enumerate(results):
            row = eval_df.iloc[position]
            if (not isinstance(item, dict)
                or item.get("statement") != str(row[claim_col])
                or item.get("gold") != row["verdict"]
                or item.get("evidence") != make_json_serializable(row[evidence_col])
                or item.get("prediction") not in LABELS):
                raise ValueError(f"Checkpoint record {position} does not match this evaluation.")
        print(f"Resuming {evidence_col}: {len(results)}/{len(eval_df)} rows.")

    try:
        for _, row in eval_df.iloc[len(results):].iterrows():
            prediction, usage = classify(str(row[claim_col]), format_evidence_for_prompt(row[evidence_col]))
            if prediction not in LABELS:
                raise ValueError(f"Invalid verdict prediction: {prediction!r}")
            results.append({
                "statement": str(row[claim_col]), "gold": row["verdict"],
                "prediction": prediction, "evidence": make_json_serializable(row[evidence_col]),
                "model": usage["resolved_model"], "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"], "total_tokens": usage["total_tokens"],
                "attempts": usage["attempts"],
            })
            if len(results) % checkpoint_every == 0:
                save_results(results, output_file, run_config)
                print(f"Checkpoint: {len(results)}/{len(eval_df)}")
    finally:
        save_results(results, output_file, run_config)
    print(f"Saved evaluation to {output_file}")
    return results


def safe_name(name):
    return "".join(char if char.isalnum() else "_" for char in name).strip("_") or "evidence"


def experiment(eval_df, claim_col, evidence_cols, output_prefix, backend, classify,
               settings, output_dir=DEFAULT_OUTPUT_DIR, overwrite=False, checkpoint_every=50):
    names = [safe_name(column) for column in evidence_cols]
    if len(set(names)) != len(names):
        raise ValueError("Evidence columns must have distinct names after filename sanitization.")
    outputs = []
    for column, name in zip(evidence_cols, names):
        output = repo_path(output_dir) / f"{output_prefix}_{name}_{backend}.json"
        run_verdict_evaluation(eval_df, claim_col, column, classify, output, settings,
                               overwrite=overwrite, checkpoint_every=checkpoint_every)
        outputs.append(output)
    return outputs


def add_verdict_arguments(parser):
    parser.add_argument("--input-path", "--input", dest="input_path", type=Path, default=DEFAULT_INPUT,
                        help="Input dataset; relative paths resolve from the repository root.")
    parser.add_argument("--claim-col", default="claim")
    parser.add_argument("--evidence-cols", nargs="+", default=["Formatted_Evidence"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", type=filename_stem, default="Results")
    parser.add_argument("--limit", type=positive_int, default=None)
    parser.add_argument("--checkpoint-every", type=positive_int, default=50)
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing checkpoint and rerun all selected rows.")
