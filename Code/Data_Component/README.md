# Data components

These scripts turn collected fact-check articles into metadata-prefixed claims, decontextualized propositions, and extracted evidence. Run commands from the **repository root**; no `cd` into the script directory is needed.

```text
Collected article dataset
  → Claim_Augmentation.py: concatenate metadata, then decontextualize the proposition
  → Evidence_Extraction.py: extract and format article evidence
```

For Snopes, you can run `Leakage_Control_Snopes.py` before this sequence to remove explicit verdict language from article text.

## Files

| File | Role |
| --- | --- |
| `Claim_Augmentation.py` | First concatenates speaker and statement metadata, then uses a model to decontextualize the proposition. `--prefix-only` runs just the local step. |
| `Evidence_Extraction.py` | Uses a model to extract supporting facts, weakening facts, and missing context; formats the result into four labeled sections. |
| `Leakage_Control_Snopes.py` | Local cleanup of explicit rating language and ruling sections; preserves the original article text and deduplicates article URLs. |
| `Leakage_Control_PolitiFact.py` | Empty placeholder; no implemented processing. It is not a required pipeline step. |
| `pipeline_utils.py` | Shared repository-relative paths, dataset validation, atomic file writes, CLI arguments, and API client initialization. |
| `requirements.txt` | Dependencies for the scripts. |
| `tests/test_pipeline.py` | Offline regression tests using sample datasets and mocked model responses. |

Prompts and few-shot examples are maintained in the repository-root [`Prompt/`](../../Prompt/README.md) folder. Processing, model calls, and response validation stay in these scripts.

## Setup and paths

Use Python 3.10 or newer:

```bash
python3 -m pip install -r Code/Data_Component/requirements.txt
```

The model-backed stages read `OPENAI_API_KEY` from the environment or a `.env` file at the repository root. Existing environment values take precedence. Follow the [official OpenAI SDK setup](https://developers.openai.com/api/docs/libraries) for API credentials. API setup happens only when a model-backed stage runs; importing a script or using `--help` does not initialize a client or read a dataset.

The existing model default is `gpt-5.6-luna`; override it with `--model MODEL_NAME` as needed for your API account. The scripts retain their existing request settings and prompts. Live model availability is not checked by the offline tests.

All relative data paths resolve from the repository root, even when the process runs elsewhere. Absolute paths and `~` paths are supported. To launch from outside the repository, use the absolute path to the script.

Every implemented stage accepts these file options:

| Option | Meaning |
| --- | --- |
| `--input-path` | A single Parquet file or records-oriented JSON file. |
| `--output-dir` | Output directory; defaults to `Data/Components` under the repository root. Created automatically. |
| `--output-name` | Filename stem without a directory or extension. Both `.parquet` and `.json` files are written. |

Changing the output directory does not change the next stage's input automatically. Pass its `--input-path` when using custom directories or names. Existing outputs with the same names are replaced, except when explicitly resuming evidence extraction.

## Run the default pipeline

First generate `Data/PolitiFact_DATA.parquet` using the scripts in `Code/Data_Collection`. Then run:

```bash
python3 Code/Data_Component/Claim_Augmentation.py
python3 Code/Data_Component/Evidence_Extraction.py
```

The defaults connect as follows:

| Stage | Default input | Default outputs |
| --- | --- | --- |
| Combined augmentation | `Data/PolitiFact_DATA.parquet` | `Data/Components/Claims_Decontextualized.parquet` and `.json` |
| Evidence extraction | `Data/Components/Claims_Decontextualized.parquet` | `Data/Components/Evidence.parquet` and `.json`, plus `Evidence.run.json` for checkpoint validation |

### 1. Prefix and decontextualize claims

```bash
python3 Code/Data_Component/Claim_Augmentation.py \
  --input-path Data/PolitiFact_DATA.parquet \
  --start-row 0 \
  --data-size 20 \
  --output-name Claims_Decontextualized_Sample
```

Required input columns: `claim` and `analysis_text`. Optional metadata columns: `speaker`, `statement_description`, and `statement_context`. A verdict column is not required or passed to the model.

The script performs both steps in order for the selected rows:

1. Build `metadata` and `augmented_claim_cat` from the original claim and available metadata. For example, `Says this bill passed.` can become `Jane Doe, stated on May 1, 2026 in a speech: this bill passed.` Without usable metadata, the original claim is retained. Missing or blank claims remain missing in the prefixed column.
2. Use article context to resolve references such as “this bill,” “they,” or “today” in the proposition. The model preserves the proposition rather than correcting or fact-checking it. The metadata prefix through the first colon must remain unchanged; changed prefixes or detected verdict leakage trigger fallback to the prefixed claim.

One output contains the original columns, `metadata`, `augmented_claim_cat`, `original_claim`, `input_prefixed_claim`, `decontextualized_claim`, raw model output, status (`augmented`, `unchanged`, or `fallback`), an error field, the model name, and token counts. No intermediate file is needed. Existing prefix columns are recomputed from `claim` and its metadata, so reusing a previous output does not prefix the claim twice.

Additional options:

- `--start-row`: zero-based first row, default `0`.
- `--data-size` (alias `--datasize`): number of rows from that position; default `-1` processes all remaining rows.
- `--model`: model name.
- `--prefix-only`: only concatenate metadata, without API credentials or LLM calls. This requires only `claim` and defaults to `Claims_Metadata.parquet` and `.json`.

For local prefixing only:

```bash
python3 Code/Data_Component/Claim_Augmentation.py --prefix-only
```

The full run defaults to `Claims_Decontextualized.parquet` and `.json`; use `--output-name` to override either mode's filenames. Per-row decontextualization failures retain the prefixed claim. Empty selections fail with an explanatory message.

The full run saves both formats every 25 rows and on normal completion. It does not automatically resume. For manual continuation, use a new output name and set `--start-row` to the original start plus the number of saved rows; preserve earlier chunks separately.

### 2. Extract and format evidence

```bash
python3 Code/Data_Component/Evidence_Extraction.py \
  --input-path Data/Components/Claims_Decontextualized_Sample.parquet \
  --k 2 \
  --output-name Evidence_Sample
```

Required input columns: `decontextualized_claim` and `analysis_text`. Rows with missing or blank values in either column are skipped.

`Extracted_Evidence` contains `supporting_facts`, `weakening_facts`, and `missing_context`. Each fact has an importance score; supporting facts also have a `full` or `partial` support label. `Formatted_Evidence` renders four sections:

```text
Full support: ...
Partial support: ...
Contradiction: ...
Context: ...
```

The formatter selects facts by importance, removes duplicate wording within each category, and uses `None extracted.` for empty sections. The support limit is shared across full and partial support. Model and token-usage fields are retained alongside the original input columns.

Additional options:

- `--k`: maximum facts per extraction category; `2` (default), `3`, or `4`.
- `--data-size` (alias `--datasize`): default `-1` processes all valid rows. A positive value preserves the original balanced-sampling behavior: it must be divisible by six, the dataset must have exactly six distinct verdict classes, and each class must have enough rows. Sampling uses seed 42. Use `-1` for datasets with other verdict schemes.
- `--model`: model name.
- `--resume`: continue a matching JSON checkpoint. Input contents and order, model, and `k` must match the saved `.run.json` manifest.

JSON checkpoints are saved every 50 rows and when a run exits its extraction loop, including on a request failure or interruption. Parquet is written on successful completion. Resume using the same arguments and add `--resume`:

```bash
python3 Code/Data_Component/Evidence_Extraction.py \
  --input-path Data/Components/Claims_Decontextualized_Sample.parquet \
  --k 2 \
  --output-name Evidence_Sample \
  --resume
```

Keep the output JSON and `.run.json` manifest together. Earlier checkpoints without a manifest cannot be resumed by this script. Use a different output name for a different dataset, model, or evidence limit.

## Snopes workflow

The local cleanup stage uses `analysis_text`, `verdict`, and `article_url`. It keeps `analysis_text_raw`, removes explicit verdict language from `analysis_text`, and deduplicates URLs. It writes a separate cleaned dataset by default. This is heuristic cleanup, not a guarantee that all verdict leakage is removed.

```bash
python3 Code/Data_Component/Leakage_Control_Snopes.py

python3 Code/Data_Component/Claim_Augmentation.py \
  --input-path Data/Components/Snopes_Cleaned.parquet \
  --output-name Snopes_Claims_Decontextualized

python3 Code/Data_Component/Evidence_Extraction.py \
  --input-path Data/Components/Snopes_Claims_Decontextualized.parquet \
  --output-name Snopes_Evidence
```

The collected Snopes schema usually lacks PolitiFact-style speaker and statement metadata, so the concatenation stage keeps those claims unchanged unless you supply metadata columns.

## Offline checks

```bash
python3 -m unittest discover -s Code/Data_Component/tests -v
```

Tests use temporary files and mocked API responses. They do not download articles or make paid model requests.

## Check the PolitiFact handoffs

The end-to-end regression test exercises the actual collection, parsing, augmentation, evidence, feasibility, and both verdict-evaluation entry points. It writes and reloads each intermediate dataset from outside the repository. HTTP and model responses are fixtures; this verifies code and file compatibility, not live model quality.

Run this PolitiFact-only integration check from the repository root:

```bash
python3 -m unittest discover -s Code/Data_Component/tests -k test_collection_to_all_evaluations_for_politifact -v
```

After real augmentation and evidence generation, select the generated claim explicitly for evaluation:

```bash
python3 Code/Evaluation/Feasibility.py --text-column decontextualized_claim
python3 Code/Evaluation/verdict_evaluation.py --claim-col decontextualized_claim
python3 Code/Evaluation/verdict_evaluation_qwen.py --claim-col decontextualized_claim
```

The evaluators otherwise default to the original `claim` column. The two verdict backends are alternatives; feasibility is an independent assessment. `Leakage_Control_PolitiFact.py` is an empty placeholder and is not run; the PolitiFact parser already separates `analysis_text` from `ruling_text`.
