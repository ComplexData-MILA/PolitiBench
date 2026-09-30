# PolitiBench

PolitiBench collects PolitiFact articles, prepares standalone claims, extracts article evidence, and evaluates claim feasibility and verdict prediction. This guide covers the **PolitiFact workflow**.

Run all commands below from the repository root. Relative data paths resolve from the repository root even when a script is launched from another directory.

## Workflow

```text
PolitiFact website
  → PolitiFact_RawScrape.py     Download article HTML into a Parquet dataset
  → PolitiFact_ScrapeData.py    Extract claims, metadata, analysis, and verdicts
  → Claim_Augmentation.py      Prefix metadata, then decontextualize propositions
  → Evidence_Extraction.py     Extract and format evidence
  → Verdict evaluation        Predict labels and compute accuracy and macro F1

Augmented claims → Feasibility.py → Assess whether each claim can be fact-checked
```

Feasibility is an independent evaluation. OpenAI and Qwen verdict evaluation are alternative backends; neither needs to run before the other.

## Repository structure

| Directory | Contents |
| --- | --- |
| [Code/Data_Collection](Code/Data_Collection/README.md) | Download article HTML and parse it into structured datasets. |
| [Code/Data_Component](Code/Data_Component/README.md) | Combined claim augmentation and evidence extraction. |
| [Code/Evaluation](Code/Evaluation/README.md) | Feasibility assessment and OpenAI/Qwen verdict evaluation. |
| [Prompt](Prompt/README.md) | Model instructions, few-shot examples, and message builders. |
| `Data/` | Default raw HTML and parsed article datasets. |
| `Data/Components/` | Default augmented claims and extracted evidence. |
| `Results/Evaluation/` | Default evaluation results and checkpoints. |

The scripts create output directories when needed. Prompt files and shared helper modules are imported by the scripts; they do not need to be run separately.

## Setup

Use Python 3.10 or newer. A virtual environment keeps project dependencies separate:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install requests \
  -r Code/Data_Component/requirements.txt \
  -r Code/Evaluation/requirements.txt
```

For local Qwen verdict evaluation or feasibility assessment, also install:

```bash
python3 -m pip install -r Code/Evaluation/requirements-local.txt
```

Augmentation, evidence extraction, and OpenAI verdict evaluation require `OPENAI_API_KEY` in the environment or a `.env` file at the repository root:

```dotenv
OPENAI_API_KEY=your_api_key_here
```

Local model loading also reads `HF_TOKEN` if needed. Environment values take precedence over `.env`. Model loading and API client initialization happen when inference is needed, not when displaying `--help`.

The configured model defaults are `gpt-5.6-luna` for API stages, `Qwen/Qwen3-4B` for local verdict evaluation, and `Qwen/Qwen3-8B` for feasibility. Use `--model` to select another compatible model available to you. Local models require sufficient memory and download uncached weights when loaded.

## Run a small PolitiFact pipeline

Run each step after the preceding step succeeds. These commands use matching default file paths.

### 1. Collect raw HTML

```bash
python3 Code/Data_Collection/PolitiFact_RawScrape.py --limit 1
```

Output: `Data/PolitiFact_HTML.parquet`.

`--limit` counts listing pages, not articles. The dataset stores each article's HTML in an `html` column alongside its URL and download metadata; it does not create separate HTML files. The scraper's default limit is 200 pages, and `-1` removes the page limit. An optional `--time-stamp YYYYMMDD` sets the earliest article date.

### 2. Parse the articles

```bash
python3 Code/Data_Collection/PolitiFact_ScrapeData.py
```

Output: `Data/PolitiFact_DATA.parquet`.

This extracts claims, verdicts, speaker and statement metadata, summaries, sources, and article analysis. `analysis_text` and `ruling_text` are separate. The empty `Leakage_Control_PolitiFact.py` placeholder is not part of the runnable pipeline.

### 3. Prefix and decontextualize claims

```bash
python3 Code/Data_Component/Claim_Augmentation.py --data-size 20
```

Outputs: `Data/Components/Claims_Decontextualized.parquet` and `.json`.

The combined script first builds `augmented_claim_cat` using available metadata, then calls the model to resolve contextual references in the proposition. It retains the original `claim` and adds `decontextualized_claim`, status/error fields, and token counts. A separate metadata-generation run is not required.

Omit `--data-size` to process all rows. Use `--start-row` to select a zero-based starting row. For local prefixing only:

```bash
python3 Code/Data_Component/Claim_Augmentation.py --prefix-only
```

That mode requires no API call and writes `Claims_Metadata.parquet` and `.json`; it does not produce the decontextualized input required by the next stage.

### 4. Extract evidence

```bash
python3 Code/Data_Component/Evidence_Extraction.py --k 2
```

Outputs: `Data/Components/Evidence.parquet`, `.json`, and `Evidence.run.json`.

This reads the decontextualized claims and `analysis_text`. It adds structured `Extracted_Evidence` and a `Formatted_Evidence` column with full support, partial support, contradiction, and context sections. `--k` accepts 2, 3, or 4 facts per extraction category. Missing or blank claims/article text are excluded.

### 5. Evaluate verdict prediction

Choose the OpenAI backend:

```bash
python3 Code/Evaluation/verdict_evaluation.py \
  --claim-col decontextualized_claim
```

Output: `Results/Evaluation/Results_Formatted_Evidence_GPT.json`.

Or use local Qwen:

```bash
python3 Code/Evaluation/verdict_evaluation_qwen.py \
  --claim-col decontextualized_claim
```

Output: `Results/Evaluation/Results_Formatted_Evidence_QWEN.json`.

Both read `Data/Components/Evidence.parquet` and use `Formatted_Evidence` by default. They report accuracy and macro F1 over PolitiFact's six verdict labels. Gold labels are used for scoring, not passed to the model. Use `--evidence-cols analysis_text Formatted_Evidence` to compare evidence representations on the same valid rows.

The default claim column is the original `claim`. The explicit `--claim-col decontextualized_claim` above evaluates the output of augmentation instead.

### Optional: assess claim feasibility

```bash
python3 Code/Evaluation/Feasibility.py \
  --text-column decontextualized_claim \
  --output-name Feasibility_Decontextualized
```

Output: `Results/Evaluation/Feasibility_Decontextualized.parquet`, plus its `.run.json` manifest.

This reads the augmented claims directly and rates whether each is sufficiently specified for fact-checking: `0` means not assessable, `1` means ambiguous or missing context, and `2` means sufficiently clear. It does not decide truth or perform web searches. Use `--text-column claim` with another output name to assess the original claims.

## Custom paths and reruns

All processing stages accept `--output-dir`; parsing, component, and evaluation stages accept `--input-path`. Changing one stage's output directory does not automatically change the next stage's input.

| Stage | Naming and resume behavior |
| --- | --- |
| Raw collection | `--output-name` specifies the full raw filename. Reusing a path overwrites the file; runs do not append. |
| Article parsing | `--output-path` specifies a complete output path and overrides `--output-dir`. |
| Augmentation | `--output-name` is a stem for both Parquet and JSON. Saves every 25 rows; no automatic resume. Use a new name and `--start-row` for continuation. |
| Evidence extraction | `--output-name` is a stem. Add `--resume` with the same input, model, and `k`; keep the JSON checkpoint and run manifest together. |
| Verdict evaluation | `--output-prefix` names results. Matching checkpoints resume automatically; `--overwrite` starts a fresh run. |
| Feasibility | `--output-name` names the default Parquet output. Add `--resume` with matching input/settings, or use `--output-path` for another format. |

Use fresh output names when changing prompts or comparing experiments. Empty crawls and parser runs with no successfully parsed records now fail explicitly rather than reporting success without a usable dataset.

## Verification

Run the PolitiFact-only end-to-end integration test:

```bash
python3 -m unittest discover \
  -s Code/Data_Component/tests \
  -k test_collection_to_all_evaluations_for_politifact -v
```

Run the evaluation regression tests:

```bash
python3 -m unittest discover -s Code/Evaluation/tests -v
```

The integration test runs the real script entry points and writes/reloads every intermediate file, with HTTP and model responses mocked. It checks schema compatibility, paths, and handoffs without paid API calls or model downloads.
