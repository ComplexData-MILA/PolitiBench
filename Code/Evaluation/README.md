# Evaluation

Run these scripts from the **repository root**. They evaluate two different things:

- **Feasibility:** whether a claim contains enough information for its truth to be assessed.
- **Verdict prediction:** whether a model can predict the gold PolitiFact verdict from a claim and supplied evidence.

These are separate evaluations. Feasibility does not need to run before verdict prediction. The OpenAI and Qwen verdict scripts are alternative model backends, not sequential stages.

## Files

| File | Purpose |
| --- | --- |
| `Feasibility.py` | Uses a local model to assign a feasibility rating of 0, 1, or 2, with an explanation. |
| `verdict_evaluation.py` | Uses the OpenAI API to predict a six-class PolitiFact verdict and compute accuracy and macro F1. |
| `verdict_evaluation_qwen.py` | Performs the same verdict evaluation with a local Qwen model. |
| `evaluation_utils.py` | Shared paths, dataset loading, label normalization, metrics, and checkpoint handling. |
| `../../Prompt/verdict_evaluation.py` | Shared verdict prompt so both backends receive the same claim/evidence instructions. |
| `local_model.py` | Lazy local model loading, device selection, and text generation. |
| `requirements.txt` | Dependencies for dataset handling, metrics, and OpenAI evaluation. |
| `requirements-local.txt` | Includes the base dependencies plus PyTorch and Transformers for local models. |
| `tests/test_evaluation.py` | Offline tests using temporary datasets and mocked model responses. |

Evaluation prompt text lives in the repository-root [`Prompt/`](../../Prompt/README.md) folder: `feasibility.py` for feasibility and `verdict_evaluation.py` for both verdict backends.

## Setup and paths

Use Python 3.10 or newer. For OpenAI verdict evaluation:

```bash
python3 -m pip install -r Code/Evaluation/requirements.txt
```

For feasibility or local Qwen evaluation:

```bash
python3 -m pip install -r Code/Evaluation/requirements-local.txt
```

OpenAI evaluation reads `OPENAI_API_KEY` from the environment or the repository-root `.env` file. Local model loading reads `HF_TOKEN` from the same locations when needed. Environment values take precedence. API credentials and models are initialized only when needed for inference; importing the scripts and using `--help` do not load models or process data.

All relative input and output paths resolve from the repository root. Absolute paths and `~` paths are also supported. Results default to `Results/Evaluation/`, which is created automatically. To launch from elsewhere, use an absolute script path.

All scripts accept Parquet, CSV, JSON, JSONL, and NDJSON inputs. JSON can be a list of records, a single record, or records wrapped under `data`, `statements`, `claims`, or `results`.

Default models retain the original scripts' choices: `gpt-5.6-luna` for OpenAI verdicts, `Qwen/Qwen3-4B` for local verdicts, and `Qwen/Qwen3-8B` for feasibility. Override them with `--model`. The local loader downloads uncached weights when an actual evaluation begins. No model weights are bundled, and live inference is not verified by the offline tests.

## Feasibility

The default input is `Data/Components/Claims_Decontextualized.parquet`, produced by `Claim_Augmentation.py`. The default text column remains `claim`, so choose `decontextualized_claim` to assess the rewritten proposition:

```bash
python3 Code/Evaluation/Feasibility.py \
  --text-column decontextualized_claim \
  --output-name Feasibility_Decontextualized \
  --limit 20
```

Compare with the original claims using a separate output:

```bash
python3 Code/Evaluation/Feasibility.py \
  --text-column claim \
  --output-name Feasibility_Original \
  --limit 20
```

Only the selected text column is required; gold verdicts and article evidence are not used. The prompt asks whether a hypothetical fact-checker with web search could assess the statement. **This script does not perform web searches itself.**

Ratings are:

| Rating | Meaning |
| --- | --- |
| `0` | Insufficient information, irrecoverable ambiguity, or no assessable factual claim. |
| `1` | Ambiguous or missing context, with some possibility of recovering the intended meaning. |
| `2` | Sufficiently clear to assess. |

The output retains input columns and adds raw model output, explanation, rating, parse-success flag, error, model name, and evaluated text-column name. Blank statements are retained with an error. Per-row generation failures are recorded and processing continues. A numeric rating alone does not count as a successful parse; an explanation is required.

| Option | Default | Purpose |
| --- | --- | --- |
| `--input-path` / `--input` | `Data/Components/Claims_Decontextualized.parquet` | Input dataset. |
| `--text-column` | `claim` | Column to assess. |
| `--output-dir` | `Results/Evaluation` | Output directory. |
| `--output-name` | `Feasibility` | Filename stem for the default `.parquet` output. |
| `--output-path` / `--output` | Unset | Explicit file path; overrides output directory/name. Supports the same formats as input. |
| `--limit` | All rows | Positive number of input rows to evaluate. |
| `--model` | `Qwen/Qwen3-8B` | Local model identifier or local model directory. |
| `--device` | `auto` | `auto`, `cuda`, `mps`, or `cpu`; automatic selection prefers CUDA, then MPS, then CPU. |
| `--max-new-tokens` | `160` | Maximum generated tokens per statement. |
| `--checkpoint-every` | `25` | Save interval in rows. |
| `--resume` | Off | Reuse successfully parsed rows from a matching output checkpoint. |

A new run replaces an output with the same name. To resume, rerun with the **same original input and options** plus `--resume`. Keep the output and its companion `<output filename>.run.json` together. Input contents/order, text column, model, device, and token limit must match. Failed or incomplete rows are retried; completed rows are skipped. Output is also saved when the processing loop exits or is interrupted.

## Verdict prediction

Both backends default to `Data/Components/Evidence.parquet`, produced by `Evidence_Extraction.py`. Required columns are `verdict`, the selected claim column, and every requested evidence column. Defaults are `claim` and `Formatted_Evidence`.

OpenAI:

```bash
python3 Code/Evaluation/verdict_evaluation.py \
  --claim-col decontextualized_claim \
  --evidence-cols Formatted_Evidence \
  --output-prefix Decontextualized \
  --limit 20
```

Local Qwen:

```bash
python3 Code/Evaluation/verdict_evaluation_qwen.py \
  --claim-col decontextualized_claim \
  --evidence-cols Formatted_Evidence \
  --output-prefix Decontextualized \
  --limit 20
```

These write `Results/Evaluation/Decontextualized_Formatted_Evidence_GPT.json` and `Results/Evaluation/Decontextualized_Formatted_Evidence_QWEN.json`, respectively.

To compare evidence representations on the same rows:

```bash
python3 Code/Evaluation/verdict_evaluation.py \
  --input-path Data/Components/Evidence.parquet \
  --evidence-cols analysis_text Formatted_Evidence \
  --output-prefix Evidence_Comparison
```

Rows missing a gold label, selected claim, or **any** selected evidence representation are excluded before applying `--limit`. This keeps the evaluated rows paired across evidence columns. Structured evidence dictionaries and arrays are serialized into JSON in the prompt.

The six gold labels are `True`, `Mostly true`, `Half true`, `Mostly false`, `False`, and `Pants on fire`. Capitalization is normalized, and `Pants on fire!` is accepted. Other gold labels cause an explanatory error; these scripts do not map the broader Snopes rating scheme.

The JSON output contains run settings, `num_examples`, `metrics`, and per-row `results`. Metrics are six-class accuracy and macro F1 over all six labels, including absent classes with zero F1. Each result records the claim, gold label, prediction, evidence, resolved model, token counts, and request attempts. The gold verdict is used for scoring, not included in the model prompt.

Both verdict scripts share these options:

| Option | Default | Purpose |
| --- | --- | --- |
| `--input-path` / `--input` | `Data/Components/Evidence.parquet` | Input dataset. |
| `--claim-col` | `claim` | Claim representation to evaluate. |
| `--evidence-cols` | `Formatted_Evidence` | One or more evidence columns; one output per column. |
| `--output-dir` | `Results/Evaluation` | Results directory. |
| `--output-prefix` | `Results` | Filename prefix; directories belong in `--output-dir`. |
| `--limit` | All valid rows | Positive maximum after paired filtering. |
| `--model` | Backend-specific | Model name. |
| `--checkpoint-every` | `50` | Save interval in completed predictions. |
| `--overwrite` | Off | Discard an existing checkpoint and start over. |

Matching verdict checkpoints resume automatically, including after failures. Checkpoints validate the full selected dataset, claim/evidence columns, model, and generation settings. An already complete checkpoint requires no inference. Changed settings or older checkpoints without run metadata require a new output prefix or explicit `--overwrite`. Completed predictions are saved on interruption or failure; the failed row is retried on the next run.

Qwen also accepts `--device`, `--thinking`, `--max-new-tokens`, and `--seed`. Defaults are automatic device selection, thinking disabled, 16 output tokens (1024 with thinking enabled), and seed 42. Thinking mode uses sampled decoding. Prompts exceeding the model context window fail instead of being silently truncated. Incomplete thinking without a final answer is not accepted as a verdict.

## Offline tests

```bash
python3 -m unittest discover -s Code/Evaluation/tests -v
```

The tests use temporary input files and mocked inference to check paths, filtering, metrics, response parsing, and checkpoint behavior. They do not call the API or download model weights.
