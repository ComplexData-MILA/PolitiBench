# Prompts

This folder is the central location for the model instructions, few-shot examples, and message builders used by the component and evaluation scripts. Edit prompts here; the scripts import them directly.

| File | Contents | Used by |
| --- | --- | --- |
| `claim_augmentation.py` | Decontextualization instructions, preservation rules, and few-shot examples. | `Code/Data_Component/Claim_Augmentation.py` |
| `evidence_extraction.py` | Evidence categories, extraction instructions, and example inputs/outputs. | `Code/Data_Component/Evidence_Extraction.py` |
| `feasibility.py` | Feasibility instructions, rating scale, and statement formatting. | `Code/Evaluation/Feasibility.py` |
| `verdict_evaluation.py` | Shared six-class PolitiFact verdict instructions. | Both verdict evaluation scripts in `Code/Evaluation/`. |

These Python files build messages from each row's claim, context, or evidence. They do not run models or load data and do not need to be executed separately. The prompt content and examples were moved without changing the rendered model messages.

The combined augmentation script first adds metadata locally, then uses the decontextualization prompt on the prefixed claim. `--prefix-only` skips the prompt and model call.

API calls, model configuration, response schemas, output parsing, and validation remain in the processing scripts. Prompt edits affect subsequent model requests; use new output names or start fresh when evaluating a changed prompt, since existing checkpoints may contain responses from an earlier version.
