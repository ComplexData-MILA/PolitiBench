# PolitiFact Dataset

This directory contains processed PolitiFact fact-checking data with claim augmentation and structured evidence extraction.

## Files

The dataset is provided in four Parquet files:

| File | Description |
|---|---|
| `PolitiFact_Data_k2_First_Half.parquet` | First half of the dataset containing all available processing components for the `k=2` evidence extraction setting. |
| `PolitiFact_Data_k2_Second_Half.parquet` | Second half dataset containing all available processing components for the `k=2` evidence extraction setting. |
| `PolitiFact_Data_k2_Simplified_First_Half.parquet` | First half of the simplified dataset containing the main article information and key generated components for `k=2`. |
| `PolitiFact_Data_k2_Simplified_Second_Half.parquet` | Second half of the simplified dataset containing the main article information and key generated components for `k=2`. |

### Evidence extraction setting (`k`)

The value of `k` specifies the maximum number of evidence items the evidence-extraction model is asked to generate for each evidence category.

For example:

- `k=2`: up to 2 evidence items per category.
- `k=4`: up to 4 evidence items per category.

The model may return fewer than `k` items when sufficient evidence is not available.

## Dataset Variants

### `Simplified`

The `Simplified` files retain the main components needed for most experiments, including:

- article and claim information,
- claim augmentation outputs,
- extracted evidence, and
- formatted evidence.

These files are intended to provide a cleaner and more convenient version of the dataset for downstream modeling and analysis.


## Article Context

The dataset also contains the original fact-checking article content in both combined and separated forms.

### `context`

The complete article context. This field contains the three main sections of the fact-checking article:

```text
Short Summary:
...

Article:
...

Our Ruling:
...
```

The `context` field preserves the article content together with the hyperlinks contained in the source article.

### `short_summary`

Contains only the **Short Summary** section of the fact-checking article.

### `analysis_text`

Contains only the main **Article** or analysis section.

This section contains the substantive fact-checking discussion and evidence used to evaluate the claim, excluding the final ruling section.

### `ruling_text`

Contains only the **Our Ruling** or conclusion section of the article.

This section contains the fact-checker's final conclusion and ruling-related discussion.

The relationship between these fields can be viewed as:

```text
context
├── Short Summary  -> short_summary
├── Article        -> analysis_text
└── Our Ruling     -> ruling_text
```

## Claim Representation

The dataset contains multiple representations of each claim.

### `claim`

The original claim as it appears in the source fact-checking data.

### `augmented_claim_cat`

The original claim with an added metadata prefix. The prefix provides contextual information associated with the claim, such as available attribution, date, or statement context.

The proposition itself is not decontextualized at this stage.

### `decontextualized_claim`

The final augmented claim.

This representation contains:

1. the metadata prefix, and
2. a decontextualized version of the original proposition.

The decontextualization step is intended to make the proposition more self-contained by resolving context-dependent references when sufficient information is available, while preserving the meaning of the original claim.

The claim representations can therefore be viewed as:

```text
claim
  -> augmented_claim_cat
  -> decontextualized_claim
```

## Evidence Representation

### `Extracted_Evidence`

The structured output produced by the LLM-based evidence extraction stage.

The extracted evidence is organized into evidence categories such as:

- supporting facts,
- weakening facts, and
- missing context.

Depending on the extraction schema, individual evidence items may also contain metadata such as importance scores or support type.

### `Formatted_Evidence`

A flattened and standardized representation of `Extracted_Evidence`.

This field converts the structured LLM output into a simpler format that is easier to provide directly to downstream models. Evidence is organized in a consistent order and represented without the nested structure of the original extraction output.

Conceptually:

```text
Extracted_Evidence
  -> formatting / flattening
  -> Formatted_Evidence
```

## Recommended Usage

Use the `Simplified` files for most downstream experiments involving claim representation, evidence-based fact verification, or verdict prediction.

Use the `ALL` files when intermediate processing fields or the complete dataset representation are required.

When comparing evidence extraction settings, use the corresponding `k=2` and `k=4` files while keeping the remaining experimental setup consistent.

## File Format

All dataset files are stored in Apache Parquet format and can be loaded with common data-processing libraries such as pandas:

```python
import pandas as pd

df = pd.read_parquet("PolitiFact_Data_k2_Simplified.parquet")
print(df.head())
```
