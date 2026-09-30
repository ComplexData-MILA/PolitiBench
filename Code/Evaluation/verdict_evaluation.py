"""Evaluate PolitiFact verdict predictions with the OpenAI API."""

import argparse
from pathlib import Path
import sys
import time
from typing import Literal

from pydantic import BaseModel

from evaluation_utils import LABELS, add_verdict_arguments, create_client, experiment, load_data
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Prompt.verdict_evaluation import build_eval_messages

DEFAULT_MODEL = "gpt-5.6-luna"


class VerdictPrediction(BaseModel):
    verdict_id: Literal[
        "True",
        "Mostly true",
        "Half true",
        "Mostly false",
        "False",
        "Pants on fire",
    ]

def parse_label(output):
    output = output.strip()
    return output if output in LABELS else "INVALID"


def classify_statement(claim, evidence, *, client, model=DEFAULT_MODEL, max_retries=5):
    from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

    messages = build_eval_messages(claim, evidence)
    for attempt in range(max_retries):
        try:
            response = client.responses.parse(
                model=model,
                input=messages,
                text_format=VerdictPrediction,
                reasoning={"effort": "medium"},
            )

            if response.status != "completed":
                raise ValueError(f"Incomplete verdict response: {response.incomplete_details}")
            parsed = response.output_parsed

            if parsed is None:
                raise ValueError(
                    "The API returned no parsed verdict. "
                    f"Raw output: {response.output_text!r}"
                )

            prediction = parse_label(parsed.verdict_id)

            if prediction == "INVALID":
                raise ValueError(
                    f"Invalid verdict ID: {parsed.verdict_id!r}"
                )

            usage = {
                "resolved_model": response.model,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
                "attempts": attempt + 1,
            }

            return prediction, usage

        except (
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
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

class OpenAIEvaluator:
    def __init__(self, model=DEFAULT_MODEL, client=None):
        self.model = model
        self.client = client

    def classify(self, claim, evidence):
        if self.client is None:
            self.client = create_client()
        return classify_statement(claim, evidence, client=self.client, model=self.model)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_verdict_arguments(parser)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser.parse_args()


def main():
    args = parse_args()
    eval_df = load_data(args.input_path, [args.claim_col, *args.evidence_cols], limit=args.limit)
    evaluator = OpenAIEvaluator(args.model)
    experiment(
        eval_df, args.claim_col, args.evidence_cols, args.output_prefix, "GPT",
        evaluator.classify, settings={"backend": "openai", "model": args.model, "reasoning_effort": "medium"},
        output_dir=args.output_dir, overwrite=args.overwrite, checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
