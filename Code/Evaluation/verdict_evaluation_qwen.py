"""Evaluate PolitiFact verdict predictions with a local Qwen model."""

import argparse
from pathlib import Path
import sys
import re

from evaluation_utils import (
    LABELS, add_verdict_arguments, experiment, load_data, normalize_label, positive_int,
)
from local_model import LocalTextModel
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Prompt.verdict_evaluation import build_eval_messages

DEFAULT_MODEL = "Qwen/Qwen3-4B"

LABEL_PATTERN = re.compile(
    r"(?<!\w)(pants\s+on\s+fire!?|mostly\s+true|half\s+true|"
    r"mostly\s+false|false|true)(?!\w)",
    flags=re.IGNORECASE,
)


def parse_label(output):
    output = output.strip()
    if "<think>" in output and "</think>" not in output:
        return "INVALID"

    # Thinking mode places the final answer after </think>.
    final_text = output.rsplit("</think>", maxsplit=1)[-1].strip()
    unwrapped = re.sub(r"^<answer>\s*", "", final_text, flags=re.IGNORECASE)
    unwrapped = re.sub(r"\s*</answer>$", "", unwrapped, flags=re.IGNORECASE)
    direct = normalize_label(unwrapped.strip().rstrip(".:;!?"))
    if direct in LABELS:
        return direct

    matches = LABEL_PATTERN.findall(final_text)
    if not matches:
        return "INVALID"

    return normalize_label(matches[-1])


class QwenEvaluator:
    def __init__(self, model_name=DEFAULT_MODEL, device="auto", thinking=False,
                 max_new_tokens=None, seed=42):
        self.generator = LocalTextModel(model_name, device=device, thinking=thinking,
                                        max_new_tokens=max_new_tokens, seed=seed)

    def classify(self, claim, evidence):
        output, usage = self.generator.generate(build_eval_messages(claim, evidence))
        prediction = parse_label(output)
        if prediction == "INVALID":
            raise ValueError(f"Qwen returned an invalid verdict: {output!r}")
        return prediction, usage


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_verdict_arguments(parser)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--thinking", action="store_true", help="Enable sampled Qwen thinking mode.")
    parser.add_argument("--max-new-tokens", type=positive_int, default=None,
                        help="Default: 16 without thinking, 1024 with thinking.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    eval_df = load_data(args.input_path, [args.claim_col, *args.evidence_cols], limit=args.limit)
    max_new_tokens = args.max_new_tokens or (1024 if args.thinking else 16)
    evaluator = QwenEvaluator(args.model, device=args.device, thinking=args.thinking,
                              max_new_tokens=max_new_tokens, seed=args.seed)
    experiment(
        eval_df, args.claim_col, args.evidence_cols, args.output_prefix, "QWEN", evaluator.classify,
        settings={"backend": "qwen", "model": args.model, "device": args.device,
                  "thinking": args.thinking, "max_new_tokens": max_new_tokens, "seed": args.seed},
        output_dir=args.output_dir, overwrite=args.overwrite, checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
