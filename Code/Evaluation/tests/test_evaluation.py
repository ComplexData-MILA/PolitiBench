"""Offline evaluation regression tests; no model downloads or paid requests."""

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import evaluation_utils as utils
import Feasibility as feasibility
import local_model
import verdict_evaluation as gpt
import verdict_evaluation_qwen as qwen


def sample_data():
    return pd.DataFrame({
        "claim": ["The bill passed.", "The measure passed unanimously."],
        "decontextualized_claim": ["The Education Act passed.", "The Education Act passed unanimously."],
        "Formatted_Evidence": ["Full support: The vote passed.", "Contradiction: Ten members opposed it."],
        "analysis_text": ["The vote passed.", "Ten members opposed it."],
        "verdict": ["True", "False"],
        "publication_date": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
    })


def usage():
    return {"resolved_model": "test-model", "input_tokens": 10,
            "output_tokens": 5, "total_tokens": 15, "attempts": 1}


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".evaluation-test-", dir=utils.REPO_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.relative = self.directory.relative_to(utils.REPO_ROOT)
        sample_data().to_parquet(self.directory / "input.parquet", index=False)

    def test_help_imports_and_defaults_do_not_initialize_models(self):
        with patch.object(pd, "read_parquet", side_effect=AssertionError("Unexpected read")), \
             patch.object(utils, "create_client", side_effect=AssertionError("Unexpected client")), \
             patch.object(local_model.LocalTextModel, "_load", side_effect=AssertionError("Unexpected model")):
            for module in (feasibility, gpt, qwen):
                importlib.reload(module)
            qwen.QwenEvaluator()
            gpt.OpenAIEvaluator()
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)
        env.pop("HF_TOKEN", None)
        for name in ("Feasibility.py", "verdict_evaluation.py", "verdict_evaluation_qwen.py"):
            process = subprocess.run([sys.executable, str(SCRIPT_DIR / name), "--help"],
                                     cwd=tempfile.gettempdir(), env=env, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn("--output-dir", process.stdout)
        for module in (feasibility, gpt, qwen):
            with patch.object(sys, "argv", [module.__file__]):
                args = module.parse_args()
            self.assertEqual(args.output_dir, Path("Results/Evaluation"))

    def test_all_clis_resolve_data_paths_from_root_and_elsewhere(self):
        original_cwd = Path.cwd()
        self.addCleanup(os.chdir, original_cwd)
        for cwd in (utils.REPO_ROOT, Path(tempfile.gettempdir())):
            os.chdir(cwd)
            for module, backend in ((gpt, "GPT"), (qwen, "QWEN")):
                evaluator = Mock()
                evaluator.classify.side_effect = [("True", usage()), ("False", usage())]
                factory = "OpenAIEvaluator" if module is gpt else "QwenEvaluator"
                args = [module.__file__, "--input-path", str(self.relative / "input.parquet"),
                        "--output-dir", str(self.relative / "out"), "--overwrite"]
                with patch.object(sys, "argv", args), patch.object(module, factory, return_value=evaluator):
                    module.main()
                result = json.loads((self.directory / "out" / f"Results_Formatted_Evidence_{backend}.json").read_text())
                self.assertEqual(result["num_examples"], 2)
                self.assertEqual(result["metrics"]["accuracy_6_class"], 1)
                self.assertAlmostEqual(result["metrics"]["macro_f1_6_class"], 2 / 6)
            generator = Mock()
            generator.generate.return_value = ("Clear statement | 2", usage())
            args = [feasibility.__file__, "--input", str(self.relative / "input.parquet"),
                    "--output", str(self.relative / "out" / "feasibility.json")]
            with patch.object(sys, "argv", args), patch.object(feasibility, "LocalTextModel", return_value=generator):
                feasibility.main()
            saved = json.loads((self.directory / "out" / "feasibility.json").read_text())
            self.assertEqual(saved[0]["feasibility_rating"], 2)
            self.assertIn("2026-01-01", saved[0]["publication_date"])

    def test_paired_filtering_and_schema_errors(self):
        df = sample_data()
        df.loc[1, "analysis_text"] = " "
        path = self.directory / "paired.json"
        utils.save_dataframe(df, path)
        selected = utils.load_data(path, ["claim", "Formatted_Evidence", "analysis_text"])
        self.assertEqual(len(selected), 1)
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            utils.load_data(path, ["missing"])
        df.loc[0, "verdict"] = "Mixture"
        utils.save_dataframe(df, path)
        with self.assertRaisesRegex(ValueError, "Mixture"):
            utils.load_data(path, ["claim"])
        df["claim"] = " "
        utils.save_dataframe(df, path)
        with self.assertRaisesRegex(ValueError, "No valid rows"):
            utils.load_data(path, ["claim"])

    def test_verdict_failure_checkpoints_resume_and_validate_settings(self):
        df = sample_data()
        output = self.directory / "checkpoint.json"
        classifier = Mock(side_effect=[("True", usage()), RuntimeError("Simulated failure")])
        settings = {"model": "test-model", "backend": "mock"}
        with self.assertRaisesRegex(RuntimeError, "Simulated failure"):
            utils.run_verdict_evaluation(df, "claim", "Formatted_Evidence", classifier, output, settings)
        self.assertEqual(json.loads(output.read_text())["num_examples"], 1)
        classifier = Mock(return_value=("False", usage()))
        results = utils.run_verdict_evaluation(df, "claim", "Formatted_Evidence", classifier, output, settings)
        self.assertEqual(len(results), 2)
        classifier.assert_called_once()
        no_inference = Mock(side_effect=AssertionError("Already complete"))
        utils.run_verdict_evaluation(df, "claim", "Formatted_Evidence", no_inference, output, settings)
        no_inference.assert_not_called()
        with self.assertRaisesRegex(ValueError, "do not match"):
            utils.run_verdict_evaluation(df, "claim", "Formatted_Evidence", no_inference, output,
                                         {**settings, "model": "different"})
        changed = df.copy()
        changed.loc[1, "claim"] = "Changed input."
        with self.assertRaisesRegex(ValueError, "do not match"):
            utils.run_verdict_evaluation(changed, "claim", "Formatted_Evidence", no_inference, output, settings)

    def test_zero_predictions_save_valid_json_with_null_metrics(self):
        output = self.directory / "failed.json"
        classifier = Mock(side_effect=ValueError("Invalid model response"))
        with self.assertRaisesRegex(ValueError, "Invalid model response"):
            utils.run_verdict_evaluation(sample_data(), "claim", "Formatted_Evidence", classifier,
                                         output, {"model": "test"})
        result = json.loads(output.read_text())
        self.assertEqual(result["num_examples"], 0)
        self.assertIsNone(result["metrics"]["accuracy_6_class"])
        self.assertNotIn("NaN", output.read_text())

    def test_feasibility_resume_retries_failed_parse_and_preserves_dates(self):
        df = sample_data()
        output = self.directory / "feasibility.parquet"
        generator = Mock()
        generator.generate.side_effect = [("Clear | 2", usage()), ("| 1", usage())]
        result = feasibility.run_feasibility(df, "claim", output, generator=generator)
        self.assertFalse(result.iloc[1].feasibility_parse_success)
        self.assertEqual(result.iloc[1].feasibility_rating, 1)
        retry = Mock()
        retry.generate.return_value = ("Ambiguous context | 1", usage())
        resumed = feasibility.run_feasibility(df, "claim", output, resume=True, generator=retry)
        retry.generate.assert_called_once()
        self.assertTrue(resumed.feasibility_parse_success.all())
        self.assertEqual(pd.read_parquet(output).publication_date.tolist(), df.publication_date.tolist())
        with patch.object(feasibility, "LocalTextModel", side_effect=AssertionError("Already complete")):
            feasibility.run_feasibility(df, "claim", output, resume=True)
        with self.assertRaisesRegex(ValueError, "settings differ"):
            feasibility.run_feasibility(df, "decontextualized_claim", output, resume=True, generator=retry)

    def test_feasibility_missing_rows_do_not_load_a_model(self):
        df = pd.DataFrame({"claim": [None, " "]})
        with patch.object(feasibility, "LocalTextModel", side_effect=AssertionError("No input")):
            result = feasibility.run_feasibility(df, "claim", self.directory / "missing.json")
        self.assertEqual(result.feasibility_error.tolist(), ["Missing statement."] * 2)
        with self.assertRaisesRegex(ValueError, "No rows selected"):
            feasibility.run_feasibility(df.iloc[:0], "claim", self.directory / "empty.json")

    def test_qwen_parsing_and_shared_prompt(self):
        self.assertEqual(qwen.parse_label("<think>Maybe false.</think><answer>Mostly true</answer>"), "Mostly true")
        self.assertEqual(qwen.parse_label("<think>Maybe false."), "INVALID")
        self.assertEqual(qwen.parse_label("<think>False</think>"), "INVALID")
        self.assertEqual(qwen.parse_label("Pants on fire!"), "Pants on fire")
        evaluator = qwen.QwenEvaluator()
        evaluator.generator = Mock()
        evaluator.generator.generate.return_value = ("False", usage())
        prediction, _ = evaluator.classify("Test claim", "Test evidence")
        self.assertEqual(prediction, "False")
        self.assertIs(gpt.build_eval_messages, qwen.build_eval_messages)
        prompt = evaluator.generator.generate.call_args.args[0]
        self.assertIn("Test claim", prompt[1]["content"])

    def test_openai_adapter_uses_injected_client(self):
        sdk = ModuleType("openai")
        for name in ("APIConnectionError", "APITimeoutError", "InternalServerError", "RateLimitError"):
            setattr(sdk, name, type(name, (Exception,), {}))
        previous = sys.modules.get("openai")
        sys.modules["openai"] = sdk
        try:
            client = Mock()
            client.responses.parse.return_value = SimpleNamespace(
                status="completed", model="resolved-test", output_parsed=gpt.VerdictPrediction(verdict_id="True"),
                usage=SimpleNamespace(input_tokens=4, output_tokens=2, total_tokens=6),
            )
            evaluator = gpt.OpenAIEvaluator(model="requested-test", client=client)
            prediction, stats = evaluator.classify("Test claim", "Test evidence")
            self.assertEqual(prediction, "True")
            self.assertEqual(stats["resolved_model"], "resolved-test")
            self.assertEqual(client.responses.parse.call_args.kwargs["model"], "requested-test")
        finally:
            if previous is None:
                sys.modules.pop("openai", None)
            else:
                sys.modules["openai"] = previous

    def test_structured_evidence_formats_and_output_name_collisions(self):
        value = {"facts": np.array(["Fact one", "Fact two"]), "missing": pd.NA}
        self.assertEqual(json.loads(utils.format_evidence_for_prompt(value))["facts"], ["Fact one", "Fact two"])
        for suffix in (".parquet", ".json", ".csv", ".jsonl", ".ndjson"):
            output = self.directory / f"format{suffix}"
            utils.save_dataframe(sample_data(), output)
            self.assertEqual(len(utils.load_dataframe(output)), 2)
        with self.assertRaisesRegex(ValueError, "distinct names"):
            utils.experiment(sample_data(), "claim", ["a-b", "a b"], "test", "GPT", Mock(), {})


if __name__ == "__main__":
    unittest.main()
