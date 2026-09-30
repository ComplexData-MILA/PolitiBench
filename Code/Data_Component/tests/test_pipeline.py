"""Offline checks for the component pipeline; no API credentials are needed."""

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

import pandas as pd
import requests

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import Claim_Augmentation as augmentation

metadata = augmentation
import Evidence_Extraction as evidence
import Leakage_Control_Snopes as leakage
import pipeline_utils as utils

sys.path.extend([str(utils.REPO_ROOT / "Code/Data_Collection"),
                 str(utils.REPO_ROOT / "Code/Evaluation")])
import PolitiFact_RawScrape as politifact_raw
import PolitiFact_ScrapeData as politifact_parser
import Snopes_RawScrape as snopes_raw
import Snopes_ScrapeData as snopes_parser
import evaluation_utils
import Feasibility as feasibility
import verdict_evaluation as verdict_gpt
import verdict_evaluation_qwen as verdict_qwen


POLITIFACT_HTML = '''<html><head>
<meta property="article:published_time" content="2026-05-01T12:00:00Z">
<meta property="og:title" content="A sample fact check">
<meta name="keywords" content="Education,Legislation">
</head><body><a href="/personalities/jane-doe/">Jane Doe</a>
<div>stated on May 1, 2026 in a speech:</div><p>Says this bill passed.</p>
<img alt="True" src="meter.png">
<div class="short-on-time"><p>The vote passed.</p></div>
<p>The Education Act passed on May 1, 2026.</p>
<h2>Our ruling</h2><p>We rate this claim True.</p>
<section id="sources"><a href="https://example.com/record">Voting record</a></section>
</body></html>'''

SNOPES_HTML = '''<html><head><script type="application/ld+json">
[{"@type":"Article","headline":"A sample fact check","datePublished":"2026-05-01T12:00:00Z",
"author":{"name":"Test Author"},"keywords":["Education"]},
{"@type":"ClaimReview","claimReviewed":"The Education Act passed.",
"reviewRating":{"alternateName":"True"}}]</script></head><body>
<article id="article-content"><p>The Education Act passed on May 1, 2026.</p>
<p>We rated the claim true.</p></article>
<div id="sources_rows"><p><a href="https://example.com/record">Voting record</a></p></div>
</body></html>'''


def html_response(url, html, status=200):
    response = requests.Response()
    response.url = url
    response.status_code = status
    response._content = html.encode("utf-8")
    response.encoding = "utf-8"
    return response


def sample_rows():
    return pd.DataFrame([
        {
            "article_url": "https://example.com/one",
            "claim": "Says this bill passed.",
            "speaker": "Jane Doe",
            "statement_description": "stated on May 1, 2026 in a speech:",
            "statement_context": "a speech",
            "analysis_text": "The Education Act passed on May 1, 2026.",
            "verdict": "True",
            "publication_date": pd.Timestamp("2026-05-01", tz="UTC"),
        },
        {
            "article_url": "https://example.com/two",
            "claim": "The Education Act passed.",
            "speaker": None,
            "statement_description": None,
            "statement_context": None,
            "analysis_text": "The Education Act passed on May 1, 2026.",
            "verdict": "True",
            "publication_date": pd.Timestamp("2026-05-02", tz="UTC"),
        },
    ])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".component-test-", dir=utils.REPO_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.relative = self.directory.relative_to(utils.REPO_ROOT)
        # Only exception imports are needed; the actual client is always mocked.
        sdk = ModuleType("openai")
        for name in ("APIConnectionError", "APIError", "RateLimitError", "APITimeoutError", "InternalServerError"):
            setattr(sdk, name, type(name, (Exception,), {}))
        previous_sdk = sys.modules.get("openai")
        sys.modules["openai"] = sdk

        def restore_sdk():
            if previous_sdk is None:
                sys.modules.pop("openai", None)
            else:
                sys.modules["openai"] = previous_sdk

        self.addCleanup(restore_sdk)

    def fake_client(self):
        usage = SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15)
        client = Mock()
        client.responses.create.return_value = SimpleNamespace(
            status="completed", usage=usage,
            output_text="Jane Doe, stated on May 1, 2026 in a speech: the Education Act passed.",
        )
        client.responses.parse.return_value = SimpleNamespace(
            status="completed", usage=usage, model="test-model",
            output_parsed=evidence.ExtractedEvidence(
                supporting_facts=[evidence.SupportingFact(
                    fact="The Education Act passed on May 1, 2026.",
                    importance=5, support_type="full",
                )],
            ),
        )
        return client

    def test_imports_and_help_have_no_processing_side_effects(self):
        with patch.object(pd, "read_parquet", side_effect=AssertionError("Unexpected read")), \
             patch.object(utils, "create_client", side_effect=AssertionError("Unexpected client")):
            for module in (augmentation, evidence, leakage):
                importlib.reload(module)
        environment = os.environ.copy()
        environment.pop("OPENAI_API_KEY", None)
        for script in ("Claim_Augmentation.py",
                       "Evidence_Extraction.py", "Leakage_Control_Snopes.py"):
            result = subprocess.run(
                [sys.executable, str(SCRIPT_DIR / script), "--help"],
                cwd=tempfile.gettempdir(), env=environment, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--input-path", result.stdout)

    def test_metadata_handles_missing_values_without_model_dependencies(self):
        result = metadata.augment_dataframe(sample_rows())
        self.assertEqual(result.iloc[0].augmented_claim_cat,
                         "Jane Doe, stated on May 1, 2026 in a speech: this bill passed.")
        self.assertEqual(result.iloc[1].augmented_claim_cat, "The Education Act passed.")
        missing = metadata.augment_dataframe(pd.DataFrame({"claim": [None, float("nan"), " "]}))
        self.assertTrue(missing.augmented_claim_cat.isna().all())

    def test_local_clis_resolve_paths_from_root_and_unrelated_directory(self):
        input_path = self.directory / "input.parquet"
        sample_rows().to_parquet(input_path, index=False)
        for cwd in (utils.REPO_ROOT, Path(tempfile.gettempdir())):
            for script, output_name in (("Claim_Augmentation.py", "metadata"),
                                        ("Leakage_Control_Snopes.py", "cleaned")):
                result = subprocess.run([
                    sys.executable, str(SCRIPT_DIR / script),
                    "--input-path", str(self.relative / "input.parquet"),
                    "--output-dir", str(self.relative / "out"), "--output-name", output_name,
                    *(["--prefix-only"] if script == "Claim_Augmentation.py" else []),
                ], cwd=cwd, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                output = self.directory / "out" / f"{output_name}.parquet"
                self.assertEqual(len(pd.read_parquet(output)), 2)
                self.assertTrue(output.with_suffix(".json").exists())
        self.assertNotIn("analysis_text_raw", pd.read_parquet(input_path).columns)

    def test_end_to_end_pipeline_and_completed_resume(self):
        raw = sample_rows().iloc[:1]
        client = self.fake_client()
        augmented = augmentation.run_augmentation(
            raw, output_dir=self.relative, client=client, model="test-model",
        )
        self.assertEqual(augmented.iloc[0].decontextualization_status, "augmented")
        self.assertEqual(augmented.iloc[0].augmented_claim_cat,
                         "Jane Doe, stated on May 1, 2026 in a speech: this bill passed.")
        self.assertIn(augmented.iloc[0].augmented_claim_cat,
                      client.responses.create.call_args.kwargs["input"])
        input_path = self.relative / "Claims_Decontextualized.parquet"
        loaded = evidence.load_data(input_path)
        result = evidence.experiment(loaded, output_dir=self.relative, client=client, model="test-model")
        self.assertIn("Full support: The Education Act passed", result.iloc[0].Formatted_Evidence)
        self.assertEqual(result.iloc[0].extraction_total_tokens, 15)
        self.assertTrue((self.directory / "Evidence.parquet").exists())
        client.reset_mock()
        with patch.object(evidence, "create_client", side_effect=AssertionError("Unexpected API client")):
            resumed = evidence.experiment(loaded, output_dir=self.relative, model="test-model", resume=True)
        self.assertEqual(len(resumed), 1)
        client.responses.parse.assert_not_called()
        with self.assertRaisesRegex(ValueError, "differs"):
            evidence.experiment(loaded, output_dir=self.relative, model="other", resume=True, client=client)
        changed = loaded.copy()
        changed.loc[0, "analysis_text"] = "Different evidence."
        with self.assertRaisesRegex(ValueError, "differs"):
            evidence.experiment(changed, output_dir=self.relative, model="test-model", resume=True, client=client)

    def test_combined_cli_and_prefix_only_defaults(self):
        input_path = self.relative / "raw.parquet"
        sample_rows().to_parquet(utils.REPO_ROOT / input_path, index=False)
        argv = [augmentation.__file__, "--input-path", str(input_path),
                "--output-dir", str(self.relative), "--data-size", "1"]
        original_cwd = Path.cwd()
        self.addCleanup(os.chdir, original_cwd)
        os.chdir(tempfile.gettempdir())
        client = self.fake_client()
        with patch.object(sys, "argv", argv), patch.object(augmentation, "create_client", return_value=client):
            augmentation.main()
        result = pd.read_parquet(self.directory / "Claims_Decontextualized.parquet")
        self.assertEqual(len(result), 1)
        self.assertIn("metadata", result.columns)
        self.assertIn("decontextualized_claim", result.columns)
        with patch.object(sys, "argv", argv + ["--prefix-only"]), \
             patch.object(augmentation, "create_client", side_effect=AssertionError("Unexpected client")):
            augmentation.main()
        prefixed = pd.read_parquet(self.directory / "Claims_Metadata.parquet")
        self.assertEqual(len(prefixed), 1)
        self.assertNotIn("decontextualized_claim", prefixed.columns)
        # Recompute the prefix from the original claim, avoiding double-prefixing.
        regenerated = augmentation.augment_dataframe(prefixed)
        self.assertEqual(regenerated.iloc[0].augmented_claim_cat, prefixed.iloc[0].augmented_claim_cat)

    def test_start_row_fallback_and_empty_selection(self):
        prefixed = metadata.augment_dataframe(sample_rows())
        client = self.fake_client()
        client.responses.create.return_value.output_text = "The Education Act passed."
        result = augmentation.run_augmentation(
            prefixed, start_row=1, datasize=1, output_dir=self.directory, client=client,
        )
        self.assertEqual(result.iloc[0].article_url, "https://example.com/two")
        self.assertEqual(result.iloc[0].decontextualization_status, "unchanged")
        client.responses.create.return_value.output_text = "A changed prefix: Education Act passed."
        fallback = augmentation.run_augmentation(
            prefixed.iloc[:1], output_dir=self.directory, client=client,
        )
        self.assertEqual(fallback.iloc[0].decontextualization_status, "fallback")
        self.assertEqual(fallback.iloc[0].decontextualized_claim, prefixed.iloc[0].augmented_claim_cat)
        with self.assertRaisesRegex(ValueError, "No rows selected"):
            augmentation.run_augmentation(prefixed, start_row=20, output_dir=self.directory, client=client)

    def test_evidence_checkpoint_survives_failure_and_resumes(self):
        df = sample_rows().assign(decontextualized_claim=lambda x: x["claim"])
        client = self.fake_client()
        response = client.responses.parse.return_value
        client.responses.parse.side_effect = [response, RuntimeError("Simulated failure")]
        with self.assertRaisesRegex(RuntimeError, "Simulated failure"):
            evidence.experiment(df, output_dir=self.directory, model="test-model", client=client)
        self.assertEqual(len(json.loads((self.directory / "Evidence.json").read_text())), 1)
        client.responses.parse.side_effect = None
        client.reset_mock()
        result = evidence.experiment(df, output_dir=self.directory, model="test-model", resume=True, client=client)
        self.assertEqual(len(result), 2)
        self.assertEqual(client.responses.parse.call_count, 1)
        saved = pd.read_parquet(self.directory / "Evidence.parquet")
        self.assertEqual(saved.publication_date.tolist(), df.publication_date.tolist())

    def test_leakage_cleanup_preserves_original_and_removes_duplicates(self):
        original = "Records show 100 votes; therefore, we rated the claim false."
        df = pd.DataFrame({"article_url": ["one", "one"], "analysis_text": [original] * 2,
                           "verdict": ["False"] * 2})
        cleaned = leakage.clean_dataframe(df)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.iloc[0].analysis_text_raw, original)
        self.assertEqual(cleaned.iloc[0].analysis_text, "Records show 100 votes")
        self.assertEqual(leakage.clean_dataframe(cleaned).iloc[0].analysis_text_raw, original)

    def test_schema_validation_sampling_and_evidence_format(self):
        path = self.directory / "input.parquet"
        pd.DataFrame({"wrong": [1]}).to_parquet(path)
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            evidence.load_data(path)
        df = pd.DataFrame({
            "decontextualized_claim": ["Claim"] * 12 + [" "],
            "analysis_text": ["Evidence"] * 13,
            "verdict": [str(i % 6) for i in range(12)] + ["0"],
        })
        df.to_parquet(path)
        selected = evidence.load_data(path, datasize=6)
        self.assertEqual(selected.verdict.nunique(), 6)
        self.assertEqual(len(evidence.load_data(path)), 12)
        with self.assertRaisesRegex(ValueError, "divisible"):
            evidence.load_data(path, datasize=5)
        formatted = evidence.format_evidence({
            "supporting_facts": [
                {"fact": "Same fact", "importance": 5, "support_type": "full"},
                {"fact": "Same fact.", "importance": 4, "support_type": "full"},
                {"fact": "Partial fact", "importance": 3, "support_type": "partial"},
            ],
        }, limit=2)
        self.assertEqual(formatted.count("Same fact"), 1)
        self.assertIn("Partial support: Partial fact.", formatted)
        self.assertIn("Contradiction: None extracted.", formatted)

    def test_collection_to_all_evaluations_for_politifact(self):
        original_cwd = Path.cwd()
        self.addCleanup(os.chdir, original_cwd)
        # Run all real entry points from outside the repository. Mock only HTTP
        # responses and model responses; every intermediate dataset is real.
        os.chdir(tempfile.gettempdir())
        source, raw, parser, html = "PolitiFact", politifact_raw, politifact_parser, POLITIFACT_HTML
        out = self.relative / source
        article_url = "https://www.politifact.com/factchecks/2026/may/01/jane-doe/sample/"
        category = "latest"
        listing = f'<a href="{article_url}">Article</a>'
        responses = {raw.URLs[category]: listing, article_url: html}
        with patch.object(sys, "argv", [raw.__file__, "--limit", "1", "--output-dir", str(out)]), \
             patch.object(raw.SESSION, "get", side_effect=lambda url, **_: html_response(url, responses[url])), \
             patch.object(raw.time, "sleep"):
            raw.main()
        raw_path = out / f"{source}_HTML.parquet"
        self.assertEqual(len(pd.read_parquet(utils.repo_path(raw_path))), 1)
        with patch.object(sys, "argv", [parser.__file__, "--input-path", str(raw_path), "--output-dir", str(out)]):
            parser.main()
        parsed_path = out / f"{source}_DATA.parquet"
        parsed = pd.read_parquet(utils.repo_path(parsed_path))
        self.assertTrue(parsed[["claim", "analysis_text", "verdict"]].notna().all().all())
        client = self.fake_client()
        with patch.object(sys, "argv", [augmentation.__file__, "--input-path", str(parsed_path), "--output-dir", str(out)]), \
             patch.object(augmentation, "create_client", return_value=client):
            augmentation.main()
        augmented_path = out / "Claims_Decontextualized.parquet"
        with patch.object(sys, "argv", [evidence.__file__, "--input-path", str(augmented_path), "--output-dir", str(out)]), \
             patch.object(evidence, "create_client", return_value=client):
            evidence.main()
        evidence_path = out / "Evidence.parquet"
        evidence_data = pd.read_parquet(utils.repo_path(evidence_path))
        self.assertEqual(len(evidence_data), 1)
        self.assertIn("Full support:", evidence_data.iloc[0].Formatted_Evidence)
        # Exercise the API adapter and shared scoring on the generated evidence file.
        client.responses.parse.return_value.output_parsed = verdict_gpt.VerdictPrediction(verdict_id="True")
        with patch.object(sys, "argv", [verdict_gpt.__file__, "--input-path", str(evidence_path),
                                       "--claim-col", "decontextualized_claim", "--output-dir", str(out)]), \
             patch.object(verdict_gpt, "create_client", return_value=client):
            verdict_gpt.main()
        generator = Mock()
        generator.generate.return_value = ("True", {"resolved_model":"mock-qwen", "input_tokens":10,
                                                    "output_tokens":1, "total_tokens":11, "attempts":1})
        with patch.object(sys, "argv", [verdict_qwen.__file__, "--input-path", str(evidence_path),
                                       "--claim-col", "decontextualized_claim", "--output-dir", str(out)]), \
             patch.object(verdict_qwen, "LocalTextModel", return_value=generator):
            verdict_qwen.main()
        for backend in ("GPT", "QWEN"):
            scored = json.loads(utils.repo_path(out / f"Results_Formatted_Evidence_{backend}.json").read_text())
            self.assertEqual(scored["num_examples"], 1)
            self.assertEqual(scored["metrics"]["accuracy_6_class"], 1)
        generator.generate.return_value = ("The claim identifies the legislation. | 2", {})
        with patch.object(sys, "argv", [feasibility.__file__, "--input-path", str(augmented_path),
                                       "--text-column", "decontextualized_claim", "--output-dir", str(out)]), \
             patch.object(feasibility, "LocalTextModel", return_value=generator):
            feasibility.main()
        self.assertEqual(pd.read_parquet(utils.repo_path(out / "Feasibility.parquet")).iloc[0].feasibility_rating, 2)

    def test_snopes_relative_article_urls(self):
        article_url = "https://www.snopes.com/fact-check/sample/"
        listing = '<div class="article_wrapper"><a class="outer_article_link_wrapper" href="/fact-check/sample/">Article</a></div>'
        def get(url, **kwargs):
            self.assertIn(url, [snopes_raw.URLs["fact-check"], article_url])
            return html_response(url, listing if url == snopes_raw.URLs["fact-check"] else SNOPES_HTML)
        with patch.object(snopes_raw.SESSION, "get", side_effect=get), patch.object(snopes_raw.time, "sleep"):
            rows = snopes_raw.getArticles(category="fact-check", limit=1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["article_url"], article_url)
        self.assertEqual(rows[0]["html"], SNOPES_HTML)

    def test_politifact_empty_raw_collection_fails_without_writing_output(self):
        for module in (politifact_raw,):
            with patch.object(module, "getArticles", return_value=[]):
                kwargs = dict(outputDirectory=self.directory, category="latest", timeStamp=None,
                              limit=1, dataset_name="empty.parquet")
                if module is snopes_raw:
                    kwargs["start"] = 1
                with self.assertRaisesRegex(ValueError, "No articles"):
                    module.getData(**kwargs)
        self.assertFalse((self.directory / "empty.parquet").exists())

    def test_snopes_failed_download_does_not_break_valid_rows(self):
        raw = pd.DataFrame([
            {"article_url":"one", "html":SNOPES_HTML, "download_error":None},
            {"article_url":"two", "html":float("nan"), "download_error":"HTTP 403"},
        ])
        path = self.directory / "raw.parquet"
        raw.to_parquet(path, index=False)
        with patch.object(sys, "argv", [snopes_parser.__file__, "--input-path", str(path), "--output-dir", str(self.directory)]):
            snopes_parser.main()
        parsed = pd.read_parquet(self.directory / "Snopes_DATA.parquet")
        self.assertEqual(parsed.iloc[0]["claim"], "The Education Act passed.")
        self.assertEqual(parsed.iloc[1]["parse_error"], "HTTP 403")

    def test_politifact_all_failed_html_reports_clear_error(self):
        path = self.directory / "failed.parquet"
        pd.DataFrame([{"html":None,"article_url":"failed","download_error":"HTTP 403"}]).to_parquet(path)
        for module in (politifact_parser,):
            with patch.object(sys, "argv", [module.__file__, "--input-path", str(path), "--output-dir", str(self.directory)]):
                with self.assertRaisesRegex(ValueError, "No .*parsed"):
                    module.main()


if __name__ == "__main__":
    unittest.main()
