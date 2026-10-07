"""CPU/client tests. No model inference and no NPU access."""
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import long_quality as q


def fixture():
    cases = []
    for document in ("A-ledger", "B-ledger", "C-ledger", "D-ledger"):
        case = dict(id=document, records=q.make_records(document, 8), token_ids=[1, 2, 3],
                    input_sha256=q.sha([1, 2, 3]), record_bands=list(range(8)), gold_tokens=8192)
        case["gold"] = q.gold_text(case)
        case["gold_sha256"] = q.sha(case["gold"])
        cases.append(case)
    pack = dict(format="long-quality-v1", batch_size=4, input_tokens=3,
                min_output_tokens=8192, tokenizer_assets={}, cases=cases, scope="unit mock")
    pack["pack_sha256"] = q.sha(pack)
    metadata = {k: "same" for k in ("weights", "tokenizer", "vllm", "torch_npu", "cann",
                                    "dtype_quantization", "devices")}
    metadata.update(tp=4, prefix_caching=False, vllm_ascend="base", server_command="serve")
    responses = [dict(id=c["id"], text=c["gold"], finish_reason="stop", done=True,
                      usage=dict(prompt_tokens=3, completion_tokens=8192)) for c in cases]
    run = dict(protocol="long-quality-v1", pack_sha256=pack["pack_sha256"], model="qwen27b",
               requested_rounds=1, sampling=dict(temperature=0, seed=42, ignore_eos=False, max_tokens=12288),
               metadata=metadata, complete=True, rounds=[dict(round=0, responses=responses)])
    return pack, run


class LongQualityTests(unittest.TestCase):
    def setUp(self):
        self.pack, self.run = fixture()

    def test_streaming_timings_use_usage_not_chunk_count(self):
        response = self.run['rounds'][0]['responses'][0]
        response['events'] = [dict(seconds=2, data={'choices':[{'text':'multiple tokens'}]}),
                              dict(seconds=6, data={'choices':[{'text':'end'}]}),
                              dict(seconds=9, data={'choices':[], 'usage':response['usage']})]
        result = self.result()
        self.assertEqual(result['latency']['mean_ttft_ms'], 2000)
        self.assertAlmostEqual(result['latency']['mean_tpot_ms'], 4000/8191)
        self.assertEqual(result['latency']['tpot_count'], 1)
        self.assertTrue(result['passed'])

    def test_single_token_and_missing_times_are_na(self):
        response = dict(usage={'completion_tokens':1},events=[dict(seconds=2,data={'choices':[{'text':'x'}]})])
        self.assertEqual(q.streaming_timings(response)['ttft_ms'],2000)
        self.assertIsNone(q.streaming_timings(response)['tpot_ms'])
        self.assertIsNone(q.streaming_timings({})['ttft_ms'])
        response['events'][0]['seconds'] = float('inf')
        self.assertIsNone(q.streaming_timings(response)['ttft_ms'])

    def test_failed_response_excluded_from_mean_latencies(self):
        response = self.run['rounds'][0]['responses'][0]
        response.update(error='connection lost', events=[dict(seconds=2,data={'choices':[{'text':'x'}]})])
        self.assertIsNone(self.result()['latency']['mean_ttft_ms'])

    def test_nonmonotonic_or_missing_timestamps_not_measured(self):
        response = dict(usage={'completion_tokens':4},events=[dict(seconds=5,data={'choices':[{'text':'x'}]}),
                                                             dict(seconds=3,data={'choices':[{'text':'y'}]})])
        self.assertIsNone(q.streaming_timings(response)['tpot_ms'])
        del response['events'][0]['seconds']
        self.assertIsNone(q.streaming_timings(response)['ttft_ms'])

    def result(self):
        return q.score_run(self.pack, self.run)

    def response(self):
        return self.run["rounds"][0]["responses"][0]

    def change_row(self, index, field, value):
        lines = self.response()["text"].splitlines()
        item = json.loads(lines[index])
        item[field] = value
        lines[index] = q.compact(item)
        self.response()["text"] = "\n".join(lines)

    def test_gold_passes_and_has_separate_eight_bands(self):
        result = self.result()
        self.assertTrue(result["passed"])
        self.assertEqual(result["correct_facts"], 4 * 8 * 8)
        self.assertTrue(all(b["correct_facts"] == b["facts"] for b in result["details"][0]["bands"]))

    def test_closed_form_arithmetic_and_branches(self):
        r = dict(id="A001", site="Aster", units=2, price=7, discount=20, shipping=3, paid=5, urgent=1)
        self.assertEqual(q.reconcile(r), dict(id="A001", gross=14, net=0, due=3,
                                             balance=0, change=2, status="credit", dispatch="express", route="north"))
        r.update(paid=0)
        self.assertEqual(q.reconcile(r)["dispatch"], "hold")
        r.update(paid=3, urgent=0)
        self.assertEqual(q.reconcile(r)["status"], "settled")
        self.assertEqual(q.reconcile(r)["dispatch"], "standard")

    def test_deterministic_non_shared_sequences(self):
        a = q.make_records("A-ledger", 8)
        self.assertEqual(a, q.make_records("A-ledger", 8))
        self.assertNotEqual(a, q.make_records("B-ledger", 8))

    def test_last_band_wrong_fact_detected(self):
        self.change_row(7, "due", 99999)
        result = self.result()
        self.assertFalse(result["passed"])
        self.assertEqual(result["correct_facts"], 255)
        self.assertEqual(result["details"][0]["bands"][7]["correct_facts"], 7)

    def test_extra_key_schema_failure_is_regression(self):
        base = copy.deepcopy(self.run)
        self.change_row(0, "extra", "bad")
        result = self.result()
        self.assertFalse(result["passed"])
        self.assertEqual(result["correct_facts"], 256)
        report, code = q.compare(self.pack, base, self.run)
        self.assertEqual((report["status"], code), ("REGRESSION", 1))

    def test_unknown_finish_never_passes(self):
        base = copy.deepcopy(self.run)
        self.response()["finish_reason"] = None
        self.assertFalse(self.result()["passed"])
        self.assertEqual(q.compare(self.pack, base, self.run)[1], 1)

    def test_length_finish_fails_without_erasing_semantic_score(self):
        self.response()["finish_reason"] = "length"
        result = self.result()
        self.assertEqual(result["factual_accuracy"], 1)
        self.assertFalse(result["passed"])

    def test_short_output_fails_workload_coverage(self):
        self.response()["usage"]["completion_tokens"] = 8191
        self.assertFalse(self.result()["passed"])
        self.assertEqual(self.result()["factual_accuracy"], 1)

    def test_missing_wrong_or_boolean_usage_fails(self):
        for usage in (None, {"prompt_tokens": 4, "completion_tokens": 8192},
                      {"prompt_tokens": 3, "completion_tokens": True}):
            self.response()["usage"] = usage
            self.assertFalse(self.result()["passed"])

    def test_missing_done_and_error_fail(self):
        self.response()["done"] = False
        self.assertFalse(self.result()["passed"])
        self.response()["done"] = True
        self.response()["error"] = "transport failure"
        self.assertFalse(self.result()["passed"])

    def test_dropped_row_and_missing_trailer_fail(self):
        lines = self.response()["text"].splitlines()
        self.response()["text"] = "\n".join(lines[:-1])
        self.assertFalse(self.result()["passed"])
        self.response()["text"] = "\n".join(lines[:3] + lines[4:])
        self.assertFalse(self.result()["passed"])

    def test_swapped_sequences_fail_identity_and_facts(self):
        self.response()["text"] = self.run["rounds"][0]["responses"][1]["text"]
        result = self.result()
        self.assertEqual(result["details"][0]["correct_facts"], 0)
        self.assertFalse(result["passed"])

    def test_repeated_rows_and_extra_prose_fail(self):
        self.response()["text"] += "\n" + self.response()["text"].splitlines()[0]
        self.assertFalse(self.result()["passed"])
        self.response()["text"] += "\nHere is the answer."
        self.assertFalse(self.result()["passed"])

    def test_duplicate_keys_nan_and_wrong_numeric_type_fail(self):
        original = self.response()["text"]
        self.response()["text"] = original.replace('"gross":', '"id":"A001","gross":', 1)
        self.assertFalse(self.result()["passed"])
        self.response()["text"] = original
        self.change_row(0, "net", float("nan"))
        self.assertFalse(self.result()["passed"])
        self.response()["text"] = original
        self.change_row(0, "net", True)
        self.assertFalse(self.result()["passed"])
        self.change_row(0, "net", 1.0)
        self.assertFalse(self.result()["passed"])

    def test_partial_rounds_and_wrong_identity_never_pass(self):
        self.run["requested_rounds"] = 3
        self.assertFalse(self.result()["passed"])
        self.run["requested_rounds"] = 1
        self.response()["id"] = "foreign"
        self.assertFalse(self.result()["passed"])

    def test_unfinished_file_and_missing_metadata_comparison_refused(self):
        base = copy.deepcopy(self.run)
        self.run["complete"] = False
        self.assertEqual(q.compare(self.pack, base, self.run)[1], 2)
        self.run = copy.deepcopy(base)
        del self.run["metadata"]["weights"]
        with self.assertRaises(ValueError):
            q.compare(self.pack, base, self.run)

    def test_compare_refuses_different_rounds_and_dependencies(self):
        base = copy.deepcopy(self.run)
        self.run["requested_rounds"] = 3
        with self.assertRaises(ValueError):
            q.compare(self.pack, base, self.run)
        self.run = copy.deepcopy(base)
        self.run["metadata"]["vllm"] = "different"
        with self.assertRaises(ValueError):
            q.compare(self.pack, base, self.run)

    def test_two_bad_models_are_not_quality_pass(self):
        self.change_row(2, "balance", 987)
        report, code = q.compare(self.pack, self.run, copy.deepcopy(self.run))
        self.assertEqual(code, 2)
        self.assertNotEqual(report["status"], "PASS_ON_THIS_SET")

    def test_pack_integrity_and_golden_recomputation(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pack.json"
            q.save(path, self.pack)
            q.load_pack(path)
            self.pack["cases"][0]["records"][0]["price"] += 1
            self.pack["pack_sha256"] = q.sha({k: v for k, v in self.pack.items() if k != "pack_sha256"})
            q.save(path, self.pack)
            with self.assertRaises(ValueError):
                q.load_pack(path)

    def test_missing_template_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            self.pack["tokenizer_assets"] = {"chat_template.jinja": "required-hash"}
            with self.assertRaises(ValueError):
                q.verify_tokenizer(self.pack, td)

    def test_sse_usage_trailer_and_multichar_chunks_retained(self):
        events = [{"choices": [{"index": 0, "text": "hello", "finish_reason": None}]},
                  {"choices": [{"index": 0, "text": " world", "finish_reason": "stop"}]},
                  {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 8192}}]
        stream = io.BytesIO(b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events)
                            + b"data: [DONE]\n\n")
        args = SimpleNamespace(model="qwen27b", base_url="http://localhost:8000", timeout=2)
        with patch("urllib.request.urlopen", return_value=stream):
            result = q.request_one(args, self.pack, self.pack["cases"][0], self.run["sampling"])
        self.assertEqual(result["text"], "hello world")
        self.assertEqual(result["usage"]["completion_tokens"], 8192)
        self.assertEqual(len(result["events"]), 3)
        self.assertTrue(result["done"])

    def test_client_partial_failure_preserves_received_text(self):
        stream = io.BytesIO(b'data: {"choices":[{"index":0,"text":"partial"}]}\n\n')
        args = SimpleNamespace(model="qwen27b", base_url="http://localhost:8000", timeout=2)
        with patch("urllib.request.urlopen", return_value=stream):
            result = q.request_one(args, self.pack, self.pack["cases"][0], self.run["sampling"])
        self.assertEqual(result["text"], "partial")
        self.assertIn("without DONE", result["error"])

    def test_unsafe_sampling_or_server_metadata_never_pass(self):
        original = copy.deepcopy(self.run)
        for key, value in (("ignore_eos", True), ("max_tokens", 8192), ("seed", 1)):
            self.run = copy.deepcopy(original)
            self.run["sampling"][key] = value
            self.assertFalse(self.result()["passed"])
        self.run = copy.deepcopy(original)
        self.run["metadata"]["tp"] = 1
        self.assertFalse(self.result()["passed"])

    def test_run_client_saves_partial_failure_and_uses_four_workers(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            q.save(p / "metadata.json", self.run["metadata"])
            args = SimpleNamespace(pack=p / "unused.json", out=p / "run.json", model="qwen27b",
                                   tag="mock", max_tokens=12288, metadata=p / "metadata.json",
                                   rounds=3, base_url="http://localhost:8000", timeout=2)
            def request(_args, _pack, case, _sampling):
                response = copy.deepcopy(self.run["rounds"][0]["responses"]
                                         [next(i for i, c in enumerate(self.pack["cases"]) if c["id"] == case["id"])])
                if case["id"] == "D-ledger":
                    response["error"] = "mock HTTP failure"
                return response
            with patch.object(q, "load_pack", return_value=self.pack), patch.object(q, "request_one", side_effect=request):
                self.assertEqual(q.run(args), 1)
            saved = q.read(args.out)
            self.assertFalse(saved["complete"])
            self.assertEqual(len(saved["rounds"][0]["responses"]), 4)
            self.assertFalse(saved["score"]["passed"])

    def test_real_prepared_pack_if_present(self):
        path = q.ROOT / "fixtures/pack.json"
        if not path.exists():
            self.skipTest("Real tokenizer fixture not downloaded yet")
        pack = q.load_pack(path)
        self.assertTrue(all(len(c["token_ids"]) == 8192 and c["gold_tokens"] >= 8192 for c in pack["cases"]))
        self.assertEqual(set(b for c in pack["cases"] for b in c["record_bands"]), set(range(8)))
        for case in pack["cases"]:
            rendered = (path.parent / "inputs" / f"{case['id']}.txt").read_text(encoding="utf-8")
            before, after = q.render(case, "").split("BEGIN ARCHIVE\n", 1)
            self.assertIn(before, rendered)
            self.assertIn(after, rendered)
            self.assertEqual((path.parent / "gold" / f"{case['id']}.jsonl").read_text(encoding="utf-8"),
                             q.gold_text(case) + "\n")
        run = copy.deepcopy(self.run)
        run["pack_sha256"] = pack["pack_sha256"]
        run["rounds"][0]["responses"] = [
            dict(id=c["id"], text=c["gold"], done=True, finish_reason="stop",
                 usage=dict(prompt_tokens=8192, completion_tokens=c["gold_tokens"])) for c in pack["cases"]]
        score = q.score_run(pack, run)
        self.assertTrue(score["passed"])
        self.assertEqual(score["correct_facts"], 5856)
        response = run["rounds"][0]["responses"][0]
        lines = response["text"].splitlines()
        last = json.loads(lines[-2])
        last["due"] = -999
        lines[-2] = q.compact(last)
        response["text"] = "\n".join(lines)
        score = q.score_run(pack, run)
        self.assertFalse(score["passed"])
        self.assertEqual(score["correct_facts"], 5855)
        self.assertEqual(score["details"][0]["bands"][7]["correct_facts"],
                         score["details"][0]["bands"][7]["facts"] - 1)

    def test_freeze_preserves_archive_delimiter_and_all_records(self):
        class CharacterTokenizer:
            def encode(self, text, **kwargs):
                return list(map(ord, text))

            def decode(self, ids, **kwargs):
                return "".join(map(chr, ids))

            def apply_chat_template(self, messages, **kwargs):
                return self.encode("SYSTEM\n" + messages[0]["content"] + "\nUSER\n"
                                   + messages[1]["content"] + "\nASSISTANT\n")
        case = dict(id="A-ledger", records=q.make_records("A-ledger", 4))
        ids, rendered = q.freeze(CharacterTokenizer(), case, "Unrelated archive text. " * 400, 8192)
        self.assertEqual(len(ids), 8192)
        self.assertIn("\nEND ARCHIVE\n", rendered)
        before, after = q.render(case, "").split("BEGIN ARCHIVE\n", 1)
        self.assertIn(before, rendered)
        self.assertIn(after, rendered)


if __name__ == "__main__":
    unittest.main()
