"""Client/metric tests only. They do not execute a language model."""
import copy
import hashlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import bs4_check as check


class FakeLexicalTokenizer:
    def __init__(self):
        self.words = []
        self.ids = {}

    def encode(self, text, **kwargs):
        result = []
        for word in re.findall(r"\w+|[^\w\s]", text):
            if word not in self.ids:
                self.ids[word] = len(self.words)
                self.words.append(word)
            result.append(self.ids[word])
        return result

    def decode(self, ids, **kwargs):
        return " ".join(self.words[i] for i in ids)

    def apply_chat_template(self, messages, **kwargs):
        return self.encode("<system> " + messages[0]["content"] + " <user> "
                           + messages[1]["content"] + " <assistant>")


class CheckerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = check.read_json(check.ROOT / "cases.json")
        cls.gold = check.read_json(check.ROOT / "expected/bs4.json")

    def rows(self):
        return [{"id": c["id"], "text": json.dumps(self.gold[c["id"]]),
                 "finish_reason": "stop"} for c in self.cases]

    def test_known_answers_match_evidence_and_individual_files(self):
        for case in self.cases:
            self.assertEqual(self.gold[case["id"]], check.read_json(check.ROOT / "expected" / f"{case['id']}.json"))
            for s in case["sections"]:
                for q in s["questions"]:
                    self.assertIn(check.norm(q["evidence"]), check.norm(s["text"]))

    def test_source_book_bytes_match_recorded_provenance(self):
        for source in check.read_json(check.ROOT / "sources/provenance.json"):
            data = (check.ROOT / "sources" / f"pg{source['id']}.txt").read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), source["normalized_utf8_sha256"])

    def test_40_correct_fields_and_separate_factual_score(self):
        score = check.score_rows(self.cases, self.rows())
        self.assertEqual((score["correct_fields"], score["correct_facts"], score["correct_identifiers"]), (40, 24, 16))
        self.assertEqual(score["correct_documents"], 4)

    def test_one_wrong_fact_partial_credit(self):
        rows = self.rows()
        value = self.gold[self.cases[0]["id"]].copy()
        value["answer_1"] = "wrong animal"
        rows[0]["text"] = json.dumps(value)
        score = check.score_rows(self.cases, rows)
        self.assertEqual(score["correct_fields"], 39)
        self.assertEqual(score["correct_facts"], 23)
        self.assertEqual(score["correct_documents"], 3)

    def test_alias_case_order_punctuation(self):
        rows = self.rows()
        value = self.gold["alice-01"].copy()
        value["answer_1"] = "THE WHITE RABBIT."
        value["answer_4"] = "caucus race"
        rows[0]["text"] = "```json\n" + json.dumps(dict(reversed(list(value.items())))) + "\n```"
        self.assertEqual(check.score_rows(self.cases, rows)["correct_fields"], 40)

    def test_cross_sequence_swap_is_wrong(self):
        rows = self.rows()
        rows[0]["text"], rows[1]["text"] = rows[1]["text"], rows[0]["text"]
        self.assertEqual(check.score_rows(self.cases, rows)["correct_documents"], 2)

    def test_invalid_repeated_duplicate_or_truncated(self):
        self.assertIsNone(check.parse('{"x":1,"x":2}'))
        self.assertIsNone(check.parse('{"x":1}\nuser\nassistant\n{"x":1}'))
        self.assertIsNone(check.parse('noise </think> {"x":1}'))
        self.assertIsNone(check.parse('<think>a</think><think>b</think>{"x":1}'))
        self.assertEqual(check.parse('<think>short reasoning</think>{"x":"a"}'), {"x": "a"})
        rows = self.rows()
        rows[0]["finish_reason"] = "length"
        self.assertEqual(check.score_rows(self.cases, rows)["correct_fields"], 30)

    def test_missing_extra_keys_and_unknown_finish(self):
        rows = self.rows()
        answer = self.gold["alice-01"].copy()
        answer["extra"] = "not requested"
        rows[0]["text"] = json.dumps(answer)
        rows[0]["finish_reason"] = None
        score = check.score_rows(self.cases, rows)
        self.assertEqual(score["correct_documents"], 3)
        self.assertEqual(score["correct_fields"], 40)
        self.assertEqual(score["finish_reason_unverified"], 1)

    def test_exact_freeze_preserves_all_evidence_and_template(self):
        tokenizer = FakeLexicalTokenizer()
        frozen = [check.freeze_case(tokenizer, case, 8192) for case in self.cases]
        self.assertEqual([len(c["token_ids"]) for c in frozen], [8192] * 4)
        self.assertEqual(len({c["input_sha256"] for c in frozen}), 4)
        for c in frozen:
            self.assertTrue(c["rendered"].endswith("< assistant >"))

    def test_compare_regression_unstable_main_and_input_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = {"pack_sha256": "test-client-only", "input_tokens": 8192,
                      "batch_size": 4, "sampling": {}, "cases_sha256": check.digest(self.cases)}
            baseline = {**common, "rows": self.rows() * 2}
            candidate = copy.deepcopy(baseline)
            check.dump(root / "main.json", baseline)
            check.dump(root / "candidate.json", candidate)
            args = SimpleNamespace(main=root / "main.json", candidate=root / "candidate.json", out=root / "report.json")
            self.assertEqual(check.compare(args), 0)
            candidate["rows"][0]["text"] = "broken"
            check.dump(args.candidate, candidate)
            self.assertEqual(check.compare(args), 1)
            self.assertEqual(len(check.read_json(args.out)["lost_stable_main_correct_fields"]), 10)
            baseline["rows"][0]["text"] = "broken"
            check.dump(args.main, baseline)
            check.dump(args.candidate, baseline)
            self.assertEqual(check.compare(args), 2)
            candidate["input_tokens"] = 4096
            check.dump(args.candidate, candidate)
            with self.assertRaises(ValueError):
                check.compare(args)

    def test_api_batch_accounting_choice_order_and_saved_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = [{"id": c["id"], "token_ids": [i] * 8192} for i, c in enumerate(self.cases)]
            pack = {"cases": frozen, "input_tokens": 8192, "batch_size": 4,
                    "cases_sha256": check.digest(self.cases),
                    "pack_sha256": check.digest([c["token_ids"] for c in frozen])}
            check.dump(root / "pack.json", pack)
            response = {"usage": {"prompt_tokens": 32768}, "choices": [
                {"index": i, "text": json.dumps(self.gold[c["id"]]), "finish_reason": "stop"}
                for i, c in reversed(list(enumerate(self.cases)))]}
            args = SimpleNamespace(pack=root / "pack.json", model="FAKE", tag="client-test",
                                   revision="NOT-A-MODEL-RUN", metadata=None, max_tokens=384,
                                   base_url="http://127.0.0.1:1", rounds=1, timeout=1, out=root / "run.json")
            with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as request:
                self.assertEqual(check.run(args), 0)
            sent = json.loads(request.call_args.args[0].data)
            self.assertEqual([len(ids) for ids in sent["prompt"]], [8192] * 4)
            self.assertFalse(sent["ignore_eos"])
            self.assertEqual(check.read_json(args.out)["score"]["correct_fields"], 40)
            args.out = root / "bad-run.json"
            response["usage"]["prompt_tokens"] = 32767
            with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())):
                with self.assertRaises(ValueError):
                    check.run(args)

    def test_truncated_main_is_inconclusive_even_with_correct_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.rows()
            rows[0]["finish_reason"] = "length"
            data = {"pack_sha256": "test-client-only", "input_tokens": 8192,
                    "batch_size": 4, "sampling": {}, "cases_sha256": check.digest(self.cases),
                    "rows": rows}
            check.dump(root / "main.json", data)
            check.dump(root / "candidate.json", data)
            args = SimpleNamespace(main=root / "main.json", candidate=root / "candidate.json",
                                   out=root / "report.json")
            self.assertEqual(check.compare(args), 2)

    def test_published_fixture_integrity_and_retained_evidence(self):
        pack, cases = check.load_pack(check.DEFAULT_PACK)
        self.assertEqual(pack["input_tokens"], 8192)
        self.assertEqual(len({c["input_sha256"] for c in pack["cases"]}), 4)
        self.assertEqual(check.read_json(check.DEFAULT_PACK.parent / "prompt_bs4.json"),
                         [c["token_ids"] for c in pack["cases"]])
        for item, case in zip(pack["cases"], cases):
            self.assertEqual(item["input_sha256"], check.digest(item["token_ids"]))
            text = (check.DEFAULT_PACK.parent / f"input_{case['id']}.txt").read_text(encoding="utf-8")
            self.assertEqual(text, item["rendered"])
            self.assertTrue(text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
            for section in case["sections"]:
                self.assertIn(check.norm(section["code"]), check.norm(text))
                for question in section["questions"]:
                    self.assertIn(check.norm(question["evidence"]), check.norm(text))

    def test_transformers_batch_encoding_chat_template_result(self):
        tokenizer = FakeLexicalTokenizer()
        original = tokenizer.apply_chat_template
        tokenizer.apply_chat_template = lambda *a, **kw: {"input_ids": original(*a, **kw)}
        item = check.freeze_case(tokenizer, self.cases[0], 8192)
        self.assertEqual(len(item["token_ids"]), 8192)
        self.assertTrue(all(isinstance(t, int) for t in item["token_ids"]))

    def test_tokenizer_fingerprint_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(tokenizer=root)
            with self.assertRaisesRegex(ValueError, "Tokenizer assets differ"):
                check.verify_tokenizer(args)


if __name__ == "__main__":
    unittest.main()
