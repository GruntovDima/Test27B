#!/usr/bin/env python3
"""Freeze four 8K inputs, record main/PR answers, and score factual accuracy."""
import argparse
import hashlib
import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_PACK = ROOT / "fixtures/qwen3.6-27b-8k/pack.json"
SYSTEM = "Read the supplied excerpts and answer their questions. Return only the requested JSON object."


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def norm(value):
    if not isinstance(value, str):
        return None
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"\w+", value))


def expected(case):
    result = {"document_id": [case["id"]]}
    for i, section in enumerate(case["sections"], 1):
        result[f"excerpt_{i}_code"] = [section["code"]]
        for j, question in enumerate(section["questions"]):
            result[f"answer_{(i - 1) * 2 + j + 1}"] = [question["answer"], *question["aliases"]]
    return result


def parse(text):
    if not isinstance(text, str):
        return None
    text = text.strip()
    if text.count("</think>") > 1:
        return None
    # Only a leading thinking block (or closing tag supplied by the template).
    text = re.sub(r"^(?:<think>.*?</think>|</think>)\s*", "", text, count=1, flags=re.DOTALL)
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        # Duplicate JSON keys are a malformed response, even if the last is right.
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate JSON key")
                result[key] = value
            return result
        result = json.loads(text, object_pairs_hook=unique)
        return result if isinstance(result, dict) else None
    except (ValueError, TypeError):
        return None


def user_text(case, texts):
    lines = [f"Document ID: {case['id']}", f"{case['title']} — {case['author']}",
             "The following are three excerpts; intervening passages may be omitted."]
    for i, (section, text) in enumerate(zip(case["sections"], texts), 1):
        lines += [f"\nEXCERPT {i}; CODE {section['code']}\n", text, "\nEND OF EXCERPT"]
    lines += ["\nReturn exactly one JSON object. All values must be short English strings. "
              "Keys: document_id, excerpt_1_code, excerpt_2_code, excerpt_3_code, "
              "answer_1, answer_2, answer_3, answer_4, answer_5, answer_6. "
              "Copy the document ID and each excerpt code from this document. "
              "No explanations or additional turns."]
    for i, section in enumerate(case["sections"]):
        for j, question in enumerate(section["questions"]):
            lines.append(f"answer_{i * 2 + j + 1} (excerpt {i + 1}): {question['question']}")
    return "\n".join(lines)


def chat_ids(tokenizer, text):
    result = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )
    # Transformers versions differ: newer versions return BatchEncoding by default.
    return list(result["input_ids"] if isinstance(result, Mapping) else result)


def freeze_case(tokenizer, case, target):
    raw = [list(tokenizer.encode(s["text"], add_special_tokens=False)) for s in case["sections"]]
    overhead = len(chat_ids(tokenizer, user_text(case, ["", "", ""])))
    budget = (target - overhead) // 3
    if budget < 100:
        raise ValueError("Input budget too small")
    if min(map(len, raw)) < budget:
        raise ValueError(f"{case['id']}: source excerpt too short for balanced 8K; no filler added")
    lengths = [budget, budget, budget]
    # Retokenization at prose boundaries changes counts slightly. Grow only real text.
    for _ in range(20):
        texts = [tokenizer.decode(ids[:n], skip_special_tokens=False)
                 for ids, n in zip(raw, lengths)]
        full = chat_ids(tokenizer, user_text(case, texts))
        if len(full) >= target:
            break
        lengths[2] += target - len(full) + 16
        if lengths[2] > len(raw[2]):
            raise ValueError("Third excerpt exhausted")
    else:
        raise ValueError("Could not build exact input")
    short = chat_ids(tokenizer, user_text(case, [texts[0], texts[1], ""]))
    prefix = 0
    while prefix < min(len(full), len(short)) and full[prefix] == short[prefix]:
        prefix += 1
    suffix = 0
    while (suffix < min(len(full), len(short)) - prefix
           and full[-suffix - 1] == short[-suffix - 1]):
        suffix += 1
    end = len(full) - suffix
    excess = len(full) - target
    if excess > end - prefix:
        raise ValueError("Clipping would touch template/questions")
    ids = full[:end - excess] + full[end:]
    rendered = tokenizer.decode(ids, skip_special_tokens=False)
    if len(ids) != target:
        raise ValueError("Wrong token count")
    for section in case["sections"]:
        if norm(section["code"]) not in norm(rendered):
            raise ValueError("Excerpt code lost")
        for question in section["questions"]:
            if norm(question["evidence"]) not in norm(rendered):
                raise ValueError(f"Clipping removed answer evidence: {case['id']} {question['evidence']}")
    return {"id": case["id"], "token_ids": ids, "rendered": rendered,
            "tokens": len(ids), "input_sha256": digest(ids)}


def prepare(args):
    from transformers import AutoTokenizer
    cases = read_json(ROOT / "cases.json")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    if args.out.exists():
        raise ValueError("Output directory exists; choose a new path to preserve fixtures")
    frozen = [freeze_case(tokenizer, c, args.tokens) for c in cases]
    args.out.mkdir(parents=True, exist_ok=False)
    for item in frozen:
        (args.out / f"input_{item['id']}.txt").write_text(item["rendered"], encoding="utf-8")
    payload = {"format": 1, "batch_size": 4, "input_tokens": args.tokens,
               "tokenizer": str(args.tokenizer), "cases_sha256": digest(cases),
               "tokenizer_class": type(tokenizer).__name__, "cases": frozen}
    payload["pack_sha256"] = digest([i["token_ids"] for i in frozen])
    dump(args.out / "pack.json", payload)
    dump(args.out / "prompt_bs4.json", [i["token_ids"] for i in frozen])
    print(f"Frozen BS4: four x {args.tokens} tokens; digest {payload['pack_sha256']}")


def load_pack(path):
    pack = read_json(path)
    cases = read_json(ROOT / "cases.json")
    if (pack.get("batch_size") != 4 or pack.get("cases_sha256") != digest(cases)
        or [c["id"] for c in pack["cases"]] != [c["id"] for c in cases]
        or pack["pack_sha256"] != digest([c["token_ids"] for c in pack["cases"]])
        or any(len(c["token_ids"]) != pack["input_tokens"] for c in pack["cases"])):
        raise ValueError("Pack/input/case mismatch; refuse incomparable run")
    return pack, cases


def verify_tokenizer(args):
    provenance = read_json(ROOT / "fixtures/qwen3.6-27b-8k/tokenizer_provenance.json")
    failures = []
    for name in ("tokenizer.json", "tokenizer_config.json"):
        path = args.tokenizer / name
        reference = next(f for f in provenance["files"] if f["name"] == name)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != reference["sha256"]:
            failures.append(name)
    template = args.tokenizer / "chat_template.jinja"
    if template.exists():
        reference = next(f for f in provenance["files"] if f["name"] == template.name)
        if hashlib.sha256(template.read_bytes()).hexdigest() != reference["sha256"]:
            failures.append(template.name)
    if failures:
        raise ValueError("Tokenizer assets differ from the frozen official revision: "
                         + ", ".join(failures) + "; prepare a new pack using the server tokenizer")
    print("Tokenizer matches frozen Qwen/Qwen3.6-27B revision " + provenance["revision"])


def score_rows(cases, rows):
    if len(rows) % 4 or not rows:
        raise ValueError("Expected complete rounds of four answers")
    correct = total = complete = valid_json = valid_schema = unknown_finish = 0
    fact_correct = fact_total = id_correct = id_total = 0
    details = []
    for index, row in enumerate(rows):
        case = cases[index % 4]
        if row.get("id") != case["id"]:
            raise ValueError("Answer order/identity mismatch")
        accepted = expected(case)
        answer = parse(row.get("text"))
        finish = row.get("finish_reason")
        unknown_finish += finish is None
        ended = finish in (None, "stop")
        schema_ok = answer is not None and set(answer) == set(accepted)
        valid_json += answer is not None
        valid_schema += schema_ok
        hits = {key: ended and answer is not None and norm(answer.get(key)) is not None
                and norm(answer[key]) in {norm(v) for v in allowed}
                for key, allowed in accepted.items()}
        total += len(hits)
        correct += sum(hits.values())
        fact_correct += sum(v for k, v in hits.items() if k.startswith("answer_"))
        fact_total += sum(k.startswith("answer_") for k in hits)
        id_correct += sum(v for k, v in hits.items() if not k.startswith("answer_"))
        id_total += sum(not k.startswith("answer_") for k in hits)
        all_correct = schema_ok and all(hits.values())
        complete += all_correct
        details.append({"id": case["id"], "round": index // 4, "parsed": answer,
                        "field_correct": hits, "all_correct": all_correct,
                        "finish_reason": finish, "valid_schema": schema_ok})
    return {"field_accuracy": correct / total, "correct_fields": correct, "total_fields": total,
            "document_accuracy": complete / len(rows), "correct_documents": complete,
            "total_documents": len(rows), "valid_json": valid_json,
            "factual_accuracy": fact_correct / fact_total,
            "correct_facts": fact_correct, "total_facts": fact_total,
            "identifier_accuracy": id_correct / id_total,
            "correct_identifiers": id_correct, "total_identifiers": id_total,
            "valid_schema": valid_schema, "finish_reason_unverified": unknown_finish,
            "details": details}


def print_score(result):
    print(f"Factual accuracy: {result['correct_facts']}/{result['total_facts']} "
          f"= {result['factual_accuracy']:.2%}; identifiers: "
          f"{result['correct_identifiers']}/{result['total_identifiers']}")
    print(f"Fact/ID accuracy: {result['correct_fields']}/{result['total_fields']} "
          f"= {result['field_accuracy']:.2%}")
    print(f"Fully correct documents: {result['correct_documents']}/{result['total_documents']} "
          f"= {result['document_accuracy']:.2%}; valid JSON {result['valid_json']}/{result['total_documents']}")
    if result["finish_reason_unverified"]:
        print("Finish reason missing: content scored, runtime completion not verified.")


def run(args):
    if args.out.exists():
        raise ValueError("Output exists; choose a new file to preserve evidence")
    pack, cases = load_pack(args.pack)
    sampling = {"temperature": 0, "seed": 42, "max_tokens": args.max_tokens,
                "ignore_eos": False, "n": 1, "stream": False}
    headers = {"Content-Type": "application/json"}
    if os.environ.get("VLLM_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["VLLM_API_KEY"]
    result = {"tag": args.tag, "revision_user_supplied": args.revision,
              "model": args.model, "pack_sha256": pack["pack_sha256"],
              "cases_sha256": pack["cases_sha256"],
              "input_tokens": pack["input_tokens"], "batch_size": 4,
              "sampling": sampling, "metadata": read_json(args.metadata) if args.metadata else {},
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "rounds": [], "rows": []}
    for round_id in range(args.rounds):
        body = {"model": args.model, "prompt": [c["token_ids"] for c in pack["cases"]], **sampling}
        req = urllib.request.Request(args.base_url.rstrip("/") + "/v1/completions",
                                     data=json.dumps(body).encode(), headers=headers)
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as response:
                data = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc
        choices = sorted(data.get("choices", []), key=lambda c: c["index"])
        if [c["index"] for c in choices] != list(range(4)):
            raise ValueError("Server did not return exactly four choices with n=1")
        if data.get("usage", {}).get("prompt_tokens") != pack["input_tokens"] * 4:
            raise ValueError("Server did not account for four complete input prompts")
        result["rounds"].append({"round": round_id, "seconds": time.monotonic() - started,
                                  "response": data})
        result["rows"] += [{"id": c["id"], "text": choice["text"],
                            "finish_reason": choice.get("finish_reason")}
                           for c, choice in zip(cases, choices)]
        result["score"] = score_rows(cases, result["rows"])
        dump(args.out, result)
        print(f"{args.tag}: completed BS4 round {round_id + 1}/{args.rounds}", flush=True)
    print_score(result["score"])
    return 0 if result["score"]["correct_documents"] == len(result["rows"]) else 1


def score(args):
    cases = read_json(ROOT / "cases.json")
    data = read_json(args.outputs)
    if isinstance(data, dict) and "rows" in data:
        rows = data["rows"]
    else:
        texts = data.get("generated_texts") if isinstance(data, dict) else data
        if not isinstance(texts, list) or len(texts) != 4 or not all(isinstance(t, str) for t in texts):
            raise ValueError("Use a list of four strings, generated_texts, or a saved run")
        rows = [{"id": c["id"], "text": t, "finish_reason": None} for c, t in zip(cases, texts)]
    result = score_rows(cases, rows)
    dump(args.out, result)
    print_score(result)
    return 0 if result["correct_documents"] == len(rows) else 1


def canonical(case, field, value):
    n = norm(value)
    if n is not None and n in {norm(v) for v in expected(case)[field]}:
        return norm(expected(case)[field][0])
    return n


def compare(args):
    main = read_json(args.main)
    candidate = read_json(args.candidate)
    cases = read_json(ROOT / "cases.json")
    if (not main.get("pack_sha256") or main.get("cases_sha256") != digest(cases)
        or candidate.get("cases_sha256") != digest(cases)):
        raise ValueError("Missing input provenance or wrong question set")
    if any(main.get(k) != candidate.get(k) for k in
           ["pack_sha256", "input_tokens", "batch_size", "sampling"]):
        raise ValueError("Inputs/BS/sampling differ; comparison refused")
    baseline = score_rows(cases, main["rows"])
    current = score_rows(cases, candidate["rows"])
    regressions, unstable = [], []
    agree = eligible = 0
    for case in cases:
        b = [r for r in baseline["details"] if r["id"] == case["id"]]
        c = [r for r in current["details"] if r["id"] == case["id"]]
        for field in expected(case):
            values = {canonical(case, field, r["parsed"].get(field)) for r in b
                      if r["parsed"] is not None and norm(r["parsed"].get(field)) is not None
                      and r["finish_reason"] in (None, "stop")}
            if len(values) != 1 or any(not r["valid_schema"]
                                      or r["finish_reason"] not in (None, "stop") for r in b):
                unstable.append(f"{case['id']}:{field}")
            stable_correct = all(r["field_correct"][field] for r in b)
            if stable_correct and any(not r["field_correct"][field] for r in c):
                regressions.append(f"{case['id']}:{field}")
            if values:
                eligible += len(c)
                agree += sum(r["parsed"] is not None and r["finish_reason"] in (None, "stop")
                             and canonical(case, field, r["parsed"].get(field)) in values for r in c)
    report = {"main_accuracy": baseline["field_accuracy"],
              "candidate_accuracy": current["field_accuracy"],
              "delta_percentage_points": 100 * (current["field_accuracy"] - baseline["field_accuracy"]),
              "main_factual_accuracy": baseline["factual_accuracy"],
              "candidate_factual_accuracy": current["factual_accuracy"],
              "factual_delta_percentage_points": 100 * (current["factual_accuracy"] - baseline["factual_accuracy"]),
              "main_agreement": agree / eligible if eligible else None,
              "agreement_fields": agree, "eligible_fields": eligible,
              "lost_stable_main_correct_fields": regressions,
              "main_unstable_or_invalid_fields": unstable,
              "main": baseline, "candidate": current,
              "scope": "Four documents; factual smoke test, not kernel numerical certification."}
    regression = bool(regressions or report["delta_percentage_points"] < 0
                      or report["factual_delta_percentage_points"] < 0)
    report["status"] = ("regression_detected" if regression else
                        "inconclusive_main_unstable_or_invalid" if unstable else
                        "no_regression_detected_on_this_set")
    dump(args.out, report)
    print(f"main={report['main_accuracy']:.2%}; candidate={report['candidate_accuracy']:.2%}; "
          f"delta={report['delta_percentage_points']:+.2f} percentage points")
    print(f"Lost stable-correct main fields: {len(regressions)}; "
          f"main unstable/invalid fields: {len(unstable)}")
    print(f"Semantic agreement with observed main answers: {agree}/{eligible}")
    print(report["status"])
    return 1 if regression else 2 if unstable else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="command", required=True)
    a = subs.add_parser("prepare", help="Freeze exact input IDs once, locally, without inference")
    a.add_argument("--tokenizer", required=True)
    a.add_argument("--tokens", type=int, default=8192)
    a.add_argument("--out", type=Path, default=Path("prepared"))
    a.set_defaults(func=prepare)
    a = subs.add_parser("verify-tokenizer", help="Check server tokenizer files against the ready pack, without dependencies")
    a.add_argument("--tokenizer", type=Path, required=True)
    a.set_defaults(func=verify_tokenizer)
    a = subs.add_parser("run", help="Record real main or candidate BS4 responses")
    a.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    a.add_argument("--model", required=True)
    a.add_argument("--tag", required=True)
    a.add_argument("--revision", required=True, help="vllm-ascend SHA; user supplied, not auto-attested")
    a.add_argument("--base-url", default="http://127.0.0.1:8000")
    a.add_argument("--rounds", type=int, default=3)
    a.add_argument("--max-tokens", type=int, default=384)
    a.add_argument("--timeout", type=float, default=1800)
    a.add_argument("--metadata", type=Path, help="Optional versions/hardware/server-command JSON")
    a.add_argument("--out", type=Path, required=True)
    a.set_defaults(func=run)
    a = subs.add_parser("score", help="Score already saved output strings or a run")
    a.add_argument("--outputs", type=Path, required=True)
    a.add_argument("--out", type=Path, default=Path("score.json"))
    a.set_defaults(func=score)
    a = subs.add_parser("compare", help="Compare actual main and candidate runs on identical inputs")
    a.add_argument("--main", type=Path, required=True)
    a.add_argument("--candidate", type=Path, required=True)
    a.add_argument("--out", type=Path, default=Path("comparison.json"))
    a.set_defaults(func=compare)
    args = p.parse_args()
    if getattr(args, "rounds", 1) < 1 or getattr(args, "tokens", 8192) < 1024:
        p.error("Positive rounds and input >=1024 required")
    result = args.func(args)
    raise SystemExit(result or 0)


if __name__ == "__main__":
    main()
