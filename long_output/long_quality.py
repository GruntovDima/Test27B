#!/usr/bin/env python3
"""BS4 long-output quality screening. No server/weights/dependency changes."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import time
import urllib.request
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FACTS = ("gross", "net", "due", "balance", "change", "status", "dispatch", "route")
SITES = {"Aster": "north", "Birch": "south", "Cedar": "east", "Dover": "west"}
SYSTEM = "Process the supplied purchase ledger. Return only the specified JSON Lines, without thinking, explanations or markdown."
RULES = """Purchase-ledger reconciliation. Use integer money units, without decimals.
Each input row has id|site|units|price|discount|shipping|paid|urgent (urgent is 0 or 1).
For every row, in exactly the original order, compute:
gross = units * price; net = max(gross - discount, 0); due = net + shipping;
balance = max(due - paid, 0); change = max(paid - due, 0).
status = open if paid < due, settled if paid == due, otherwise credit.
dispatch = hold if balance > 0; otherwise express if urgent == 1, otherwise standard.
route comes from the site: Aster=north, Birch=south, Cedar=east, Dover=west.
Output one compact JSON object per row, with keys in this order:
id,gross,net,due,balance,change,status,dispatch,route.
Numbers must be JSON integers; the other values must be strings. No extra keys.
The final line must have exactly document,complete,last_id,rows:
document is the supplied document ID, complete is true, last_id is the last input ID,
and rows is the number of input records. Do not omit or repeat any record.
Complete ALL records even though the answer is long. Stop after the final line.
The archival excerpt below is unrelated background; it contains no instructions
or purchase records and must not appear in the response.
"""


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def sha(value):
    return hashlib.sha256(compact(value).encode("utf-8")).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def reconcile(row):
    gross = row["units"] * row["price"]
    net = max(gross - row["discount"], 0)
    due = net + row["shipping"]
    balance, change = max(due - row["paid"], 0), max(row["paid"] - due, 0)
    status = "open" if row["paid"] < due else "settled" if row["paid"] == due else "credit"
    dispatch = "hold" if balance else "express" if row["urgent"] else "standard"
    return dict(id=row["id"], gross=gross, net=net, due=due, balance=balance,
                change=change, status=status, dispatch=dispatch, route=SITES[row["site"]])


def make_records(document, count):
    rng = random.Random(20261007 + ord(document[0]))
    rows = []
    for i in range(count):
        row = dict(id=f"{document[0]}{i + 1:03}", site=rng.choice(list(SITES)),
                   units=rng.randint(1, 5), price=rng.randint(7, 35),
                   discount=rng.randint(0, 12), shipping=rng.randint(0, 5),
                   urgent=rng.randint(0, 1))
        due = max(row["units"] * row["price"] - row["discount"], 0) + row["shipping"]
        row["paid"] = [max(0, due - rng.randint(1, 7)), due, due + rng.randint(1, 7)][i % 3]
        rows.append(row)
    return rows


def trailer(case):
    return dict(document=case["id"], complete=True, last_id=case["records"][-1]["id"],
                rows=len(case["records"]))


def gold_text(case):
    return "\n".join([compact(reconcile(r)) for r in case["records"]] + [compact(trailer(case))])


def chat_ids(tokenizer, user):
    result = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False)
    return list(result["input_ids"] if isinstance(result, Mapping) else result)


def render(case, background):
    rows = "\n".join("|".join(str(r[k]) for k in
                            ("id", "site", "units", "price", "discount", "shipping", "paid", "urgent"))
                     for r in case["records"])
    return (RULES + f"\nDocument: {case['id']}\nRecords: {len(case['records'])}\n"
            + "BEGIN LEDGER\n" + rows + "\nEND LEDGER\nBEGIN ARCHIVE\n" + background
            + "\nEND ARCHIVE\nReconcile the complete ledger above now. Output JSON Lines only.")


def freeze(tokenizer, case, background, target):
    # Only cut unscored archival context. Never cut a ledger, rule or chat prefix.
    full, short = chat_ids(tokenizer, render(case, background)), chat_ids(tokenizer, render(case, ""))
    if len(short) >= target or len(full) < target:
        raise ValueError("Input budget cannot preserve all ledger rows; change task parameters")
    prefix = 0
    while prefix < min(len(full), len(short)) and full[prefix] == short[prefix]:
        prefix += 1
    suffix = 0
    while suffix < min(len(full), len(short)) - prefix and full[-suffix - 1] == short[-suffix - 1]:
        suffix += 1
    end = len(full) - suffix
    # The common token suffix can start at END ARCHIVE, not its preceding newline
    # (the empty and nonempty archives tokenize that boundary differently).
    # Preserve a real line boundary, budgeted inside the archive cut.
    boundary = tokenizer.encode("\n", add_special_tokens=False)
    excess = len(full) - target + len(boundary)
    if excess > end - prefix:
        raise ValueError("Clipping would remove required evidence")
    ids = full[:end - excess] + boundary + full[end:]
    rendered = tokenizer.decode(ids, skip_special_tokens=False)
    # Verify the protected segments survived verbatim, and lossless saved ID decode.
    before, after = render(case, "").split("BEGIN ARCHIVE\n", 1)
    if before not in rendered or after not in rendered or len(ids) != target:
        raise ValueError("Required ledger/rules/template were altered")
    return ids, rendered


def prepare(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    if args.out.exists():
        raise ValueError("Pack destination exists; preserve it and choose a new directory")
    background_path = Path(args.background)
    background = background_path.read_text(encoding="utf-8-sig")
    cases = []
    for document in ("A-ledger", "B-ledger", "C-ledger", "D-ledger"):
        # Choose an honest semantic output volume, not EOS padding or copied gold.
        for count in range(16, 400):
            case = dict(id=document, records=make_records(document, count))
            gold = gold_text(case)
            gold_ids = tokenizer.encode(gold, add_special_tokens=False)
            if len(gold_ids) >= args.output_tokens:
                break
        else:
            raise ValueError("Output target too large")
        ids, text = freeze(tokenizer, case, background, args.input_tokens)
        case.update(token_ids=ids, input_sha256=sha(ids), gold=gold,
                    gold_tokens=len(gold_ids), gold_sha256=sha(gold))
        starts = []
        for i in range(count):
            preceding = "\n".join(compact(reconcile(r)) for r in case["records"][:i])
            starts.append(len(tokenizer.encode(preceding + ("\n" if i else ""), add_special_tokens=False)))
        case["record_bands"] = [min(7, start * 8 // len(gold_ids)) for start in starts]
        cases.append(case)
        (args.out / "inputs").mkdir(parents=True, exist_ok=True)
        (args.out / "gold").mkdir(exist_ok=True)
        (args.out / "inputs" / f"{document}.txt").write_text(text, encoding="utf-8")
        (args.out / "gold" / f"{document}.jsonl").write_text(gold + "\n", encoding="utf-8")
    assets = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        path = Path(args.tokenizer) / name
        assets[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    pack = dict(format="long-quality-v1", batch_size=4, input_tokens=args.input_tokens,
                min_output_tokens=args.output_tokens, tokenizer_assets=assets,
                background_sha256=hashlib.sha256(background_path.read_bytes()).hexdigest(),
                scope="Synthetic ledger transformation, not general model accuracy", cases=cases)
    pack["pack_sha256"] = sha(pack)
    save(args.out / "pack.json", pack)
    report = {"pack_sha256": pack["pack_sha256"], "inference_executed": False,
              "tokenizer_class": type(tokenizer).__name__, "tokenizer_assets": assets,
              "cases": [{"id": c["id"], "rows": len(c["records"]),
                         "facts": len(c["records"]) * len(FACTS),
                         "input_tokens": len(c["token_ids"]), "gold_tokens": c["gold_tokens"]}
                        for c in cases]}
    save(args.out / "preparation.json", report)
    print(json.dumps(report, indent=2))


def load_pack(path):
    pack = read(path)
    unhashed = {k: v for k, v in pack.items() if k != "pack_sha256"}
    if pack.get("format") != "long-quality-v1" or pack.get("pack_sha256") != sha(unhashed):
        raise ValueError("Pack corruption or unsupported format")
    if pack.get("batch_size") != 4 or len(pack["cases"]) != 4:
        raise ValueError("Expected BS4")
    if len({c["id"] for c in pack["cases"]}) != 4:
        raise ValueError("Duplicate case identity")
    for c in pack["cases"]:
        if (len(c["token_ids"]) != pack["input_tokens"] or sha(c["token_ids"]) != c["input_sha256"]
                or any(type(i) is not int or i < 0 for i in c["token_ids"])
                or c["gold"] != gold_text(c) or c["gold_sha256"] != sha(c["gold"])
                or c["gold_tokens"] < pack["min_output_tokens"]
                or len(c["record_bands"]) != len(c["records"])
                or any(type(b) is not int or not 0 <= b < 8 for b in c["record_bands"])):
            raise ValueError("Invalid frozen case")
    return pack


def verify_tokenizer(pack, path):
    for name, expected in pack["tokenizer_assets"].items():
        asset = Path(path) / name
        actual = hashlib.sha256(asset.read_bytes()).hexdigest() if asset.is_file() else None
        if actual != expected:
            raise ValueError(f"Tokenizer mismatch (including presence/absence): {name}")


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("Duplicate key")
        obj[key] = value
    return obj


def same(actual, expected):
    # Do not accept True as 1 or floats as integer money values.
    return type(actual) is type(expected) and actual == expected


def streaming_timings(response):
    """Client first/last nonempty text events; never infer token count from chunks."""
    times = [event.get('seconds') for event in response.get('events', [])
             if any(choice.get('text') for choice in event.get('data', {}).get('choices', []))]
    count = (response.get('usage') or {}).get('completion_tokens')
    valid = (bool(times) and all(type(t) in (float, int) and math.isfinite(t) and t >= 0 for t in times)
             and all(a <= b for a, b in zip(times, times[1:])))
    if not valid:
        return dict(ttft_ms=None, tpot_ms=None, decode_seconds=None, token_intervals=None)
    intervals = count - 1 if type(count) is int and count > 1 else None
    elapsed = times[-1] - times[0]
    return dict(ttft_ms=1000 * times[0], tpot_ms=1000 * elapsed / intervals if intervals else None,
                decode_seconds=elapsed if intervals else None, token_intervals=intervals)


def score_case(pack, case, response):
    gold = [reconcile(r) for r in case["records"]]
    lines = response.get("text", "").strip().splitlines()
    objects, parse_errors = [], []
    for i, line in enumerate(lines):
        try:
            value = json.loads(line, object_pairs_hook=unique_object,
                               parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
            if not isinstance(value, dict):
                raise ValueError("Expected a JSON object")
            objects.append(value)
        except (ValueError, TypeError):
            objects.append(None)
            parse_errors.append(i)
    bands = [{"band": i, "correct_facts": 0, "facts": 0, "correct_records": 0, "records": 0}
             for i in range(8)]
    details = []
    for i, expected in enumerate(gold):
        actual = objects[i] if i < len(objects) else None
        identity = actual is not None and same(actual.get("id"), expected["id"])
        schema = actual is not None and set(actual) == set(expected)
        hits = {key: bool(identity and same(actual.get(key), expected[key])) for key in FACTS}
        correct = bool(schema and identity and all(hits.values()))
        band = bands[case["record_bands"][i]]
        band["correct_facts"] += sum(hits.values())
        band["facts"] += len(FACTS)
        band["correct_records"] += correct
        band["records"] += 1
        details.append(dict(id=expected["id"], band=band["band"], schema_ok=schema,
                            identity_ok=identity, fact_correct=hits, correct=correct))
    final = objects[-1] if objects else None
    wanted = trailer(case)
    trailer_ok = bool(final is not None and set(final) == set(wanted)
                      and all(same(final.get(k), v) for k, v in wanted.items()))
    structure_ok = bool(len(objects) == len(gold) + 1 and not parse_errors and trailer_ok
                        and all(d["schema_ok"] and d["identity_ok"] for d in details))
    usage = response.get("usage") or {}
    prompt, output = usage.get("prompt_tokens"), usage.get("completion_tokens")
    usage_ok = (type(prompt) is int and prompt == pack["input_tokens"]
                and type(output) is int and output > 0)
    length_ok = bool(usage_ok and output >= pack["min_output_tokens"])
    completed = (response.get("finish_reason") == "stop" and response.get("done") is True
                 and not response.get("error"))
    fact_correct = sum(b["correct_facts"] for b in bands)
    facts = sum(b["facts"] for b in bands)
    return dict(id=case["id"], correct_facts=fact_correct, facts=facts,
                factual_accuracy=fact_correct / facts, correct_records=sum(d["correct"] for d in details),
                records=len(gold), structure_ok=structure_ok, trailer_ok=trailer_ok,
                completed=completed, usage_ok=usage_ok, length_ok=length_ok,
                prompt_tokens=prompt, output_tokens=output, finish_reason=response.get("finish_reason"),
                parse_errors=parse_errors, bands=bands, details=details,
                passed=bool(structure_ok and completed and length_ok and fact_correct == facts),
                timing=streaming_timings(response))


def score_run(pack, run):
    rounds = run.get("rounds", [])
    expected_rounds = run.get("requested_rounds")
    count_ok = type(expected_rounds) is int and expected_rounds > 0 and len(rounds) == expected_rounds
    protocol_ok = (run.get("protocol") == "long-quality-v1" and run.get("pack_sha256") == pack["pack_sha256"]
                   and run.get("sampling", {}).get("ignore_eos") is False
                   and run.get("sampling", {}).get("temperature") == 0
                   and run.get("sampling", {}).get("seed") == 42
                   and type(run.get("sampling", {}).get("max_tokens")) is int
                   and run["sampling"]["max_tokens"] > max(c["gold_tokens"] for c in pack["cases"])
                   and run.get("metadata", {}).get("tp") == 4
                   and run.get("metadata", {}).get("prefix_caching") is False)
    results = []
    for index, round_ in enumerate(rounds):
        responses = round_.get("responses", [])
        if len(responses) != 4 or round_.get("round") != index:
            count_ok = False
        for case, response in zip(pack["cases"], responses):
            if response.get("id") != case["id"]:
                count_ok = False
            results.append(dict(round=index, **score_case(pack, case, response)))
    good = sum(r["passed"] for r in results)
    correct, total = sum(r["correct_facts"] for r in results), sum(r["facts"] for r in results)
    timings = [r['timing'] for r in results if r['completed'] and r['usage_ok']]
    ttfts = [t['ttft_ms'] for t in timings if t['ttft_ms'] is not None]
    tpots = [t['tpot_ms'] for t in timings if t['tpot_ms'] is not None]
    weighted = [t for t in timings if t['token_intervals'] is not None]
    latency = dict(mean_ttft_ms=sum(ttfts)/len(ttfts) if ttfts else None,
                   mean_tpot_ms=sum(tpots)/len(tpots) if tpots else None,
                   token_weighted_tpot_ms=1000*sum(t['decode_seconds'] for t in weighted)
                   /sum(t['token_intervals'] for t in weighted) if weighted else None,
                   ttft_count=len(ttfts), tpot_count=len(tpots),
                   definition='Client first/last nonempty text SSE event, usage token count; EOS/empty-event timing not observable')
    return dict(passed=bool(count_ok and protocol_ok and run.get("complete") is True
                            and len(results) == expected_rounds * 4 and good == len(results)),
                run_complete=run.get("complete") is True, round_count_ok=count_ok,
                protocol_ok=protocol_ok, passed_responses=good, responses=len(results),
                correct_facts=correct, facts=total, factual_accuracy=correct / total if total else None,
                details=results, scope=pack["scope"], latency=latency)


def request_one(args, pack, case, sampling):
    body = dict(model=args.model, prompt=case["token_ids"], n=1, stream=True,
                stream_options={"include_usage": True}, **sampling)
    headers = {"Content-Type": "application/json"}
    if os.environ.get("VLLM_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["VLLM_API_KEY"]
    req = urllib.request.Request(args.base_url.rstrip("/") + "/v1/completions",
                                 data=compact(body).encode("utf-8"), headers=headers)
    started = time.monotonic()
    result = dict(id=case["id"], request=body, text="", events=[], usage=None,
                  finish_reason=None, done=False, started_monotonic=started)
    last_checkpoint = started

    def checkpoint():
        directory = getattr(args, "progress_dir", None)
        if directory is not None:
            path = Path(directory) / f"{case['id']}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
            temporary.replace(path)

    checkpoint()
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as response:
            for raw in response:
                if not raw.startswith(b"data:"):
                    continue
                now = time.monotonic()
                payload = raw[5:].strip()
                if payload == b"[DONE]":
                    result["done"] = True
                    break
                event = json.loads(payload)
                result["events"].append(dict(seconds=now - started, data=event))
                if event.get("error"):
                    raise ValueError(str(event["error"]))
                if event.get("usage"):
                    result["usage"] = event["usage"]
                for choice in event.get("choices", []):
                    if choice.get("index") != 0:
                        raise ValueError("Expected exactly one sequence per HTTP request")
                    result["text"] += choice.get("text") or ""
                    if choice.get("finish_reason") is not None:
                        result["finish_reason"] = choice["finish_reason"]
                if now - last_checkpoint >= 30:
                    checkpoint()
                    last_checkpoint = now
        if not result["done"]:
            raise ValueError("SSE stream ended without DONE")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["seconds"] = time.monotonic() - started
    checkpoint()
    return result


def run(args):
    pack = load_pack(args.pack)
    if args.out.exists():
        raise ValueError("Output exists; use a new filename")
    sampling = dict(temperature=0, seed=42, max_tokens=args.max_tokens, ignore_eos=False)
    if args.max_tokens <= max(c["gold_tokens"] for c in pack["cases"]):
        raise ValueError("Output cap must leave room for a complete natural answer")
    metadata = read(args.metadata)
    required = ("weights", "tokenizer", "vllm", "vllm_ascend", "torch_npu", "cann",
                "dtype_quantization", "tp", "devices", "prefix_caching", "server_command")
    if any(k not in metadata or metadata[k] in (None, "", "FILL_ME") for k in required):
        raise ValueError("Fill actual server provenance in metadata JSON; no guessed versions")
    if metadata["tp"] != 4 or metadata["prefix_caching"] is not False:
        raise ValueError("Expected TP4 and disabled prefix caching")
    result = dict(protocol="long-quality-v1", tag=args.tag, model=args.model,
                  pack_sha256=pack["pack_sha256"], requested_rounds=args.rounds,
                  sampling=sampling, metadata=metadata, rounds=[], complete=False,
                  started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    save(args.out, result)
    for i in range(args.rounds):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(pool.map(lambda c: request_one(args, pack, c, sampling), pack["cases"]))
        result["rounds"].append(dict(round=i, responses=responses))
        save(args.out, result)
        if any(r.get("error") for r in responses):
            print("Client/server failure: raw partial responses retained; NOT PASS", flush=True)
            break
        print(f"{args.tag}: completed round {i + 1}/{args.rounds}", flush=True)
    result["complete"] = (len(result["rounds"]) == args.rounds
                           and all(not r.get("error") and r.get("done") for wave in result["rounds"]
                                   for r in wave["responses"]))
    result["score"] = score_run(pack, result)
    save(args.out, result)
    print_summary(result["score"])
    return 0 if result["score"]["passed"] else 1


def print_summary(score):
    print(f"PASS responses: {score['passed_responses']}/{score['responses']}; "
          f"facts: {score['correct_facts']}/{score['facts']}; complete={score['run_complete']}")
    for case in score["details"]:
        print(f"round {case['round'] + 1} {case['id']}: output={case['output_tokens']}, "
              f"facts={case['correct_facts']}/{case['facts']}, finish={case['finish_reason']}, "
              f"PASS={case['passed']}")
    latency = score.get('latency', {})
    for name in ('mean_ttft_ms', 'mean_tpot_ms', 'token_weighted_tpot_ms'):
        value = latency.get(name)
        print(f"{name}: {value:.3f} ms" if value is not None else f"{name}: N/A")


def compare(pack, baseline, candidate):
    if any(baseline.get(k) != candidate.get(k) for k in
           ("protocol", "pack_sha256", "requested_rounds", "sampling", "model")):
        raise ValueError("Different protocol/input/repeats/sampling/model; comparison refused")
    # The only permitted provenance difference is the tested plugin revision/command.
    for key in ("weights", "tokenizer", "vllm", "torch_npu", "cann", "dtype_quantization", "tp",
                "devices", "prefix_caching"):
        if (key not in baseline.get("metadata", {}) or key not in candidate.get("metadata", {})
                or baseline["metadata"][key] != candidate["metadata"][key]):
            raise ValueError(f"Missing/different provenance: {key}")
    b, c = score_run(pack, baseline), score_run(pack, candidate)
    comparable = (b["run_complete"] and c["run_complete"] and b["round_count_ok"] and c["round_count_ok"]
                  and b["protocol_ok"] and c["protocol_ok"])
    stable, lost, unstable = [], [], []
    for case in pack["cases"]:
        rows_b = [r for r in b["details"] if r["id"] == case["id"]]
        rows_c = [r for r in c["details"] if r["id"] == case["id"]]
        for index, record in enumerate(case["records"]):
            for field in FACTS:
                key = f"{case['id']}:{record['id']}:{field}"
                if rows_b and all(r["details"][index]["fact_correct"][field] for r in rows_b):
                    stable.append(key)
                    if any(not r["details"][index]["fact_correct"][field] for r in rows_c):
                        lost.append(key)
                else:
                    unstable.append(key)
    regression = bool(comparable and (lost or c["passed_responses"] < b["passed_responses"]
                                      or c["correct_facts"] < b["correct_facts"]))
    # Never turn a consistently bad or short candidate into PASS merely because main is also bad.
    status = ("INCONCLUSIVE_INCOMPLETE_RUN" if not comparable else
              "REGRESSION" if regression else "PASS_ON_THIS_SET" if b["passed"] and c["passed"] else
              "INCONCLUSIVE_QUALITY_OR_LENGTH_FAILURE")
    return dict(status=status, main=b, candidate=c, lost_stable_correct_facts=lost,
                baseline_not_always_correct_facts=unstable, stable_correct_facts=len(stable),
                metadata_is_user_attested=True), (0 if status == "PASS_ON_THIS_SET" else 1 if regression else 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--background", required=True)
    p.add_argument("--input-tokens", type=int, default=8192)
    p.add_argument("--output-tokens", type=int, default=8192)
    p.add_argument("--out", type=Path, default=ROOT / "prepared")
    p = sub.add_parser("verify-tokenizer")
    p.add_argument("--pack", type=Path, default=ROOT / "fixtures/pack.json")
    p.add_argument("--tokenizer", required=True)
    p = sub.add_parser("run")
    p.add_argument("--pack", type=Path, default=ROOT / "fixtures/pack.json")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=12288)
    p.add_argument("--timeout", type=float, default=7200)
    p.add_argument("--progress-dir", type=Path, help="Optional atomic per-sequence SSE evidence checkpoints")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("score")
    p.add_argument("--pack", type=Path, default=ROOT / "fixtures/pack.json")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("compare")
    p.add_argument("--pack", type=Path, default=ROOT / "fixtures/pack.json")
    p.add_argument("--main", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if getattr(args, "rounds", 1) < 1 or getattr(args, "timeout", 1) <= 0:
        parser.error("Positive repeats and timeout required")
    if args.command == "prepare":
        if args.input_tokens < 1024 or args.output_tokens < 1024:
            parser.error("Token targets must be at least 1024")
        prepare(args)
        return 0
    pack = load_pack(args.pack)
    if args.command == "verify-tokenizer":
        verify_tokenizer(pack, args.tokenizer)
        print("Tokenizer fingerprints match (including absent external template).")
        return 0
    if args.command == "run":
        return run(args)
    if args.out.exists():
        raise ValueError("Report exists; use a new filename")
    if args.command == "score":
        result = score_run(pack, read(args.run))
        save(args.out, result)
        print_summary(result)
        return 0 if result["passed"] else 1
    report, code = compare(pack, read(args.main), read(args.candidate))
    save(args.out, report)
    print(report["status"])
    return code


if __name__ == "__main__":
    raise SystemExit(main())
