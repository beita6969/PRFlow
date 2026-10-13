#!/usr/bin/env python3
"""Build the six out-of-distribution test sets (128 tasks each) of the paper from pinned public sources.

  build_ood.py text      --raw RAW [--out data/r2orch/test/ood] [--download]   TriviaQA, MuSiQue, MATH-Hard
  build_ood.py env       --raw RAW [--out data/r2orch/test/ood] [--download]   SWE-Bench Verified, WebShop
  build_ood.py gpqa-ids  --csv gpqa_diamond.csv --out data/r2orch/test/ood/gpqa_diamond.ids.json
  build_ood.py gpqa      --csv gpqa_diamond.csv --ids data/r2orch/test/ood/gpqa_diamond.ids.json --out gpqa_diamond.jsonl

Every raw file is checked against the sha256 recorded in RAW_FILES.  Each benchmark keeps the first 128 candidates by
ascending sha256('test:<benchmark>:<source id>').  A row says which IID task type its benchmark is posed as
(``posed_as``: the dataset key of configs/r2orch.toml).  GPQA is gated: the repository holds only the Record IDs and
option-order seeds; accept the terms of https://huggingface.co/datasets/Idavidrein/gpqa, download gpqa_diamond.csv
yourself and run ``gpqa`` to materialize the questions locally (do not redistribute them).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
import urllib.request
from pathlib import Path

COUNT = 128
LETTERS = ("A", "B", "C", "D")

POSED_AS = {
    "triviaqa": "nq_open",
    "musique": "hotpotqa",
    "math_hard": "aime",
    "gpqa_diamond": "medqa",
    "swe_bench_verified": "mbppplus",
    "webshop": "alfworld",
}
ORIGIN = {
    "triviaqa": "mandarjoshi/trivia_qa/rc.nocontext/validation",
    "musique": "dgslibisey/MuSiQue/default/validation (musique_ans_v1.0_dev)",
    "math_hard": "lighteval/MATH-Hard/default/test",
    "gpqa_diamond": "Idavidrein/gpqa/gpqa_diamond/train",
    "swe_bench_verified": "princeton-nlp/SWE-bench_Verified/default/test",
    "webshop": "princeton-nlp/WebShop/human_goals/test",
}

# name -> (pinned URL, sha256)
RAW_FILES = {
    "triviaqa_rc_nocontext_validation.parquet": (
        "https://huggingface.co/datasets/mandarjoshi/trivia_qa/resolve/0f7faf33a3908546c6fd5b73a660e0f8ff173c2f/rc.nocontext/validation-00000-of-00001.parquet",
        "48a5005c0eb4f8a4ae5b9868644297fd5bf1e694aeb3dc9c8ab958cac0d5b201",
    ),
    "musique_validation.parquet": (
        "https://huggingface.co/datasets/dgslibisey/MuSiQue/resolve/1bd32e8b3f4f29a11f8240a9f666ef52e159c24e/default/validation/0000.parquet",
        "2d671592a7674a0920d2a595df2e13acd74c8a80997f763ca81484a273f8adfe",
    ),
    "math_hard_test.parquet": (
        "https://huggingface.co/datasets/lighteval/MATH-Hard/resolve/69bc17a93d6754e3c3a22084a58f6ad2c24f1f07/default/test/0000.parquet",
        "90ba4ce7f108cd511c8eb716ea7ca906d5dc1621f5e804565765a368ca02ab01",
    ),
    "swe_bench_verified_test.parquet": (
        "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/resolve/c104f840cc67f8b6eec6f759ebc8b2693d585d4a/data/test-00000-of-00001.parquet",
        "a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd",
    ),
    "webshop_items_human_ins.json": (
        "https://huggingface.co/datasets/YWZBrandon/webshop-data/resolve/ce990fff5aee388db2706f07820c578ab68e0453/items_human_ins.json",
        "cf78667548a71786e1d9049c24b802e48e1084ad4bb021cae56ce1f6d96954a3",
    ),
    # 5.5 GB: only its product order is read, to reproduce the official goal order.
    "webshop_items_shuffle.json": (
        "https://huggingface.co/datasets/YWZBrandon/webshop-data/resolve/ce990fff5aee388db2706f07820c578ab68e0453/items_shuffle.json",
        "2ef591d65df3af89e972ab72468eb82cbf124d876552d9f3678667edd620a6c8",
    ),
}
GPQA_SOURCE = {
    "dataset": "Idavidrein/gpqa",
    "config": "gpqa_diamond",
    "file": "gpqa_diamond.csv",
    "revision": "83022cefff930aea54f654c0b282e74b9eeda5c6",
    "git_blob_sha1": "7589e3e467d69a1dceb126a60c4108d6d4f1d166",
    "url": "https://huggingface.co/datasets/Idavidrein/gpqa/blob/83022cefff930aea54f654c0b282e74b9eeda5c6/gpqa_diamond.csv",
}
NEEDS = {
    "text": ["triviaqa_rc_nocontext_validation.parquet", "musique_validation.parquet", "math_hard_test.parquet"],
    "swe_bench_verified": ["swe_bench_verified_test.parquet"],
    "webshop": ["webshop_items_human_ins.json", "webshop_items_shuffle.json"],
}
WEBSHOP_GOAL_SHUFFLE_SEED = 233
WEBSHOP_TEST_GOALS = range(0, 500)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def check_raw(raw: Path, names: list[str], download: bool) -> None:
    for name in names:
        url, expected = RAW_FILES[name]
        path = raw / name
        if not path.is_file():
            if not download:
                raise SystemExit(f"missing {path}; download {url} or pass --download")
            raw.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(path.suffix + ".part")
            print(f"downloading {url}", file=sys.stderr)
            with urllib.request.urlopen(url) as response, partial.open("wb") as handle:
                shutil.copyfileobj(response, handle, 1 << 22)
            partial.rename(path)
        found = sha256_file(path)
        if found != expected:
            raise SystemExit(f"sha256 mismatch for {path}: {found} != {expected}")


def rank(kind: str, source_id: str) -> str:
    return hashlib.sha256(f"test:{kind}:{source_id}".encode()).hexdigest()


def select(kind: str, items: list[tuple[str, dict]]) -> list[dict]:
    ids = [source_id for source_id, _ in items]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{kind}: duplicate source ids")
    chosen = sorted(items, key=lambda item: rank(kind, item[0]))[:COUNT]
    if len(chosen) != COUNT:
        raise SystemExit(f"{kind}: only {len(chosen)} candidates")
    return [
        {
            **row,
            "kind": kind,
            "posed_as": POSED_AS[kind],
            "source_id": source_id,
            "task_id": f"{kind}/test-{source_id}",
            "split": "test",
            "origin": ORIGIN[kind],
        }
        for source_id, row in chosen
    ]


def dedupe(values: list[str]) -> list[str]:
    seen: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return seen


def triviaqa(raw: Path) -> list[dict]:
    import pyarrow.parquet as pq

    by_id: dict[str, dict] = {}
    conflicting: set[str] = set()
    for row in pq.read_table(raw / "triviaqa_rc_nocontext_validation.parquet").to_pylist():
        answer = row["answer"]
        aliases = dedupe([answer["normalized_value"], *answer["normalized_aliases"]])
        item = {
            "prompt": f"Answer the question. Give a short answer only.\nQuestion: {row['question']}",
            "answer": aliases,
            "answers": aliases,
        }
        if by_id.setdefault(row["question_id"], item) != item:
            conflicting.add(row["question_id"])
    items = sorted((qid, item) for qid, item in by_id.items() if qid not in conflicting)
    return select("triviaqa", items)


def musique(raw: Path) -> list[dict]:
    import pyarrow.parquet as pq

    items = []
    for row in pq.read_table(raw / "musique_validation.parquet").to_pylist():
        if not row["answerable"]:
            continue
        paragraphs = sorted(row["paragraphs"], key=lambda p: p["idx"])
        context = "\n".join(f"[{p['title']}] {p['paragraph_text']}" for p in paragraphs)
        prompt = (
            "Answer the question using the passages. Give a short answer only.\n"
            f"Question: {row['question']}\nPassages:\n{context}"
        )
        answers = dedupe([row["answer"], *(row["answer_aliases"] or [])])
        items.append((row["id"], {"prompt": prompt, "answer": row["answer"], "answers": answers}))
    return select("musique", items)


def last_boxed(text: str) -> str | None:
    start = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if start < 0:
        return None
    rest = text[start:]
    if rest.startswith("\\boxed "):
        return rest[len("\\boxed "):].split("$")[0].strip()
    open_at = rest.find("{")
    if open_at < 0:
        return None
    depth = 0
    for index in range(open_at, len(rest)):
        if rest[index] == "{":
            depth += 1
        elif rest[index] == "}":
            depth -= 1
            if depth == 0:
                return rest[open_at + 1:index].strip()
    return None


MATH_SUBJECTS = {
    "Algebra": "algebra",
    "Counting & Probability": "counting_and_probability",
    "Geometry": "geometry",
    "Intermediate Algebra": "intermediate_algebra",
    "Number Theory": "number_theory",
    "Prealgebra": "prealgebra",
    "Precalculus": "precalculus",
}


def math_hard(raw: Path) -> list[dict]:
    import pyarrow.parquet as pq

    items = []
    counters: dict[str, int] = {}
    for row in pq.read_table(raw / "math_hard_test.parquet").to_pylist():
        if row["level"] != "Level 5":
            raise SystemExit("math_hard: row outside Level 5")
        subject = MATH_SUBJECTS[row["type"]]
        index = counters.get(subject, 0)
        counters[subject] = index + 1
        answer = last_boxed(row["solution"])
        if not answer:
            raise SystemExit(f"math_hard: no boxed answer in {subject}/{index}")
        prompt = f"Solve the following competition math problem. Put the final answer in \\boxed{{}}.\n{row['problem']}"
        items.append((f"{subject}_{index:03d}", {"prompt": prompt, "answer": answer}))
    return select("math_hard", items)


def gpqa_seed(record_id: str) -> int:
    return int(hashlib.sha256(f"options:gpqa_diamond:{record_id}".encode()).hexdigest()[:8], 16)


def gpqa_order(seed: int) -> list[int]:
    return sorted(range(4), key=lambda k: hashlib.sha256(f"{seed}:{k}".encode()).hexdigest())


def gpqa_ids(record_ids: list[str], raw_sha256: str | None) -> dict:
    chosen = sorted(record_ids, key=lambda rid: rank("gpqa_diamond", rid))[:COUNT]
    return {
        "kind": "gpqa_diamond",
        "posed_as": POSED_AS["gpqa_diamond"],
        "source": {**GPQA_SOURCE, "rows": len(record_ids), "sha256": raw_sha256},
        "selection": "first 128 Record IDs by ascending sha256('test:gpqa_diamond:<Record ID>')",
        "option_seed": "int(sha256('options:gpqa_diamond:<Record ID>').hexdigest()[:8], 16)",
        "option_order": "options [Correct Answer, Incorrect Answer 1, Incorrect Answer 2, Incorrect Answer 3] "
        "sorted by sha256('<seed>:<k>') for k=0..3 give letters A-D",
        "items": [{"record_id": rid, "option_seed": gpqa_seed(rid)} for rid in chosen],
    }


def gpqa_csv(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = {row["Record ID"].strip(): row for row in csv.DictReader(stream)}
    if len(rows) != 198:
        raise SystemExit(f"gpqa_diamond: expected 198 records, got {len(rows)}")
    return rows


def gpqa(path: Path, ids_path: Path) -> list[dict]:
    records = gpqa_csv(path)
    spec = json.loads(ids_path.read_text())
    if gpqa_ids(sorted(records), spec["source"]["sha256"])["items"] != spec["items"]:
        raise SystemExit("gpqa_diamond: Record IDs or option seeds differ from the ids file")
    rows = []
    for item in spec["items"]:
        record = records[item["record_id"]]
        texts = [record["Correct Answer"], record["Incorrect Answer 1"], record["Incorrect Answer 2"], record["Incorrect Answer 3"]]
        order = gpqa_order(item["option_seed"])
        listed = "\n".join(f"{letter}. {texts[k].strip()}" for letter, k in zip(LETTERS, order))
        prompt = (
            "Answer the following question by choosing one option.\n"
            f"Question: {record['Question'].strip()}\nOptions:\n{listed}\n"
            "End your reply with 'Answer: <letter>'."
        )
        rid = item["record_id"]
        rows.append(
            {
                "answer": LETTERS[order.index(0)],
                "kind": "gpqa_diamond",
                "options": list(LETTERS),
                "origin": ORIGIN["gpqa_diamond"],
                "posed_as": POSED_AS["gpqa_diamond"],
                "prompt": prompt,
                "source_id": rid,
                "split": "test",
                "task_id": f"gpqa_diamond/test-{rid}",
            }
        )
    return rows


def swe_bench_verified(raw: Path) -> list[dict]:
    import pyarrow.parquet as pq

    rows = pq.read_table(raw / "swe_bench_verified_test.parquet", columns=["instance_id", "repo", "base_commit"]).to_pylist()
    if len(rows) != 500:
        raise SystemExit(f"swe_bench_verified: expected 500 instances, found {len(rows)}")
    items = [(row["instance_id"], {"repo": row["repo"], "base_commit": row["base_commit"]}) for row in rows]
    return select("swe_bench_verified", items)


def iter_product_asins(path: Path):
    try:
        import ijson
    except ImportError:
        with path.open() as handle:
            for product in json.load(handle):
                yield product["asin"]
        return
    with path.open("rb") as handle:
        yield from ijson.items(handle, "item.asin")


def webshop(raw: Path) -> list[dict]:
    with (raw / "webshop_items_human_ins.json").open() as handle:
        human = json.load(handle)
    seen: set[str] = set()
    goals: list[dict] = []
    for asin in iter_product_asins(raw / "webshop_items_shuffle.json"):
        if asin == "nan" or len(asin) > 10 or asin in seen:
            continue
        seen.add(asin)
        for position, instruction in enumerate(human.get(asin, [])):
            if len(instruction["instruction_attributes"]) == 0:
                continue
            goals.append({"asin": asin, "instruction_position": position, "instruction": instruction["instruction"].strip(".")})
    random.seed(WEBSHOP_GOAL_SHUFFLE_SEED)
    random.shuffle(goals)
    if len(goals) != 12087:
        raise SystemExit(f"webshop: expected 12087 human goals, found {len(goals)}")
    items = [
        (
            str(index),
            {
                "goal_index": index,
                "goal_shuffle_seed": WEBSHOP_GOAL_SHUFFLE_SEED,
                "asin": goals[index]["asin"],
                "instruction_position": goals[index]["instruction_position"],
                "instruction": goals[index]["instruction"],
            },
        )
        for index in WEBSHOP_TEST_GOALS
    ]
    return select("webshop", items)


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"{path.name}: {len(rows)} rows sha256={sha256_file(path)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="part", required=True)
    for name in ("text", "env"):
        part = sub.add_parser(name)
        part.add_argument("--raw", type=Path, required=True, help="directory holding the raw files of RAW_FILES")
        part.add_argument("--out", type=Path, default=Path("data/r2orch/test/ood"))
        part.add_argument("--download", action="store_true", help="fetch missing raw files from the pinned URLs")
    sub.choices["env"].add_argument("--only", nargs="+", choices=("swe_bench_verified", "webshop"), default=["swe_bench_verified", "webshop"])
    ids = sub.add_parser("gpqa-ids")
    ids.add_argument("--csv", type=Path, required=True)
    ids.add_argument("--out", type=Path, required=True)
    text = sub.add_parser("gpqa")
    text.add_argument("--csv", type=Path, required=True)
    text.add_argument("--ids", type=Path, required=True)
    text.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.part == "text":
        check_raw(args.raw, NEEDS["text"], args.download)
        write(args.out / "triviaqa.jsonl", triviaqa(args.raw))
        write(args.out / "musique.jsonl", musique(args.raw))
        write(args.out / "math_hard.jsonl", math_hard(args.raw))
    elif args.part == "env":
        for kind in args.only:
            check_raw(args.raw, NEEDS[kind], args.download)
            write(args.out / f"{kind}.jsonl", {"swe_bench_verified": swe_bench_verified, "webshop": webshop}[kind](args.raw))
    elif args.part == "gpqa-ids":
        spec = gpqa_ids(sorted(gpqa_csv(args.csv)), sha256_file(args.csv))
        args.out.write_text(json.dumps(spec, indent=1) + "\n")
        print(f"{args.out.name}: {len(spec['items'])} ids")
    else:
        write(args.out, gpqa(args.csv, args.ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
