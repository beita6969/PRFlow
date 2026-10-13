#!/usr/bin/env python3
"""Check that the training pool, the held-out set and the IID / OOD test sets share no task.

  check_disjoint.py [--data data/r2orch] [--gpqa-jsonl gpqa_diamond.jsonl]

Every held-out and test task is compared with EVERY training task (all task types), and every test task with every
held-out task, on its question text:
  exact   the same normalized question (MBPP+: the same statement with numbers masked as well)
  near    5-word-shingle containment >= 0.9 (questions of >= 8 words); 0.8-0.9 is listed for review
  game    ALFWorld only: the same game file (goal sentences repeat across scenes by design)
Question text = the prompt without its instruction header, passages and options; WebShop uses its instruction.
The held-out files are read from <data>/heldout/*.jsonl and the OOD files from <data>/test/ood/*.jsonl; GPQA
questions are not distributed, so pass the file built by build_ood.py gpqa to include them.  Exit status 1 when a
task has an exact or near hit or a game hit.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

N_SHINGLE, FAIL, REVIEW, MIN_WORDS = 5, 0.9, 0.8, 8
HEADERS = (
    "Solve the following competition math problem. The answer is an integer from 0 to 999. Put the final answer in \\boxed{}.\n",
    "Solve the following competition math problem. Put the final answer in \\boxed{}.\n",
    "Answer the question using the passages. Give a short answer only.\n",
    "Answer the question. Give a short answer only.\n",
    "Answer the following medical question by choosing one option.\n",
    "Answer the following question by choosing one option.\n",
)


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def question(row: dict) -> str:
    text = row.get("prompt") or row.get("instruction") or ""
    for header in HEADERS:
        if text.startswith(header):
            text = text[len(header):]
            break
    if "\nYour code should pass this test" in text:
        text = text.split("\nYour code should pass this test")[0]
    if text.startswith("Question:"):
        text = text[len("Question:"):]
    return re.split(r"\nPassages:|\nOptions?:|\n[A-D][.)] ", text)[0].strip()


def norm(text: str, mask_numbers: bool = False) -> str:
    text = re.sub(r"[^0-9a-z\u00c0-\uffff]+", " ", unicodedata.normalize("NFKC", text).lower())
    return " ".join((re.sub(r"\d+", "#", text) if mask_numbers else text).split())


def shingles(text: str) -> frozenset[str]:
    words = text.split()
    if len(words) < N_SHINGLE:
        return frozenset([" ".join(words)]) if words else frozenset()
    return frozenset(" ".join(words[i:i + N_SHINGLE]) for i in range(len(words) - N_SHINGLE + 1))


def game(row: dict) -> str | None:
    path = (row.get("metadata") or {}).get("game_path")
    return re.sub(r"^.*?(train|valid_unseen|valid_seen)/", "", str(path)) if path else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "r2orch")
    parser.add_argument("--gpqa-jsonl", type=Path)
    args = parser.parse_args()

    train = load(args.data / "train" / "train_pool.jsonl")
    heldout = {f"heldout/{p.stem}": load(p) for p in sorted((args.data / "heldout").glob("*.jsonl"))}
    tests = {f"iid/{p.stem}": load(p) for p in sorted((args.data / "test" / "iid").glob("*.jsonl"))}
    tests |= {f"ood/{p.stem}": load(p) for p in sorted((args.data / "test" / "ood").glob("*.jsonl"))}
    if args.gpqa_jsonl:
        tests["ood/gpqa_diamond"] = load(args.gpqa_jsonl)

    class Corpus:
        def __init__(self, rows: list[dict]) -> None:
            self.rows = rows
            self.texts = [norm(question(row)) for row in rows]
            self.exact, self.masked, self.index = (collections.defaultdict(list) for _ in range(3))
            for i, row in enumerate(rows):
                self.exact[self.texts[i]].append(i)
                self.masked[norm(question(row), True)].append(i)
                for piece in shingles(self.texts[i]):
                    self.index[piece].append(i)
            self.sets = [shingles(text) for text in self.texts]
            self.games = {game(row) for row in rows if row.get("dataset") == "alfworld"} - {None}

        def compare(self, rows: list[dict], name: str, against: str, notes: list[str]) -> dict:
            counts = {"exact": 0, "near": 0, "game": 0, "review": 0}
            for row in rows:
                ident = row.get("id") or row.get("task_id")
                if row.get("dataset") == "alfworld":
                    counts["game"] += game(row) in self.games
                    continue
                text = norm(question(row))
                if not text:
                    continue
                masked = norm(question(row), True) if row.get("dataset") == "mbppplus" else None
                if self.exact.get(text) or (masked and self.masked.get(masked)):
                    counts["exact"] += 1
                    notes.append(f"exact  {name} {ident} in {against}: {text[:90]}")
                    continue
                query = shingles(text)
                if len(text.split()) < MIN_WORDS:
                    continue
                shared = collections.Counter(j for piece in query for j in self.index.get(piece, ()))
                score, j = max(((max(c / len(query), c / max(1, len(self.sets[j]))), j) for j, c in shared.items()), default=(0.0, None))
                if score >= FAIL:
                    counts["near"] += 1
                    notes.append(f"near   {name} {ident} ~ {against} {self.rows[j]['id']} ({score:.2f}): {text[:80]}")
                elif score >= REVIEW:
                    counts["review"] += 1
                    notes.append(f"review {name} {ident} ~ {against} {self.rows[j]['id']} ({score:.2f}): {text[:80]}")
            return counts

    print(f"training pool: {len(train)} tasks " + str(dict(collections.Counter(row["dataset"] for row in train))))
    print(f"{'set (against)':34s} {'tasks':>5s} {'exact':>5s} {'near':>5s} {'game':>5s} {'review':>6s}")
    failed, notes = False, []
    train_corpus = Corpus(train)
    held_corpus = Corpus([row for rows in heldout.values() for row in rows])
    jobs = [(name, rows, train_corpus, "train") for name, rows in {**heldout, **tests}.items()]
    jobs += [(name, rows, held_corpus, "heldout") for name, rows in tests.items()]
    for name, rows, corpus, against in jobs:
        counts = corpus.compare(rows, name, against, notes)
        failed |= bool(counts["exact"] or counts["near"] or counts["game"])
        print(f"{name + ' (' + against + ')':34s} {len(rows):5d} {counts['exact']:5d} {counts['near']:5d} {counts['game']:5d} {counts['review']:6d}")
    for note in notes:
        print("  " + note)
    print("DISJOINT" if not failed else "OVERLAP FOUND")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
