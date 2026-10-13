"""Replay every logged Proposer list through a served model and record its distribution over the candidate labels
The list's own prompt token ids are sent unchanged; the next-token log-probabilities of the
label tokens (top-20, renormalised over the labels) give the model's choice distribution for that list.
With --model r2orch-base this is the same-prompt reference the Proposer trainer uses (LoRA disabled); with a Proposer
adapter it reproduces the logged behaviour log-probabilities (a check of the method).
Output CSV, one row per list: cycle, list id, dataset, per label: previous_success, behaviour log-prob, replayed log-prob.
usage: proposer_reference_replay.py <run_root> <base_url> <model> <out.csv> [--cycles a-b] [--workers 16]"""
import csv, glob, json, math, os, sys, urllib.request
from multiprocessing.pool import ThreadPool

RUN, URL, MODEL, OUT = sys.argv[1:5]
lo, hi = 0, 10 ** 6
if "--cycles" in sys.argv:
    lo, hi = (int(x) for x in sys.argv[sys.argv.index("--cycles") + 1].split("-"))
WORKERS = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 16


def load_lists():
    out = []
    for d in sorted(glob.glob(RUN + "/cycle-*")):
        c = int(d.rsplit("-", 1)[1])
        p = d + "/proposer_lists.jsonl"
        if not (lo <= c <= hi) or not os.path.exists(p):
            continue
        for line in open(p):
            r = json.loads(line)
            if r.get("status") != "ok" or not r.get("prompt_token_ids") or not r.get("choices"):
                continue
            out.append(r)
    return out


def replay(r):
    body = json.dumps({"model": MODEL, "prompt": r["prompt_token_ids"], "max_tokens": 1, "temperature": 0.0,
                       "logprobs": 20}).encode()
    req = urllib.request.Request(URL + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    for attempt in range(3):
        try:
            resp = json.loads(urllib.request.urlopen(req, timeout=300).read().decode())
            break
        except Exception as exc:  # retry transient failures
            err = str(exc)
    else:
        return r, None, err
    top = resp["choices"][0]["logprobs"]["top_logprobs"][0]  # token text -> logprob
    labels = r["labels"]
    lp = {lab: top.get(lab, top.get(" " + lab)) for lab in labels}
    known = {k: v for k, v in lp.items() if v is not None}
    if not known:
        return r, None, "no label in top-20"
    floor = min(known.values()) - 2.0  # a label outside the top 20 is far less likely than the least likely listed one
    vals = {k: (v if v is not None else floor) for k, v in lp.items()}
    z = math.log(sum(math.exp(v) for v in vals.values()))
    return r, {k: v - z for k, v in vals.items()}, sum(1 for v in lp.values() if v is None)


lists = load_lists()
print("lists", len(lists), "model", MODEL, flush=True)
rows, failed = [], 0
with ThreadPool(WORKERS) as pool:
    for i, (r, rep, info) in enumerate(pool.imap(replay, lists)):
        if rep is None:
            failed += 1
            continue
        beh = r["choices"][0].get("behavior_logprobs") or {}
        cands = {c["label"]: c for c in r["candidates"]}
        for lab in r["labels"]:
            rows.append([r["cycle"], r["list_id"], r["dataset"], lab, cands[lab].get("previous_success"),
                         cands[lab].get("source"), beh.get(lab), round(rep[lab], 6), info])
        if (i + 1) % 200 == 0:
            print("  %d / %d lists" % (i + 1, len(lists)), flush=True)
with open(OUT, "w") as f:
    w = csv.writer(f)
    w.writerow(["cycle", "list_id", "dataset", "label", "previous_success", "source", "behavior_logprob",
                "replayed_logprob", "labels_outside_top20"])
    w.writerows(rows)
print("wrote", OUT, "rows", len(rows), "failed lists", failed)
