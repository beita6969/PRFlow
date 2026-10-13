"""Mechanism metrics of one run, per cycle, written as CSV (one row per cycle) plus a JSON of
calibration bins.  From curriculum_observations (GRPO groups = the 5 siblings of a pick):
  groups, histogram of successes per group (0..5 of 5), mixed share, share of rollouts in mixed groups (non-zero
  advantage), mean Bernoulli variance p(1-p) of the groups.
From solver_rollouts:
  distinct graph structures per group, share of groups whose siblings differ in structure AND in outcome (a structure
  contrast GRPO can learn from), Director action rejections per Director call, typed policy failures per rollout,
  success of the visible (card shown) and hidden (card withheld) arms.
From proposer_lists + observations: calibration of previous_success (shown to the Proposer) against the realised
success of the new group, in bins.
usage: mechanism_metrics.py <run_root> <out.csv> <calibration.json>"""
import csv, glob, json, os, sys
from collections import defaultdict

RUN, OUT, CAL = sys.argv[1:4]


def success(r):
    md = r["metadata"]
    s = md.get("answer_score")
    if s is None:
        s = md.get("evaluation_score")
    if s is None:
        return 1 if (r.get("reward") or 0) > 0.5 else 0
    return 1 if float(s) > 0.5 else 0


def bin_of(p):
    if p is None:
        return "unseen"
    if p <= 0:
        return "p=0"
    if p >= 1:
        return "p=1"
    return "0<p<=0.4" if p <= 0.4 else ("0.4<p<0.6" if p < 0.6 else "0.6<=p<1")


rows, cal = [], defaultdict(lambda: [0, 0.0, 0])  # (window, bin) -> [groups, realised success sum, mixed]
for d in sorted(glob.glob(RUN + "/cycle-*")):
    c = int(d.rsplit("-", 1)[1])
    op, sp, lp = d + "/curriculum_observations.jsonl", d + "/solver_rollouts.jsonl", d + "/proposer_lists.jsonl"
    if not (os.path.exists(op) and os.path.exists(sp) and os.path.exists(d + "/solver_batch.json")):
        continue
    obs = {}
    for line in open(op):
        o = json.loads(line)
        rw = [x for x in (o.get("rewards") or []) if x is not None]
        if rw:
            obs[o["task_id"]] = (sum(1 for x in rw if x > 0), len(rw))
    hist = [0] * 6; mixed = 0; roll_mixed = 0; roll = 0; var = 0.0
    for s, k in obs.values():
        hist[min(5, round(5 * s / k))] += 1
        m = 0 < s < k; mixed += m; roll += k; roll_mixed += k if m else 0; var += (s / k) * (1 - s / k)
    n = len(obs)
    # rollouts: structure per group, rejections, failures, arms
    groups = defaultdict(list); calls = rej = fails = nroll = 0; arm = defaultdict(lambda: [0, 0])
    for line in open(sp):
        r = json.loads(line); md = r["metadata"]; pv = md.get("r2orch") or {}
        nroll += 1
        fails += str(md.get("terminal_status")) == "typed_policy_failure"
        for pc in r.get("policy_calls") or []:
            meta = pc.get("metadata") or {}
            calls += 1; rej += meta.get("accepted") is False
        g = r.get("graph") or {}
        sig = (len(g.get("nodes") or []), len(g.get("relations") or []))
        ok = success(r)
        groups[pv.get("task_id")].append((sig, ok))
        if pv.get("skill_arm") in ("visible", "hidden"):
            a = arm[pv["skill_arm"]]; a[0] += ok; a[1] += 1
    multi = [v for v in groups.values() if len(v) >= 2]
    distinct = sum(len({s for s, _ in v}) for v in multi) / len(multi) if multi else None
    contrast = sum(1 for v in multi if len({s for s, _ in v}) > 1 and 0 < sum(o for _, o in v) < len(v)) / len(multi) if multi else None
    struct_var = sum(1 for v in multi if len({s for s, _ in v}) > 1) / len(multi) if multi else None
    # calibration: previous_success shown to the Proposer vs the realised group
    if os.path.exists(lp):
        for line in open(lp):
            r = json.loads(line)
            cands = {x["label"]: x for x in r.get("candidates") or []}
            for ch in r.get("choices") or []:
                o = obs.get(ch.get("task_id"))
                if not o or o[1] < 2 or ch.get("label") not in cands:
                    continue
                b = cal[("%d-%d" % ((c // 50) * 50, (c // 50) * 50 + 49), bin_of(cands[ch["label"]].get("previous_success")))]
                b[0] += 1; b[1] += o[0] / o[1]; b[2] += 0 < o[0] < o[1]
    rows.append({"cycle": c, "groups": n, **{"groups_%d_of_5" % i: hist[i] for i in range(6)},
                 "mixed_share": round(mixed / n, 4) if n else None,
                 "rollouts_with_signal_share": round(roll_mixed / roll, 4) if roll else None,
                 "mean_bernoulli_var": round(var / n, 4) if n else None,
                 "distinct_structures_per_group": round(distinct, 3) if distinct is not None else None,
                 "groups_structure_varies": round(struct_var, 4) if struct_var is not None else None,
                 "groups_structure_and_outcome_vary": round(contrast, 4) if contrast is not None else None,
                 "director_calls": calls, "director_rejection_rate": round(rej / calls, 4) if calls else None,
                 "policy_failure_rate": round(fails / nroll, 4) if nroll else None,
                 "visible_success": round(arm["visible"][0] / arm["visible"][1], 4) if arm["visible"][1] else None,
                 "hidden_success": round(arm["hidden"][0] / arm["hidden"][1], 4) if arm["hidden"][1] else None,
                 "visible_n": arm["visible"][1], "hidden_n": arm["hidden"][1]})
    if c % 20 == 0:
        print("cycle", c, flush=True)
with open(OUT, "w") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
json.dump({"%s %s" % k: {"groups": v[0], "realised_success": round(v[1] / v[0], 4), "mixed_rate": round(v[2] / v[0], 4)}
           for k, v in sorted(cal.items())}, open(CAL, "w"), indent=1)
print("wrote", OUT, len(rows), "cycles")
