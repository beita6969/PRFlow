"""Measured per-cycle cost of a training run: rollouts, Director tokens (prompt / generated), Worker
tokens, rollout seconds, collection wall-clock, Director update seconds, Proposer selection calls and the share of
mixed-outcome (gradient-carrying) task groups.  Prints one JSON object per cycle and a summary line.
usage: cost_profile.py <run_root> <first_cycle> <last_cycle> [<parent_run_root> for cycles before a fork]"""
import json, os, sys
from collections import defaultdict
run, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
parent = sys.argv[4] if len(sys.argv) > 4 else None
state = json.load(open(run + "/training_state.json"))
upd = {x["cycle"]: x for x in state["history"] if x.get("role") == "solver"}
rows = []
for c in range(lo, hi + 1):
    C = "%s/cycle-%04d" % (run, c)
    if not os.path.exists(C + "/solver_batch.json") and parent:
        C = "%s/cycle-%04d" % (parent, c)
    if not os.path.exists(C + "/solver_batch.json"):
        continue
    n = dp = dg = wi = wo = 0; dur = 0.0; calls = 0
    groups = defaultdict(list)
    for line in open(C + "/solver_rollouts.jsonl"):
        r = json.loads(line); md = r.get("metadata") or {}
        n += 1
        for pc in r.get("policy_calls") or []:
            ids = pc.get("token_ids") or []; mask = pc.get("action_mask") or []
            gen = sum(1 for m in mask if m) if mask else 0
            dg += gen; dp += len(ids) - gen; calls += 1
        wi += int(md.get("token_in") or 0); wo += int(md.get("token_out") or 0)
        dur += float(md.get("duration_s") or 0.0)
        groups[r["task_id"]].append(float(r.get("reward") or 0.0))
    mixed = sum(1 for v in groups.values() if len(set(v)) > 1)
    t0 = os.path.getmtime(C + "/proposer_lists.jsonl") if os.path.exists(C + "/proposer_lists.jsonl") else None
    t1 = os.path.getmtime(C + "/solver_batch.json")
    lists = sum(1 for _ in open(C + "/proposer_lists.jsonl")) if os.path.exists(C + "/proposer_lists.jsonl") else 0
    u = upd.get(c) or {}
    rows.append({"cycle": c, "rollouts": n, "groups": len(groups), "mixed_groups": mixed,
                 "director_calls": calls, "director_prompt_tokens": dp, "director_gen_tokens": dg,
                 "worker_in_tokens": wi, "worker_out_tokens": wo, "rollout_seconds": round(dur, 1),
                 "collection_wall_s": round(t1 - t0, 1) if t0 else None, "proposer_lists": lists,
                 "proposer_choice_calls": 2 * lists, "update_seconds": u.get("duration_seconds"),
                 "trained_samples": u.get("samples")})
for r in rows:
    print(json.dumps(r))
k = len(rows)
if k:
    tot = {key: sum((r[key] or 0) for r in rows) for key in rows[0] if key != "cycle"}
    print("SUMMARY cycles %d-%d (%d): per cycle rollouts %.1f, mixed groups %.1f / %.1f, Director tokens %.0f prompt + %.0f gen, "
          "Worker tokens %.0f in + %.0f out, rollout-s %.0f, collection wall %.0f s, update %.0f s" % (
              rows[0]["cycle"], rows[-1]["cycle"], k, tot["rollouts"] / k, tot["mixed_groups"] / k, tot["groups"] / k,
              tot["director_prompt_tokens"] / k, tot["director_gen_tokens"] / k, tot["worker_in_tokens"] / k,
              tot["worker_out_tokens"] / k, tot["rollout_seconds"] / k, tot["collection_wall_s"] / k, tot["update_seconds"] / k))
