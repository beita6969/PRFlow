"""Cost and behaviour records of a training run, kept wide so metrics can be chosen later without re-running
Writes into <out_dir>:
  rollouts.jsonl  one record per rollout: ids, arm/structure, outcome, Director calls/tokens/rejections/actions,
                  Worker tokens/requests/latency, environment steps, graph shape, turns, timing, training use;
  cycles.jsonl    per cycle: totals and per-dataset totals of every additive field, group outcome counts, collection
                  wall-clock, Director/Proposer update seconds, trained samples and tokens, Proposer selection calls
                  and prompt tokens, per-cycle GPU-seconds by role (from --gpus role=count).
usage: cost_records.py <run_root> <first_cycle> <last_cycle> <out_dir> [--parent <run_root>] [--gpus rollout=1,worker=3,trainer=1]
"""
import argparse, json, os
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("lo", type=int); ap.add_argument("hi", type=int); ap.add_argument("out")
ap.add_argument("--parent", default=None)
ap.add_argument("--gpus", default="rollout=1,worker=3,trainer=1")
args = ap.parse_args()
gpus = dict((k, float(v)) for k, v in (item.split("=") for item in args.gpus.split(",")))
os.makedirs(args.out, exist_ok=True)
state = json.load(open(args.run + "/training_state.json"))
hist = defaultdict(dict)
for x in state["history"]:
    hist[x.get("cycle")][x.get("role")] = x


def events(md, key):
    ev = md.get(key) or []
    return (len(ev), sum(float(e.get("upstream_elapsed_s") or 0) for e in ev),
            sum(float(e.get("queue_wait_s") or 0) for e in ev), Counter(str(e.get("route")) for e in ev))


def rollout_record(c, r, trained):
    md = r.get("metadata") or {}; pv = md.get("r2orch") or {}
    calls = r.get("policy_calls") or []
    rej, acts = Counter(), Counter()
    dp = dg = think_req = think_eff = 0
    for pc in calls:
        m = pc.get("metadata") or {}
        ids = pc.get("token_ids") or []; mask = pc.get("action_mask") or []
        gen = sum(1 for x in mask if x); dg += gen; dp += len(ids) - gen
        acts[str(m.get("action_type"))] += 1
        if m.get("accepted") is False:
            rej[str(m.get("rejection_code"))] += 1
        think_req += bool(m.get("thinking_requested")); think_eff += bool(m.get("thinking_effective"))
    b_n, b_up, b_q, b_routes = events(md, "backend_request_events")
    d_n, d_up, d_q, _ = events(md, "director_request_events")
    e_n, e_up, _, _ = events(md, "environment_request_events")
    j_n, j_up, _, _ = events(md, "judge_request_events")
    g = r.get("graph") or {}
    nodes = g.get("nodes") if isinstance(g, dict) else None
    nodes = list(nodes.values()) if isinstance(nodes, dict) else (nodes or [])
    rel = g.get("relations") if isinstance(g, dict) else None
    od = md.get("outcome_decision") or {}
    t = trained.get(r.get("rollout_id"))
    return {
        "cycle": c, "rollout_id": r.get("rollout_id"), "task_id": r.get("task_id"), "dataset": md.get("dataset"),
        "sibling_index": pv.get("sibling_index"), "skill_arm": pv.get("skill_arm"), "arm_card": pv.get("arm_card"),
        "scaffold_condition": pv.get("scaffold_condition"),
        "list_id": pv.get("list_id"), "choice_index": pv.get("choice_index"),
        "reward": r.get("reward"), "director_reward": md.get("director_reward"),
        "base_director_reward": md.get("base_director_reward"), "evaluation_score": md.get("evaluation_score"),
        "answer_score": md.get("answer_score"), "finished": md.get("finished"), "terminal_status": md.get("terminal_status"),
        "failure_mode": md.get("failure_mode"), "outcome_status": od.get("status"), "outcome_reason": od.get("reason"),
        "director_calls": len(calls), "director_rejected": sum(rej.values()), "rejection_codes": dict(rej),
        "action_types": dict(acts), "director_prompt_tokens": dp, "director_gen_tokens": dg,
        "thinking_requested_calls": think_req, "thinking_effective_calls": think_eff,
        "director_requests": d_n, "director_upstream_s": round(d_up, 3), "director_queue_s": round(d_q, 3),
        "worker_in_tokens": int(md.get("token_in") or 0), "worker_out_tokens": int(md.get("token_out") or 0),
        "worker_cost": pv.get("worker_cost", md.get("worker_cost")), "worker_requests": b_n,
        "worker_upstream_s": round(b_up, 3), "worker_queue_s": round(b_q, 3), "worker_routes": dict(b_routes),
        "environment_steps": e_n, "environment_s": round(e_up, 3), "judge_requests": j_n,
        "agents": len(nodes), "relations": len(rel) if isinstance(rel, (list, dict)) else 0,
        "layers": 1 + max([int(n.get("layer") or 0) for n in nodes if isinstance(n, dict)] or [0]),
        "accepted_turns": md.get("accepted_turns"), "interactive_turns": md.get("interactive_turns"),
        "final_execution_count": md.get("final_execution_count"),
        "incremental_execution_count": md.get("incremental_execution_count"),
        "prompt_revision_count": md.get("prompt_revision_count"),
        "prompt_revision_worker_calls": md.get("prompt_revision_worker_model_call_count"),
        "duration_s": md.get("duration_s"), "postsolve_deadline_exceeded": md.get("postsolve_deadline_exceeded"),
        "trained": t is not None, "advantage": (t or {}).get("advantage"), "trained_tokens": (t or {}).get("tokens", 0),
    }


ADD = ["director_calls", "director_rejected", "director_prompt_tokens", "director_gen_tokens", "director_requests",
       "director_upstream_s", "director_queue_s", "worker_in_tokens", "worker_out_tokens", "worker_requests",
       "worker_upstream_s", "worker_queue_s", "environment_steps", "environment_s", "judge_requests", "agents",
       "relations", "duration_s", "trained_tokens", "thinking_requested_calls", "thinking_effective_calls"]
fr = open(os.path.join(args.out, "rollouts.jsonl"), "w"); fc = open(os.path.join(args.out, "cycles.jsonl"), "w")
for c in range(args.lo, args.hi + 1):
    C = "%s/cycle-%04d" % (args.run, c)
    if not os.path.exists(C + "/solver_batch.json") and args.parent:
        C = "%s/cycle-%04d" % (args.parent, c)
    if not os.path.exists(C + "/solver_batch.json"):
        continue
    trained = {}
    for s in json.load(open(C + "/solver_batch.json")).get("samples") or []:
        trained[s.get("rollout_id")] = {"advantage": s.get("advantage"),
                                        "tokens": sum(1 for x in (s.get("action_mask") or []) if x)}
    recs = [rollout_record(c, json.loads(l), trained) for l in open(C + "/solver_rollouts.jsonl")]
    for x in recs:
        fr.write(json.dumps(x) + "\n")
    def total(sel):
        out = {k: round(sum(float(x.get(k) or 0) for x in sel), 3) for k in ADD}
        out["rollouts"] = len(sel); out["successes"] = sum(1 for x in sel if (x.get("reward") or 0) > 0)
        out["trained_rollouts"] = sum(1 for x in sel if x["trained"])
        out["rejection_codes"] = dict(sum((Counter(x["rejection_codes"]) for x in sel), Counter()))
        groups = defaultdict(list)
        for x in sel:
            groups[x["task_id"]].append(float(x.get("reward") or 0))
        out["groups"] = len(groups)
        out["groups_mixed"] = sum(1 for v in groups.values() if len(set(v)) > 1)
        out["groups_all_success"] = sum(1 for v in groups.values() if len(set(v)) == 1 and v[0] > 0)
        out["groups_all_fail"] = sum(1 for v in groups.values() if len(set(v)) == 1 and v[0] <= 0)
        return out
    row = {"cycle": c, "total": total(recs),
           "by_dataset": {ds: total([x for x in recs if x["dataset"] == ds]) for ds in sorted({x["dataset"] for x in recs})},
           "by_arm": {arm: total([x for x in recs if x["skill_arm"] == arm]) for arm in sorted({str(x["skill_arm"]) for x in recs})}}
    lp = C + "/proposer_lists.jsonl"
    t0 = os.path.getmtime(lp) if os.path.exists(lp) else None
    t1 = os.path.getmtime(C + "/solver_batch.json")
    sels = [json.loads(l) for l in open(lp)] if os.path.exists(lp) else []
    row["proposer_lists"] = len(sels)
    row["proposer_choice_calls"] = sum(len([ch for ch in (s.get("choices") or [])]) for s in sels)
    row["proposer_prompt_tokens"] = sum(len(s.get("prompt_token_ids") or []) * len(s.get("choices") or []) for s in sels)
    row["collection_wall_s"] = round(t1 - t0, 1) if t0 else None
    h = hist.get(c, {})
    row["director_update_s"] = (h.get("solver") or {}).get("duration_seconds")
    row["director_trained_samples"] = (h.get("solver") or {}).get("samples")
    row["director_masked_tokens"] = (h.get("solver") or {}).get("masked_tokens")
    row["proposer_update_s"] = (h.get("proposer") or {}).get("duration_seconds")
    row["proposer_trained_samples"] = (h.get("proposer") or {}).get("samples")
    if row["collection_wall_s"]:
        row["gpu_s_collection"] = round(row["collection_wall_s"] * (gpus.get("rollout", 0) + gpus.get("worker", 0)), 1)
    row["gpu_s_update"] = round(((row["director_update_s"] or 0) + (row["proposer_update_s"] or 0)) * gpus.get("trainer", 0), 1)
    fc.write(json.dumps(row) + "\n")
fr.close(); fc.close()
print("wrote", args.out)
