"""Gradient-level evidence per Director update, from each cycle's solver_batch.json:
rollouts that carry a gradient (|advantage| >= the low-signal threshold; GRPO groups with identical outcomes give zero
advantage and are dropped), their share of the batch, action tokens that receive gradient, total |advantage| (the
gradient weight), plus the update's grad norm and KL from training_state.json.
usage: gradient_metrics.py <run_root> <out.csv>"""
import csv, glob, json, os, sys

RUN, OUT = sys.argv[1:3]
st = json.load(open(RUN + "/training_state.json"))
upd = {x.get("cycle"): x for x in st["history"] if x.get("role") == "solver"}
rows = []
for d in sorted(glob.glob(RUN + "/cycle-*")):
    c = int(d.rsplit("-", 1)[1]); p = d + "/solver_batch.json"
    if not os.path.exists(p):
        continue
    b = json.load(open(p)); m = b.get("metadata") or {}
    ls = m.get("r2orch_low_signal") or {}; wh = m.get("r2orch_whiten") or {}
    smp = b.get("samples") or []
    kept, dropped = ls.get("kept"), ls.get("dropped")
    abs_adv = [abs(s.get("advantage") or 0.0) for s in smp]
    u = upd.get(c) or {}
    rows.append({"cycle": c, "rollouts_in_batch": (kept or 0) + (dropped or 0) if kept is not None else None,
                 "rollouts_with_gradient": kept, "rollouts_dropped_no_signal": dropped,
                 "gradient_share": round(kept / (kept + dropped), 4) if kept is not None and (kept + dropped) else None,
                 "action_tokens_with_gradient": wh.get("tokens"), "advantage_std_before_whiten": wh.get("std"),
                 "sum_abs_advantage": round(sum(abs_adv), 3), "mean_abs_advantage": round(sum(abs_adv) / len(abs_adv), 4) if abs_adv else None,
                 "grad_norm": u.get("grad_norm"), "kl": u.get("kl"), "update_samples": u.get("samples"),
                 "masked_tokens": u.get("masked_tokens")})
with open(OUT, "w") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("wrote", OUT, len(rows), "cycles")
