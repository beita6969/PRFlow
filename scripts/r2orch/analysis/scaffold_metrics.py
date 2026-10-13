"""Outer scaffold loop per cycle: the share of tasks whose scope still has a card to show, the share of siblings
shown a card (visible arm), ledger card counts by status (trial / admitted / retired, cumulative retirements and
reinstatements, version of each live card), the editor steps of the PATS review (REVISE / DISTILL: scopes updated,
unchanged, rejected or skipped) and the editor / checker calls.
usage: scaffold_metrics.py <run_root> <out.csv>"""
import csv, glob, json, os, sys
from collections import Counter

RUN, OUT = sys.argv[1:3]
rows = []
for d in sorted(glob.glob(RUN + "/cycle-*")):
    c = int(d.rsplit("-", 1)[1])
    sa = d + "/scaffold_assignments.json"
    if not os.path.exists(sa) or not os.path.exists(d + "/solver_batch.json"):
        continue
    a = json.load(open(sa))
    tasks = list(a.values())
    arms = Counter(x for v in tasks for x in (v.get("skill_arms") or []))
    with_card = sum(1 for v in tasks if (v.get("arm_candidate_cards") or 0) > 0)
    row = {"cycle": c, "tasks": len(tasks)}
    row["tasks_with_card_to_show"] = round(with_card / len(tasks), 4) if tasks else None
    nsib = sum(arms.values())
    row["siblings_visible"] = round(arms.get("visible", 0) / nsib, 4) if nsib else None
    row["siblings_hidden"] = round(arms.get("hidden", 0) / nsib, 4) if nsib else None
    ls = d + "/skill_ledger_summary.json"
    if os.path.exists(ls):
        s = json.load(open(ls)); cards = s.get("cards") or {}
        st = Counter(v.get("status") for v in cards.values())
        row.update({"cards_trial": st.get("trial", 0), "cards_admitted": st.get("admitted", 0),
                    "cards_replaced": st.get("replaced", 0), "cards_retired_total": len(s.get("retired") or {}),
                    "reinstated_total": len(s.get("reinstated") or []),
                    "audits_this_cycle": s.get("audits_this_cycle"),
                    "mean_live_card_version": round(sum(v.get("version") or 0 for v in cards.values()) / len(cards), 2) if cards else None})
    pr = d + "/pats_review.json"
    if os.path.exists(pr):
        p = json.load(open(pr))
        outcome = lambda r: r["status"] if r.get("status") in ("updated", "unchanged", "rejected") else "skipped"
        steps = Counter((r.get("step"), outcome(r)) for r in p.get("reviews") or [])
        for name in ("REVISE", "DISTILL"):
            for status in ("updated", "unchanged", "rejected", "skipped"):
                row[name.lower() + "_" + status] = steps.get((name, status), 0)
        row.update({"pats_refiner_calls": p.get("refiner_calls"), "pats_checker_calls": p.get("checker_calls"),
                    "pats_credible_groups": p.get("credible_groups")})
    rows.append(row)
keys = []
for r in rows:
    keys += [k for k in r if k not in keys]
with open(OUT, "w") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
print("wrote", OUT, len(rows), "cycles")
