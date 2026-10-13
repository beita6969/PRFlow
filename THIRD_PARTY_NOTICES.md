# Third-Party Notices

R²Orch adapts code from the projects below. Their license texts are kept unchanged in `third_party/notices/`.

| Component | Upstream and license | Adapted in `src/r2orch/` |
| --- | --- | --- |
| Policy-aware training scaffolding: grouped rollout evidence, bounded skill edits | PATS, [shi-yipeng/PATS](https://github.com/shi-yipeng/PATS) revision `bad468b5c73081c2f5aa74c4e0011c6fb2872dbf`; Apache License 2.0 (`PATS-LICENSE`) | `pats.py`, `pats_evidence.py`, `pats_refiner.py`, `pats_semantics.py` |
| Mini-cluster curriculum state, boundary movement, clustering and difficulty preprocessing | ADS (Adaptive Data Scheduling), Copyright 2026 The ADS Authors; Apache License 2.0 (`ADS-LICENSE`, `ADS-NOTICE`) | `curriculum.py`, `ads_preprocessing.py` |
| KNN-KDE probability assignment (exact NumPy nearest neighbours) | TSDS, Copyright (c) 2024 Zifan Liu; MIT License (`TSDS-LICENSE`) | `curriculum.py` |
| Structured agent artifacts and bounded relay packets | MANTA, Copyright (c) 2026 The MANTA Authors; MIT License (`MANTA-LICENSE`) | `contracts.py` |

The adaptations change the upstream code to fit this project (Director-only scaffold, rollout-group evidence,
JSONL task pools); they do not vendor the upstream training stacks.
