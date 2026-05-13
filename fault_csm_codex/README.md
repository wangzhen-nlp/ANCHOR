# fault_csm_codex

`fault_csm_codex` is an independent Continuous Subgraph Matching fault grouping path.
It keeps the existing alarm input, rule config, and JSONL group output shape,
but does not call the `match_rules.py` temporal engine.

Run:

```bash
python -m fault_csm_codex <alarms> <output.jsonl> --algorithm graphflow --compact-output
```

Use `--alarm-active-sec N` to add an automatic clear update for every inserted
alarm edge at `occurrence_ts + N`. The alarm is no longer considered after that
time even if the real clear event has not arrived. If another active alarm with
the same type still exists on the site, the site-level alarm predicate remains
satisfiable.

Main design:

- Site vertices and topology reachability edges are the static part of the data
  graph.
- Alarm arrivals insert dynamic alarm vertices plus
  `site --HAS_ALARM--> alarm` dynamic edges; clear events delete them.
- Each rule is compiled as a query graph of role vertices, implicit alarm
  predicate vertices, `HAS_ALARM` query edges, and directed topology edges.
- Role/site structural compatibility is indexed once and reused by incremental
  matching; dynamic alarm predicates are evaluated at match time so context
  roles without alarms keep the same semantics as `fault_grouping`.
- Matching starts from the real updated `HAS_ALARM` data edge. The backtracking
  phase extends through original topology edges that match the rule
  direction/hop labels, with role/site indexes and algorithm-specific support
  checks pruning candidates.
- Clear events, TTL pruning, and `--alarm-active-sec` expiry remove dynamic
  alarm edges and invalidate any retained output groups that depended on the
  removed alarm IDs.
- `--algorithm incisomatch` uses the lightest incremental DFS order after the
  updated edge is bound.
- `--algorithm sjtree` uses a selective edge-decomposition order before joining
  through the common backtracker.
- `--algorithm graphflow` uses updated-edge driven matching orders similar to
  Graphflow.
- `--algorithm iedyn` uses DAG/DCS-style candidate support pruning with a
  selective incremental order.
- `--algorithm turboflux` uses a selective query DAG order plus DCS
  candidate-support ordering.
- `--algorithm symbi` uses a SymBi-style DAG order and bidirectional
  candidate-support checks.
- Produced matches are merged by overlapping alarm IDs and written with the
  same group-output builder used by the existing visualization/evaluation path.

This version intentionally uses the ContinuousSubgraphMatching execution shape
instead of the `match_rules.py` temporal-engine pending-role expansion path.
The Python implementation keeps the same rule semantics as `fault_grouping`
while using CSM-style add/remove-triggered incremental matching.
