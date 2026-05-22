# Alarm Cascade DHP

This directory contains a streaming alarm-cascade grouper built from the
online Dirichlet-Hawkes clustering shape used by the local DHP code, with
alarm-specific stream controls and topology support.

It estimates cascade membership. It does not claim that an assigned edge is a
root-cause direction or a learned propagation graph.

## What is implemented

1. `features.py` converts alarm title, alarm code, severity, source, site, NE
   metadata, resource fields, and topology context into document-style tokens.
2. `config.py` and `model.py` use short-burst and long-tail alarm time kernels,
   active/cooling/closed windows, and online kernel-weight updates.
3. `streaming.py` buffers bounded out-of-order input, compresses duplicate
   raises and clear/reopen flaps, and treats clear alarms as control events.
4. `topology.py` exposes baseline topology tokens such as site/device/hop
   context, so topology is present in DHP content likelihoods.
5. `model.py` also multiplies candidate cluster scores by an explicit
   cluster-local topology affinity over same-device, same-site, hop, domain,
   unknown, and disconnected relations.
6. `time_power` in `AlarmDHPConfig` and the CLI is the PDHP-style powered
   temporal prior for existing cascade intensities.

## Use with match_rules-style stream items

```python
from alarm_cascade_dhp import AlarmCascadeEngine

engine = AlarmCascadeEngine.from_topology_files(
    site_graph_path="topology_resources/site_graph_by_ne.json",
    ne_graph_path="topology_resources/ne_graph.json",
)

for item in valid_alarms:
    for decision in engine.observe_match_rules_item(item):
        print(decision.to_dict())

for decision in engine.flush():
    print(decision.to_dict())

groups = engine.cascade_snapshots()
```

`observe_match_rules_item` accepts the same normalized event shape used by
`fault_grouping.match_rules`:

```python
{
    "alarm": raw_alarm_dict,
    "site_id": "site-a",
    "alarm_source": "ne-a",
    "alarm_title": "link down",
    "ts": 1716200000.0,
}
```

`observe_alarm_record` accepts raw CSV/JSON alarm rows. `observe` accepts
either shape.

## CLI

```bash
python3 -m alarm_cascade_dhp.run_cascades \
  alarms.jsonl \
  output/cascade_decisions.jsonl \
  --topo topology_resources/site_graph_by_ne.json \
  --ne-graph topology_resources/ne_graph.json \
  --visual-output output/cascade_visual_groups.jsonl \
  --time-power 1.4 \
  --assignment map
```

The decisions JSONL records clustered raises, clear controls, and stream-policy
skips. The final group snapshot defaults to
`output/cascade_decisions.groups.json` unless `--visual-output` is set.
Pass `--groups-output` explicitly when both snapshot JSON and visual JSONL are
needed.
When `--visual-output` is set, the CLI also writes match_rules-compatible
group JSONL for `visualization/fault_group_browser.html` and
`visualization/ne_propagation_visualizer.html`. Closed cascades are appended
while the stream runs; cascades still active when the input ends are finalized
with `cascade_info.finalization_reason=stream_end` during the final flush.
`--site-graph` controls the site metadata used by that visualization output.
The visual group includes only devices with clustered cascade alarms by
default. Pass `--visual-ne-scope site-context` when the browser and propagation
view should also include other devices at the cascade sites.

The CLI reads the same raw CSV, JSONL, ZIP, or directory inputs supported by
`alarm_tools.alarm_inputs`. File inputs are loaded and sorted by event time by
default before they are pushed through the same online engine, matching the
offline ordering expectation of `fault_grouping.match_rules`. Pass
`--preserve-input-order` only when the source order is already a real live
stream and bounded disorder should be handled by the reorder buffer through
`--reorder-lag-sec`; very late live records are surfaced as `skipped`
decisions. The default offline loading path shows source-file read progress
while it builds sortable cascade events; `--show-progress` enables the same
source read display for `--preserve-input-order`.
While the stream runs, the CLI prints a processing progress line with read
alarm count, clustered/clear/skipped decisions, current cascade count, and
reorder-buffer depth. It prints a flush message and a final run summary after
the input stream is exhausted.

Add `--profile` to print cumulative timings for the main online path after the
run: input parsing, feature construction, reorder/dedupe handling, model
assignment, existing-cascade scoring, cluster updates, topology calls, and
output writing.

## Main tuning knobs

- Raise `--time-power` when cascades should be more burst-driven; lower it when
  content and topology should dominate.
- Raise `--base-intensity` when the stream should open new cascades more often;
  lower it when early related alarms are being split too aggressively.
- `--max-candidate-cascades` scores the most recently updated active cascades
  first. The default `1024` keeps candidate recall broad during early
  evaluation; reduce it to `64`, `128`, or `256` when online latency matters,
  or use `0` only for unbounded comparison runs.
- Tune `--active-window-sec`, `--cooling-after-sec`, and `--close-after-sec`
  from measured alarm delay distributions.
- Tune `--topology-strength` down if the topology graph has many missing or
  noisy links.
- Keep `--assignment map` for reproducible evaluation. Use particles with the
  default sampled assignment for online uncertainty experiments.
