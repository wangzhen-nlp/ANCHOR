# Alarm Flow ISAHP Notes

This directory contains an exploratory alarm-stream adaptation of ISAHP.
The current data assumption is roughly 350,000 alarm sources and three modeled
alarm types: `link`, `power`, and `offline`.

## Current Blockers

1. The default event type definition is not scalable for 350,000 alarm sources.

   The default `type_fields` is `alarm_source + alarm_type`. With three modeled
   alarm types, the type vocabulary can reach about:

   ```text
   350,000 sources * 3 alarm types = 1,050,000 event types
   ```

   The source embedding itself is not the main problem. The problem is that the
   current implementation also uses this device-level type vocabulary as the
   Hawkes output type space.

2. The per-pair ISAHP heads materialize all output event types.

   `mu_proj`, `alpha_proj`, `gamma_proj`, and `baseline_logits` all scale with
   `n_types`. More importantly, `alpha` and `gamma` are materialized as dense
   tensors shaped roughly like:

   ```text
   [batch_size, history_len, n_types]
   ```

   With the current defaults `batch_size=16` and `max_history_events=128`, one
   FP32 `alpha` tensor at `n_types=1,050,000` is about 8 GiB. `alpha` plus
   `gamma` is already about 16 GiB before gradients, Monte Carlo integration,
   other activations, or optimizer state. Bounded history attention does not
   remove this output-space blowup.

3. Type-level regularization and score export are quadratic in `n_types`.

   `type_regularization()` and `average_type_score_matrix()` allocate or reshape
   type-pair accumulators with size:

   ```text
   n_types * n_types
   ```

   At `n_types=1,050,000`, one FP32 type matrix is about 4.1 TiB. The training
   regularizer currently uses several such tensors. This path is not usable
   with device-level event types.

4. The interval integral still sums dense intensity over all event types.

   The integral is now evaluated in closed form for the neural exponential
   kernel, so the Monte Carlo sample axis is gone. But the loss still evaluates
   total intensity over the full output type vocabulary: with `batch_size=16`,
   `max_history_events=128`, and `n_types=1,050,000`, the `alpha` / `gamma` /
   `decay_integral` tensors are still on the order of 8 GiB each before
   gradients.

## Modeling Risks

1. Device ID is currently both a condition and a predicted type.

   The current input embedding already carries `alarm_source`, and topology pair
   features also depend on source and target devices. Encoding
   `alarm_source + alarm_type` again as the predicted Hawkes type creates a huge
   sparse output classification/intensity space. For this scale, device identity
   is more likely to belong in conditional features or candidate edges than in a
   dense global type head.

2. Device-level types will be sparse and brittle.

   Even if memory were available, many `alarm_source + alarm_type` combinations
   may have few observations. Baselines and type-pair summaries for rare devices
   can be unstable. Training and inference vocab mismatch is also likely when a
   device source is new, renamed, normalized differently, or absent from the
   training slice.

3. Global history competition can hide plausible device parents.

   The model intentionally uses one global alarm flow for device-to-device
   relations. Each target still keeps at most `max_history_events` strictly
   earlier alarms inside `history_window_sec`. In a dense global stream, the
   most recent alarms from unrelated devices can fill the cap and push out other
   alarms that are still within the 15 minute window.

4. Windowed history changes the paper semantics.

   The current implementation builds one bounded target window per alarm rather
   than retaining the paper-style full prior history of a sequence. This removes
   chunk-boundary behavior and makes compute bounded, but it cannot learn effects
   that require history outside the configured window.

5. Equal-timestamp alarms need a policy decision.

   Equal-timestamp alarms remain in the globally ordered flow, but only strictly
   earlier timestamps enter the target history. The query event also prefers the
   latest strictly earlier alarm. This avoids imposing arbitrary same-time causal
   parents, but it removes same-time interaction signals.

6. Topology is only a soft pair feature.

   `ne_graph.json` features are concatenated into pair projections for `alpha`
   and `gamma`. They do not prune candidate histories and they do not provide a
   learned graph encoder. The model can still spend history capacity and output
   capacity on device pairs that topology would make implausible.

7. Alarm title filtering controls data coverage.

   `alarm_type` is derived from `alarm_title`. Only titles mapped to `link`,
   `power`, or `offline` are kept; all other titles are dropped before vocab and
   target-window construction. Mapping coverage therefore directly changes the
   training population.

8. Target windows are eagerly materialized.

   Preprocessing stores every target window with history ids, event objects, and
   topology pair features. This scales with roughly the number of modeled alarms
   times the kept history length. A large global alarm stream may need lazy
   dataset construction or an indexed/on-the-fly collator even after the output
   type issue is fixed.

## Likely Next Direction

The first redesign should remove device identity from the dense Hawkes output
type space. A candidate direction is:

1. Keep the predicted type space small, for example alarm type only.
2. Keep `alarm_source` as an input embedding and device-pair condition.
3. Use topology and bounded-history candidates to score instance edges without
   allocating dense outputs or dense type-pair matrices over every device type.

## Verification Gap

The implementation environment used so far did not have `torch` installed.
Compilation, CLI, preprocessing, and target-window smoke tests were run, but a
real model forward pass, training step, artifact round-trip, and edge export run
still need verification in the training environment.
