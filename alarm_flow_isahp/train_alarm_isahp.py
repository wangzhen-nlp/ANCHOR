#!/usr/bin/env python3
"""Train an ISAHP-style model from the ordered alarm stream used by match_rules."""

import copy
import json
import os
from argparse import ArgumentParser

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from alarm_flow_isahp.alarm_io import load_ordered_alarm_events
from alarm_flow_isahp.model import (
    AlarmFlowISAHP,
    AlarmISAHPConfig,
    AlarmTargetWindowDataset,
    average_type_score_matrix,
    collate_alarm_target_windows,
    move_batch_to_device,
    require_torch,
    save_alarm_isahp_artifact,
)
from alarm_flow_isahp.ne_topology import NETopologyIndex, PAIR_FEATURE_NAMES
from alarm_flow_isahp.sequences import (
    AlarmSequenceConfig,
    build_alarm_sequences,
    build_alarm_vocabs,
    parse_type_fields,
)
from topology_resources import NE_GRAPH_JSON, SITE_GRAPH_BY_NE_JSON, resource_display


def _derive_type_score_path(model_output):
    stem, _ext = os.path.splitext(model_output)
    return stem + ".type_scores.json"


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _build_sequence_config(args):
    return AlarmSequenceConfig(
        type_fields=parse_type_fields(args.type_fields),
        history_window_sec=args.history_window_sec,
        max_history_events=args.max_history_events,
        min_events=args.min_events,
        time_scale_sec=args.time_scale_sec,
        include_clear=args.include_clear,
    )


def _split_target_windows(sequences, valid_fraction):
    sequences = list(sequences)
    if valid_fraction <= 0 or not sequences:
        return sequences, []

    # The model now uses one global alarm flow. Keep validation at the tail so
    # training windows do not depend on future validation targets.
    windows = list(sequences[0].target_windows)
    if len(windows) < 2:
        return sequences, []
    valid_count = max(1, round(len(windows) * valid_fraction))
    valid_count = min(valid_count, len(windows) - 1)
    train_sequence = copy.copy(sequences[0])
    valid_sequence = copy.copy(sequences[0])
    train_sequence.target_windows = windows[:-valid_count]
    valid_sequence.target_windows = windows[-valid_count:]
    return [train_sequence], [valid_sequence]


def _target_window_count(sequences):
    return sum(len(sequence.target_windows) for sequence in sequences)


def _make_loader(sequences, *, batch_size, shuffle, torch):
    return torch.utils.data.DataLoader(
        AlarmTargetWindowDataset(sequences),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_alarm_target_windows,
    )


def _prediction_count(batch):
    return int(batch["target_type_ids"].numel())


def _run_epoch(
    model,
    dataloader,
    device,
    *,
    optimizer=None,
    l1_reg=0.0,
    variance_reg=0.0,
    num_mc_samples=20,
    torch,
):
    is_training = optimizer is not None
    model.train(is_training)
    total_predictions = 0
    totals = {
        "loss": 0.0,
        "nll": 0.0,
        "integral": 0.0,
        "negative_log_term": 0.0,
        "l1_reg": 0.0,
        "variance_reg": 0.0,
    }
    grad_context = torch.enable_grad() if is_training else torch.no_grad()
    with grad_context:
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            nll, nll_metrics = model.negative_log_likelihood(
                batch["target_type_ids"],
                batch["target_times"],
                batch["interval_dts"],
                batch["query_dts"],
                batch["query_alarm_source_ids"],
                batch["query_alarm_type_ids"],
                batch["history_times"],
                batch["history_dts"],
                batch["history_alarm_source_ids"],
                batch["history_alarm_type_ids"],
                batch["history_mask"],
                topology_pair_features=batch["topology_pair_features"],
                num_mc_samples=num_mc_samples,
            )
            _intensities, _mu, alpha, _gamma, pair_mask = model.intensity_at_events(
                batch["target_times"],
                batch["query_dts"],
                batch["query_alarm_source_ids"],
                batch["query_alarm_type_ids"],
                batch["history_times"],
                batch["history_dts"],
                batch["history_alarm_source_ids"],
                batch["history_alarm_type_ids"],
                batch["history_mask"],
                topology_pair_features=batch["topology_pair_features"],
            )
            l1_mean, variance = model.type_regularization(
                alpha,
                batch["target_type_ids"],
                batch["history_type_ids"],
                pair_mask,
            )
            loss = nll + l1_reg * l1_mean + variance_reg * variance
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            count = _prediction_count(batch)
            total_predictions += count
            totals["loss"] += float(loss.detach()) * count
            totals["nll"] += float(nll_metrics["nll"]) * count
            totals["integral"] += float(nll_metrics["integral"]) * count
            totals["negative_log_term"] += float(nll_metrics["negative_log_term"]) * count
            totals["l1_reg"] += float(l1_mean.detach()) * count
            totals["variance_reg"] += float(variance.detach()) * count

    denominator = max(1, total_predictions)
    return {
        **{name: value / denominator for name, value in totals.items()},
        "prediction_count": total_predictions,
    }


def _type_score_payload(labels, scores, counts):
    return {
        "matrix_axes": {
            "rows": "target_alarm_type",
            "columns": "historical_source_alarm_type",
        },
        "labels": list(labels),
        "scores": scores.tolist(),
        "counts": counts.tolist(),
    }


def main():
    parser = ArgumentParser(description="Train an alarm-flow ISAHP model from ordered alarms.")
    parser.add_argument("alarms", help="Raw alarms or prepare_sorted_alarms cache consumed by match_rules.")
    parser.add_argument("-o", "--output", required=True, help="Output PyTorch model artifact (.pt).")
    parser.add_argument(
        "--type-score-output",
        default="",
        help="Type-level causal score JSON. Default: <output>.type_scores.json.",
    )
    parser.add_argument(
        "--topo",
        default=SITE_GRAPH_BY_NE_JSON,
        help=f"Site topology for raw alarm inputs. Default: {resource_display('site_graph_by_ne.json')}.",
    )
    parser.add_argument(
        "--ne-graph",
        default=NE_GRAPH_JSON,
        help=f"NE graph for raw alarm inputs. Default: {resource_display('ne_graph.json')}.",
    )
    parser.add_argument("--start-time", default="", help="Raw-input first occurrence lower bound.")
    parser.add_argument("--end-time", default="", help="Raw-input first occurrence upper bound.")
    parser.add_argument("--clear-delay-sec", type=float, default=0.0, help="Raw-input clear delay.")
    parser.add_argument(
        "--topology-max-hops",
        type=int,
        default=2,
        help="Maximum NE graph hops encoded as soft pair features. Default: 2.",
    )
    parser.add_argument(
        "--type-fields",
        default="alarm_source,alarm_type",
        help=(
            "Comma-separated alarm event fields forming the event type. "
            "Supported: alarm_source,alarm_type,alarm_title,site_id. "
            "alarm_type is derived as link/power/offline from alarm_title."
        ),
    )
    parser.add_argument(
        "--history-window-sec",
        type=float,
        default=900.0,
        help="Only strictly earlier alarms within this time window can influence a target alarm. Default: 900.",
    )
    parser.add_argument(
        "--max-history-events",
        type=int,
        default=128,
        help="Keep at most this many strictly earlier alarms per target history window. Default: 128.",
    )
    parser.add_argument("--min-events", type=int, default=2, help="Minimum global flow event count.")
    parser.add_argument("--time-scale-sec", type=float, default=60.0, help="Divide alarm timestamps by this scale.")
    parser.add_argument("--include-clear", action="store_true", help="Include synthetic clear events in the model stream.")
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=32,
        help="Attention feature size; time plus alarm source/type embedding dims must fit this size.",
    )
    parser.add_argument(
        "--alarm-type-embedding-dim",
        type=int,
        default=4,
        help="Embedding dim for derived link/power/offline alarm_type. Alarm source uses the remaining feature dim.",
    )
    parser.add_argument("--num-heads", type=int, default=4, help="Even attention head count.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Attention dropout.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Target-window batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--l1-reg", type=float, default=0.025, help="Type-level mean alpha sparsity weight.")
    parser.add_argument("--variance-reg", type=float, default=0.25, help="Within type-pair alpha variance weight.")
    parser.add_argument("--num-mc-samples", type=int, default=20, help="Monte Carlo samples for interval integral.")
    parser.add_argument(
        "--valid-fraction",
        type=float,
        default=0.1,
        help="Holdout fraction from the tail of global target windows.",
    )
    parser.add_argument("--patience", type=int, default=6, help="Stop after this many non-improving validation epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default="auto", help="PyTorch device, for example auto, cpu, cuda.")
    args = parser.parse_args()

    require_torch()
    import torch

    if not 0 <= args.valid_fraction < 1:
        parser.error("--valid-fraction must be in [0, 1)")
    if args.history_window_sec <= 0:
        parser.error("--history-window-sec must be positive")
    if args.max_history_events < 1:
        parser.error("--max-history-events must be >= 1")
    torch.manual_seed(args.seed)

    sequence_config = _build_sequence_config(args)
    alarm_events, alarm_metadata = load_ordered_alarm_events(
        args.alarms,
        topo_path=args.topo,
        ne_graph_path=args.ne_graph,
        start_time=args.start_time or None,
        end_time=args.end_time or None,
        clear_delay_sec=args.clear_delay_sec,
    )
    vocabs, considered_event_count = build_alarm_vocabs(alarm_events, sequence_config)
    topology_index = NETopologyIndex.from_file(args.ne_graph, max_hops=args.topology_max_hops)
    sequences, sequence_stats = build_alarm_sequences(
        alarm_events,
        vocabs,
        sequence_config,
        topology_index=topology_index,
    )
    if not sequences:
        raise ValueError("no global alarm flow survived preprocessing; relax min-events or inspect input alarms")

    train_sequences, valid_sequences = _split_target_windows(sequences, args.valid_fraction)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    alarm_source_embedding_dim = args.hidden_size - args.alarm_type_embedding_dim - 1
    if alarm_source_embedding_dim < 1:
        parser.error("--hidden-size must leave at least one feature for alarm_source embedding")
    model = AlarmFlowISAHP(
        AlarmISAHPConfig(
            n_types=len(vocabs.type_vocab),
            n_alarm_sources=len(vocabs.alarm_source_vocab),
            n_alarm_types=len(vocabs.alarm_type_vocab),
            alarm_source_embedding_dim=alarm_source_embedding_dim,
            alarm_type_embedding_dim=args.alarm_type_embedding_dim,
            topology_pair_feature_dim=topology_index.feature_dim,
            history_window_sec=sequence_config.history_window_sec,
            time_scale_sec=sequence_config.time_scale_sec,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            dropout=args.dropout,
        )
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train_loader = _make_loader(train_sequences, batch_size=args.batch_size, shuffle=True, torch=torch)
    valid_loader = (
        _make_loader(valid_sequences, batch_size=args.batch_size, shuffle=False, torch=torch)
        if valid_sequences else None
    )

    print(
        f"modeled alarm types={len(vocabs.type_vocab)}, "
        f"alarm sources={len(vocabs.alarm_source_vocab)}, "
        f"derived alarm types={len(vocabs.alarm_type_vocab)}, "
        f"input events={considered_event_count}, "
        f"target windows={_target_window_count(sequences)}, "
        f"train windows={_target_window_count(train_sequences)}, "
        f"valid windows={_target_window_count(valid_sequences)}, device={device}"
    )
    history = []
    best_metric = float("inf")
    best_epoch = -1
    best_state = None
    stale_epochs = 0
    for epoch in range(args.epochs):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            l1_reg=args.l1_reg,
            variance_reg=args.variance_reg,
            num_mc_samples=args.num_mc_samples,
            torch=torch,
        )
        valid_metrics = (
            _run_epoch(
                model,
                valid_loader,
                device,
                l1_reg=args.l1_reg,
                variance_reg=args.variance_reg,
                num_mc_samples=args.num_mc_samples,
                torch=torch,
            )
            if valid_loader else None
        )
        metric = (valid_metrics or train_metrics)["nll"]
        history.append({"epoch": epoch, "train": train_metrics, "valid": valid_metrics})
        valid_text = f", valid_nll={valid_metrics['nll']:.6f}" if valid_metrics else ""
        print(f"epoch={epoch}, train_nll={train_metrics['nll']:.6f}, train_loss={train_metrics['loss']:.6f}{valid_text}")
        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if valid_loader and args.patience > 0 and stale_epochs >= args.patience:
                print(f"early stopping at epoch={epoch}; best_epoch={best_epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)
    all_loader = _make_loader(sequences, batch_size=args.batch_size, shuffle=False, torch=torch)
    type_scores, type_counts = average_type_score_matrix(model, all_loader, device)
    training_payload = {
        "input": os.path.abspath(args.alarms),
        "alarm_metadata": alarm_metadata,
        "considered_event_count": considered_event_count,
        "sequence_stats": sequence_stats,
        "topology_pair_feature_names": list(PAIR_FEATURE_NAMES),
        "train_target_window_count": _target_window_count(train_sequences),
        "valid_target_window_count": _target_window_count(valid_sequences),
        "best_epoch": best_epoch,
        "best_nll": best_metric,
        "history": history,
        "config": vars(args),
    }
    save_alarm_isahp_artifact(args.output, model, vocabs, sequence_config, training_payload)
    type_score_output = args.type_score_output or _derive_type_score_path(args.output)
    _write_json(type_score_output, _type_score_payload(vocabs.type_vocab.labels, type_scores, type_counts))
    print(f"model artifact written to: {args.output}")
    print(f"type causal score matrix written to: {type_score_output}")


if __name__ == "__main__":
    main()
