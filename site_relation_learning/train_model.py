#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from site_relation_learning.core import (
    RELATION_CLASSES,
    build_feature_importance,
    evaluate_dense_samples,
    fit_standardizer,
    infer_feature_names,
    load_dataset_samples,
    train_gbdt_classifier,
    train_mlp_classifier,
    train_softmax_regression,
    vectorize_samples,
    write_json,
)


def _format_label_counts(samples):
    counts = {label: 0 for label in RELATION_CLASSES}
    for sample in samples:
        label = sample.get("label", "none")
        if label in counts:
            counts[label] += 1
    return ", ".join(f"{label}={counts[label]}" for label in RELATION_CLASSES)


def _derive_metrics_path(model_output):
    if model_output.endswith(".json"):
        return model_output[:-5] + ".metrics.json"
    return model_output + ".metrics.json"


MODEL_DEFAULTS = {
    # 不同模型最佳的训练超参经验区间，未在 CLI 显式指定时用这里的值
    "softmax": {"epochs": 30, "learning_rate": 0.03, "early_stop_patience": 5},
    "mlp":     {"epochs": 50, "learning_rate": 0.01, "early_stop_patience": 8},
    "gbdt":    {"epochs": 200, "learning_rate": 0.05, "early_stop_patience": 20},
}


def main():
    parser = ArgumentParser(description="训练站点关系四分类模型")
    parser.add_argument("--train", required=True, help="训练集 JSONL")
    parser.add_argument("--valid", default="", help="验证集 JSONL")
    parser.add_argument("-o", "--output", default="site_relation_model.json", help="模型输出 JSON")
    parser.add_argument("--metrics-output", default="", help="指标输出 JSON")
    parser.add_argument("--model-type", choices=("softmax", "mlp", "gbdt"), default="softmax", help="模型类型，默认: softmax")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数；默认按 model-type 取 softmax=30/mlp=50/gbdt=200")
    parser.add_argument(
        "--mlp-hidden-dim", "--hidden-dim", dest="mlp_hidden_dim", type=int, default=64,
        help="MLP 隐层维度，仅 --model-type mlp 生效，默认: 64",
    )
    parser.add_argument("--gbdt-num-leaves", type=int, default=31, help="GBDT num_leaves，仅 --model-type gbdt 生效，默认: 31")
    parser.add_argument("--gbdt-min-data-in-leaf", type=int, default=20, help="GBDT min_data_in_leaf，仅 --model-type gbdt 生效，默认: 20")
    parser.add_argument(
        "--learning-rate", type=float, default=None,
        help="学习率；默认按 model-type 取 softmax=0.03/mlp=0.01/gbdt=0.05",
    )
    parser.add_argument("--l2", type=float, default=1e-4, help="L2 正则，默认: 1e-4")
    parser.add_argument("--class-weight", choices=("balanced", "none"), default="balanced", help="类别权重，默认: balanced")
    parser.add_argument(
        "--early-stop-patience", type=int, default=None,
        help="early stopping patience；<=0 表示关闭；默认按 model-type 取 softmax=5/mlp=8/gbdt=20",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="mini-batch SGD 批大小，默认: 512")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认: 42")
    parser.add_argument("--no-progress", action="store_true", help="关闭进度条")
    args = parser.parse_args()

    # 按 model-type 应用未显式指定的默认值
    defaults = MODEL_DEFAULTS[args.model_type]
    if args.epochs is None:
        args.epochs = defaults["epochs"]
    if args.learning_rate is None:
        args.learning_rate = defaults["learning_rate"]
    if args.early_stop_patience is None:
        args.early_stop_patience = defaults["early_stop_patience"]

    print(f"加载训练集: {args.train}")
    train_samples = load_dataset_samples(args.train)
    if not train_samples:
        raise ValueError("训练集为空")
    print(f"训练样本数: {len(train_samples)}")
    print(f"训练类别分布: {_format_label_counts(train_samples)}")

    valid_samples = []
    if args.valid:
        print(f"加载验证集: {args.valid}")
        valid_samples = load_dataset_samples(args.valid)
        print(f"验证样本数: {len(valid_samples)}")
        print(f"验证类别分布: {_format_label_counts(valid_samples)}")

    print("推断特征集合...")
    feature_names = infer_feature_names(train_samples)
    print(f"特征数: {len(feature_names)}")

    standardizer = fit_standardizer(
        train_samples,
        feature_names,
        show_progress=not args.no_progress,
    )
    train_dense = vectorize_samples(
        train_samples,
        feature_names,
        standardizer,
        show_progress=not args.no_progress,
        progress_label="向量化训练样本",
    )
    valid_dense = (
        vectorize_samples(
            valid_samples,
            feature_names,
            standardizer,
            show_progress=not args.no_progress,
            progress_label="向量化验证样本",
        )
        if valid_samples else []
    )

    print(f"开始训练模型: {args.model_type}")
    if args.model_type == "gbdt":
        model_state = train_gbdt_classifier(
            train_dense,
            valid_dense_samples=valid_dense,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            class_weight=args.class_weight,
            early_stop_patience=args.early_stop_patience,
            seed=args.seed,
            show_progress=not args.no_progress,
            num_leaves=args.gbdt_num_leaves,
            min_data_in_leaf=args.gbdt_min_data_in_leaf,
        )
        weights = {
            "model_type": "gbdt",
            "booster": model_state["booster"],
            "model_string": model_state["model_string"],
            "best_iteration": model_state["best_iteration"],
        }
        biases = None
    elif args.model_type == "mlp":
        model_state = train_mlp_classifier(
            train_dense,
            valid_dense_samples=valid_dense,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            class_weight=args.class_weight,
            early_stop_patience=args.early_stop_patience,
            seed=args.seed,
            show_progress=not args.no_progress,
            batch_size=args.batch_size,
            hidden_dim=args.mlp_hidden_dim,
        )
        weights = {
            "model_type": "mlp",
            "hidden_weights": model_state["hidden_weights"],
            "hidden_biases": model_state["hidden_biases"],
            "output_weights": model_state["output_weights"],
            "output_biases": model_state["output_biases"],
        }
        biases = None
    else:
        model_state = train_softmax_regression(
            train_dense,
            valid_dense_samples=valid_dense,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            class_weight=args.class_weight,
            early_stop_patience=args.early_stop_patience,
            seed=args.seed,
            show_progress=not args.no_progress,
            batch_size=args.batch_size,
        )
        weights = model_state["weights"]
        biases = model_state["biases"]
    # 直接复用 trainer 内部已经算过的指标，避免再次全量预测
    train_metrics = model_state.get("train_metrics")
    valid_metrics = model_state.get("valid_metrics")
    if train_metrics is None:
        train_metrics, _ = evaluate_dense_samples(train_dense, weights, biases)
    if valid_dense and valid_metrics is None:
        valid_metrics, _ = evaluate_dense_samples(valid_dense, weights, biases)

    model_payload = {
        "model_type": {
            "softmax": "site_relation_softmax_v1",
            "mlp": "site_relation_mlp_v1",
            "gbdt": "site_relation_gbdt_v1",
        }[args.model_type],
        "classes": list(RELATION_CLASSES),
        "train_file": args.train,
        "valid_file": args.valid,
        "feature_names": feature_names,
        "standardizer": standardizer,
        "best_epoch": model_state["best_epoch"],
        "class_weights": {
            RELATION_CLASSES[class_idx]: value
            for class_idx, value in model_state["class_weights"].items()
        },
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "training_config": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
            "class_weight": args.class_weight,
            "model_type": args.model_type,
            "mlp_hidden_dim": args.mlp_hidden_dim if args.model_type == "mlp" else None,
            "gbdt_num_leaves": args.gbdt_num_leaves if args.model_type == "gbdt" else None,
            "gbdt_min_data_in_leaf": args.gbdt_min_data_in_leaf if args.model_type == "gbdt" else None,
            "early_stop_patience": args.early_stop_patience,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "stopped_epoch": model_state.get("stopped_epoch", args.epochs),
        "feature_importance": build_feature_importance(feature_names, weights),
        "training_history": model_state["history"],
    }
    if args.model_type == "gbdt":
        model_payload.update({
            "gbdt_model_string": model_state["model_string"],
            "gbdt_best_iteration": model_state["best_iteration"],
            "gbdt_params": model_state["params"],
        })
    elif args.model_type == "mlp":
        model_payload.update({
            "hidden_dim": args.mlp_hidden_dim,
            "hidden_weights": model_state["hidden_weights"],
            "hidden_biases": model_state["hidden_biases"],
            "output_weights": model_state["output_weights"],
            "output_biases": model_state["output_biases"],
        })
    else:
        model_payload.update({
            "weights": {
                label: {
                    feature_name: weight
                    for feature_name, weight in zip(feature_names, weights[class_idx])
                }
                for class_idx, label in enumerate(RELATION_CLASSES)
            },
            "biases": {
                label: biases[class_idx]
                for class_idx, label in enumerate(RELATION_CLASSES)
            },
        })
    write_json(args.output, model_payload)

    metrics_output = args.metrics_output or _derive_metrics_path(args.output)
    write_json(
        metrics_output,
        {
            "train_metrics": train_metrics,
            "valid_metrics": valid_metrics,
            "best_epoch": model_state["best_epoch"],
            "feature_importance": model_payload["feature_importance"],
        },
    )

    print(f"特征数: {len(feature_names)}")
    print(f"最佳 epoch: {model_state['best_epoch']}")
    print(f"train: accuracy={train_metrics['accuracy']:.4f}, macro_f1={train_metrics['macro_f1']:.4f}")
    if valid_metrics:
        print(f"valid: accuracy={valid_metrics['accuracy']:.4f}, macro_f1={valid_metrics['macro_f1']:.4f}")
    print(f"模型已输出到: {args.output}")
    print(f"指标已输出到: {metrics_output}")


if __name__ == "__main__":
    main()
