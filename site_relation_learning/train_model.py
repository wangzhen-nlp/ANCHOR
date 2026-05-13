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
    train_softmax_regression,
    vectorize_samples,
    write_json,
)


def _derive_metrics_path(model_output):
    if model_output.endswith(".json"):
        return model_output[:-5] + ".metrics.json"
    return model_output + ".metrics.json"


def main():
    parser = ArgumentParser(description="训练站点关系四分类 softmax 模型")
    parser.add_argument("--train", required=True, help="训练集 JSONL")
    parser.add_argument("--valid", default="", help="验证集 JSONL")
    parser.add_argument("-o", "--output", default="site_relation_model.json", help="模型输出 JSON")
    parser.add_argument("--metrics-output", default="", help="指标输出 JSON")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数，默认: 30")
    parser.add_argument("--learning-rate", type=float, default=0.03, help="学习率，默认: 0.03")
    parser.add_argument("--l2", type=float, default=1e-4, help="L2 正则，默认: 1e-4")
    parser.add_argument("--class-weight", choices=("balanced", "none"), default="balanced", help="类别权重，默认: balanced")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认: 42")
    parser.add_argument("--no-progress", action="store_true", help="关闭进度条")
    args = parser.parse_args()

    train_samples = load_dataset_samples(args.train)
    if not train_samples:
        raise ValueError("训练集为空")
    valid_samples = load_dataset_samples(args.valid) if args.valid else []
    feature_names = infer_feature_names(train_samples)
    standardizer = fit_standardizer(train_samples, feature_names)
    train_dense = vectorize_samples(train_samples, feature_names, standardizer, show_progress=not args.no_progress)
    valid_dense = vectorize_samples(valid_samples, feature_names, standardizer, show_progress=not args.no_progress) if valid_samples else []

    model_state = train_softmax_regression(
        train_dense,
        valid_dense_samples=valid_dense,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        class_weight=args.class_weight,
        seed=args.seed,
        show_progress=not args.no_progress,
    )
    weights = model_state["weights"]
    biases = model_state["biases"]
    train_metrics, _ = evaluate_dense_samples(train_dense, weights, biases)
    valid_metrics = None
    if valid_dense:
        valid_metrics, _ = evaluate_dense_samples(valid_dense, weights, biases)

    model_payload = {
        "model_type": "site_relation_softmax_v1",
        "classes": list(RELATION_CLASSES),
        "train_file": args.train,
        "valid_file": args.valid,
        "feature_names": feature_names,
        "standardizer": standardizer,
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
            "seed": args.seed,
        },
        "feature_importance": build_feature_importance(feature_names, weights),
        "training_history": model_state["history"],
    }
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

