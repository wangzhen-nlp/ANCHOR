#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from ne_link_learning.core import (
    build_feature_importance,
    choose_best_threshold,
    evaluate_dense_samples,
    fit_standardizer,
    infer_feature_names,
    load_dataset_samples,
    train_logistic_regression,
    vectorize_samples,
    write_json,
)


def _derive_metrics_path(model_output):
    if model_output.endswith(".json"):
        return model_output[:-5] + ".metrics.json"
    return model_output + ".metrics.json"


def main():
    parser = ArgumentParser(description="训练 topology link 纯 Python 逻辑回归模型")
    parser.add_argument("--train", required=True, help="训练集 JSONL")
    parser.add_argument("--valid", default="", help="验证集 JSONL")
    parser.add_argument(
        "-o",
        "--output",
        default="topology_link_model.json",
        help="模型输出 JSON，默认: topology_link_model.json",
    )
    parser.add_argument(
        "--metrics-output",
        default="",
        help="训练指标输出 JSON；默认与模型同名前缀",
    )
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数，默认: 20")
    parser.add_argument("--learning-rate", type=float, default=0.03, help="学习率，默认: 0.03")
    parser.add_argument("--l2", type=float, default=1e-4, help="L2 正则，默认: 1e-4")
    parser.add_argument(
        "--positive-weight",
        type=float,
        default=0.0,
        help="正样本权重；<=0 时自动按负正样本比计算，默认: 0",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认: 42")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=5,
        help="验证集早停 patience，默认: 5",
    )
    args = parser.parse_args()

    print(f"加载训练集: {args.train}")
    train_samples = load_dataset_samples(args.train)
    if not train_samples:
        raise ValueError("训练集为空")

    if args.valid:
        print(f"加载验证集: {args.valid}")
    valid_samples = load_dataset_samples(args.valid) if args.valid else []
    print("推断特征集合...")
    feature_names = infer_feature_names(train_samples)
    print("拟合标准化参数...")
    standardizer = fit_standardizer(train_samples, feature_names)
    print("向量化训练集...")
    train_dense = vectorize_samples(
        train_samples,
        feature_names,
        standardizer,
        show_progress=True,
        progress_label="向量化训练样本",
    )
    valid_dense = []
    if valid_samples:
        print("向量化验证集...")
        valid_dense = vectorize_samples(
            valid_samples,
            feature_names,
            standardizer,
            show_progress=True,
            progress_label="向量化验证样本",
        )

    print("训练模型...")
    model_state = train_logistic_regression(
        train_dense_samples=train_dense,
        valid_dense_samples=valid_dense,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        positive_weight=(args.positive_weight if args.positive_weight > 0 else None),
        seed=args.seed,
        early_stop_patience=args.early_stop_patience,
        show_progress=True,
        progress_label="训练轮次",
    )

    if valid_dense:
        print("搜索最佳阈值...")
        threshold, valid_metrics, _ = choose_best_threshold(
            valid_dense,
            model_state["weights"],
            model_state["bias"],
            show_progress=True,
            progress_label="搜索最佳阈值",
        )
    else:
        threshold = 0.5
        valid_metrics = None

    print("评估训练集...")
    train_metrics, _ = evaluate_dense_samples(
        train_dense,
        model_state["weights"],
        model_state["bias"],
        threshold=threshold,
        show_progress=True,
        progress_label="评估训练样本",
    )
    if valid_dense:
        print("评估验证集...")
        valid_metrics, _ = evaluate_dense_samples(
            valid_dense,
            model_state["weights"],
            model_state["bias"],
            threshold=threshold,
            show_progress=True,
            progress_label="评估验证样本",
        )

    model_payload = {
        "model_type": "python_logistic_regression_v1",
        "train_file": args.train,
        "valid_file": args.valid,
        "feature_names": feature_names,
        "standardizer": standardizer,
        "weights": {
            feature_name: weight
            for feature_name, weight in zip(feature_names, model_state["weights"])
        },
        "bias": model_state["bias"],
        "threshold": threshold,
        "best_epoch": model_state["best_epoch"],
        "positive_weight": model_state["positive_weight"],
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "training_config": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
            "seed": args.seed,
            "early_stop_patience": args.early_stop_patience,
        },
        "feature_importance": build_feature_importance(feature_names, model_state["weights"]),
        "training_history": model_state["history"],
    }
    print(f"写出模型文件: {args.output}")
    write_json(args.output, model_payload)

    metrics_output = args.metrics_output or _derive_metrics_path(args.output)
    print(f"写出指标文件: {metrics_output}")
    write_json(
        metrics_output,
        {
            "train_metrics": train_metrics,
            "valid_metrics": valid_metrics,
            "threshold": threshold,
            "best_epoch": model_state["best_epoch"],
            "feature_importance": model_payload["feature_importance"],
        },
    )

    print(f"特征数: {len(feature_names)}")
    print(f"最佳 epoch: {model_state['best_epoch']}")
    print(
        f"train: precision={train_metrics['precision']:.4f}, "
        f"recall={train_metrics['recall']:.4f}, "
        f"f1={train_metrics['f1']:.4f}, auc={train_metrics['roc_auc']:.4f}"
    )
    if valid_metrics:
        print(
            f"valid: precision={valid_metrics['precision']:.4f}, "
            f"recall={valid_metrics['recall']:.4f}, "
            f"f1={valid_metrics['f1']:.4f}, auc={valid_metrics['roc_auc']:.4f}"
        )
    print(f"模型已输出到: {args.output}")
    print(f"指标已输出到: {metrics_output}")


if __name__ == "__main__":
    main()
