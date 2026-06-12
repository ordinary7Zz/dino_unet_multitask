from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


SUPPORTED_SAMPLE_SCHEMAS = (
    ("true_label", "prob_class_1"),
    ("ground_truth_label", "malignant_probability"),
)

METRIC_ORDER = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auroc",
    "auprc",
    "sensitivity",
    "specificity",
    "youden",
    "ece",
)


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_records(path: Path) -> list[dict[str, Any]]:
    data = _load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"results JSON 必须是列表格式: {path}")
    return [item for item in data if isinstance(item, dict)]


def _extract_sample_arrays(records: list[dict[str, Any]], source: Path) -> tuple[np.ndarray, np.ndarray, str]:
    y_true: list[int] = []
    y_prob: list[float] = []
    source_schema = "unknown"

    for item in records:
        if item.get("record_type") == "roc_summary":
            continue

        matched_schema = None
        for label_key, prob_key in SUPPORTED_SAMPLE_SCHEMAS:
            if item.get(label_key) is not None and item.get(prob_key) is not None:
                matched_schema = (label_key, prob_key)
                break

        if matched_schema is None:
            continue

        label_key, prob_key = matched_schema
        y_true.append(int(item[label_key]))
        y_prob.append(float(item[prob_key]))
        source_schema = "current" if label_key == "true_label" else "classification_agent"

    if not y_true:
        raise ValueError(
            f"未找到可用于完整分类指标计算的样本级字段: {source}. "
            "请提供 true_label + prob_class_1 或 ground_truth_label + malignant_probability。"
        )

    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)

    unique_labels = np.unique(y_true_arr)
    if unique_labels.size < 2:
        raise ValueError(f"真实标签只有一个类别，无法计算 AUROC/AUPRC: {source}")

    return y_true_arr, y_prob_arr, source_schema


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return tp, tn, fp, fn


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _binary_threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pos_total = int(np.sum(y_true == 1))
    neg_total = int(np.sum(y_true == 0))
    if pos_total == 0 or neg_total == 0:
        raise ValueError("真实标签只有一个类别，无法计算 AUROC/AUPRC。")

    unique_thresholds = np.unique(y_prob)[::-1]
    thresholds = np.concatenate(([np.inf], unique_thresholds))

    tpr: list[float] = []
    fpr: list[float] = []
    precision: list[float] = []
    recall: list[float] = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(np.int32)
        tp, tn, fp, fn = _confusion_counts(y_true, y_pred)
        tpr.append(_safe_div(tp, pos_total))
        fpr.append(_safe_div(fp, neg_total))
        recall.append(_safe_div(tp, pos_total))
        precision.append(_safe_div(tp, tp + fp) if (tp + fp) > 0 else 1.0)

    return thresholds, np.asarray(tpr, dtype=np.float64), np.asarray(fpr, dtype=np.float64), np.asarray(precision, dtype=np.float64), np.asarray(recall, dtype=np.float64)


def _roc_auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    _, tpr, fpr, _, _ = _binary_threshold_sweep(y_true, y_prob)
    return float(np.trapz(tpr, fpr))


def _average_precision_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    _, _, _, precision, recall = _binary_threshold_sweep(y_true, y_prob)
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def _expected_calibration_error_binary(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)

    valid = y_true != -1
    y_true = y_true[valid]
    y_prob = y_prob[valid]

    if y_true.size == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = y_true.size

    for i in range(n_bins):
        start = bin_edges[i]
        end = bin_edges[i + 1]
        if i == 0:
            mask = (y_prob >= start) & (y_prob <= end)
        else:
            mask = (y_prob > start) & (y_prob <= end)

        if not np.any(mask):
            continue

        prob_bin = y_prob[mask]
        true_bin = y_true[mask]
        avg_conf = float(prob_bin.mean())
        avg_acc = float((true_bin == 1).mean())
        weight = prob_bin.size / n
        ece += weight * abs(avg_acc - avg_conf)

    return float(ece)


def _compute_threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= float(threshold)).astype(np.int32)
    tp, tn, fp, fn = _confusion_counts(y_true, y_pred)

    accuracy = _safe_div(tp + tn, y_true.size)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
    sensitivity = recall
    specificity = _safe_div(tn, tn + fp)
    youden = sensitivity + specificity - 1.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "youden": float(youden),
        "ece": _expected_calibration_error_binary(y_true, y_prob),
    }


def _compute_single_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    metrics = _compute_threshold_metrics(y_true, y_prob, threshold)
    metrics["auroc"] = _roc_auc_score(y_true, y_prob)
    metrics["auprc"] = _average_precision_score(y_true, y_prob)
    return metrics


def classification_bootstrap_metrics(
    y_probs: np.ndarray,
    y_labels: np.ndarray,
    threshold: float = 0.5,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, tuple[float, tuple[float, float]]]:
    y_probs = np.asarray(y_probs, dtype=np.float64)
    y_labels = np.asarray(y_labels, dtype=np.int32)

    valid_mask = y_labels != -1
    y_probs = y_probs[valid_mask]
    y_labels = y_labels[valid_mask]

    if y_labels.size == 0:
        zero_ci = (0.0, (0.0, 0.0))
        return {k: zero_ci for k in METRIC_ORDER}

    rng = np.random.default_rng(seed)
    n = y_labels.size

    metrics_samples: dict[str, list[float]] = {k: [] for k in METRIC_ORDER}

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        probs_s = y_probs[idx]
        labels_s = y_labels[idx]

        if np.unique(labels_s).size < 2:
            single_metrics = _compute_threshold_metrics(labels_s, probs_s, threshold)
            for key in METRIC_ORDER:
                if key in ("auroc", "auprc"):
                    metrics_samples[key].append(float("nan"))
                else:
                    metrics_samples[key].append(float(single_metrics[key]))
            continue

        single_metrics = _compute_single_metrics(labels_s, probs_s, threshold)
        for key in METRIC_ORDER:
            metrics_samples[key].append(float(single_metrics[key]))

    results: dict[str, tuple[float, tuple[float, float]]] = {}
    alpha = 1.0 - ci
    for key, vals in metrics_samples.items():
        arr = np.asarray(vals, dtype=np.float64)
        arr_valid = arr[~np.isnan(arr)]
        if arr_valid.size == 0:
            results[key] = (0.0, (0.0, 0.0))
            continue
        mean = float(arr_valid.mean())
        lower = float(np.percentile(arr_valid, 100 * alpha / 2))
        upper = float(np.percentile(arr_valid, 100 * (1 - alpha / 2)))
        results[key] = (mean, (lower, upper))

    return results


def find_best_threshold_by_youden_index(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_true_np = np.asarray(y_true, dtype=np.int32).reshape(-1)
    y_prob_np = np.asarray(y_prob, dtype=np.float64).reshape(-1)

    valid_mask = y_true_np != -1
    y_true_np = y_true_np[valid_mask]
    y_prob_np = y_prob_np[valid_mask]

    if y_true_np.size == 0 or np.unique(y_true_np).size < 2:
        return {
            "best_threshold": 0.5,
            "youden": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
        }

    thresholds, tpr, fpr, _, _ = _binary_threshold_sweep(y_true_np, y_prob_np)
    youden = tpr - fpr
    max_j = np.max(youden)
    candidate_idx = np.where(youden == max_j)[0]

    if candidate_idx.size > 1:
        best_tpr = np.max(tpr[candidate_idx])
        candidate_idx = candidate_idx[np.where(tpr[candidate_idx] == best_tpr)[0]]

    idx = int(candidate_idx[np.argmin(thresholds[candidate_idx])])
    return {
        "best_threshold": float(thresholds[idx]),
        "youden": float(youden[idx]),
        "sensitivity": float(tpr[idx]),
        "specificity": float(1.0 - fpr[idx]),
    }


def _metric_block(metric_value: tuple[float, tuple[float, float]]) -> dict[str, Any]:
    mean_v, (low_v, high_v) = metric_value
    return {
        "mean": round(float(mean_v), 4),
        "CI95": [round(float(low_v), 4), round(float(high_v), 4)],
    }


def _round_float(value: Any) -> float:
    return round(float(value), 4)


def _build_summary(source: Path, threshold: float, n_boot: int, ci: float, seed: int) -> dict[str, Any]:
    records = _load_records(source)
    y_true, y_prob, schema = _extract_sample_arrays(records, source)

    metrics = classification_bootstrap_metrics(
        y_prob,
        y_true,
        threshold=threshold,
        n_boot=n_boot,
        ci=ci,
        seed=seed,
    )

    youden = find_best_threshold_by_youden_index(y_true, y_prob)

    summary: dict[str, Any] = {
        "source": str(source),
        "schema": schema,
        "n_samples": int(y_true.size),
        "n_positive": int(np.sum(y_true == 1)),
        "n_negative": int(np.sum(y_true == 0)),
        "threshold": round(float(threshold), 4),
        "bootstrap": {
            "n_boot": int(n_boot),
            "ci": round(float(ci), 4),
            "seed": int(seed),
        },
        "metrics": {},
        "youden_threshold": {
            "best_threshold": _round_float(youden.get("best_threshold", 0.5)),
            "youden": _round_float(youden.get("youden", 0.0)),
            "sensitivity": _round_float(youden.get("sensitivity", 0.0)),
            "specificity": _round_float(youden.get("specificity", 0.0)),
        },
    }

    for key in METRIC_ORDER:
        if key in metrics:
            summary["metrics"][key] = _metric_block(metrics[key])

    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"Source: {summary['source']}")
    print(f"Schema: {summary['schema']}")
    print(
        f"Samples: {summary['n_samples']}  "
        f"(pos={summary['n_positive']}, neg={summary['n_negative']})"
    )
    print(f"Threshold: {summary['threshold']:.4f}")
    print("Metrics:")
    for key in METRIC_ORDER:
        metric = summary["metrics"].get(key)
        if not metric:
            continue
        low_v, high_v = metric["CI95"]
        print(f"  {key.upper():<11} mean={metric['mean']:.4f}  CI95=({low_v:.4f}, {high_v:.4f})")
    yt = summary["youden_threshold"]
    print(
        "Youden threshold: "
        f"best={yt['best_threshold']:.4f}, youden={yt['youden']:.4f}, "
        f"sens={yt['sensitivity']:.4f}, spec={yt['specificity']:.4f}"
    )


def _default_output_path(source: Path) -> Path:
    return source.with_name(f"{source.stem}_binary_metrics.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute binary classification metrics from standardized results JSON.")
    parser.add_argument("input_json", type=Path, help="Input results JSON file")
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for precision/recall-style metrics")
    parser.add_argument("--n-boot", type=int, default=2000, help="Number of bootstrap samples")
    parser.add_argument("--ci", type=float, default=0.95, help="Confidence interval level")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.input_json

    if not source.exists():
        raise FileNotFoundError(f"Input JSON not found: {source}")

    summary = _build_summary(
        source=source,
        threshold=args.threshold,
        n_boot=args.n_boot,
        ci=args.ci,
        seed=args.seed,
    )

    _print_summary(summary)

    output_path = args.output or _default_output_path(source)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved metrics summary to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
