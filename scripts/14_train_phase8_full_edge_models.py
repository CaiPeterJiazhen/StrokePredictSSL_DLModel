from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stroke_predict.config import load_yaml_mapping
from stroke_predict.phase8_evaluation import bootstrap_phase8_ci, permutation_phase8_test
from stroke_predict.phase8_features import align_full_edge_features, merge_feature_tables, tag_feature_table
from stroke_predict.phase8_1_validation import (
    apply_multiple_comparison_correction,
    audit_comparison_models,
    audit_source_mode,
    build_patient_error_audit,
    build_threshold_calibration_table,
    write_phase8_1_validation_outputs,
)
from stroke_predict.phase8_models import run_phase8_lopo_models
from stroke_predict.phase8_reports import write_phase8_model_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--feature-set", choices=["reduced32", "full62"], default="reduced32")
    parser.add_argument("--fold-limit", type=int, default=None)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_yaml_mapping(config_path)
    requested = [str(model_id) for model_id in config.get("models", [])]
    if args.feature_set == "full62" and args.run_mode == "fast" and not any(model_id.startswith("M16") for model_id in requested):
        config["models"] = ["M16a_prop_full62_fullfc_ridge_logistic", "M16b_prop_full62_fullfc_pls_da"]
        requested = [str(model_id) for model_id in config["models"]]
    if args.run_mode == "full" and args.feature_set == "full62" and any(model_id.startswith("M16") for model_id in requested):
        if not bool(config.get("m16_full62_full_mode_enabled", False)):
            raise ValueError("M16 full62 full-mode is not planned for Phase 8 unless explicitly enabled")

    input_paths = dict(config.get("input_paths", {}))
    output_dir = _resolve(config_path, str(config.get("output_dir", "outputs")))
    labels = pd.read_csv(output_dir / "evaluation" / "phase8_label_audit.csv")
    folds = json.loads(_resolve(config_path, str(input_paths.get("folds", "outputs/folds/outer_folds.json"))).read_text(encoding="utf-8"))
    features = _load_phase8_features(config_path, input_paths, output_dir, labels, args.feature_set)

    result = run_phase8_lopo_models(
        config,
        features=features,
        labels=labels,
        folds=folds,
        run_mode=args.run_mode,
        fold_limit=args.fold_limit,
        feature_set=args.feature_set,
    )
    settings = dict(config.get(args.run_mode, {}))
    bootstrap_n = int(settings.get("bootstrap_resamples", 25 if args.run_mode == "fast" else 1000))
    permutation_n = int(settings.get("permutation_resamples", 25 if args.run_mode == "fast" else 1000))
    ci = bootstrap_phase8_ci(result.predictions, n_bootstrap=bootstrap_n, random_seed=int(config.get("random_seed", 42)))
    perm = permutation_phase8_test(result.predictions, n_permutations=permutation_n, random_seed=int(config.get("random_seed", 42)))
    metrics = _attach_uncertainty(result.metrics, ci, perm)

    (output_dir / "evaluation").mkdir(parents=True, exist_ok=True)
    result.no_leakage_audit.to_csv(output_dir / "evaluation" / "phase8_no_leakage_audit.csv", index=False)
    fc_audit = _fc_audit(output_dir, args.feature_set)
    write_phase8_model_report(
        result.predictions,
        metrics,
        fc_audit,
        output_dir=output_dir,
        label_audit=_label_audit(labels),
    )
    correction = apply_multiple_comparison_correction(metrics[["model_id", "permutation_p_value"]])
    best_model_id = _best_model_id(metrics)
    best_predictions = result.predictions.loc[result.predictions["model_id"].astype(str).eq(best_model_id)].copy()
    source_audit = audit_source_mode(fc_audit)
    duplicate_audit = _comparison_audit(features, result.predictions)
    write_phase8_1_validation_outputs(
        output_dir=output_dir,
        source_audit=source_audit,
        duplicate_audit=duplicate_audit,
        correction_table=correction,
        threshold_calibration=build_threshold_calibration_table(best_predictions),
        patient_error_audit=build_patient_error_audit(labels, best_predictions),
        no_leakage_audit=result.no_leakage_audit,
        best_model_id=best_model_id,
        real_time_series_reproduced=_real_time_series_reproduced(source_audit, metrics, best_model_id),
    )

    print("PHASE8_MODELS_OK")
    print(f"run_mode={args.run_mode}")
    print(f"feature_set={args.feature_set}")
    print(f"n_predictions={len(result.predictions)}")
    print(f"n_models={result.predictions['model_id'].nunique()}")
    return 0


def _load_phase8_features(
    config_path: Path,
    input_paths: dict[str, object],
    output_dir: Path,
    labels: pd.DataFrame,
    feature_set: str,
) -> pd.DataFrame:
    subject_index = pd.read_csv(output_dir / "matrices" / "phase8_matrix_subject_index.csv")
    if feature_set == "full62":
        eo_path = output_dir / "matrices" / "phase8_fc_full62_eo.npy"
        ec_path = output_dir / "matrices" / "phase8_fc_full62_ec.npy"
    else:
        eo_path = output_dir / "matrices" / "phase8_fc_full_reduced32_eo.npy"
        ec_path = output_dir / "matrices" / "phase8_fc_full_reduced32_ec.npy"
    eo = align_full_edge_features(np.load(eo_path), subject_index, labels, feature_prefix="fullfc_eo")
    ec = align_full_edge_features(np.load(ec_path), subject_index, labels, feature_prefix="fullfc_ec")
    base = merge_feature_tables(subject_index, eo, ec)

    optional_tables = []
    for key, prefix in (("roi_features", "roi_fc"), ("summary_features", "summary")):
        value = input_paths.get(key)
        if value:
            path = _resolve(config_path, str(value))
            if path.exists():
                optional_tables.append(tag_feature_table(pd.read_csv(path), prefix))
    return merge_feature_tables(base, *optional_tables)


def _attach_uncertainty(metrics: pd.DataFrame, ci: pd.DataFrame, perm: pd.DataFrame) -> pd.DataFrame:
    result = metrics.copy()
    roc_ci = ci.loc[ci["metric"].eq("roc_auc"), ["model_id", "ci_lower", "ci_upper"]].rename(
        columns={"ci_lower": "bootstrap_ci_lower", "ci_upper": "bootstrap_ci_upper"}
    )
    roc_perm = perm.loc[perm["metric"].eq("roc_auc"), ["model_id", "p_value"]].rename(
        columns={"p_value": "permutation_p_value"}
    )
    result = result.merge(roc_ci, on="model_id", how="left")
    result = result.merge(roc_perm, on="model_id", how="left")
    return result


def _best_model_id(metrics: pd.DataFrame) -> str:
    if metrics.empty:
        raise ValueError("Cannot choose best Phase 8 model from empty metrics")
    for column, ascending in (("roc_auc", False), ("pr_auc", False), ("permutation_p_value", True)):
        if column in metrics.columns:
            values = pd.to_numeric(metrics[column], errors="coerce")
            if values.notna().any():
                index = values.idxmin() if ascending else values.idxmax()
                return str(metrics.loc[index, "model_id"])
    return str(metrics.iloc[0]["model_id"])


def _comparison_audit(features: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, object]:
    models = set(predictions["model_id"].astype(str))
    if {"M15a_prop_roi_fc_best_ml", "M15b_prop_summary_eeg_best_ml"} <= models:
        return audit_comparison_models(
            features=features,
            predictions=predictions,
            model_a="M15a_prop_roi_fc_best_ml",
            model_b="M15b_prop_summary_eeg_best_ml",
        )
    return {
        "model_a": "M15a_prop_roi_fc_best_ml",
        "model_b": "M15b_prop_summary_eeg_best_ml",
        "feature_matrices_identical": False,
        "predictions_identical": False,
        "explanation": "M15a/M15b duplicate audit was not applicable because both comparison models were not run.",
    }


def _real_time_series_reproduced(source_audit: dict[str, object], metrics: pd.DataFrame, best_model_id: str) -> bool:
    if not bool(source_audit.get("is_real_time_series_fc", False)):
        return False
    row = metrics.loc[metrics["model_id"].astype(str).eq(best_model_id)]
    if row.empty:
        return False
    auc = float(pd.to_numeric(row["roc_auc"], errors="coerce").iloc[0]) if "roc_auc" in row else np.nan
    p_value = (
        float(pd.to_numeric(row["permutation_p_value"], errors="coerce").iloc[0])
        if "permutation_p_value" in row
        else np.nan
    )
    return bool(np.isfinite(auc) and auc >= 0.744 and np.isfinite(p_value) and p_value < 0.05)


def _fc_audit(output_dir: Path, feature_set: str) -> dict[str, object]:
    if feature_set == "full62":
        edge_path = output_dir / "features" / "phase8_full62_full_edge_index.csv"
        channel_path = output_dir / "features" / "phase8_full62_channels.csv"
    else:
        edge_path = output_dir / "features" / "phase8_reduced32_full_edge_index.csv"
        channel_path = output_dir / "features" / "phase8_reduced32_channels.csv"
    full62_edge_path = output_dir / "features" / "phase8_full62_full_edge_index.csv"
    n_edges = len(pd.read_csv(edge_path)) if edge_path.exists() else 0
    n_channels = len(pd.read_csv(channel_path)) if channel_path.exists() else 0
    audit_path = output_dir / "features" / f"phase8_{feature_set}_full_edge_audit.csv"
    source_mode = "not_available"
    if audit_path.exists():
        audit = pd.read_csv(audit_path)
        if "source_mode" in audit.columns and not audit.empty:
            source_mode = str(audit.iloc[0]["source_mode"])
    return {
        "reduced32_n_channels": n_channels,
        "reduced32_n_edges": n_edges,
        "metrics": "coherence, imaginary_coherence, wpli",
        "bands": "delta, theta, alpha_mu, low_beta, high_beta, broad_beta",
        "conditions": "EO, EC",
        "source_mode": source_mode,
        "full62_smoke_status": "run" if feature_set == "full62" or full62_edge_path.exists() else "not_run",
    }


def _label_audit(labels: pd.DataFrame) -> dict[str, object]:
    analyzable = labels.loc[labels["phase8_label_status"].eq("analyzable")]
    return {
        "n_analyzable": int(len(analyzable)),
        "median_residual": float(analyzable["median_residual"].dropna().iloc[0]) if "median_residual" in labels.columns and analyzable["median_residual"].notna().any() else np.nan,
        "n_proportional_recovery": int(analyzable["primary_label_prop_residual"].eq("ProportionalRecovery").sum()),
        "n_poor_recovery": int(analyzable["primary_label_prop_residual"].eq("PoorRecovery").sum()),
    }


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


if __name__ == "__main__":
    raise SystemExit(main())
