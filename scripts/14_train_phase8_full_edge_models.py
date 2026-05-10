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
from stroke_predict.phase8_features import align_full_edge_features, merge_feature_tables
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
    write_phase8_model_report(
        result.predictions,
        metrics,
        _fc_audit(output_dir, args.feature_set),
        output_dir=output_dir,
        label_audit=_label_audit(labels),
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
    for key in ("roi_features", "summary_features"):
        value = input_paths.get(key)
        if value:
            path = _resolve(config_path, str(value))
            if path.exists():
                optional_tables.append(pd.read_csv(path))
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
