from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import torch

from tests.test_matrixnet_no_leakage import _write_minimal_inputs


def test_pretrain_cli_supports_fast_fold_limit_cpu(tmp_path: Path) -> None:
    _write_minimal_inputs(tmp_path)
    config_path = _write_ssl_config(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/10_pretrain_ssl_matrixnet.py",
            "--config",
            str(config_path),
            "--run-mode",
            "fast",
            "--fold-limit",
            "1",
            "--ssl-variant",
            "stroke_baseline",
            "--device",
            "cpu",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "SSL_MATRIXNET_PRETRAIN_OK" in completed.stdout


def test_train_cli_supports_fast_fold_limit_cpu(tmp_path: Path) -> None:
    _write_minimal_inputs(tmp_path)
    config_path = _write_ssl_config(tmp_path)
    subprocess.run(
        [
            sys.executable,
            "scripts/10_pretrain_ssl_matrixnet.py",
            "--config",
            str(config_path),
            "--run-mode",
            "fast",
            "--fold-limit",
            "1",
            "--ssl-variant",
            "stroke_baseline",
            "--device",
            "cpu",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/11_train_ssl_matrixnet.py",
            "--config",
            str(config_path),
            "--run-mode",
            "fast",
            "--fold-limit",
            "1",
            "--ssl-variant",
            "stroke_baseline",
            "--device",
            "cpu",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "SSL_MATRIXNET_TRAIN_OK" in completed.stdout


def test_cuda_full_mode_refuses_when_required_and_unavailable(tmp_path: Path) -> None:
    if torch.cuda.is_available():
        return
    _write_minimal_inputs(tmp_path)
    config_path = _write_ssl_config(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/10_pretrain_ssl_matrixnet.py",
            "--config",
            str(config_path),
            "--run-mode",
            "full",
            "--ssl-variant",
            "stroke_baseline",
            "--device",
            "cuda",
            "--require-cuda",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "CUDA is required" in completed.stderr


def _write_ssl_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "ssl_matrixnet.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"output_dir: {tmp_path.as_posix()}",
                "run_id: unit",
                "run_modes:",
                "  fast:",
                "    seeds: [0]",
                "    ssl_epochs: 1",
                "    finetune_epochs: 2",
                "    patience: 1",
                "    batch_size: 2",
                "    learning_rates: [0.001]",
                "    weight_decays: [0.01]",
                "    dropouts: [0.3]",
                "    embedding_dims: [8]",
                "    hidden_dims: [16]",
                "    mask_ratio: 0.25",
                "    device: cpu",
                "    require_cuda: false",
                "    bootstrap_resamples: 10",
                "    permutation_resamples: 10",
                "  full:",
                "    seeds: [0]",
                "    ssl_epochs: 1",
                "    finetune_epochs: 1",
                "    patience: 1",
                "    batch_size: 2",
                "    learning_rates: [0.001]",
                "    weight_decays: [0.01]",
                "    dropouts: [0.3]",
                "    embedding_dims: [8]",
                "    hidden_dims: [16]",
                "    mask_ratio: 0.25",
                "    device: cuda",
                "    require_cuda: true",
                "models:",
                "  fast:",
                "    stroke_baseline:",
                "      - M9a_sslA_fc_only",
                "  full:",
                "    stroke_baseline:",
                "      - M9a_sslA_fc_only",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config_path
