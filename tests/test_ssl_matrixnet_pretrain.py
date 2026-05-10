from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from stroke_predict.matrixnet import MatrixNet, MatrixNetConfig
from stroke_predict.ssl_matrixnet import load_pretrained_matrixnet_branches
from stroke_predict.ssl_matrixnet_training import SSLPretrainConfig, pretrain_ssl_matrixnet


def test_tiny_synthetic_psd_fc_dataset_can_pretrain_for_one_epoch(tmp_path: Path) -> None:
    psd, fc = _tiny_matrices()
    checkpoint_path = tmp_path / "ssl_encoder.pt"
    config = SSLPretrainConfig(
        ssl_variant="stroke_baseline",
        run_mode="fast",
        epochs=1,
        batch_size=2,
        mask_ratio=0.25,
        embedding_dim=8,
        hidden_dim=16,
        seed=0,
        device="cpu",
    )

    result = pretrain_ssl_matrixnet(psd=psd, fc=fc, checkpoint_path=checkpoint_path, config=config)

    assert np.isfinite(result.final_loss)
    assert checkpoint_path.exists()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert "psd_encoder" in checkpoint
    assert "fc_encoder" in checkpoint


def test_checkpoint_can_load_into_matrixnet_branches(tmp_path: Path) -> None:
    psd, fc = _tiny_matrices()
    checkpoint_path = tmp_path / "ssl_encoder.pt"
    config = SSLPretrainConfig(
        ssl_variant="stroke_baseline",
        run_mode="fast",
        epochs=1,
        batch_size=2,
        mask_ratio=0.25,
        embedding_dim=8,
        hidden_dim=16,
        seed=0,
        device="cpu",
    )
    pretrain_ssl_matrixnet(psd=psd, fc=fc, checkpoint_path=checkpoint_path, config=config)
    model = MatrixNet(
        MatrixNetConfig(
            use_psd=True,
            use_fc=True,
            embedding_dim=8,
            hidden_dim=16,
            dropout=0.0,
        )
    )
    before = {name: tensor.detach().clone() for name, tensor in model.fc_branch.state_dict().items()}

    loaded = load_pretrained_matrixnet_branches(model, checkpoint_path, load_psd=True, load_fc=True)

    assert loaded == {"psd": True, "fc": True}
    after = model.fc_branch.state_dict()
    assert any(not torch.equal(before[name], after[name]) for name in before)


def _tiny_matrices() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    psd = rng.normal(size=(4, 2, 4, 5)).astype(np.float32)
    fc = rng.normal(size=(4, 4, 3, 2)).astype(np.float32)
    return psd, fc
