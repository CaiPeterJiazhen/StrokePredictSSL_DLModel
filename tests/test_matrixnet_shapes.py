from __future__ import annotations

import torch

from stroke_predict.matrixnet import MatrixNet, MatrixNetConfig


def test_matrixnet_accepts_psd_only_inputs() -> None:
    model = MatrixNet(MatrixNetConfig(use_psd=True, use_fc=False, use_tacs=False, use_clinical=False))
    logits = model(
        psd_eo=torch.randn(4, 2, 62, 90),
        psd_ec=torch.randn(4, 2, 62, 90),
    )
    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()


def test_matrixnet_accepts_fc_only_inputs() -> None:
    model = MatrixNet(MatrixNetConfig(use_psd=False, use_fc=True, use_tacs=False, use_clinical=False))
    logits = model(
        fc_eo=torch.randn(3, 2, 36, 6, 2),
        fc_ec=torch.randn(3, 2, 36, 6, 2),
    )
    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()


def test_fc_canonicalization_preserves_edge_by_band_structure() -> None:
    model = MatrixNet(MatrixNetConfig(use_psd=False, use_fc=True, use_tacs=False, use_clinical=False))
    canonical = model.canonicalize_fc(torch.randn(3, 2, 36, 6, 2))
    assert canonical.shape == (3, 4, 36, 6)


def test_matrixnet_accepts_psd_fc_tacs_clinical_inputs() -> None:
    model = MatrixNet(
        MatrixNetConfig(
            use_psd=True,
            use_fc=True,
            use_tacs=True,
            use_clinical=True,
            tacs_dim=7,
            clinical_dim=5,
            embedding_dim=16,
            hidden_dim=32,
            dropout=0.2,
        )
    )
    logits = model(
        psd_eo=torch.randn(2, 2, 62, 90),
        psd_ec=torch.randn(2, 2, 62, 90),
        fc_eo=torch.randn(2, 2, 36, 6, 2),
        fc_ec=torch.randn(2, 2, 36, 6, 2),
        tacs=torch.randn(2, 7),
        clinical=torch.randn(2, 5),
    )
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
