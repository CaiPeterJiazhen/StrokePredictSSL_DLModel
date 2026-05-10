from __future__ import annotations

from math import prod
from pathlib import Path
from typing import Any

import torch
from torch import nn

from stroke_predict.matrixnet import MatrixBranch, MatrixNet

SUPPORTED_MASK_RATIOS = (0.15, 0.25, 0.40)


def generate_mask(
    shape: tuple[int, ...] | torch.Size,
    *,
    mask_ratio: float = 0.25,
    seed: int | None = None,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    ratio = float(mask_ratio)
    if ratio not in SUPPORTED_MASK_RATIOS:
        raise ValueError(f"mask_ratio must be one of {SUPPORTED_MASK_RATIOS}")
    shape_tuple = tuple(int(value) for value in shape)
    total = int(prod(shape_tuple))
    if total <= 0:
        raise ValueError("mask shape must contain at least one element")
    n_mask = max(1, int(round(total * ratio)))
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(int(seed))
    flat = torch.zeros(total, dtype=torch.bool)
    flat[torch.randperm(total, generator=generator)[:n_mask]] = True
    mask = flat.reshape(shape_tuple)
    return mask.to(device=device) if device is not None else mask


def masked_mse_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    prediction, target = torch.broadcast_tensors(prediction.float(), target.float())
    mask = torch.broadcast_to(mask.bool(), prediction.shape)
    if not bool(mask.any()):
        return prediction.new_tensor(0.0)
    diff = prediction[mask] - target[mask]
    return torch.mean(diff * diff)


class SSLMatrixAutoencoder(nn.Module):
    def __init__(
        self,
        *,
        psd_shape: tuple[int, int, int] | None,
        fc_shape: tuple[int, int, int] | None,
        embedding_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.psd_shape = psd_shape
        self.fc_shape = fc_shape
        self.psd_encoder = MatrixBranch(psd_shape[0], embedding_dim, dropout) if psd_shape is not None else None
        self.fc_encoder = MatrixBranch(fc_shape[0], embedding_dim, dropout) if fc_shape is not None else None
        self.psd_decoder = (
            nn.Linear(embedding_dim, int(prod(psd_shape))) if psd_shape is not None else None
        )
        self.fc_decoder = nn.Linear(embedding_dim, int(prod(fc_shape))) if fc_shape is not None else None

    def forward(
        self,
        *,
        psd: torch.Tensor | None = None,
        fc: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        if self.psd_encoder is not None and self.psd_decoder is not None:
            if psd is None:
                raise ValueError("PSD input is required for this SSL autoencoder")
            embedding = self.psd_encoder(psd)
            outputs["psd"] = self.psd_decoder(embedding).reshape(psd.shape)
        if self.fc_encoder is not None and self.fc_decoder is not None:
            if fc is None:
                raise ValueError("FC input is required for this SSL autoencoder")
            embedding = self.fc_encoder(fc)
            outputs["fc"] = self.fc_decoder(embedding).reshape(fc.shape)
        return outputs


def load_pretrained_matrixnet_branches(
    model: MatrixNet,
    checkpoint_path: str | Path,
    *,
    load_psd: bool,
    load_fc: bool,
) -> dict[str, bool]:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    loaded = {"psd": False, "fc": False}
    if load_psd:
        if model.psd_branch is None:
            raise ValueError("Requested PSD checkpoint load for a MatrixNet without PSD branch")
        if "psd_encoder" not in checkpoint:
            raise ValueError("SSL checkpoint does not contain psd_encoder")
        model.psd_branch.load_state_dict(checkpoint["psd_encoder"])
        loaded["psd"] = True
    if load_fc:
        if model.fc_branch is None:
            raise ValueError("Requested FC checkpoint load for a MatrixNet without FC branch")
        if "fc_encoder" not in checkpoint:
            raise ValueError("SSL checkpoint does not contain fc_encoder")
        model.fc_branch.load_state_dict(checkpoint["fc_encoder"])
        loaded["fc"] = True
    return loaded


def checkpoint_metadata(
    *,
    ssl_variant: str,
    run_mode: str,
    final_loss: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ssl_variant": ssl_variant,
        "run_mode": run_mode,
        "final_loss": float(final_loss),
        "config": dict(config),
    }


def redacted_checkpoint_path(path: str | Path) -> str:
    path_obj = Path(path)
    parts = list(path_obj.parts)
    if "ssl_matrixnet" in parts:
        index = parts.index("ssl_matrixnet")
        return "/".join(parts[index:])
    return f"<redacted-checkpoint>/{path_obj.name}"
