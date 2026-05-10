from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MatrixNetConfig:
    use_psd: bool = True
    use_fc: bool = True
    use_tacs: bool = False
    use_clinical: bool = False
    tacs_dim: int = 0
    clinical_dim: int = 0
    embedding_dim: int = 32
    hidden_dim: int = 64
    dropout: float = 0.5


class MatrixBranch(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.projection = nn.Linear(32, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(x.float()))


class VectorBranch(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("VectorBranch input_dim must be positive")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class MatrixNet(nn.Module):
    def __init__(self, config: MatrixNetConfig) -> None:
        super().__init__()
        if not any([config.use_psd, config.use_fc, config.use_tacs, config.use_clinical]):
            raise ValueError("At least one input family must be enabled")
        self.config = config
        self.psd_branch = MatrixBranch(2, config.embedding_dim, config.dropout) if config.use_psd else None
        self.fc_branch = MatrixBranch(4, config.embedding_dim, config.dropout) if config.use_fc else None
        self.tacs_branch = (
            VectorBranch(config.tacs_dim, config.embedding_dim, config.hidden_dim, config.dropout)
            if config.use_tacs
            else None
        )
        self.clinical_branch = (
            VectorBranch(config.clinical_dim, config.embedding_dim, config.hidden_dim, config.dropout)
            if config.use_clinical
            else None
        )
        n_embeddings = int(config.use_psd) * 2 + int(config.use_fc) * 2 + int(config.use_tacs) + int(config.use_clinical)
        fusion_dim = n_embeddings * config.embedding_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self,
        *,
        psd_eo: torch.Tensor | None = None,
        psd_ec: torch.Tensor | None = None,
        fc_eo: torch.Tensor | None = None,
        fc_ec: torch.Tensor | None = None,
        tacs: torch.Tensor | None = None,
        clinical: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings: list[torch.Tensor] = []
        if self.psd_branch is not None:
            if psd_eo is None or psd_ec is None:
                raise ValueError("PSD inputs are required for this MatrixNet")
            embeddings.extend(
                [
                    self.psd_branch(self.canonicalize_psd(psd_eo)),
                    self.psd_branch(self.canonicalize_psd(psd_ec)),
                ]
            )
        if self.fc_branch is not None:
            if fc_eo is None or fc_ec is None:
                raise ValueError("FC inputs are required for this MatrixNet")
            embeddings.extend(
                [
                    self.fc_branch(self.canonicalize_fc(fc_eo)),
                    self.fc_branch(self.canonicalize_fc(fc_ec)),
                ]
            )
        if self.tacs_branch is not None:
            if tacs is None:
                raise ValueError("tACS input is required for this MatrixNet")
            embeddings.append(self.tacs_branch(tacs))
        if self.clinical_branch is not None:
            if clinical is None:
                raise ValueError("Clinical input is required for this MatrixNet")
            embeddings.append(self.clinical_branch(clinical))
        fused = torch.cat(embeddings, dim=1)
        return self.classifier(fused).squeeze(-1)

    def canonicalize_psd(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.ndim == 3:
            return x.unsqueeze(1).repeat(1, 2, 1, 1)
        if x.ndim == 4:
            if x.shape[1] == 1:
                return x.repeat(1, 2, 1, 1)
            return x
        raise ValueError(f"PSD input must be [N,H,W] or [N,C,H,W], found {tuple(x.shape)}")

    def canonicalize_fc(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.ndim == 4:
            return x
        if x.ndim == 5:
            batch, views, edges, bands, metrics = x.shape
            return x.permute(0, 1, 4, 2, 3).reshape(batch, views * metrics, edges, bands)
        raise ValueError(f"FC input must be [N,C,E,B] or [N,V,E,B,M], found {tuple(x.shape)}")
