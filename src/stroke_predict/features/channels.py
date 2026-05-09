from __future__ import annotations

from typing import Iterable

import numpy as np


_BASE_PAIRS = {
    "Fp1": "Fp2",
    "AF3": "AF4",
    "F7": "F8",
    "F5": "F6",
    "F3": "F4",
    "F1": "F2",
    "FT7": "FT8",
    "FC5": "FC6",
    "FC3": "FC4",
    "FC1": "FC2",
    "T7": "T8",
    "C5": "C6",
    "C3": "C4",
    "C1": "C2",
    "TP7": "TP8",
    "CP5": "CP6",
    "CP3": "CP4",
    "CP1": "CP2",
    "P7": "P8",
    "P5": "P6",
    "P3": "P4",
    "P1": "P2",
    "PO7": "PO8",
    "PO5": "PO6",
    "PO3": "PO4",
    "O1": "O2",
}

DEFAULT_CHANNEL_PAIR_MAP: dict[str, str] = {}
for _left, _right in _BASE_PAIRS.items():
    DEFAULT_CHANNEL_PAIR_MAP[_left] = _right
    DEFAULT_CHANNEL_PAIR_MAP[_right] = _left


def build_flip_indices(channels: Iterable[str], pair_map: dict[str, str] | None = None) -> list[int]:
    labels = [str(channel) for channel in channels]
    lookup = {label.upper(): index for index, label in enumerate(labels)}
    pairs = {key.upper(): value.upper() for key, value in (pair_map or DEFAULT_CHANNEL_PAIR_MAP).items()}
    indices: list[int] = []
    for label in labels:
        mapped = pairs.get(label.upper(), label.upper())
        indices.append(lookup.get(mapped, lookup[label.upper()]))
    return indices


def flip_psd_matrix(psd: np.ndarray, channels: Iterable[str], pair_map: dict[str, str] | None = None) -> np.ndarray:
    indices = build_flip_indices(channels, pair_map)
    return np.asarray(psd)[indices, ...]


def flip_fc_edges(
    edges: list[tuple[str, str]],
    values: np.ndarray,
    pair_map: dict[str, str] | None = None,
) -> tuple[list[tuple[str, str]], np.ndarray]:
    pairs = pair_map or DEFAULT_CHANNEL_PAIR_MAP
    remapped_edges = [_canonical_edge(_map_label(a, pairs), _map_label(b, pairs)) for a, b in edges]
    return remapped_edges, np.asarray(values).copy()


def should_flip_for_hand(hand: str | None) -> bool:
    return str(hand or "").strip().lower() == "left"


def normalized_channel(channel: str, hand: str | None, pair_map: dict[str, str] | None = None) -> str:
    if not should_flip_for_hand(hand):
        return channel
    return _map_label(channel, pair_map or DEFAULT_CHANNEL_PAIR_MAP)


def _map_label(label: str, pair_map: dict[str, str]) -> str:
    lookup = {key.upper(): value for key, value in pair_map.items()}
    return lookup.get(str(label).upper(), str(label))


def _canonical_edge(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b), key=lambda item: item.upper()))  # type: ignore[return-value]

