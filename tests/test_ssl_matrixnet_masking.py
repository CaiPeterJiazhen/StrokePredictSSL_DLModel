from __future__ import annotations

import torch

from stroke_predict.ssl_matrixnet import generate_mask, masked_mse_loss


def test_mask_generator_produces_expected_shape_and_ratio() -> None:
    mask = generate_mask((4, 5), mask_ratio=0.25, seed=7)

    assert mask.shape == (4, 5)
    assert mask.dtype == torch.bool
    assert int(mask.sum().item()) == 5


def test_mask_generator_is_deterministic_for_seed() -> None:
    first = generate_mask((2, 3, 4), mask_ratio=0.40, seed=11)
    second = generate_mask((2, 3, 4), mask_ratio=0.40, seed=11)
    different = generate_mask((2, 3, 4), mask_ratio=0.40, seed=12)

    assert torch.equal(first, second)
    assert not torch.equal(first, different)


def test_masked_reconstruction_loss_uses_only_masked_elements() -> None:
    prediction = torch.tensor([[1.0, 100.0], [3.0, 4.0]])
    target = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    mask = torch.tensor([[True, False], [True, False]])

    loss = masked_mse_loss(prediction, target, mask)

    expected = torch.tensor(((1.0 - 0.0) ** 2 + (3.0 - 1.0) ** 2) / 2.0)
    assert torch.isclose(loss, expected)


def test_unmasked_entries_do_not_contribute_to_loss() -> None:
    target = torch.zeros((2, 2))
    mask = torch.tensor([[True, False], [False, True]])
    calm_prediction = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    noisy_unmasked_prediction = torch.tensor([[1.0, 9999.0], [-9999.0, 1.0]])

    calm_loss = masked_mse_loss(calm_prediction, target, mask)
    noisy_loss = masked_mse_loss(noisy_unmasked_prediction, target, mask)

    assert torch.isclose(calm_loss, noisy_loss)
