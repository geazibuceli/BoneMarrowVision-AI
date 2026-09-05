import random

import numpy as np
import torch

from bone_marrow_vision.training.reproducibility import (
    capture_rng_state,
    make_generator,
    restore_rng_state,
    seed_everything,
)


def _draw_values():
    return random.random(), float(np.random.random()), torch.rand(3)


def test_seed_everything_reproduces_all_rngs():
    seed_everything(123)
    first = _draw_values()
    seed_everything(123)
    second = _draw_values()
    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])
    assert torch.are_deterministic_algorithms_enabled()


def test_rng_state_round_trip_is_weights_only_safe(tmp_path):
    seed_everything(321)
    state = capture_rng_state()
    expected = _draw_values()
    restore_rng_state(state)
    actual = _draw_values()
    assert expected[:2] == actual[:2]
    assert torch.equal(expected[2], actual[2])

    path = tmp_path / "rng.pt"
    torch.save({"rng": state}, path)
    loaded = torch.load(path, weights_only=True)
    assert torch.equal(loaded["rng"]["torch_cpu"], state["torch_cpu"])


def test_seeded_generators_have_identical_sequences():
    first = torch.randperm(20, generator=make_generator(9))
    second = torch.randperm(20, generator=make_generator(9))
    assert torch.equal(first, second)
