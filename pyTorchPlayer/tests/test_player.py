import math
import pytest
import torch
from ogame_env.foo_torch import Player, Resources


def make_resources(m, c, d):
    return Resources(torch.tensor([m, c, d], dtype=torch.float64))


@pytest.mark.parametrize(
    "level, count",
    [
        (0, 1),
        (1, 2),
        (2, 2),
        (3, 3),
        (13, 8),
        (14, 8),
        (22, 12),
        (23, 13),
        (30, 16),
        (31, 17),
    ],
)
def test_player_planet_count(level, count):
    player = Player()
    for _ in range(level):
        player.astrophysics.upgrade()
    player._update_planets()
    assert len(player.planets) == count


def test_player_has_no_resources():
    player = Player()
    assert player.resources.values.tolist() == [0.0, 0.0, 0.0]


def test_player_gains_resources():
    player = Player()
    player.planets[0].metal_mine.upgrade()
    player.proceed_to_next_day()
    assert player.resources.values[0].item() == 792.0


def test_player_gains_modified_resources():
    player = Player()
    player.planets[0].metal_mine.upgrade()
    player.plasma.upgrade()
    player.proceed_to_next_day()
    assert player.resources.values[0].item() == 799.0


def test_player_spends_resources():
    player = Player()
    player.add_resources(make_resources(10, 20, 30))
    result = player.try_spend_resources(make_resources(1, 2, 3))
    assert result is True
    assert player.resources.values.tolist() == [126.0, 0.0, 0.0]


def test_player_spends_exact_resources():
    player = Player()
    player.add_resources(make_resources(1, 2, 3))
    result = player.try_spend_resources(make_resources(1, 2, 3))
    assert result is True
    assert player.resources.values.tolist() == [0.0, 0.0, 0.0]


def test_player_does_not_overspend_resources():
    player = Player()
    player.add_resources(make_resources(1, 2, 3))
    result = player.try_spend_resources(make_resources(10, 20, 30))
    assert result is False
    assert player.resources.values.tolist() == [1.0, 2.0, 3.0]
