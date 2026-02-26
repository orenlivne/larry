"""Tests for player configuration."""

import json
import tempfile
from pathlib import Path

import pytest

from larrybot.config import MAX_UCI_ELO, MIN_UCI_ELO, PlayerConfig, StyleVector


class TestStyleVector:

    def test_defaults(self):
        sv = StyleVector()
        assert sv.aggression == 0.5
        assert sv.material_weight == 0.5

    def test_custom_values(self):
        sv = StyleVector(aggression=0.8, centrality=0.2)
        assert sv.aggression == 0.8
        assert sv.centrality == 0.2

    def test_boundary_values(self):
        sv = StyleVector(aggression=0.0, material_weight=1.0)
        assert sv.aggression == 0.0
        assert sv.material_weight == 1.0

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            StyleVector(aggression=1.1)
        with pytest.raises(ValueError):
            StyleVector(centrality=-0.5)


class TestPlayerConfig:

    def test_target_elo(self):
        pc = PlayerConfig(player_name="Larry", classical_elo=2620, elo_offset=-200)
        assert pc.target_elo == 2420

    def test_clamped_elo_within_range(self):
        pc = PlayerConfig(player_name="Larry", classical_elo=2620, elo_offset=-200)
        assert pc.clamped_elo == 2420

    def test_clamped_elo_low(self):
        pc = PlayerConfig(player_name="Beginner", classical_elo=1000, elo_offset=0)
        assert pc.clamped_elo == MIN_UCI_ELO

    def test_clamped_elo_high(self):
        pc = PlayerConfig(player_name="Engine", classical_elo=3500, elo_offset=0)
        assert pc.clamped_elo == MAX_UCI_ELO

    def test_save_load_roundtrip(self):
        style = StyleVector(aggression=0.7, material_weight=0.3)
        config = PlayerConfig(
            player_name="TestPlayer",
            classical_elo=2500,
            elo_offset=-100,
            style=style,
            candidate_threshold_cp=60,
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            config.save(path)
            loaded = PlayerConfig.load(path)
            assert loaded.player_name == "TestPlayer"
            assert loaded.classical_elo == 2500
            assert loaded.elo_offset == -100
            assert loaded.target_elo == 2400
            assert loaded.style is not None
            assert loaded.style.aggression == 0.7
            assert loaded.candidate_threshold_cp == 60
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_load_no_style(self):
        config = PlayerConfig(player_name="NoStyle", classical_elo=2000)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            config.save(path)
            loaded = PlayerConfig.load(path)
            assert loaded.style is None
        finally:
            Path(path).unlink(missing_ok=True)
