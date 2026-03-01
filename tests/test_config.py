"""Tests for player configuration."""

import json
import tempfile
from pathlib import Path

import pytest

from larrybot.config import MAX_UCI_ELO, MIN_UCI_ELO, PlayerConfig, StyleVector, convert_to_fide, elo_to_temperature


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


class TestEloToTemperature:

    def test_low_elo_high_temperature(self):
        temp = elo_to_temperature(1320)
        assert temp > 0.8

    def test_high_elo_low_temperature(self):
        temp = elo_to_temperature(3190)
        assert temp < 0.1

    def test_monotonically_decreasing(self):
        """Higher ELO should always produce lower temperature."""
        elos = [1320, 1500, 1800, 2000, 2200, 2500, 2800, 3190]
        temps = [elo_to_temperature(e) for e in elos]
        for i in range(len(temps) - 1):
            assert temps[i] > temps[i + 1], f"temp({elos[i]}) should be > temp({elos[i+1]})"

    def test_never_negative(self):
        for elo in range(MIN_UCI_ELO, MAX_UCI_ELO + 1, 100):
            assert elo_to_temperature(elo) > 0

    def test_mid_range_reasonable(self):
        """2000 ELO should have moderate temperature (not too low, not too high)."""
        temp = elo_to_temperature(2000)
        assert 0.3 < temp < 0.8


class TestConvertToFide:

    def test_fide_is_identity(self):
        assert convert_to_fide(2000, "fide") == 2000
        assert convert_to_fide(1500, "fide") == 1500

    def test_lichess_lower_than_input(self):
        """Lichess ratings are inflated vs FIDE."""
        for rating in [1500, 1800, 2000, 2200]:
            assert convert_to_fide(rating, "lichess") < rating

    def test_chesscom_lower_than_input(self):
        for rating in [1500, 1800, 2000, 2200]:
            assert convert_to_fide(rating, "chesscom") < rating

    def test_uscf_lower_than_input(self):
        for rating in [1500, 1800, 2000, 2200]:
            assert convert_to_fide(rating, "uscf") < rating

    def test_unknown_system_raises(self):
        with pytest.raises(ValueError, match="Unknown rating system"):
            convert_to_fide(2000, "icc")

    def test_lichess_known_values(self):
        """Spot-check against ChessGoals cross-rating data."""
        # Lichess rapid 2085 ≈ FIDE 1816
        assert abs(convert_to_fide(2085, "lichess") - 1816) <= 5
        # Lichess rapid 1990 ≈ FIDE 1737
        assert abs(convert_to_fide(1990, "lichess") - 1737) <= 5

    def test_deflation_ordering(self):
        """Lichess inflates the most, then chess.com/USCF, then FIDE (no change)."""
        rating = 2000
        fide = convert_to_fide(rating, "fide")
        uscf = convert_to_fide(rating, "uscf")
        chesscom = convert_to_fide(rating, "chesscom")
        lichess = convert_to_fide(rating, "lichess")
        assert lichess < chesscom <= uscf < fide
