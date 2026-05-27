"""Tests for the scientific reasoning diary."""

from src.reasoning.diary import format_obs_table, format_candidate_table, format_alternatives


def test_format_obs_table():
    X = [[0.1, 0.5, 0.3, 0.8], [0.2, 0.6, 0.4, 0.9]]
    Y = [[0.9, 0.3, 0.7], [0.8, 0.4, 0.6]]
    param_names = ["a", "b", "c", "d"]
    obj_names = ["s", "c", "k"]
    table = format_obs_table(X, Y, param_names, obj_names)
    assert "a | b | c | d | s | c | k" in table
    assert "0.1000" in table
    assert "0.8000" in table


def test_format_candidate_table():
    candidates = [
        {"params": [0.3, 0.4], "objectives": [0.95, 0.2]},
        {"params": [0.7, 0.2], "objectives": [0.85, 0.3]},
    ]
    table = format_candidate_table(candidates, ["strength", "cost"])
    assert "Candidate 1" in table
    assert "Candidate 2" in table
    assert "0.9500" in table


def test_format_alternatives():
    alternatives = [
        {"params": [0.1, 0.9], "objectives": [0.5, 0.6]},
    ]
    param_names = ["x", "y"]
    obj_names = ["s", "c"]
    table = format_alternatives(alternatives, param_names, obj_names)
    assert "0.1000" in table
    assert "0.5000" in table
