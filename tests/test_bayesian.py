"""Tests for the Bayesian optimization module."""

import torch
from src.bayesian_opt.search_space import DEFAULT_SPACE, MaterialSearchSpace
from src.bayesian_opt.optimizer import BayesianOptimizer


def test_search_space_bounds():
    bounds = DEFAULT_SPACE.bounds_tensor()
    assert bounds.shape == (2, 4)
    assert torch.all(bounds[0] < bounds[1])


def test_random_sample():
    samples = DEFAULT_SPACE.random_sample(20)
    assert samples.shape == (20, 4)
    lb, ub = DEFAULT_SPACE.bounds_tensor()
    assert torch.all(samples >= lb)
    assert torch.all(samples <= ub)


def test_synthetic_evaluation():
    X = DEFAULT_SPACE.random_sample(5)
    Y = DEFAULT_SPACE.evaluate_synthetic(X)
    assert Y.shape == (5, 3)


def test_optimizer_initialization():
    opt = BayesianOptimizer(search_space=DEFAULT_SPACE)
    assert opt.n_observations == 0


def test_optimizer_add_observation():
    opt = BayesianOptimizer(search_space=DEFAULT_SPACE)
    x = DEFAULT_SPACE.random_sample(1)[0]
    y = DEFAULT_SPACE.evaluate_synthetic(x.unsqueeze(0))[0]
    opt.add_observation(x, y)
    assert opt.n_observations == 1


def test_optimizer_suggest():
    opt = BayesianOptimizer(search_space=DEFAULT_SPACE)
    n_init = 8
    X_init = DEFAULT_SPACE.random_sample(n_init)
    Y_init = DEFAULT_SPACE.evaluate_synthetic(X_init)
    for xi, yi in zip(X_init, Y_init):
        opt.add_observation(xi, yi)
    candidates = opt.suggest_next(n_candidates=3)
    assert candidates.shape == (3, 4)
    lb, ub = DEFAULT_SPACE.bounds_tensor()
    assert torch.all(candidates >= lb)
    assert torch.all(candidates <= ub)


def test_full_optimization_loop():
    opt = BayesianOptimizer(search_space=DEFAULT_SPACE)
    result = opt.run_optimization_loop(n_initial=6, n_rounds=2, candidates_per_round=2)
    assert len(result["rounds"]) == 2
    assert len(result["final_X"]) >= 6 + 4
    assert "pareto_frontier" in result


def test_custom_search_space():
    from dataclasses import dataclass
    @dataclass
    class Param:
        name: str = "x"
        low: float = 0.0
        high: float = 1.0
        description: str = ""
    @dataclass
    class Obj:
        name: str = "y"
        minimize: bool = True
        description: str = ""

    space = MaterialSearchSpace(
        parameters=[Param()],
        objectives=[Obj()],
    )
    opt = BayesianOptimizer(search_space=space)
    X_init = space.random_sample(5)
    Y_init = space.evaluate_synthetic(X_init)
    assert Y_init.shape == (5, 1)
    for xi, yi in zip(X_init, Y_init):
        opt.add_observation(xi, yi)
    assert opt.n_observations == 5
