"""Multi-objective Bayesian optimization loop using BoTorch."""

import logging
from typing import Optional

import torch
from botorch.acquisition.multi_objective import qLogExpectedHypervolumeImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective import is_non_dominated
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from src.bayesian_opt.search_space import MaterialSearchSpace

logger = logging.getLogger(__name__)


class BayesianOptimizer:
    """Multi-objective Bayesian optimizer for material discovery."""

    def __init__(
        self,
        search_space: MaterialSearchSpace,
        X_init: Optional[torch.Tensor] = None,
        Y_init: Optional[torch.Tensor] = None,
    ):
        self.space = search_space
        self.device = torch.device("cpu")

        self.X = X_init if X_init is not None else torch.zeros(0, self.space.dim)
        self.Y = Y_init if Y_init is not None else torch.zeros(0, self.space.n_objectives)

    @property
    def n_observations(self) -> int:
        return self.X.shape[0]

    def add_observation(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Add a single or batch of observations."""
        x_2d = x.unsqueeze(0) if x.dim() == 1 else x
        y_2d = y.unsqueeze(0) if y.dim() == 1 else y
        self.X = torch.cat([self.X, x_2d], dim=0)
        self.Y = torch.cat([self.Y, y_2d], dim=0)

    def _build_model(self) -> ModelListGP:
        """Build a multi-output GP using a separate SingleTaskGP per objective."""
        n = self.n_observations
        if n < 3:
            raise ValueError(f"Need at least 3 observations, got {n}")

        models = []
        for i in range(self.space.n_objectives):
            model = SingleTaskGP(
                train_X=self.X,
                train_Y=self.Y[:, i : i + 1],
                outcome_transform=Standardize(m=1),
            )
            models.append(model)

        return ModelListGP(*models)

    def suggest_next(
        self,
        n_candidates: int = 3,
        raw_samples: int = 512,
        num_restarts: int = 10,
    ) -> torch.Tensor:
        """Suggest the next n_candidate experiments using qEHVI."""
        model = self._build_model()

        # Compute the Pareto frontier from observed data
        # BoTorch assumes minimization → negate maximization objectives
        Y_for_ref = self.Y.clone()
        for i, obj in enumerate(self.space.objectives):
            if not obj.minimize:
                Y_for_ref[:, i] = -Y_for_ref[:, i]

        partitioning = FastNondominatedPartitioning(
            ref_point=Y_for_ref.min(dim=0).values - 0.1,
            Y=Y_for_ref,
        )

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        acq = qLogExpectedHypervolumeImprovement(
            model=model,
            ref_point=partitioning.ref_point - 1e-4,
            partitioning=partitioning,
            sampler=sampler,
        )

        bounds = self.space.bounds_tensor()
        candidates, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=n_candidates,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={"batch_limit": 5, "maxiter": 200},
        )

        return candidates.detach()

    def run_optimization_loop(
        self,
        n_initial: int = 8,
        n_rounds: int = 5,
        candidates_per_round: int = 3,
    ) -> dict:
        """Run a full optimization loop with synthetic evaluations."""
        if self.n_observations == 0:
            X_init = self.space.random_sample(n_initial)
            Y_init = self.space.evaluate_synthetic(X_init)
            for xi, yi in zip(X_init, Y_init):
                self.add_observation(xi, yi)

        rounds = []
        for r in range(n_rounds):
            logger.info("Optimization round %d/%d", r + 1, n_rounds)
            X_next = self.suggest_next(n_candidates=candidates_per_round)
            Y_next = self.space.evaluate_synthetic(X_next)

            for xi, yi in zip(X_next, Y_next):
                self.add_observation(xi, yi)

            rounds.append({
                "round": r + 1,
                "suggested_params": X_next.tolist(),
                "suggested_objectives": Y_next.tolist(),
                "n_total_observations": self.n_observations,
            })

        return {
            "search_space": {
                "parameters": [p.name for p in self.space.parameters],
                "objectives": [o.name for o in self.space.objectives],
            },
            "rounds": rounds,
            "final_X": self.X.tolist(),
            "final_Y": self.Y.tolist(),
            "pareto_frontier": self._pareto_frontier(),
        }

    def _pareto_frontier(self) -> list[list[float]]:
        """Return the Pareto-optimal points from observed data."""
        if self.n_observations == 0:
            return []
        Y = self.Y.clone()
        for i, obj in enumerate(self.space.objectives):
            if not obj.minimize:
                Y[:, i] = -Y[:, i]
        pareto_mask = is_non_dominated(Y)
        return self.Y[pareto_mask].tolist()
