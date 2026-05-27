"""Material search space definitions for Bayesian optimization."""

from dataclasses import dataclass, field

import torch


@dataclass
class Parameter:
    name: str
    low: float
    high: float
    description: str = ""


@dataclass
class Objective:
    name: str
    minimize: bool = False  # False = maximize
    description: str = ""


@dataclass
class MaterialSearchSpace:
    """Defines a constrained material formulation space."""

    parameters: list[Parameter] = field(default_factory=list)
    objectives: list[Objective] = field(default_factory=list)

    @property
    def dim(self) -> int:
        return len(self.parameters)

    @property
    def n_objectives(self) -> int:
        return len(self.objectives)

    def bounds_tensor(self) -> torch.Tensor:
        """Return (2, d) tensor of [low, high] bounds."""
        lows = [p.low for p in self.parameters]
        highs = [p.high for p in self.parameters]
        return torch.tensor([lows, highs], dtype=torch.float64)

    def random_sample(self, n: int = 10) -> torch.Tensor:
        """Draw uniform random samples from the search space."""
        lb, ub = self.bounds_tensor()
        return lb + (ub - lb) * torch.rand(n, self.dim)

    def evaluate_synthetic(self, X: torch.Tensor) -> torch.Tensor:
        """Synthetic objective functions for demonstration.

        Uses available dimensions adaptively; fills missing dimensions with 0.
        """
        d = X.shape[-1]
        cols = [X[:, i] if i < d else torch.zeros_like(X[:, 0]) for i in range(4)]

        strength = (
            0.4 * torch.sin(3.0 * cols[0]) * torch.cos(2.5 * cols[1])
            + 0.3 * torch.exp(-((cols[2] - 0.5) ** 2) / 0.1)
            - 0.2 * cols[3]
            + 0.5 * torch.sigmoid(5.0 * (cols[0] - 0.3))
        )

        cost = (
            0.6 * cols[0]
            + 0.3 * cols[1] ** 2
            + 0.2 * cols[2]
            + 0.1 * cols[3]
            + 0.15 * torch.exp(-((cols[0] - 0.8) ** 2) / 0.05)
        )

        conductivity = (
            0.5 * torch.sin(4.0 * cols[2]) * torch.cos(2.0 * cols[0])
            - 0.3 * (cols[1] - 0.4) ** 2
            + 0.4 * cols[3]
            + 0.2
        )

        objectives = [strength, -cost, conductivity]
        n_obj = self.n_objectives
        return torch.stack(objectives[:n_obj], dim=-1)


# Default search space for demonstration
DEFAULT_SPACE = MaterialSearchSpace(
    parameters=[
        Parameter("precursor_ratio_A_B", 0.1, 1.0, "Molar ratio of precursor A to precursor B"),
        Parameter("annealing_temp", 0.0, 1.0, "Annealing temperature (normalized 100-500°C → 0-1)"),
        Parameter("doping_concentration", 0.0, 1.0, "Dopant concentration (normalized 0-10 mol%)"),
        Parameter("solvent_ratio", 0.0, 1.0, "Co-solvent volume ratio"),
    ],
    objectives=[
        Objective("mechanical_strength", minimize=False, description="Material tensile strength (MPa)"),
        Objective("production_cost", minimize=True, description="Normalized production cost"),
        Objective("ionic_conductivity", minimize=False, description="Ionic conductivity (mS/cm)"),
    ],
)
