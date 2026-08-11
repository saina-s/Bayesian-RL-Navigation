"""
standalone program for Figures 8 and 9 of:

    Alali & Imani (2024), "Bayesian reinforcement learning for
    navigation planning in unknown environments".

implementation:

1. the figure 8 (4x4) and Figure 9 (6x6) mazes
2. their 27 possible environment models
3. bayesian posterior updates over environment models (paper Eq. 9)
4. the belief-MDP transition simulator (paper Eqs. 8-11 and 23)
5. the injury-location reward with auxiliary eta variables
6. the proposed DQN policy (paper Eqs. 15-20 and reported hyperparameters)
7. model specific DQN policies used by Baseline, MAP, and Active Learning
   (paper Eqs. 2, 21, and 22)
8. evaluation over independent trials and 95% confidence intervals
9. layout and policy-comparison plots
10. internal self-tests -> to make sure Of the algorithms
------------------

paper experiment settings:

both Figure 8 and Figure 9 use three unknown cells, a uniform prior,
the movement probabilities 0.8/0.1/0.1, 5,000 training episodes,
a training horizon of 250, and evaluation over 1,000 trials and 50 steps.
-----------------

important implementation boundaries:
paper does not publish every low-level implementation choice.
This program uses these assumptions (instead of hiding them):

* robot position is one hot encoded
* collision with an unknown wall produces an explicit WALL observation
* MAP ties use a CLI-controlled convention. The recommended reproducible
  convention is ``sticky-random`` because the paper does not specify one
* Baseline/MAP/Active Learning use one model-specific DQN per environment
* comparator DQNs use a common seed by default to avoid arbitrary asymmetry
  in the posterior weighted average of Eq. 22.
* confidence intervals use mean +/- 1.96 * sample_std/sqrt(N).
----------------------

Figure 8:
python final_bayesian_navigation.py --figure 8 --mode self-test
python final_bayesian_navigation.py --figure 8 --mode train-proposed --episodes 5000 --horizon 250
python final_bayesian_navigation.py --figure 8 --mode evaluate-proposed --trials 1000 --eval-horizon 50 --base-eval-seed 10000
python final_bayesian_navigation.py --figure 8 --mode train-neural-comparators --comparator-episodes 5000 --comparator-horizon 250
python final_bayesian_navigation.py --figure 8 --mode evaluate-all --map-tie-rule sticky-random --trials 1000 --eval-horizon 50

Figure 9:
python final_bayesian_navigation.py --figure 9 --mode self-test
python final_bayesian_navigation.py --figure 9 --mode plot-layout
python final_bayesian_navigation.py --figure 9 --mode train-proposed --episodes 5000 --horizon 250
python final_bayesian_navigation.py --figure 9 --mode evaluate-proposed --trials 1000 --eval-horizon 50 --base-eval-seed 10000
python final_bayesian_navigation.py --figure 9 --mode train-neural-comparators --comparator-episodes 5000 --comparator-horizon 250
python final_bayesian_navigation.py --figure 9 --mode evaluate-all --map-tie-rule sticky-random --trials 1000 --eval-horizon 50

* only NumPy, PyTorch, and Matplotlib are required
----------------------
few notes:

* as the paper says, if the posterior distribution is uniform, 
"""

from __future__ import annotations
import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from enum import Enum, IntEnum
from itertools import product
from pathlib import Path
from typing import Iterable, Protocol, Sequence
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# Reproducibility

def set_global_seed(seed: int) -> None:
    """seed Python, NumPy, and PyTorch once for a complete run"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if you have cuda, it will use it but CPU is also ok
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    return device



# maze and environment models

Position = tuple[int, int]
EnvironmentModel = tuple["CellType", ...]

class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


ACTION_DELTA: dict[Action, Position] = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}

# considering uncertainty
PERPENDICULAR_ACTIONS: dict[Action, tuple[Action, Action]] = {
    Action.UP: (Action.LEFT, Action.RIGHT),
    Action.DOWN: (Action.LEFT, Action.RIGHT),
    Action.LEFT: (Action.UP, Action.DOWN),
    Action.RIGHT: (Action.UP, Action.DOWN),
}


class CellType(str, Enum):
    WALL = "W"
    EMPTY = "E"
    INJURY = "I"


@dataclass(frozen=True)
class MazeConfig:
    rows: int
    cols: int
    start: Position
    fixed_walls: tuple[Position, ...]
    unknown_cells: tuple[Position, ...]
    intended_probability: float = 0.8
    side_probability: float = 0.1

# not necessary, but to make sure there is no mistake and everything is acceptable
    def __post_init__(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("Maze dimensions must be positive.")

        if not self.contains(self.start):
            raise ValueError("Start position lies outside the maze.")

        if len(set(self.fixed_walls)) != len(self.fixed_walls):
            raise ValueError("Fixed walls must be unique.")

        if len(set(self.unknown_cells)) != len(self.unknown_cells):
            raise ValueError("Unknown cells must be unique.")

        if set(self.fixed_walls) & set(self.unknown_cells):
            raise ValueError("Fixed walls and unknown cells cannot overlap.")

        for position in (*self.fixed_walls, *self.unknown_cells):
            if not self.contains(position):
                raise ValueError(f"Maze position is outside grid: {position}")

        if self.start in self.fixed_walls or self.start in self.unknown_cells:
            raise ValueError("Start must be a known free cell.")

        total = self.intended_probability + 2.0 * self.side_probability
        if not np.isclose(total, 1.0):
            raise ValueError("Movement probabilities must sum to one.")

        if self.intended_probability < 0.0 or self.side_probability < 0.0:
            raise ValueError("Movement probabilities cannot be negative.")

    def contains(self, position: Position) -> bool:
        row, col = position
        return 0 <= row < self.rows and 0 <= col < self.cols


class MazeEnvironment:
    """stochastic movement model used by Eq. 23."""

    def __init__(self, config: MazeConfig) -> None:
        self.config = config
        self._unknown_index = {
            position: index for index, position in enumerate(config.unknown_cells)
        }

    def action_outcomes(self, selected_action: Action | int) -> tuple[tuple[Action, float], ...]:
        action = Action(selected_action)
        side_1, side_2 = PERPENDICULAR_ACTIONS[action]
        return (
            (action, self.config.intended_probability),
            (side_1, self.config.side_probability),
            (side_2, self.config.side_probability),
        )

    @staticmethod
    def candidate_position(position: Position, action: Action | int) -> Position:
        delta = ACTION_DELTA[Action(action)]
        return position[0] + delta[0], position[1] + delta[1]

    def cell_type(self, position: Position, model: EnvironmentModel) -> CellType:
        if position in self.config.fixed_walls:
            return CellType.WALL

        unknown_index = self._unknown_index.get(position)
        if unknown_index is not None:
            return model[unknown_index]

        return CellType.EMPTY


@dataclass(frozen=True)
class NavigationScenario:
    figure_number: int
    config: MazeConfig
    models: tuple[EnvironmentModel, ...]
    model_prior: np.ndarray
    true_model: EnvironmentModel

    @property
    def slug(self) -> str:
        return f"figure{self.figure_number}"

    @property
    def true_injury_positions(self) -> tuple[Position, ...]:
        return tuple(
            position
            for position, cell_type in zip(
                self.config.unknown_cells, self.true_model, strict=True
            )
            if cell_type is CellType.INJURY
        )

    @property
    def true_wall_positions(self) -> tuple[Position, ...]:
        return tuple(
            position
            for position, cell_type in zip(
                self.config.unknown_cells, self.true_model, strict=True
            )
            if cell_type is CellType.WALL
        )


# making all possible models (27 in figure 8 and 9)
def generate_models(number_of_unknown_cells: int) -> tuple[EnvironmentModel, ...]:
    # just to make sure
    if number_of_unknown_cells <= 0:
        raise ValueError("Number of unknown cells must be positive.")

    # model order is deterministic: W, E, I for each cell.
    return tuple(
        tuple(values)
        for values in product(
            (CellType.WALL, CellType.EMPTY, CellType.INJURY),
            repeat=number_of_unknown_cells,
        )
    )


# initializing the prior
def independent_cell_priors_to_model_prior(
    cell_priors: Sequence[Sequence[float]],
    models: Sequence[EnvironmentModel],
) -> np.ndarray:
    prior_arrays = [np.asarray(values, dtype=np.float64) for values in cell_priors]
    # again, just to make sure everything is okay and to avoid errors
    for prior in prior_arrays:
        if prior.shape != (3,):
            raise ValueError("Each cell prior must contain [P(W), P(E), P(I)].")
        if np.any(prior < 0.0) or not np.isclose(prior.sum(), 1.0):
            raise ValueError("Each cell prior must be a probability vector.")

    type_index = {
        CellType.WALL: 0,
        CellType.EMPTY: 1,
        CellType.INJURY: 2,
    }

    probabilities = np.asarray(
        [
            math.prod(
                prior_arrays[cell_index][type_index[cell_type]]
                for cell_index, cell_type in enumerate(model)
            )
            for model in models
        ],
        dtype=np.float64,
    )

    probabilities /= probabilities.sum() # to have sum 1
    probabilities.setflags(write=False)
    return probabilities


def create_figure8_scenario() -> NavigationScenario:
    """from Figure 8A/8B:
    Row 0 is the top row and column 0 is the left column.
    Unknown-cell ordering:
        cell 1 -> (3, 3)
        cell 2 -> (3, 1)
        cell 3 -> (2, 0)
    True model: [I, W, I].
    """

    config = MazeConfig(
        rows=4,
        cols=4,
        start=(0, 0),
        fixed_walls=((1, 2), (2, 1), (2, 2)),
        unknown_cells=((3, 3), (3, 1), (2, 0)),
        intended_probability=0.8,
        side_probability=0.1,
    )

    models = generate_models(3)
    model_prior = independent_cell_priors_to_model_prior(
        cell_priors=((1 / 3, 1 / 3, 1 / 3),) * 3,
        models=models,
    )

    return NavigationScenario(
        figure_number=8,
        config=config,
        models=models,
        model_prior=model_prior,
        true_model=(CellType.INJURY, CellType.WALL, CellType.INJURY),
    )


def create_figure9_scenario() -> NavigationScenario:
    """from Figure 9A/9B.
    row 0 is the top row and column 0 is the left column.
    paper shows seven fixed walls and 29 possible robot positions.
    unknown-cell ordering follows the labels in Figure 9A:
        cell 1 -> (2, 2)
        cell 2 -> (4, 0)
        cell 3 -> (3, 0)

    The true model in Figure 9B is [W, I, I].
    """

    config = MazeConfig(
        rows=6,
        cols=6,
        start=(5, 4),
        fixed_walls=(
            (1, 3),
            (1, 4),
            (2, 3),
            (3, 3),
            (4, 1),
            (4, 2),
            (4, 3),
        ),
        unknown_cells=(
            (2, 2),
            (4, 0),
            (3, 0),
        ),
        intended_probability=0.8,
        side_probability=0.1,
    )

    models = generate_models(3)
    model_prior = independent_cell_priors_to_model_prior(
        cell_priors=((1 / 3, 1 / 3, 1 / 3),) * 3,
        models=models,
    )

    return NavigationScenario(
        figure_number=9,
        config=config,
        models=models,
        model_prior=model_prior,
        true_model=(CellType.WALL, CellType.INJURY, CellType.INJURY),
    )

# can be updated and add extra figures :)
def create_scenario(figure_number: int) -> NavigationScenario:
    if figure_number == 8:
        return create_figure8_scenario()
    if figure_number == 9:
        return create_figure9_scenario()
    raise ValueError("Only Figure 8 and Figure 9 are supported.")



# observation likelihood and Bayes update (paper Eqs. 9 and 23)
@dataclass(frozen=True)
class MazeObservation:
    next_position: Position
    observed_cell: Position | None = None
    observed_type: CellType | None = None

     # to avoid mistake if we needed to update code or add extra figurse, also to make sure data is received
    def __post_init__(self) -> None:
        if (self.observed_cell is None) != (self.observed_type is None):
            raise ValueError("Observed cell and observed type must appear together.")


def observation_for_actual_action(
    environment: MazeEnvironment,
    state: Position,
    actual_action: Action | int,
    model: EnvironmentModel,
) -> MazeObservation:
    """
    observation produced by one realized movement direction
    """

    config = environment.config
    candidate = environment.candidate_position(state, actual_action)

    if not config.contains(candidate) or candidate in config.fixed_walls:
        return MazeObservation(next_position=state)

    unknown_index = environment._unknown_index.get(candidate)
    if unknown_index is None:
        return MazeObservation(next_position=candidate)

    cell_type = model[unknown_index]

    # we do not know how the paper observed an unknown wall, but we used this; if the unknow:
    # 1. robot stays in the previous positions
    # 2. robot will be sure the cell is wall (prob = 1)
    if cell_type is CellType.WALL:
        return MazeObservation(
            next_position=state,
            observed_cell=candidate,
            observed_type=CellType.WALL,
        )

    return MazeObservation(
        next_position=candidate,
        observed_cell=candidate,
        observed_type=cell_type,
    )


def observation_distribution(
    environment: MazeEnvironment,
    state: Position,
    selected_action: Action | int,
    model: EnvironmentModel,
) -> dict[MazeObservation, float]:
    distribution: dict[MazeObservation, float] = {}

    for actual_action, probability in environment.action_outcomes(selected_action):
        if probability <= 0.0:
            continue

        observation = observation_for_actual_action(
            environment=environment,
            state=state,
            actual_action=actual_action,
            model=model,
        )

        distribution[observation] = distribution.get(observation, 0.0) + probability

    total_probability = float(sum(distribution.values()))
    # make sure to have sum prob 1
    if not np.isclose(total_probability, 1.0):
        raise RuntimeError(
            f"Observation probabilities sum to {total_probability}, not one."
        )

    return distribution # it is a dictionary of obseravtions and their probs

# P(o|s, a, \theta_{i}) for each model theta_{i}
# in fact, we are considering the possibility of the observation in each model
def observation_likelihoods(
    environment: MazeEnvironment,
    models: Sequence[EnvironmentModel],
    state: Position,
    selected_action: Action | int,
    observation: MazeObservation,
) -> np.ndarray:
    return np.asarray(
        [
            observation_distribution(
                environment=environment,
                state=state,
                selected_action=selected_action,
                model=model,
            ).get(observation, 0.0)
            for model in models
        ],
        dtype=np.float64,
    )


# updating bayesian prior (this is simply bayes rule)
def bayesian_posterior_update(prior: np.ndarray, likelihoods: np.ndarray) -> np.ndarray:
    prior_array = np.asarray(prior, dtype=np.float64)
    likelihood_array = np.asarray(likelihoods, dtype=np.float64)

    if prior_array.shape != likelihood_array.shape:
        raise ValueError("Prior and likelihood vectors must have the same shape.")

    unnormalized = prior_array * likelihood_array
    normalizer = float(unnormalized.sum())

    if normalizer <= 0.0 or not np.isfinite(normalizer):
        raise ValueError("Observation is impossible under the current posterior.")

    posterior = unnormalized / normalizer
    posterior = np.clip(posterior, 0.0, 1.0)
    posterior /= posterior.sum()
    return posterior



# belief state, eta tracking, and belief-MDP simulator (paper Eqs. 8-11)

# explanation:
# belief state in this code has 3 parts: 1.position  2.eta  3. posterior (on 27 models)
# eta = [1,1,1] shows if there is an injury or not. if eta_{i} = 1 it means not checked or not found but eta_{i} = 0 means the injury in this cell is already found (to avoid rewarding multiple times)

@dataclass(frozen=True)
class InjurySearchState:
    position: Position
    model_probabilities: np.ndarray
    eta: np.ndarray

    def __post_init__(self) -> None:
        probabilities = np.array(
            self.model_probabilities, dtype=np.float64, copy=True
        )
        eta = np.array(self.eta, dtype=np.int8, copy=True)

        #again, just to make sure
        if probabilities.ndim != 1 or probabilities.size == 0:
            raise ValueError("Posterior must be a non-empty vector.")
        if np.any(probabilities < 0.0) or not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("Posterior must be a normalized probability vector.")
        if eta.ndim != 1 or eta.size == 0 or not np.all(np.isin(eta, (0, 1))):
            raise ValueError("Eta must be a non-empty binary vector.")

        probabilities.setflags(write=False)
        eta.setflags(write=False)

        object.__setattr__(self, "model_probabilities", probabilities)
        object.__setattr__(self, "eta", eta)

    @property
    def located_injury_count(self) -> int:
        return int(np.count_nonzero(self.eta == 0))


@dataclass(frozen=True)
class InjurySearchOutcome:
    observation: MazeObservation
    next_state: InjurySearchState
    probability: float
    reward: float
    newly_located_injury_index: int | None

    def __post_init__(self) -> None:
        probability = float(self.probability)
        tolerance = 1e-12

        if not np.isfinite(probability):
            raise ValueError("Outcome probability must be finite.")
        if probability < -tolerance or probability > 1.0 + tolerance:
            raise ValueError("Outcome probability lies outside [0, 1].")

        object.__setattr__(
            self, "probability", float(np.clip(probability, 0.0, 1.0))
        )

        reward = float(self.reward)
        if reward not in (0.0, 1.0):
            raise ValueError("Injury reward must be zero or one.")
        object.__setattr__(self, "reward", reward)


# making belief ransitions
class BeliefInjurySimulator:
    """known belief-MDP simulator used for offline DQN training"""

    def __init__(
        self,
        environment: MazeEnvironment,
        models: Sequence[EnvironmentModel],
        unknown_cells: Sequence[Position],
        seed: int,
    ) -> None:
        self.environment = environment
        self.models = tuple(models)
        self.unknown_cells = tuple(unknown_cells)
        self._unknown_index = {
            position: index for index, position in enumerate(self.unknown_cells)
        }
        self._rng = np.random.default_rng(seed)

    def initial_state(self, position: Position, model_prior: np.ndarray) -> InjurySearchState:
        return InjurySearchState(
            position=position,
            model_probabilities=model_prior,
            eta=np.ones(len(self.unknown_cells), dtype=np.int8),
        )

    def all_observations(
        self, state: InjurySearchState, action: Action | int
    ) -> tuple[MazeObservation, ...]:
        """return possible observations in a deterministic order.

        Python set is not be used here: hash iteration order can differ
        between processes, which changes the mapping from random draws to
        observations and so changes the complete DQN training run even
        when all numerical seeds are identical. Dict insertion order is
        deterministic because model order and action-outcome order are fixed (no uncertainty is considered here)
        """

        observations: dict[MazeObservation, None] = {}
        for model in self.models:
            for observation in observation_distribution(
                self.environment, state.position, action, model
            ):
                observations.setdefault(observation, None)
        return tuple(observations.keys())

    def _update_eta_reward(
        self, state: InjurySearchState, observation: MazeObservation
    ) -> tuple[np.ndarray, float, int | None]:
        next_eta = np.array(state.eta, copy=True)

        if observation.observed_type is not CellType.INJURY:
            return next_eta, 0.0, None

        if observation.observed_cell is None:
            raise RuntimeError("Injury observation lacks a cell position.")

        injury_index = self._unknown_index[observation.observed_cell]

        if next_eta[injury_index] == 0:
            return next_eta, 0.0, injury_index

        next_eta[injury_index] = 0
        return next_eta, 1.0, injury_index

    def outcomes(
        self, state: InjurySearchState, action: Action | int
    ) -> tuple[InjurySearchOutcome, ...]:
        outcomes: list[InjurySearchOutcome] = []

        for observation in self.all_observations(state, action):
            likelihoods = observation_likelihoods(
                environment=self.environment,
                models=self.models,
                state=state.position,
                selected_action=action,
                observation=observation,
            )
            # P(o|b, a) = sigma_{j}{P(\theta_{j}|b) * P(o | s, a, \theta_{j})}
            predictive_probability = float(
                np.dot(state.model_probabilities, likelihoods)
            )

            if predictive_probability <= 1e-15:
                continue

            if predictive_probability > 1.0 + 1e-12:
                raise RuntimeError("Predictive probability exceeds one.")

            predictive_probability = float(
                np.clip(predictive_probability, 0.0, 1.0)
            )

            next_posterior = bayesian_posterior_update(
                state.model_probabilities, likelihoods
            )
            next_eta, reward, injury_index = self._update_eta_reward(
                state, observation
            )

            outcomes.append(
                InjurySearchOutcome(
                    observation=observation,
                    next_state=InjurySearchState(
                        position=observation.next_position,
                        model_probabilities=next_posterior,
                        eta=next_eta,
                    ),
                    probability=predictive_probability,
                    reward=reward,
                    newly_located_injury_index=injury_index,
                )
            )

        probability_sum = float(sum(outcome.probability for outcome in outcomes))
        if not np.isclose(probability_sum, 1.0):
            raise RuntimeError(
                f"Belief transition probabilities sum to {probability_sum}."
            )

        return tuple(outcomes)

    def sample(
        self, state: InjurySearchState, action: Action | int
    ) -> InjurySearchOutcome:
        # choosing one outcomes in belief transition due to their probability
        outcomes = self.outcomes(state, action)
        probabilities = np.asarray(
            [outcome.probability for outcome in outcomes], dtype=np.float64
        )
        probabilities /= probabilities.sum()
        selected_index = int(self._rng.choice(len(outcomes), p=probabilities))
        return outcomes[selected_index]

"""
difference between BeliefInjurySimulator and TrueEnvironmentSimulator:
1. BeliefInjurySimulator: for training proposed and doesn't have a real constant model. 
and transition is done with a combo of posterior if all models (27 models in here)

2. TrueEnvironmentSimulator: to evaluate. the real environment is set ( but the agent does not know).
policy just observe outcomes and update its posterior
"""

class TrueEnvironmentSimulator:
    """evaluation simulator with one fixed hidden true model"""

    def __init__(
        self,
        environment: MazeEnvironment,
        models: Sequence[EnvironmentModel],
        true_model: EnvironmentModel,
        unknown_cells: Sequence[Position],
    ) -> None:
        self.environment = environment
        self.models = tuple(models)
        self.true_model = tuple(true_model)
        self.unknown_cells = tuple(unknown_cells)
        self._unknown_index = {
            position: index for index, position in enumerate(self.unknown_cells)
        }
        self._rng = np.random.default_rng(0)

    def reseed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def initial_state(self, position: Position, model_prior: np.ndarray) -> InjurySearchState:
        return InjurySearchState(
            position=position,
            model_probabilities=model_prior,
            eta=np.ones(len(self.unknown_cells), dtype=np.int8),
        )

    def step(self, state: InjurySearchState, action: Action | int) -> InjurySearchOutcome:
        distribution = observation_distribution(
            self.environment, state.position, action, self.true_model
        )
        observations = tuple(distribution)
        probabilities = np.asarray(
            [distribution[observation] for observation in observations],
            dtype=np.float64,
        )
        probabilities /= probabilities.sum()

        selected_index = int(
            self._rng.choice(len(observations), p=probabilities)
        )
        observation = observations[selected_index]

        likelihoods = observation_likelihoods(
            environment=self.environment,
            models=self.models,
            state=state.position,
            selected_action=action,
            observation=observation,
        )
        next_posterior = bayesian_posterior_update(
            state.model_probabilities, likelihoods
        )

        next_eta = np.array(state.eta, copy=True)
        reward = 0.0
        injury_index: int | None = None

        if observation.observed_type is CellType.INJURY:
            if observation.observed_cell is None:
                raise RuntimeError("Injury observation lacks cell position.")
            injury_index = self._unknown_index[observation.observed_cell]
            if next_eta[injury_index] == 1:
                next_eta[injury_index] = 0
                reward = 1.0

        return InjurySearchOutcome(
            observation=observation,
            next_state=InjurySearchState(
                position=observation.next_position,
                model_probabilities=next_posterior,
                eta=next_eta,
            ),
            probability=float(probabilities[selected_index]),
            reward=reward,
            newly_located_injury_index=injury_index,
        )



# state encoding and DQN (paper Eqs. 15-19)
# belief state must be a vector for NN

class StateEncoder:
    """one-hot(position) + eta + posterior.

    paper defines the belief state but does not describe a neural input
    encoding so we used one-hot 
    """

    def __init__(self, config: MazeConfig, number_of_models: int) -> None:
        self.config = config
        self.number_of_models = number_of_models
        self.positions = tuple(
            (row, col)
            for row in range(config.rows)
            for col in range(config.cols)
            if (row, col) not in config.fixed_walls
        )
        self.position_to_index = {
            position: index for index, position in enumerate(self.positions)
        }
        self.number_of_unknown_cells = len(config.unknown_cells)

    @property
    def input_size(self) -> int:
        return len(self.positions) + self.number_of_unknown_cells + self.number_of_models

    def encode(self, state: InjurySearchState) -> np.ndarray:
        if state.position not in self.position_to_index:
            raise ValueError(f"Cannot encode invalid position {state.position}.")
        if state.model_probabilities.size != self.number_of_models:
            raise ValueError("Posterior size does not match encoder.")
        if state.eta.size != self.number_of_unknown_cells:
            raise ValueError("Eta size does not match encoder.")

        position_features = np.zeros(len(self.positions), dtype=np.float32)
        position_features[self.position_to_index[state.position]] = 1.0

        return np.concatenate(
            (
                position_features,
                np.asarray(state.eta, dtype=np.float32),
                np.asarray(state.model_probabilities, dtype=np.float32),
            )
        )

"""
Input -> Linear(input_size, 128) -> ReLU -> Linear(128, 128) -> ReLU -> Linear(128, 128) -> ReLU -> Linear(128, 4)
"""

class QNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        number_of_actions: int = 4,
        hidden_sizes: Sequence[int] = (128, 128, 128),
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.number_of_actions = int(number_of_actions)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)

        layers: list[nn.Module] = []
        previous_size = self.input_size
        for hidden_size in self.hidden_sizes:
            layers.extend((nn.Linear(previous_size, hidden_size), nn.ReLU()))
            previous_size = hidden_size
        layers.append(nn.Linear(previous_size, self.number_of_actions))
        self.network = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected final input dimension {self.input_size}, got {states.shape[-1]}."
            )
        return self.network(states)


def soft_update(q_network: QNetwork, target_network: QNetwork, tau: float) -> None:
    with torch.no_grad():
        for source, target in zip(
            q_network.parameters(), target_network.parameters(), strict=True
        ):
            target.mul_(1.0 - tau)
            target.add_(source, alpha=tau)


@dataclass(frozen=True)
class DQNConfig:
    learning_rate: float = 5e-4
    # gamma = discount
    gamma: float = 0.95
    epsilon: float = 0.1
    batch_size: int = 64
    replay_capacity: int = 100_000
    tau: float = 1e-3
    update_frequency: int = 4
    hidden_sizes: tuple[int, ...] = (128, 128, 128)


class ReplayBuffer:
    def __init__(self, capacity: int, state_size: int, seed: int) -> None:
        self.capacity = int(capacity)
        self.state_size = int(state_size)
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.size = 0
        self.next_index = 0
        self.rng = np.random.default_rng(seed)

    def add(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray
    ) -> None:
        self.states[self.next_index] = state
        self.actions[self.next_index] = action
        self.rewards[self.next_index] = reward
        self.next_states[self.next_index] = next_state
        self.next_index = (self.next_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        if self.size < batch_size:
            raise RuntimeError("Not enough replay samples.")
        # minibatch
        indices = self.rng.choice(self.size, size=batch_size, replace=False)
        return (
            self.states[indices].copy(),
            self.actions[indices].copy(),
            self.rewards[indices].copy(),
            self.next_states[indices].copy(),
        )


class DQNAgent:
    def __init__(
        self,
        encoder: StateEncoder,
        config: DQNConfig,
        device: torch.device,
        action_seed: int,
        replay_seed: int,
    ) -> None:
        self.encoder = encoder
        self.config = config
        self.device = device
        self.q_network = QNetwork(
            encoder.input_size, len(Action), config.hidden_sizes
        ).to(device)
        self.target_network = QNetwork(
            encoder.input_size, len(Action), config.hidden_sizes
        ).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        for parameter in self.target_network.parameters():
            parameter.requires_grad_(False)

        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=config.learning_rate
        )
        self.replay = ReplayBuffer(config.replay_capacity, encoder.input_size, replay_seed)
        self.rng = np.random.default_rng(action_seed)
        self.environment_steps = 0
        self.gradient_steps = 0

    def select_action(self, encoded_state: np.ndarray, explore: bool) -> int:
        # if training: explore = true and if evaluation: explore = false
        if explore and self.rng.random() < self.config.epsilon:
            return int(self.rng.integers(0, len(Action)))

        state_tensor = torch.as_tensor(
            encoded_state, dtype=torch.float32, device=self.device
        )
        self.q_network.eval()
        with torch.no_grad():
            values = self.q_network(state_tensor)
        self.q_network.train()
        return int(torch.argmax(values).item())

    def record(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray
    ) -> None:
        self.replay.add(state, action, reward, next_state)
        self.environment_steps += 1

    def maybe_update(self) -> float | None:
        if self.replay.size < self.config.batch_size:
            return None
        if self.environment_steps % self.config.update_frequency != 0:
            return None

        states, actions, rewards, next_states = self.replay.sample(
            self.config.batch_size
        )
        states_tensor = torch.as_tensor(
            states, dtype=torch.float32, device=self.device
        )
        actions_tensor = torch.as_tensor(
            actions, dtype=torch.int64, device=self.device
        )
        rewards_tensor = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.device
        )
        next_states_tensor = torch.as_tensor(
            next_states, dtype=torch.float32, device=self.device
        )

        current_values = self.q_network(states_tensor).gather(
            1, actions_tensor.unsqueeze(1)
        ).squeeze(1)

        # y = r + \gamma * max_{a'} Q_{target}(s', a')
        with torch.no_grad():
            maximum_next_values = self.target_network(next_states_tensor).max(1).values
            targets = rewards_tensor + self.config.gamma * maximum_next_values

        # loss = MSE(Q(s,a),y)
        loss = F.mse_loss(current_values, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        # soft update: w^(-) <- (1 - teu)*w^(-) + tau*w
        soft_update(self.q_network, self.target_network, self.config.tau)
        self.gradient_steps += 1
        return float(loss.item())


@dataclass(frozen=True)
class ProposedTrainingResult:
    episode_returns: np.ndarray
    final_injuries: np.ndarray
    losses: np.ndarray

# to train proposed:
#5000 episodes, 250 horizon, progress shows every 50 steps
# the reason horizon in training is 250: DQN should see further rewards and choose proper Q-values
def train_proposed_policy(
    scenario: NavigationScenario,
    episodes: int,
    horizon: int,
    device: torch.device,
    global_seed: int,
    transition_seed: int,
    replay_seed: int,
    action_seed: int,
    progress_every: int,
) -> tuple[DQNAgent, ProposedTrainingResult]:
    set_global_seed(global_seed)
    environment = MazeEnvironment(scenario.config)
    simulator = BeliefInjurySimulator(
        environment,
        scenario.models,
        scenario.config.unknown_cells,
        seed=transition_seed,
    )
    encoder = StateEncoder(scenario.config, len(scenario.models))
    dqn_config = DQNConfig()
    agent = DQNAgent(
        encoder,
        dqn_config,
        device,
        action_seed=action_seed,
        replay_seed=replay_seed,
    )

    returns = np.zeros(episodes, dtype=np.float64)
    final_injuries = np.zeros(episodes, dtype=np.int64)
    losses: list[float] = []

    for episode_index in range(episodes):
        state = simulator.initial_state(scenario.config.start, scenario.model_prior)
        episode_return = 0.0

        for _ in range(horizon):
            encoded = encoder.encode(state)
            action = agent.select_action(encoded, explore=True)
            outcome = simulator.sample(state, action)
            encoded_next = encoder.encode(outcome.next_state)
            agent.record(encoded, action, outcome.reward, encoded_next)
            loss = agent.maybe_update()
            if loss is not None:
                losses.append(loss)
            episode_return += outcome.reward
            state = outcome.next_state

        returns[episode_index] = episode_return
        final_injuries[episode_index] = state.located_injury_count

        completed = episode_index + 1
        if progress_every > 0 and (
            completed % progress_every == 0 or completed == episodes
        ):
            recent = returns[max(0, completed - progress_every) : completed]
            print(
                f"episodes={completed}/{episodes} "
                f"steps={agent.environment_steps} updates={agent.gradient_steps} "
                f"recent_mean_return={recent.mean():.3f}",
                flush=True,
            )

    return agent, ProposedTrainingResult(
        episode_returns=returns,
        final_injuries=final_injuries,
        losses=np.asarray(losses, dtype=np.float64),
    )



# model-specific MDP and tabular Q-learning (paper Eqs. 2, 21, 22)

@dataclass(frozen=True)
class ModelState:
    position: Position
    eta: tuple[int, ...]


@dataclass(frozen=True)
class ModelOutcome:
    next_state: ModelState
    probability: float
    reward: float

# consider model is const and known so there is no posterior
class ModelSpecificMDP:
    def __init__(
        self,
        environment: MazeEnvironment,
        model: EnvironmentModel,
        unknown_cells: Sequence[Position],
    ) -> None:
        self.environment = environment
        self.model = tuple(model)
        self.unknown_cells = tuple(unknown_cells)
        self._unknown_index = {
            position: index for index, position in enumerate(self.unknown_cells)
        }

        # shared state ordering across all models
        self.positions = tuple(
            (row, col)
            for row in range(environment.config.rows)
            for col in range(environment.config.cols)
            if (row, col) not in environment.config.fixed_walls
        )
        eta_values = tuple(product((0, 1), repeat=len(self.unknown_cells)))
        self.states = tuple(
            ModelState(position, tuple(int(value) for value in eta))
            for position in self.positions
            for eta in eta_values
        )
        self.state_to_index = {
            state: index for index, state in enumerate(self.states)
        }

    def initial_state(self, position: Position) -> ModelState:
        return ModelState(position, tuple(1 for _ in self.unknown_cells))

    def outcomes(self, state: ModelState, action: Action | int) -> tuple[ModelOutcome, ...]:
        distribution = observation_distribution(
            self.environment, state.position, action, self.model
        )
        outcomes: list[ModelOutcome] = []

        for observation, probability in distribution.items():
            eta = list(state.eta)
            reward = 0.0
            if observation.observed_type is CellType.INJURY:
                if observation.observed_cell is None:
                    raise RuntimeError("Injury observation lacks cell position.")
                index = self._unknown_index[observation.observed_cell]
                if eta[index] == 1:
                    eta[index] = 0
                    reward = 1.0

            outcomes.append(
                ModelOutcome(
                    next_state=ModelState(observation.next_position, tuple(eta)),
                    probability=float(probability),
                    reward=reward,
                )
            )

        return tuple(outcomes)


@dataclass(frozen=True)
class ComparatorConfig:
    episodes: int = 5000
    horizon: int = 250
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon: float = 0.1


@dataclass(frozen=True)
class ComparatorBank:
    models: tuple[EnvironmentModel, ...]
    states: tuple[ModelState, ...]
    q_values: np.ndarray
    visit_counts: np.ndarray
    model_seeds: np.ndarray
    config: ComparatorConfig

    def __post_init__(self) -> None:
        q_values = np.asarray(self.q_values, dtype=np.float64)
        visit_counts = np.asarray(self.visit_counts, dtype=np.int64)
        expected = (len(self.models), len(self.states), len(Action))
        if q_values.shape != expected or visit_counts.shape != expected:
            raise ValueError(f"Comparator bank must have shape {expected}.")

        object.__setattr__(self, "q_values", q_values)
        object.__setattr__(self, "visit_counts", visit_counts)
        object.__setattr__(
            self, "_model_to_index", {model: i for i, model in enumerate(self.models)}
        )
        object.__setattr__(
            self, "_state_to_index", {state: i for i, state in enumerate(self.states)}
        )

    def model_index(self, model: EnvironmentModel) -> int:
        return int(self._model_to_index[tuple(model)])

    def model_state(self, state: InjurySearchState) -> ModelState:
        return ModelState(state.position, tuple(int(value) for value in state.eta))

    def values_for_model(self, model_index: int, state: InjurySearchState) -> np.ndarray:
        state_index = self._state_to_index[self.model_state(state)]
        return self.q_values[model_index, state_index]

    def values_for_all_models(self, state: InjurySearchState) -> np.ndarray:
        state_index = self._state_to_index[self.model_state(state)]
        return self.q_values[:, state_index, :]


def random_argmax(values: np.ndarray, rng: np.random.Generator) -> int:
    maximum = float(np.max(values))
    candidates = np.flatnonzero(
        np.isclose(values, maximum, rtol=0.0, atol=1e-12)
    )
    return int(rng.choice(candidates))


def build_model_kernel(
    mdp: ModelSpecificMDP,
) -> tuple[tuple[tuple[tuple[int, float, float], ...], ...], ...]:
    kernel = []
    for state in mdp.states:
        action_rows = []
        for action in Action:
            rows = tuple(
                (
                    mdp.state_to_index[outcome.next_state],
                    outcome.probability,
                    outcome.reward,
                )
                for outcome in mdp.outcomes(state, action)
                if outcome.probability > 0.0
            )
            action_rows.append(rows)
        kernel.append(tuple(action_rows))
    return tuple(kernel)


def train_one_model_q_table(
    mdp: ModelSpecificMDP,
    initial_position: Position,
    config: ComparatorConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    kernel = build_model_kernel(mdp)
    q_values = np.zeros((len(mdp.states), len(Action)), dtype=np.float64)
    visit_counts = np.zeros_like(q_values, dtype=np.int64)
    initial_index = mdp.state_to_index[mdp.initial_state(initial_position)]

    for _ in range(config.episodes):
        state_index = initial_index
        for _ in range(config.horizon):
            if rng.random() < config.epsilon:
                action_index = int(rng.integers(0, len(Action)))
            else:
                action_index = random_argmax(q_values[state_index], rng)

            possible = kernel[state_index][action_index]
            if len(possible) == 1:
                next_index, _, reward = possible[0]
            else:
                probabilities = np.asarray([row[1] for row in possible], dtype=np.float64)
                probabilities /= probabilities.sum()
                selected = int(rng.choice(len(possible), p=probabilities))
                next_index, _, reward = possible[selected]

            target = reward + config.gamma * float(np.max(q_values[next_index]))
            error = target - q_values[state_index, action_index]
            q_values[state_index, action_index] += config.alpha * error
            visit_counts[state_index, action_index] += 1
            state_index = next_index

    return q_values, visit_counts


def train_comparator_bank(
    scenario: NavigationScenario,
    config: ComparatorConfig,
    base_seed: int,
) -> ComparatorBank:
    environment = MazeEnvironment(scenario.config)
    q_tables: list[np.ndarray] = []
    visits: list[np.ndarray] = []
    reference_states: tuple[ModelState, ...] | None = None
    model_seeds = base_seed + np.arange(len(scenario.models), dtype=np.int64)

    for model_index, model in enumerate(scenario.models):
        mdp = ModelSpecificMDP(environment, model, scenario.config.unknown_cells)
        if reference_states is None:
            reference_states = mdp.states
        elif mdp.states != reference_states:
            raise RuntimeError("Model-specific state ordering changed.")

        q_values, visit_counts = train_one_model_q_table(
            mdp,
            scenario.config.start,
            config,
            int(model_seeds[model_index]),
        )
        q_tables.append(q_values)
        visits.append(visit_counts)
        print(
            f"trained_model={model_index + 1}/{len(scenario.models)} "
            f"model={[cell.value for cell in model]} "
            f"visited_pairs={np.count_nonzero(visit_counts)}",
            flush=True,
        )

    if reference_states is None:
        raise RuntimeError("No comparator model was trained.")

    return ComparatorBank(
        models=scenario.models,
        states=reference_states,
        q_values=np.stack(q_tables),
        visit_counts=np.stack(visits),
        model_seeds=model_seeds,
        config=config,
    )



# model-specific neural Q-learning bank

class ModelStateEncoder:
    """one hot(position) + eta for known-model Q-functions q*_theta(s, a)

    The paper uses the same original task state for the model-specific policies
    in Equations (2), (21), and (22).  For the injury task this state contains
    the agent location and the eta variables, but not the model posterior.
    """

    def __init__(self, config: MazeConfig) -> None:
        self.config = config
        self.positions = tuple(
            (row, col)
            for row in range(config.rows)
            for col in range(config.cols)
            if (row, col) not in config.fixed_walls
        )
        self.position_to_index = {
            position: index for index, position in enumerate(self.positions)
        }
        self.number_of_unknown_cells = len(config.unknown_cells)

    @property
    def input_size(self) -> int:
        return len(self.positions) + self.number_of_unknown_cells

    def encode(self, state: ModelState | InjurySearchState) -> np.ndarray:
        position = state.position
        eta = np.asarray(state.eta, dtype=np.float32)

        if position not in self.position_to_index:
            raise ValueError(f"Cannot encode invalid model state position {position}.")
        if eta.shape != (self.number_of_unknown_cells,):
            raise ValueError("Model-state eta size does not match encoder.")

        position_features = np.zeros(len(self.positions), dtype=np.float32)
        position_features[self.position_to_index[position]] = 1.0
        return np.concatenate((position_features, eta))


@dataclass(frozen=True)
class NeuralComparatorTrainingResult:
    episode_returns: np.ndarray
    losses: np.ndarray
    environment_steps: int
    gradient_steps: int


class NeuralComparatorBank:
    """model-specific DQN approximations used by Baseline, MAP, and AL.

    Each network approximates q*_theta(s, a) for one fixed maze model.  This
    mirrors the paper's statement that MAP and Active Learning use the same
    offline trained model-specific policies.  It deliberately avoids the
    tabular approximation used in the first standalone draft.
    """

    def __init__(
        self,
        *,
        models: Sequence[EnvironmentModel],
        networks: Sequence[QNetwork],
        encoder: ModelStateEncoder,
        device: torch.device,
        dqn_config: DQNConfig,
        episodes: int,
        horizon: int,
        seed_mode: str,
        base_seed: int,
    ) -> None:
        self.models = tuple(tuple(model) for model in models)
        self.networks = tuple(network.to(device).eval() for network in networks)
        self.encoder = encoder
        self.device = device
        self.dqn_config = dqn_config
        self.episodes = int(episodes)
        self.horizon = int(horizon)
        self.seed_mode = str(seed_mode)
        self.base_seed = int(base_seed)

        if len(self.models) == 0 or len(self.networks) != len(self.models):
            raise ValueError("Neural comparator bank has inconsistent model/network counts.")

        self._model_to_index = {
            model: index for index, model in enumerate(self.models)
        }

    def model_index(self, model: EnvironmentModel) -> int:
        normalized = tuple(model)
        if normalized not in self._model_to_index:
            raise ValueError("Model is not present in neural comparator bank.")
        return int(self._model_to_index[normalized])

    def values_for_model(
        self, model_index: int, state: InjurySearchState
    ) -> np.ndarray:
        if not 0 <= model_index < len(self.networks):
            raise ValueError("Model index is outside neural comparator bank.")

        encoded = self.encoder.encode(state)
        tensor = torch.as_tensor(
            encoded, dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            values = self.networks[model_index](tensor)
        return values.detach().cpu().numpy().astype(np.float64, copy=False)

    def values_for_all_models(self, state: InjurySearchState) -> np.ndarray:
        encoded = self.encoder.encode(state)
        tensor = torch.as_tensor(
            encoded, dtype=torch.float32, device=self.device
        )
        rows: list[np.ndarray] = []
        with torch.no_grad():
            for network in self.networks:
                rows.append(
                    network(tensor).detach().cpu().numpy().astype(np.float64)
                )
        return np.stack(rows, axis=0)


def _comparator_model_seeds(
    *, base_seed: int, model_index: int, seed_mode: str
) -> dict[str, int]:
    """Return deterministic seeds for one model-specific DQN.

    common:
        Every model receives the same initialization, exploration, replay, and
        transition seeds.  This is the default because independent seeds can
        inject arbitrary asymmetry into Equation (22)'s posterior average.

    difference:
        model i receives an offset seed.  This mode is retained as a sensitivity
        analysis, not as the default reproduction setting.
    """

    if seed_mode not in {"common", "distinct"}:
        raise ValueError("Comparator seed mode must be 'common' or 'distinct'.")

    offset = 0 if seed_mode == "common" else 1000 * int(model_index)
    return {
        "global": int(base_seed + offset),
        "transition": int(base_seed + offset + 101),
        "replay": int(base_seed + offset + 202),
        "action": int(base_seed + offset + 303),
    }


def train_one_model_dqn(
    *,
    scenario: NavigationScenario,
    model: EnvironmentModel,
    device: torch.device,
    episodes: int,
    horizon: int,
    dqn_config: DQNConfig,
    seeds: dict[str, int],
    progress_every: int = 0,
) -> tuple[QNetwork, NeuralComparatorTrainingResult]:
    """Train q*_theta(s,a) for one fixed environment model using DQN."""

    set_global_seed(seeds["global"])
    environment = MazeEnvironment(scenario.config)
    mdp = ModelSpecificMDP(environment, model, scenario.config.unknown_cells)
    encoder = ModelStateEncoder(scenario.config)
    agent = DQNAgent(
        encoder,  # type: ignore[arg-type]
        dqn_config,
        device,
        action_seed=seeds["action"],
        replay_seed=seeds["replay"],
    )

    kernel = build_model_kernel(mdp)
    transition_rng = np.random.default_rng(seeds["transition"])
    initial_index = mdp.state_to_index[mdp.initial_state(scenario.config.start)]
    returns = np.zeros(episodes, dtype=np.float64)
    losses: list[float] = []

    for episode_index in range(episodes):
        state_index = initial_index
        episode_return = 0.0

        for _ in range(horizon):
            state = mdp.states[state_index]
            encoded = encoder.encode(state)
            action_index = agent.select_action(encoded, explore=True)

            possible = kernel[state_index][action_index]
            if len(possible) == 1:
                next_index, _, reward = possible[0]
            else:
                probabilities = np.asarray(
                    [row[1] for row in possible], dtype=np.float64
                )
                probabilities /= probabilities.sum()
                selected = int(
                    transition_rng.choice(len(possible), p=probabilities)
                )
                next_index, _, reward = possible[selected]

            next_state = mdp.states[next_index]
            encoded_next = encoder.encode(next_state)
            agent.record(encoded, action_index, reward, encoded_next)
            loss = agent.maybe_update()
            if loss is not None:
                losses.append(loss)

            episode_return += reward
            state_index = next_index

        returns[episode_index] = episode_return
        completed = episode_index + 1
        if progress_every > 0 and (
            completed % progress_every == 0 or completed == episodes
        ):
            recent = returns[max(0, completed - progress_every):completed]
            print(
                f"model_dqn_episodes={completed}/{episodes} "
                f"recent_mean_return={recent.mean():.3f}",
                flush=True,
            )

    return agent.q_network.eval(), NeuralComparatorTrainingResult(
        episode_returns=returns,
        losses=np.asarray(losses, dtype=np.float64),
        environment_steps=agent.environment_steps,
        gradient_steps=agent.gradient_steps,
    )


def save_neural_comparator_checkpoint(
    *,
    path: Path,
    scenario: NavigationScenario,
    networks: Sequence[QNetwork],
    completed_models: int,
    dqn_config: DQNConfig,
    episodes: int,
    horizon: int,
    seed_mode: str,
    base_seed: int,
    training_summaries: Sequence[dict[str, float | int]],
) -> None:
    ensure_parent(path)
    torch.save(
        {
            "backend": "model-specific-dqn",
            "models": [
                [cell.value for cell in model] for model in scenario.models
            ],
            "network_state_dicts": [
                network.state_dict() for network in networks
            ],
            "completed_models": int(completed_models),
            "input_size": ModelStateEncoder(scenario.config).input_size,
            "number_of_actions": len(Action),
            "hidden_sizes": dqn_config.hidden_sizes,
            "dqn_config": asdict(dqn_config),
            "episodes": int(episodes),
            "horizon": int(horizon),
            "seed_mode": str(seed_mode),
            "base_seed": int(base_seed),
            "training_summaries": list(training_summaries),
            "standalone_format_version": 2,
        },
        path,
    )


def train_neural_comparator_bank(
    *,
    scenario: NavigationScenario,
    device: torch.device,
    episodes: int,
    horizon: int,
    base_seed: int,
    seed_mode: str,
    output_path: Path,
    progress_every: int,
    resume: bool,
) -> NeuralComparatorBank:
    """Train all 27 model-specific DQNs, saving after every model."""

    dqn_config = DQNConfig()
    networks: list[QNetwork] = []
    summaries: list[dict[str, float | int]] = []
    start_index = 0

    if resume and output_path.exists():
        checkpoint = torch.load(output_path, map_location=device, weights_only=True)
        if checkpoint.get("backend") != "model-specific-dqn":
            raise ValueError("Existing comparator checkpoint uses another backend.")
        if int(checkpoint["episodes"]) != episodes or int(checkpoint["horizon"]) != horizon:
            raise ValueError("Resume checkpoint training configuration does not match.")
        if str(checkpoint["seed_mode"]) != seed_mode or int(checkpoint["base_seed"]) != base_seed:
            raise ValueError("Resume checkpoint seed configuration does not match.")

        encoder = ModelStateEncoder(scenario.config)
        for state_dict in checkpoint["network_state_dicts"]:
            network = QNetwork(
                encoder.input_size, len(Action), tuple(checkpoint["hidden_sizes"])
            ).to(device)
            network.load_state_dict(state_dict)
            networks.append(network.eval())
        summaries = list(checkpoint.get("training_summaries", []))
        start_index = int(checkpoint.get("completed_models", len(networks)))
        print(f"resuming_comparator_models={start_index}/{len(scenario.models)}")

    for model_index in range(start_index, len(scenario.models)):
        model = scenario.models[model_index]
        seeds = _comparator_model_seeds(
            base_seed=base_seed, model_index=model_index, seed_mode=seed_mode
        )
        print(
            f"training_neural_model={model_index + 1}/{len(scenario.models)} "
            f"model={[cell.value for cell in model]} seeds={seeds}",
            flush=True,
        )
        network, result = train_one_model_dqn(
            scenario=scenario,
            model=model,
            device=device,
            episodes=episodes,
            horizon=horizon,
            dqn_config=dqn_config,
            seeds=seeds,
            progress_every=progress_every,
        )
        networks.append(network)
        summaries.append(
            {
                "model_index": model_index,
                "mean_return": float(result.episode_returns.mean()),
                "last_100_mean_return": float(result.episode_returns[-100:].mean()),
                "environment_steps": int(result.environment_steps),
                "gradient_steps": int(result.gradient_steps),
            }
        )
        save_neural_comparator_checkpoint(
            path=output_path,
            scenario=scenario,
            networks=networks,
            completed_models=model_index + 1,
            dqn_config=dqn_config,
            episodes=episodes,
            horizon=horizon,
            seed_mode=seed_mode,
            base_seed=base_seed,
            training_summaries=summaries,
        )

    return NeuralComparatorBank(
        models=scenario.models,
        networks=networks,
        encoder=ModelStateEncoder(scenario.config),
        device=device,
        dqn_config=dqn_config,
        episodes=episodes,
        horizon=horizon,
        seed_mode=seed_mode,
        base_seed=base_seed,
    )


def load_neural_comparator_bank(
    path: Path, scenario: NavigationScenario, device: torch.device
) -> NeuralComparatorBank:
    if not path.exists():
        raise FileNotFoundError(f"Neural comparator checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("backend") != "model-specific-dqn":
        raise ValueError("Comparator checkpoint is not a model-specific DQN bank.")
    if int(checkpoint.get("completed_models", 0)) != len(scenario.models):
        raise ValueError("Comparator bank training is incomplete.")

    encoder = ModelStateEncoder(scenario.config)
    networks: list[QNetwork] = []
    for state_dict in checkpoint["network_state_dicts"]:
        network = QNetwork(
            encoder.input_size,
            int(checkpoint["number_of_actions"]),
            tuple(checkpoint["hidden_sizes"]),
        ).to(device)
        network.load_state_dict(state_dict)
        networks.append(network.eval())

    return NeuralComparatorBank(
        models=scenario.models,
        networks=networks,
        encoder=encoder,
        device=device,
        dqn_config=DQNConfig(**checkpoint["dqn_config"]),
        episodes=int(checkpoint["episodes"]),
        horizon=int(checkpoint["horizon"]),
        seed_mode=str(checkpoint["seed_mode"]),
        base_seed=int(checkpoint["base_seed"]),
    )




# evaluation policies

class EvaluationPolicy(Protocol):
    def select_action(self, state: InjurySearchState) -> int: ...


class GreedyNeuralPolicy:
    def __init__(self, network: QNetwork, encoder: StateEncoder, device: torch.device) -> None:
        self.network = network.to(device)
        self.network.eval()
        self.encoder = encoder
        self.device = device

    def select_action(self, state: InjurySearchState) -> int:
        encoded = self.encoder.encode(state)
        tensor = torch.as_tensor(encoded, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return int(torch.argmax(self.network(tensor)).item())


class BaselinePolicy:
    # knows the true model na dis ideal benchmark
    def __init__(self, bank, true_model: EnvironmentModel) -> None:
        self.bank = bank
        self.true_model_index = bank.model_index(true_model)

    def select_action(self, state: InjurySearchState) -> int:
        return int(np.argmax(self.bank.values_for_model(self.true_model_index, state)))


class MAPPolicy:
    """Paper Eq. 21 with an explicit, selectable tie convention.

    Equation (21) does not state how argmax ties are resolved.  The literal
    deterministic implementation is ``first`` (NumPy argmax).  ``sticky-random``
    is retained only as a sensitivity analysis.  The chosen rule is stored in
    result files and printed by the diagnostic mode.
    note: sticky-random had more similar output to paper
    """

    def __init__(self, bank, base_seed: int, tie_rule: str = "first") -> None:
        if tie_rule not in {"first", "sticky-random", "random-each-step"}:
            raise ValueError(
                "MAP tie rule must be first, sticky-random, or random-each-step."
            )
        self.bank = bank
        self.base_seed = int(base_seed)
        self.tie_rule = tie_rule
        self.rng = np.random.default_rng(base_seed)
        self.selected_index: int | None = None

    def reset(self, trial_seed: int) -> None:
        self.rng = np.random.default_rng(self.base_seed + int(trial_seed))
        self.selected_index = None

    def _model_index(self, posterior: np.ndarray) -> int:
        maximum = float(np.max(posterior))
        candidates = np.flatnonzero(
            np.isclose(posterior, maximum, rtol=0.0, atol=1e-12)
        )

        if self.tie_rule == "first":
            return int(candidates[0])

        if self.tie_rule == "random-each-step":
            return int(self.rng.choice(candidates))

        if self.selected_index is None or not np.any(candidates == self.selected_index):
            self.selected_index = int(self.rng.choice(candidates))
        return int(self.selected_index)

    def select_action(self, state: InjurySearchState) -> int:
        model_index = self._model_index(state.model_probabilities)
        q_values = self.bank.values_for_model(model_index, state)
        return int(np.argmax(q_values))


class ActiveLearningPolicy:
    """paper Eq. 22: posterior-weighted model-specific Q-values.
    one problem is that due to the observation of the wall that we set, it is stronger and faster than the paper.
    """

    def __init__(self, bank) -> None:
        self.bank = bank

    def select_action(self, state: InjurySearchState) -> int:
        all_values = self.bank.values_for_all_models(state)
        weighted = state.model_probabilities @ all_values
        return int(np.argmax(weighted))


@dataclass(frozen=True)
class EvaluationResult:
    cumulative_injuries: np.ndarray
    trial_seeds: np.ndarray

    @property
    def mean(self) -> np.ndarray:
        return self.cumulative_injuries.mean(axis=0)

    @property
    def ci95(self) -> np.ndarray:
        if self.cumulative_injuries.shape[0] == 1:
            return np.zeros(self.cumulative_injuries.shape[1])
        standard_error = self.cumulative_injuries.std(axis=0, ddof=1) / math.sqrt(
            self.cumulative_injuries.shape[0]
        )
        return 1.96 * standard_error


def evaluate_policy(
    scenario: NavigationScenario,
    policy: EvaluationPolicy,
    trials: int,
    horizon: int,
    base_seed: int,
) -> EvaluationResult:
    simulator = TrueEnvironmentSimulator(
        MazeEnvironment(scenario.config),
        scenario.models,
        scenario.true_model,
        scenario.config.unknown_cells,
    )
    trial_seeds = base_seed + np.arange(trials, dtype=np.int64)
    cumulative = np.zeros((trials, horizon), dtype=np.float64)

    for trial_index, trial_seed in enumerate(trial_seeds):
        integer_seed = int(trial_seed)
        simulator.reseed(integer_seed)
        reset = getattr(policy, "reset", None)
        if callable(reset):
            reset(integer_seed)

        state = simulator.initial_state(scenario.config.start, scenario.model_prior)
        for step_index in range(horizon):
            action = policy.select_action(state)
            outcome = simulator.step(state, action)
            state = outcome.next_state
            cumulative[trial_index, step_index] = state.located_injury_count

    return EvaluationResult(cumulative, trial_seeds)



# saving and loading

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_proposed_checkpoint(
    path: Path,
    agent: DQNAgent,
    result: ProposedTrainingResult,
    episodes: int,
    horizon: int,
    seeds: dict[str, int],
) -> None:
    ensure_parent(path)
    torch.save(
        {
            "q_network_state_dict": agent.q_network.state_dict(),
            "target_network_state_dict": agent.target_network.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "input_size": agent.encoder.input_size,
            "number_of_actions": len(Action),
            "hidden_sizes": agent.config.hidden_sizes,
            "completed_episodes": episodes,
            "horizon": horizon,
            "environment_steps": agent.environment_steps,
            "gradient_steps": agent.gradient_steps,
            "dqn_config": asdict(agent.config),
            "seeds": seeds,
            "standalone_format_version": 1,
        },
        path,
    )

    results_path = path.with_suffix(".training_results.npz")
    np.savez_compressed(
        results_path,
        episode_returns=result.episode_returns,
        final_injuries=result.final_injuries,
        losses=result.losses,
    )


def load_proposed_network(
    path: Path,
    scenario: NavigationScenario,
    device: torch.device,
    expected_global_seed: int | None = None,
) -> tuple[QNetwork, StateEncoder, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=True)

    if expected_global_seed is not None:
        saved_seeds = checkpoint.get("seeds")
        if not isinstance(saved_seeds, dict) or "global" not in saved_seeds:
            raise ValueError(
                "The proposed checkpoint does not contain training-seed metadata. "
                "Retrain it with this final script."
            )
        saved_global_seed = int(saved_seeds["global"])
        if saved_global_seed != int(expected_global_seed): # check seed date in check point
            raise ValueError(
                "Proposed checkpoint seed mismatch: "
                f"requested global seed {expected_global_seed}, "
                f"but {path} was trained with global seed {saved_global_seed}."
            )

    encoder = StateEncoder(scenario.config, len(scenario.models))
    hidden_sizes = tuple(checkpoint.get("hidden_sizes", (128, 128, 128)))
    network = QNetwork(
        input_size=int(checkpoint["input_size"]),
        number_of_actions=int(checkpoint["number_of_actions"]),
        hidden_sizes=hidden_sizes,
    ).to(device)
    network.load_state_dict(checkpoint["q_network_state_dict"])
    network.eval()

    if network.input_size != encoder.input_size:
        raise ValueError("Checkpoint input size does not match the selected scenario encoder.")

    return network, encoder, checkpoint


def save_comparator_bank(path: Path, bank: ComparatorBank) -> None:
    ensure_parent(path)
    models = np.asarray(
        [[cell.value for cell in model] for model in bank.models], dtype="U1"
    )
    positions = np.asarray([state.position for state in bank.states], dtype=np.int64)
    eta = np.asarray([state.eta for state in bank.states], dtype=np.int8)

    np.savez_compressed(
        path,
        models=models,
        positions=positions,
        eta=eta,
        q_values=bank.q_values,
        visit_counts=bank.visit_counts,
        model_seeds=bank.model_seeds,
        config_json=json.dumps(asdict(bank.config)),
        standalone_format_version=np.int64(1),
    )


def load_comparator_bank(path: Path) -> ComparatorBank:
    if not path.exists():
        raise FileNotFoundError(f"Comparator bank not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        models = tuple(
            tuple(CellType(value) for value in row.tolist()) for row in data["models"]
        )
        states = tuple(
            ModelState(
                position=tuple(int(value) for value in position),
                eta=tuple(int(value) for value in eta),
            )
            for position, eta in zip(data["positions"], data["eta"], strict=True)
        )
        config = ComparatorConfig(**json.loads(str(data["config_json"])))
        return ComparatorBank(
            models=models,
            states=states,
            q_values=data["q_values"],
            visit_counts=data["visit_counts"],
            model_seeds=data["model_seeds"],
            config=config,
        )


def save_evaluation_result(path: Path, result: EvaluationResult) -> None:
    ensure_parent(path)
    np.savez_compressed(
        path,
        cumulative_injuries=result.cumulative_injuries,
        trial_seeds=result.trial_seeds,
        mean=result.mean,
        ci95=result.ci95,
    )



# plotting in different path

def draw_maze(
    axis: plt.Axes,
    scenario: NavigationScenario,
    true_environment: bool,
) -> None:
    config = scenario.config
    axis.set_xlim(0, config.cols)
    axis.set_ylim(config.rows, 0)
    axis.set_aspect("equal")
    axis.set_xticks(range(config.cols + 1))
    axis.set_yticks(range(config.rows + 1))
    axis.grid(True, linewidth=1.2)
    axis.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for row, col in config.fixed_walls:
        axis.add_patch(plt.Rectangle((col, row), 1, 1, facecolor="black"))

    for index, position in enumerate(config.unknown_cells):
        row, col = position
        if true_environment:
            cell_type = scenario.true_model[index]
            if cell_type is CellType.WALL:
                facecolor = "black"
                text = "W"
                text_color = "white"
            elif cell_type is CellType.INJURY:
                facecolor = "gold"
                text = "I"
                text_color = "red"
            else:
                facecolor = "white"
                text = "E"
                text_color = "black"
        else:
            facecolor = "yellow"
            text = str(index + 1)
            text_color = "red"

        axis.add_patch(plt.Rectangle((col, row), 1, 1, facecolor=facecolor))
        axis.text(col + 0.5, row + 0.58, text, ha="center", va="center", color=text_color, fontsize=14)

    start_row, start_col = config.start
    axis.text(start_col + 0.5, start_row + 0.55, "R", ha="center", va="center", fontsize=15, fontweight="bold")


def plot_scenario_layout(scenario: NavigationScenario, output: Path) -> None:
    ensure_parent(output)
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 4.5))
    draw_maze(axes[0], scenario, true_environment=False)
    axes[0].set_title(f"Figure {scenario.figure_number}A - Unknown cells")
    draw_maze(axes[1], scenario, true_environment=True)
    model_text = ", ".join(cell.value for cell in scenario.true_model)
    axes[1].set_title(
        f"Figure {scenario.figure_number}B - True model [{model_text}]"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_policy_comparison(
    scenario: NavigationScenario,
    results: dict[str, EvaluationResult],
    output: Path,
) -> None:
    ensure_parent(output)
    horizon = next(iter(results.values())).mean.size
    steps = np.arange(1, horizon + 1)
    figure, axis = plt.subplots(figsize=(8.2, 5.4))

    for name, result in results.items():
        line = axis.plot(steps, result.mean, linewidth=2.0, label=name)[0]
        axis.fill_between(
            steps,
            np.maximum(0.0, result.mean - result.ci95),
            result.mean + result.ci95,
            color=line.get_color(),
            alpha=0.14,
        )

    axis.set_xlabel("Steps")
    axis.set_ylabel("Average Located Injuries")
    axis.set_xlim(1, horizon)
    axis.set_ylim(0.0, 2.05)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    axis.set_title(
        f"Figure {scenario.figure_number}C - Navigation Policy Comparison"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(figure)



# Audit and self-tests -> used AI for this part

def print_audit(scenario: NavigationScenario) -> None:
    reachable_positions = (
        scenario.config.rows
        * scenario.config.cols
        - len(scenario.config.fixed_walls)
    )

    print("PAPER-SPECIFIED ITEMS")
    print(
        f"- Figure {scenario.figure_number} maze: "
        f"{scenario.config.rows}x{scenario.config.cols}"
    )
    print(f"- Fixed walls: {len(scenario.config.fixed_walls)}")
    print(f"- Potential robot positions: {reachable_positions}")
    print("- Unknown cells: 3 -> 27 models")
    print("- Unknown-cell prior: uniform [1/3, 1/3, 1/3]")
    print(
        "- True test model: "
        f"{[cell.value for cell in scenario.true_model]}"
    )
    print("- Movement: intended 0.8, perpendicular 0.1 + 0.1")
    print("- Reward: 1 only when a new injury is located")
    print("- Proposed DQN: 3 hidden layers, 128 neurons each")
    print("- DQN lr=5e-4, replay=1e5, batch=64, gamma=.95, epsilon=.1")
    print("- Target tau=1e-3, Q update frequency=4")
    print("- Proposed training: 5000 episodes, horizon 250")
    print("- Evaluation: 1000 trials, first 50 steps, 95% bounds")
    print("- MAP: Eq. 21; Active Learning: Eq. 22")
    print()
    print("IMPLEMENTATION ASSUMPTIONS (EXPOSED, NOT CLAIMED AS PAPER FACTS)")
    print("- Neural position encoding: one-hot")
    print("- Unknown-wall collision yields explicit W observation")
    print("- MAP tie rule is CLI-controlled; use sticky-random for reproduction")
    print("- Model-specific comparators use DQN with reported Q-network settings")
    print("- Comparator DQNs use a common seed by default")
    print("- 95% CI: normal approximation")
    print("- The paper does not publish input encoding, MAP tie rule, or seeds")
    print()
    print(f"FIGURE {scenario.figure_number} GEOMETRY USED")
    print(f"- Start: {scenario.config.start}")
    print(f"- Fixed walls: {scenario.config.fixed_walls}")
    print(f"- Unknown cells [1,2,3]: {scenario.config.unknown_cells}")
    print(f"- True injuries: {scenario.true_injury_positions}")
    print(f"- True unknown wall: {scenario.true_wall_positions}")


def _run_scenario_geometry_tests(scenario: NavigationScenario) -> None:
    environment = MazeEnvironment(scenario.config)
    assert len(scenario.models) == 27
    assert np.isclose(scenario.model_prior.sum(), 1.0)

    expected_positions = (
        scenario.config.rows
        * scenario.config.cols
        - len(scenario.config.fixed_walls)
    )

    encoder = StateEncoder(scenario.config, len(scenario.models))
    assert len(encoder.positions) == expected_positions
    assert encoder.input_size == expected_positions + 3 + 27

    simulator_a = BeliefInjurySimulator(
        environment, scenario.models, scenario.config.unknown_cells, seed=1
    )
    simulator_b = BeliefInjurySimulator(
        environment, scenario.models, scenario.config.unknown_cells, seed=1
    )
    initial = simulator_a.initial_state(
        scenario.config.start, scenario.model_prior
    )
    assert (
        simulator_a.all_observations(initial, Action.DOWN)
        == simulator_b.all_observations(initial, Action.DOWN)
    )

    for selected_action in Action:
        distribution = observation_distribution(
            environment,
            state=scenario.config.start,
            selected_action=selected_action,
            model=scenario.true_model,
        )
        assert np.isclose(sum(distribution.values()), 1.0)


def run_self_tests() -> None:
    figure8 = create_figure8_scenario()
    figure9 = create_figure9_scenario()

    _run_scenario_geometry_tests(figure8)
    _run_scenario_geometry_tests(figure9)

    assert len(figure8.config.fixed_walls) == 3
    assert len(figure9.config.fixed_walls) == 7
    assert (
        figure9.config.rows * figure9.config.cols
        - len(figure9.config.fixed_walls)
        == 29
    )
    assert figure9.config.start == (5, 4)
    assert figure9.config.unknown_cells == ((2, 2), (4, 0), (3, 0))
    assert figure9.true_model == (
        CellType.WALL,
        CellType.INJURY,
        CellType.INJURY,
    )

    # Figure 8 explicit unknown-wall observation.
    environment8 = MazeEnvironment(figure8.config)
    wall_observation = MazeObservation(
        next_position=(3, 0),
        observed_cell=(3, 1),
        observed_type=CellType.WALL,
    )
    distribution8 = observation_distribution(
        environment8,
        state=(3, 0),
        selected_action=Action.RIGHT,
        model=figure8.true_model,
    )
    assert np.isclose(distribution8[wall_observation], 0.8)

    likelihoods = observation_likelihoods(
        environment8,
        figure8.models,
        state=(3, 0),
        selected_action=Action.RIGHT,
        observation=wall_observation,
    )
    posterior = bayesian_posterior_update(
        figure8.model_prior, likelihoods
    )
    consistent = [
        index
        for index, model in enumerate(figure8.models)
        if model[1] is CellType.WALL
    ]
    inconsistent = [
        index for index in range(27) if index not in consistent
    ]
    assert np.allclose(posterior[inconsistent], 0.0)
    assert np.isclose(posterior[consistent].sum(), 1.0)

    # Figure 9 cell 1 is a wall in the true model.
    environment9 = MazeEnvironment(figure9.config)
    figure9_wall_observation = MazeObservation(
        next_position=(2, 1),
        observed_cell=(2, 2),
        observed_type=CellType.WALL,
    )
    distribution9 = observation_distribution(
        environment9,
        state=(2, 1),
        selected_action=Action.RIGHT,
        model=figure9.true_model,
    )
    assert np.isclose(
        distribution9[figure9_wall_observation],
        0.8,
    )

    # Tiny tabular Q-learning problem learns RIGHT.
    tiny_config = MazeConfig(
        rows=1,
        cols=2,
        start=(0, 0),
        fixed_walls=(),
        unknown_cells=((0, 1),),
        intended_probability=1.0,
        side_probability=0.0,
    )
    tiny_mdp = ModelSpecificMDP(
        MazeEnvironment(tiny_config),
        (CellType.INJURY,),
        tiny_config.unknown_cells,
    )
    q_values, visits = train_one_model_q_table(
        tiny_mdp,
        tiny_config.start,
        ComparatorConfig(
            episodes=300,
            horizon=4,
            alpha=0.2,
            gamma=0.95,
            epsilon=0.2,
        ),
        seed=42,
    )
    initial_index = tiny_mdp.state_to_index[
        tiny_mdp.initial_state(tiny_config.start)
    ]
    assert int(np.argmax(q_values[initial_index])) == int(Action.RIGHT)
    assert q_values[initial_index, int(Action.RIGHT)] > 0.9
    assert visits.sum() == 1200

    # Tiny DQN integration smoke test on Figure 9's larger encoder.
    set_global_seed(123)
    environment = MazeEnvironment(figure9.config)
    encoder = StateEncoder(figure9.config, len(figure9.models))
    belief_simulator = BeliefInjurySimulator(
        environment,
        figure9.models,
        figure9.config.unknown_cells,
        seed=100,
    )
    smoke_agent = DQNAgent(
        encoder,
        DQNConfig(
            batch_size=4,
            replay_capacity=100,
            update_frequency=1,
            hidden_sizes=(16, 16),
        ),
        torch.device("cpu"),
        action_seed=101,
        replay_seed=102,
    )
    state = belief_simulator.initial_state(
        figure9.config.start, figure9.model_prior
    )
    for _ in range(8):
        encoded = encoder.encode(state)
        action = smoke_agent.select_action(encoded, explore=True)
        outcome = belief_simulator.sample(state, action)
        next_encoded = encoder.encode(outcome.next_state)
        smoke_agent.record(
            encoded, action, outcome.reward, next_encoded
        )
        smoke_agent.maybe_update()
        state = outcome.next_state

    assert smoke_agent.environment_steps == 8
    assert smoke_agent.gradient_steps > 0
    print("All standalone self-tests passed for Figures 8 and 9.")


def _action_label(action_index: int) -> str:
    return Action(int(action_index)).name


def diagnose_policy_values(
    *,
    scenario: NavigationScenario,
    proposed_network: QNetwork,
    proposed_encoder: StateEncoder,
    comparator_bank,
    device: torch.device,
    map_tie_rule: str,
    map_tie_seed: int,
) -> None:
    """Print the exact initial action values driving the selected figure.

    This diagnostic is intentionally numerical rather than visual.  It reveals
    whether Active Learning moves because Equation (22) has a genuine preference
    or because independently trained approximators introduced asymmetric noise.
    """

    initial_state = InjurySearchState(
        position=scenario.config.start,
        model_probabilities=scenario.model_prior,
        eta=np.ones(len(scenario.config.unknown_cells), dtype=np.int8),
    )

    proposed_policy = GreedyNeuralPolicy(
        proposed_network, proposed_encoder, device
    )
    baseline_policy = BaselinePolicy(comparator_bank, scenario.true_model)
    map_policy = MAPPolicy(
        comparator_bank, map_tie_seed, tie_rule=map_tie_rule
    )
    map_policy.reset(10_000)
    active_policy = ActiveLearningPolicy(comparator_bank)

    proposed_encoded = proposed_encoder.encode(initial_state)
    with torch.no_grad():
        proposed_values = (
            proposed_network(
                torch.as_tensor(
                    proposed_encoded, dtype=torch.float32, device=device
                )
            )
            .detach()
            .cpu()
            .numpy()
        )

    true_model_index = comparator_bank.model_index(scenario.true_model)
    baseline_values = comparator_bank.values_for_model(
        true_model_index, initial_state
    )
    all_values = comparator_bank.values_for_all_models(initial_state)
    active_values = initial_state.model_probabilities @ all_values

    maximum = float(np.max(initial_state.model_probabilities))
    map_candidates = np.flatnonzero(
        np.isclose(
            initial_state.model_probabilities,
            maximum,
            rtol=0.0,
            atol=1e-12,
        )
    )
    map_model_index = map_policy._model_index(initial_state.model_probabilities)
    map_values = comparator_bank.values_for_model(map_model_index, initial_state)

    def print_values(name: str, values: np.ndarray) -> None:
        pairs = ", ".join(
            f"{Action(index).name}={float(value):.6f}"
            for index, value in enumerate(values)
        )
        print(f"{name}: {pairs} -> {_action_label(int(np.argmax(values)))}")

    print("=" * 78)
    print(f"Figure {scenario.figure_number} policy diagnostic at the initial belief state")
    print("=" * 78)
    print(f"true_model={[cell.value for cell in scenario.true_model]}")
    print(f"map_tie_rule={map_tie_rule}")
    print(f"number_of_map_candidates={len(map_candidates)}")
    print(f"selected_map_model_index={map_model_index}")
    print(
        "selected_map_model="
        f"{[cell.value for cell in scenario.models[map_model_index]]}"
    )
    print_values("Proposed", proposed_values)
    print_values("Baseline", baseline_values)
    print_values("MAP", map_values)
    print_values("Active weighted mean", active_values)
    print(
        "Active per-action across-model std: "
        + ", ".join(
            f"{Action(index).name}={float(value):.6f}"
            for index, value in enumerate(all_values.std(axis=0))
        )
    )
    model_greedy_actions = np.argmax(all_values, axis=1)
    counts = np.bincount(model_greedy_actions, minlength=len(Action))
    print(
        "Model-specific initial greedy-action counts: "
        + ", ".join(
            f"{Action(index).name}={int(count)}"
            for index, count in enumerate(counts)
        )
    )
    print(
        "Selected actions: "
        f"proposed={_action_label(proposed_policy.select_action(initial_state))}, "
        f"baseline={_action_label(baseline_policy.select_action(initial_state))}, "
        f"active={_action_label(active_policy.select_action(initial_state))}, "
        f"map={_action_label(map_policy.select_action(initial_state))}"
    )
    print("=" * 78)


def train_and_evaluate_proposed_seed_sweep(
    *,
    scenario: NavigationScenario,
    device: torch.device,
    seeds: Sequence[int],
    episodes: int,
    horizon: int,
    trials: int,
    eval_horizon: int,
    base_eval_seed: int,
    checkpoint_template: Path,
    output_dir: Path,
    progress_every: int,
) -> None:
    """Measure training-seed sensitivity without averaging policies together."""

    summaries: list[dict[str, float | int | str]] = []
    curves: list[np.ndarray] = []

    for seed in seeds:
        transition_seed = int(seed + 1001)
        replay_seed = int(seed + 2002)
        action_seed = int(seed + 3003)
        checkpoint = checkpoint_template.with_name(
            f"{checkpoint_template.stem}_seed{seed}{checkpoint_template.suffix}"
        )
        print(f"starting_proposed_seed={seed} checkpoint={checkpoint}")
        agent, training_result = train_proposed_policy(
            scenario=scenario,
            episodes=episodes,
            horizon=horizon,
            device=device,
            global_seed=int(seed),
            transition_seed=transition_seed,
            replay_seed=replay_seed,
            action_seed=action_seed,
            progress_every=progress_every,
        )
        save_proposed_checkpoint(
            checkpoint,
            agent,
            training_result,
            episodes,
            horizon,
            {
                "global": int(seed),
                "transition": transition_seed,
                "replay": replay_seed,
                "action": action_seed,
            },
        )
        evaluation = evaluate_policy(
            scenario,
            GreedyNeuralPolicy(agent.q_network, agent.encoder, device),
            trials,
            eval_horizon,
            base_eval_seed,
        )
        curves.append(evaluation.mean)
        summaries.append(
            {
                "seed": int(seed),
                "checkpoint": str(checkpoint),
                "training_mean_return": float(training_result.episode_returns.mean()),
                "final_evaluation_mean": float(evaluation.mean[-1]),
            }
        )
        print(
            f"seed={seed} final_evaluation_mean={evaluation.mean[-1]:.4f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    curve_array = np.stack(curves, axis=0)
    output = output_dir / f"{scenario.slug}_proposed_training_seed_sweep.npz"
    np.savez_compressed(
        output,
        seeds=np.asarray(seeds, dtype=np.int64),
        mean_curves=curve_array,
        across_seed_mean=curve_array.mean(axis=0),
        across_seed_std=curve_array.std(axis=0, ddof=1)
        if len(seeds) > 1
        else np.zeros(eval_horizon),
        summaries_json=json.dumps(summaries),
    )
    print(f"saved_seed_sweep={output}")



# CLI modes

def summarize_result(name: str, result: EvaluationResult) -> None:
    print(name)
    for timestep in (10, 20, 30, 40, 50):
        if timestep <= result.mean.size:
            print(
                f"  step {timestep:>2}: {result.mean[timestep - 1]:.4f} "
                f"+/- {result.ci95[timestep - 1]:.4f}"
            )
    print(f"  final mean: {result.mean[-1]:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone Bayesian navigation reproduction for Figures 8 and 9."
        )
    )
    parser.add_argument(
        "--figure",
        type=int,
        choices=(8, 9),
        default=8,
        help="Select the paper experiment geometry.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "audit",
            "self-test",
            "plot-layout",
            "plot-figure8-layout",
            "train-proposed",
            "train-proposed-multiseed",
            "evaluate-proposed",
            "train-comparators",
            "train-neural-comparators",
            "diagnose-policies",
            "evaluate-all",
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--horizon", type=int, default=250)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--eval-horizon", type=int, default=50)
    parser.add_argument("--base-eval-seed", type=int, default=10000)

    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--transition-seed", type=int, default=1001)
    parser.add_argument("--replay-seed", type=int, default=1002)
    parser.add_argument("--action-seed", type=int, default=1003)
    parser.add_argument("--map-tie-seed", type=int, default=30001)
    parser.add_argument(
        "--map-tie-rule",
        choices=("first", "sticky-random", "random-each-step"),
        default="sticky-random",
        help=(
            "Equation (21) tie convention. The paper does not specify one; "
            "sticky-random is the documented reproduction convention."
        ),
    )

    parser.add_argument("--comparator-episodes", type=int, default=5000)
    parser.add_argument("--comparator-horizon", type=int, default=250)
    parser.add_argument("--comparator-alpha", type=float, default=0.1)
    parser.add_argument("--comparator-epsilon", type=float, default=0.1)
    parser.add_argument("--comparator-base-seed", type=int, default=40000)
    parser.add_argument(
        "--comparator-backend",
        choices=("neural", "tabular"),
        default="neural",
        help="Neural is the main reproduction backend; tabular is diagnostic.",
    )
    parser.add_argument(
        "--comparator-seed-mode",
        choices=("common", "distinct"),
        default="common",
        help="Common seeds avoid arbitrary Eq. 22 asymmetry.",
    )
    parser.add_argument("--resume-comparators", action="store_true")
    parser.add_argument(
        "--proposed-seeds",
        default="11,22,33",
        help="Comma-separated seeds for train-proposed-multiseed.",
    )

    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--comparator-bank", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _resolve_runtime_paths(
    args: argparse.Namespace,
    scenario: NavigationScenario,
) -> tuple[Path, Path, Path]:
    checkpoint = (
        args.checkpoint
        if args.checkpoint is not None
        else Path(
            f"outputs/models/{scenario.slug}_proposed_seed{args.global_seed}.pt"
        )
    )

    if args.comparator_bank is not None:
        comparator_bank = args.comparator_bank
    elif (
        args.mode == "train-comparators"
        or args.comparator_backend == "tabular"
    ):
        comparator_bank = Path(
            f"outputs/planning/{scenario.slug}_comparator_qlearning_standalone.npz"
        )
    else:
        comparator_bank = Path(
            f"outputs/planning/{scenario.slug}_comparator_dqn_standalone.pt"
        )

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path(f"outputs/standalone/{scenario.slug}")
    )
    return checkpoint, comparator_bank, output_dir


def main() -> None:
    args = parse_args()
    scenario = create_scenario(args.figure)
    checkpoint_path, comparator_bank_path, output_dir = (
        _resolve_runtime_paths(args, scenario)
    )

    if args.mode == "audit":
        print_audit(scenario)
        return

    if args.mode == "self-test":
        run_self_tests()
        return

    if args.mode in {"plot-layout", "plot-figure8-layout"}:
        output = output_dir / f"{scenario.slug}_ab_layout.png"
        plot_scenario_layout(scenario, output)
        print(f"saved={output}")
        return

    device = get_device(args.device)

    if args.mode == "train-proposed":
        started = time.time()
        agent, result = train_proposed_policy(
            scenario=scenario,
            episodes=args.episodes,
            horizon=args.horizon,
            device=device,
            global_seed=args.global_seed,
            transition_seed=args.transition_seed,
            replay_seed=args.replay_seed,
            action_seed=args.action_seed,
            progress_every=args.progress_every,
        )
        seeds = {
            "global": args.global_seed,
            "transition": args.transition_seed,
            "replay": args.replay_seed,
            "action": args.action_seed,
        }
        save_proposed_checkpoint(
            checkpoint_path,
            agent,
            result,
            args.episodes,
            args.horizon,
            seeds,
        )
        print(f"figure={scenario.figure_number}")
        print(f"checkpoint={checkpoint_path}")
        print(f"input_size={agent.encoder.input_size}")
        print(f"mean_training_return={result.episode_returns.mean():.4f}")
        print(f"elapsed_seconds={time.time() - started:.1f}")
        return

    if args.mode == "train-proposed-multiseed":
        seeds = [
            int(value.strip())
            for value in args.proposed_seeds.split(",")
            if value.strip()
        ]
        if not seeds:
            raise ValueError(
                "At least one proposed training seed is required."
            )
        train_and_evaluate_proposed_seed_sweep(
            scenario=scenario,
            device=device,
            seeds=seeds,
            episodes=args.episodes,
            horizon=args.horizon,
            trials=args.trials,
            eval_horizon=args.eval_horizon,
            base_eval_seed=args.base_eval_seed,
            checkpoint_template=checkpoint_path,
            output_dir=output_dir,
            progress_every=args.progress_every,
        )
        return

    if args.mode == "evaluate-proposed":
        network, encoder, checkpoint = load_proposed_network(
            checkpoint_path,
            scenario,
            device,
            expected_global_seed=args.global_seed,
        )
        result = evaluate_policy(
            scenario,
            GreedyNeuralPolicy(network, encoder, device),
            args.trials,
            args.eval_horizon,
            args.base_eval_seed,
        )
        output = output_dir / f"{scenario.slug}_proposed_evaluation.npz"
        save_evaluation_result(output, result)
        summarize_result("Proposed Policy", result)
        print(f"figure={scenario.figure_number}")
        print(
            "checkpoint_episodes="
            f"{checkpoint.get('completed_episodes', 'unknown')}"
        )
        print(f"saved={output}")
        return

    if args.mode == "train-comparators":
        comparator_config = ComparatorConfig(
            episodes=args.comparator_episodes,
            horizon=args.comparator_horizon,
            alpha=args.comparator_alpha,
            gamma=0.95,
            epsilon=args.comparator_epsilon,
        )
        started = time.time()
        bank = train_comparator_bank(
            scenario,
            comparator_config,
            args.comparator_base_seed,
        )
        save_comparator_bank(comparator_bank_path, bank)
        print(f"figure={scenario.figure_number}")
        print(f"bank={comparator_bank_path}")
        print(f"shape={bank.q_values.shape}")
        print(f"total_updates={bank.visit_counts.sum()}")
        print(f"elapsed_seconds={time.time() - started:.1f}")
        return

    if args.mode == "train-neural-comparators":
        started = time.time()
        bank = train_neural_comparator_bank(
            scenario=scenario,
            device=device,
            episodes=args.comparator_episodes,
            horizon=args.comparator_horizon,
            base_seed=args.comparator_base_seed,
            seed_mode=args.comparator_seed_mode,
            output_path=comparator_bank_path,
            progress_every=args.progress_every,
            resume=args.resume_comparators,
        )
        print(f"figure={scenario.figure_number}")
        print(f"bank={comparator_bank_path}")
        print(f"models={len(bank.models)}")
        print(f"input_size={bank.encoder.input_size}")
        print(f"seed_mode={bank.seed_mode}")
        print(f"elapsed_seconds={time.time() - started:.1f}")
        return

    if args.mode in {"diagnose-policies", "evaluate-all"}:
        network, encoder, _ = load_proposed_network(
            checkpoint_path,
            scenario,
            device,
            expected_global_seed=args.global_seed,
        )

        if args.comparator_backend == "neural":
            bank = load_neural_comparator_bank(
                comparator_bank_path, scenario, device
            )
        else:
            bank = load_comparator_bank(comparator_bank_path)

        if args.mode == "diagnose-policies":
            diagnose_policy_values(
                scenario=scenario,
                proposed_network=network,
                proposed_encoder=encoder,
                comparator_bank=bank,
                device=device,
                map_tie_rule=args.map_tie_rule,
                map_tie_seed=args.map_tie_seed,
            )
            return

        results = {
            "Baseline Policy": evaluate_policy(
                scenario,
                BaselinePolicy(bank, scenario.true_model),
                args.trials,
                args.eval_horizon,
                args.base_eval_seed,
            ),
            "Proposed Bayesian Planning Policy": evaluate_policy(
                scenario,
                GreedyNeuralPolicy(network, encoder, device),
                args.trials,
                args.eval_horizon,
                args.base_eval_seed,
            ),
            "Active Learning Policy": evaluate_policy(
                scenario,
                ActiveLearningPolicy(bank),
                args.trials,
                args.eval_horizon,
                args.base_eval_seed,
            ),
            "MAP Policy": evaluate_policy(
                scenario,
                MAPPolicy(
                    bank,
                    args.map_tie_seed,
                    tie_rule=args.map_tie_rule,
                ),
                args.trials,
                args.eval_horizon,
                args.base_eval_seed,
            ),
        }

        output_npz = (
            output_dir
            / f"{scenario.slug}_all_policies_evaluation.npz"
        )
        ensure_parent(output_npz)
        payload: dict[str, np.ndarray] = {
            "figure_number": np.asarray(scenario.figure_number),
            "trial_seeds": next(iter(results.values())).trial_seeds,
            "true_model": np.asarray(
                [cell.value for cell in scenario.true_model]
            ),
            "comparator_backend": np.asarray(args.comparator_backend),
            "comparator_episodes": np.asarray(
                bank.episodes
                if args.comparator_backend == "neural"
                else bank.config.episodes
            ),
            "comparator_horizon": np.asarray(
                bank.horizon
                if args.comparator_backend == "neural"
                else bank.config.horizon
            ),
            "comparator_seed_mode": np.asarray(
                bank.seed_mode
                if args.comparator_backend == "neural"
                else "distinct-tabular"
            ),
            "map_tie_seed": np.asarray(args.map_tie_seed),
            "map_tie_rule": np.asarray(args.map_tie_rule),
        }

        for name, result in results.items():
            key = name.lower().replace(" ", "_")
            payload[f"{key}_cumulative"] = result.cumulative_injuries
            payload[f"{key}_mean"] = result.mean
            payload[f"{key}_ci95"] = result.ci95

        np.savez_compressed(output_npz, **payload)

        figure_output = (
            output_dir / f"{scenario.slug}c_all_policies.png"
        )
        plot_policy_comparison(
            scenario,
            results,
            figure_output,
        )

        for name, result in results.items():
            summarize_result(name, result)
        print(f"figure={scenario.figure_number}")
        print(f"saved_results={output_npz}")
        print(f"saved_figure={figure_output}")
        return

    raise RuntimeError(f"Unhandled mode: {args.mode}")


if __name__ == "__main__":
    main()
