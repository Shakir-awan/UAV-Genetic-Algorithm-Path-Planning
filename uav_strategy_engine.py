"""Multi-objective UAV fleet path planning with a custom genetic algorithm.

Designed for a 50 x 50 grid and Google Colab / local Python execution.
The script creates three figures and prints both the supplied reference figures
and the results measured during the current run. Since GA operators are stochastic,
observed scores can differ from the supplied reference run.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# ---------------------------------------------------------------------------
# Assignment parameters
# ---------------------------------------------------------------------------
GRID_SIZE = 50
N_OBSTACLES = 90
N_HAZARD_POCKETS = 4
N_TARGETS = 20
N_DRONES = 3
WAYPOINTS_PER_DRONE = 5
SPEED_RANGE = (12.0, 28.0)       # metres per second
SENSOR_POWER_RANGE = (1.5, 4.5)  # abstract sensor power level
POPULATION_SIZE = 40
GENERATIONS = 75
MUTATION_RATE = 0.25
ELITISM = 2
BASELINE_EVALUATIONS = 3200
TOURNAMENT_K = 3
INTERMEDIATE_SAMPLES = 12
CELL_SIZE_M = 20.0
MAX_FLIGHT_RANGE_M = 5000.0

# Requested objective weights.
W_COV = 3.0
W_ENG = 0.08
W_TIME = 0.15
W_HAZ = 1.1

# The score scale converts priority-weighted detections into the score units
# used in the supplied reference results. It does not alter the four weights.
COVERAGE_SCORE_SCALE = 58.0
COLLIDED_SEGMENT_PENALTY = 200.0
RANGE_EXCESS_PENALTY = 1200.0

REFERENCE_RESULTS = {
    "generation_1_best_fitness": 6522.16,
    "generation_75_best_fitness": 8811.78,
    "random_search_baseline_score": 7223.11,
    "random_search_evaluations": BASELINE_EVALUATIONS,
    "reported_ga_improvement_percent": 22.0,
}


@dataclass
class Target:
    x: float
    y: float
    priority: float


@dataclass
class HazardPocket:
    x: float
    y: float
    sigma: float
    amplitude: float


class GridEnvironment:
    """Grid occupancy, continuous Gaussian hazards, starts, and target points."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.size = GRID_SIZE
        self.blocked = np.zeros((self.size, self.size), dtype=bool)
        self.starts = np.array([[2.0, 2.0], [2.0, 47.0], [47.0, 25.0]], dtype=float)

        # Reserve a small area around each launch point so a start is never blocked.
        reserved: set[tuple[int, int]] = set()
        for sx, sy in self.starts:
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    x, y = int(round(sx + dx)), int(round(sy + dy))
                    if 0 <= x < self.size and 0 <= y < self.size:
                        reserved.add((x, y))
        available = [(x, y) for x in range(self.size) for y in range(self.size)
                     if (x, y) not in reserved]
        chosen = self.rng.choice(len(available), size=N_OBSTACLES, replace=False)
        for idx in chosen:
            x, y = available[int(idx)]
            self.blocked[y, x] = True

        self.hazards: list[HazardPocket] = []
        for _ in range(N_HAZARD_POCKETS):
            self.hazards.append(HazardPocket(
                x=float(self.rng.uniform(6.0, 44.0)),
                y=float(self.rng.uniform(6.0, 44.0)),
                sigma=float(self.rng.uniform(4.0, 8.5)),
                amplitude=float(self.rng.uniform(0.55, 1.0)),
            ))

        self.targets: list[Target] = []
        while len(self.targets) < N_TARGETS:
            x, y = self.rng.uniform(1.0, self.size - 2.0, size=2)
            if not self.is_blocked(x, y):
                self.targets.append(Target(float(x), float(y),
                                           float(self.rng.uniform(1.5, 5.0))))

        self.hazard_map = self._make_hazard_map()

    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        """Return the grid array indices (row=y, column=x) for a coordinate."""
        cx = int(np.clip(round(float(x)), 0, self.size - 1))
        cy = int(np.clip(round(float(y)), 0, self.size - 1))
        return cy, cx

    def is_blocked(self, x: float, y: float) -> bool:
        row, col = self.cell_of(x, y)
        return bool(self.blocked[row, col])

    def hazard_at(self, x: float, y: float) -> float:
        value = 0.0
        for pocket in self.hazards:
            d2 = (x - pocket.x) ** 2 + (y - pocket.y) ** 2
            value += pocket.amplitude * math.exp(-d2 / (2.0 * pocket.sigma ** 2))
        return float(value)

    def _make_hazard_map(self) -> np.ndarray:
        yy, xx = np.mgrid[0:self.size, 0:self.size]
        result = np.zeros((self.size, self.size), dtype=float)
        for pocket in self.hazards:
            d2 = (xx - pocket.x) ** 2 + (yy - pocket.y) ** 2
            result += pocket.amplitude * np.exp(-d2 / (2.0 * pocket.sigma ** 2))
        return result


class UAVChromosome:
    """One candidate fleet plan: float waypoints, per-drone speed and power."""

    def __init__(self, waypoints: np.ndarray, speeds: np.ndarray,
                 sensor_power: np.ndarray) -> None:
        self.waypoints = np.asarray(waypoints, dtype=float).reshape(
            N_DRONES, WAYPOINTS_PER_DRONE, 2).copy()
        self.speeds = np.asarray(speeds, dtype=float).reshape(N_DRONES).copy()
        self.sensor_power = np.asarray(sensor_power, dtype=float).reshape(N_DRONES).copy()
        self.fitness: float | None = None
        self.details: dict[str, float] = {}

    @classmethod
    def random(cls, env: GridEnvironment, rng: np.random.Generator) -> "UAVChromosome":
        waypoints = rng.uniform(0.0, GRID_SIZE - 1.0,
                                size=(N_DRONES, WAYPOINTS_PER_DRONE, 2))
        speeds = rng.uniform(*SPEED_RANGE, size=N_DRONES)
        powers = rng.uniform(*SENSOR_POWER_RANGE, size=N_DRONES)
        chromosome = cls(waypoints, speeds, powers)
        chromosome.apply_repair(env)
        return chromosome

    def copy(self) -> "UAVChromosome":
        child = UAVChromosome(self.waypoints, self.speeds, self.sensor_power)
        child.fitness = self.fitness
        child.details = dict(self.details)
        return child

    def apply_repair(self, env: GridEnvironment) -> None:
        """Clip genes to the grid and move blocked waypoints via 8-neighbour lookup."""
        self.waypoints[:, :, 0] = np.clip(self.waypoints[:, :, 0], 0.0, env.size - 1.0)
        self.waypoints[:, :, 1] = np.clip(self.waypoints[:, :, 1], 0.0, env.size - 1.0)
        self.speeds[:] = np.clip(self.speeds, *SPEED_RANGE)
        self.sensor_power[:] = np.clip(self.sensor_power, *SENSOR_POWER_RANGE)

        for drone in range(N_DRONES):
            for waypoint_idx in range(WAYPOINTS_PER_DRONE):
                x, y = self.waypoints[drone, waypoint_idx]
                if not env.is_blocked(x, y):
                    continue
                row, col = env.cell_of(x, y)
                replacement: tuple[int, int] | None = None
                max_radius = max(env.size, 2)
                for radius in range(1, max_radius):
                    candidates: list[tuple[float, int, int]] = []
                    for dx in range(-radius, radius + 1):
                        for dy in range(-radius, radius + 1):
                            if radius == 1 and dx == 0 and dy == 0:
                                continue
                            if radius > 1 and max(abs(dx), abs(dy)) != radius:
                                continue
                            nx, ny = col + dx, row + dy
                            if 0 <= nx < env.size and 0 <= ny < env.size and not env.blocked[ny, nx]:
                                dist2 = (nx - x) ** 2 + (ny - y) ** 2
                                candidates.append((dist2, nx, ny))
                    if candidates:
                        _, nx, ny = min(candidates, key=lambda item: item[0])
                        replacement = (nx, ny)
                        break
                if replacement is not None:
                    self.waypoints[drone, waypoint_idx] = replacement


# ---------------------------------------------------------------------------
# Fitness calculation and route utilities
# ---------------------------------------------------------------------------
def sample_segment(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return endpoints plus exactly 12 intermediate spatial samples."""
    t = np.linspace(0.0, 1.0, INTERMEDIATE_SAMPLES + 2)
    return a[None, :] + (b - a)[None, :] * t[:, None]


def chromosome_routes(chromosome: UAVChromosome, env: GridEnvironment) -> list[np.ndarray]:
    routes: list[np.ndarray] = []
    for drone in range(N_DRONES):
        routes.append(np.vstack([env.starts[drone], chromosome.waypoints[drone]]))
    return routes


def compute_fitness(chromosome: UAVChromosome, env: GridEnvironment,
                    return_details: bool = False) -> float | tuple[float, dict[str, float]]:
    """Evaluate coverage reward against energy, time, hazard and hard-constraint costs."""
    routes = chromosome_routes(chromosome, env)
    target_detected = np.zeros(len(env.targets), dtype=bool)
    energy_total = 0.0
    time_total = 0.0
    hazard_integral = 0.0
    route_distance_m = np.zeros(N_DRONES, dtype=float)
    collided_segments = 0
    blocked_sample_count = 0
    total_samples = 0

    target_xy = np.array([[t.x, t.y] for t in env.targets], dtype=float)
    target_priority = np.array([t.priority for t in env.targets], dtype=float)

    for drone, route in enumerate(routes):
        speed = float(chromosome.speeds[drone])
        power = float(chromosome.sensor_power[drone])
        coverage_radius = power * 2.0
        sample_points_for_drone: list[np.ndarray] = []

        for segment_idx in range(len(route) - 1):
            a, b = route[segment_idx], route[segment_idx + 1]
            points = sample_segment(a, b)
            sample_points_for_drone.append(points)
            length_cells = float(np.linalg.norm(b - a))
            length_m = length_cells * CELL_SIZE_M
            segment_time = length_m / speed if speed > 0 else float("inf")
            route_distance_m[drone] += length_m
            time_total += segment_time

            # Kinetic/drag model
            energy_total += (0.04 * speed ** 2 + 1.8 * power) * segment_time

            # Hazard exposure integral
            hazards_here = np.array([env.hazard_at(float(px), float(py)) for px, py in points])
            hazard_integral += float(hazards_here.mean()) * segment_time

            hit_count = 0
            for px, py in points:
                total_samples += 1
                if env.is_blocked(float(px), float(py)):
                    hit_count += 1
                    blocked_sample_count += 1
            if hit_count:
                collided_segments += 1

        if sample_points_for_drone:
            path_points = np.vstack(sample_points_for_drone)
            distances = np.linalg.norm(target_xy[:, None, :] - path_points[None, :, :], axis=2)
            target_detected |= np.any(distances <= coverage_radius, axis=1)

    weighted_detected_priority = float(np.sum(target_priority[target_detected]))
    coverage_gain = COVERAGE_SCORE_SCALE * weighted_detected_priority
    coverage_reward = W_COV * coverage_gain

    range_penalty = 0.0
    for travelled in route_distance_m:
        if travelled > MAX_FLIGHT_RANGE_M:
            range_penalty += RANGE_EXCESS_PENALTY * ((travelled - MAX_FLIGHT_RANGE_M) / MAX_FLIGHT_RANGE_M)

    obstacle_penalty = COLLIDED_SEGMENT_PENALTY * collided_segments
    fitness = (coverage_reward
               - W_ENG * energy_total
               - W_TIME * time_total
               - W_HAZ * hazard_integral
               - obstacle_penalty
               - range_penalty)

    details = {
        "fitness": float(fitness),
        "coverage_reward": float(coverage_reward),
        "weighted_detected_priority": weighted_detected_priority,
        "targets_detected": float(np.count_nonzero(target_detected)),
        "energy": float(energy_total),
        "flight_time": float(time_total),
        "hazard_exposure": float(hazard_integral),
        "collided_segments": float(collided_segments),
        "blocked_samples": float(blocked_sample_count),
        "sample_count": float(total_samples),
        "range_penalty": float(range_penalty),
        "obstacle_penalty": float(obstacle_penalty),
    }
    chromosome.fitness = float(fitness)
    chromosome.details = details
    if return_details:
        return float(fitness), details
    return float(fitness)


# ---------------------------------------------------------------------------
# Genetic operators
# ---------------------------------------------------------------------------
def tournament_selection(population: list[UAVChromosome],
                         rng: np.random.Generator, k: int = TOURNAMENT_K) -> UAVChromosome:
    contestants = rng.choice(len(population), size=k, replace=False)
    winner = max((population[int(i)] for i in contestants),
                 key=lambda candidate: -math.inf if candidate.fitness is None else candidate.fitness)
    return winner


def custom_segment_crossover(parent_a: UAVChromosome, parent_b: UAVChromosome,
                             env: GridEnvironment, rng: np.random.Generator
                             ) -> tuple[UAVChromosome, UAVChromosome]:
    child_a, child_b = parent_a.copy(), parent_b.copy()
    drone = int(rng.integers(0, N_DRONES))
    start = int(rng.integers(0, WAYPOINTS_PER_DRONE))
    stop = int(rng.integers(start + 1, WAYPOINTS_PER_DRONE + 1))
    child_a.waypoints[drone, start:stop] = parent_b.waypoints[drone, start:stop]
    child_b.waypoints[drone, start:stop] = parent_a.waypoints[drone, start:stop]

    if rng.random() < 0.5:
        child_a.speeds[drone], child_b.speeds[drone] = parent_b.speeds[drone], parent_a.speeds[drone]
    if rng.random() < 0.5:
        child_a.sensor_power[drone], child_b.sensor_power[drone] = (
            parent_b.sensor_power[drone], parent_a.sensor_power[drone])
    child_a.apply_repair(env)
    child_b.apply_repair(env)
    child_a.fitness = child_b.fitness = None
    return child_a, child_b


def spatial_mutation(chromosome: UAVChromosome, env: GridEnvironment,
                     rng: np.random.Generator, mutation_rate: float = MUTATION_RATE) -> None:
    for drone in range(N_DRONES):
        for waypoint_idx in range(WAYPOINTS_PER_DRONE):
            if rng.random() < mutation_rate:
                chromosome.waypoints[drone, waypoint_idx] += rng.normal(0.0, 4.0, size=2)
        if rng.random() < mutation_rate:
            chromosome.speeds[drone] += float(rng.normal(0.0, 2.4))
        if rng.random() < mutation_rate:
            chromosome.sensor_power[drone] += float(rng.normal(0.0, 0.55))
    chromosome.apply_repair(env)
    chromosome.fitness = None
    chromosome.details = {}


def run_genetic_algorithm(env: GridEnvironment, rng: np.random.Generator
                          ) -> tuple[UAVChromosome, list[dict[str, float]], int]:
    population = [UAVChromosome.random(env, rng) for _ in range(POPULATION_SIZE)]
    evaluations = 0
    for candidate in population:
        compute_fitness(candidate, env)
        evaluations += 1

    history: list[dict[str, float]] = []
    for generation in range(1, GENERATIONS + 1):
        scores = np.array([float(p.fitness) for p in population])
        history.append({
            "generation": float(generation),
            "best": float(np.max(scores)),
            "average": float(np.mean(scores)),
            "worst": float(np.min(scores)),
        })
        if generation == GENERATIONS:
            break

        ranked = sorted(population, key=lambda candidate: float(candidate.fitness), reverse=True)
        next_population = [ranked[i].copy() for i in range(ELITISM)]
        while len(next_population) < POPULATION_SIZE:
            parent_a = tournament_selection(population, rng)
            parent_b = tournament_selection(population, rng)
            child_a, child_b = custom_segment_crossover(parent_a, parent_b, env, rng)
            spatial_mutation(child_a, env, rng)
            spatial_mutation(child_b, env, rng)
            for child in (child_a, child_b):
                if len(next_population) >= POPULATION_SIZE:
                    break
                compute_fitness(child, env)
                evaluations += 1
                next_population.append(child)
        population = next_population

    best = max(population, key=lambda candidate: float(candidate.fitness))
    return best, history, evaluations


def run_random_search(env: GridEnvironment, rng: np.random.Generator,
                      evaluations: int = BASELINE_EVALUATIONS
                      ) -> tuple[UAVChromosome, float]:
    best: UAVChromosome | None = None
    best_score = -math.inf
    for _ in range(evaluations):
        candidate = UAVChromosome.random(env, rng)
        score = float(compute_fitness(candidate, env))
        if score > best_score:
            best, best_score = candidate, score
    assert best is not None
    return best, best_score


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def coverage_field(chromosome: UAVChromosome, env: GridEnvironment) -> np.ndarray:
    yy, xx = np.mgrid[0:env.size, 0:env.size]
    field = np.zeros((env.size, env.size), dtype=float)
    routes = chromosome_routes(chromosome, env)
    for drone, route in enumerate(routes):
        radius = float(chromosome.sensor_power[drone] * 2.0)
        for segment_idx in range(len(route) - 1):
            samples = sample_segment(route[segment_idx], route[segment_idx + 1])
            for px, py in samples:
                dist = np.sqrt((xx - px) ** 2 + (yy - py) ** 2)
                field += np.clip(1.0 - dist / max(radius, 1e-6), 0.0, 1.0)
    return field


def plot_convergence(history: list[dict[str, float]], output: Path, show: bool = True) -> None:
    generations = [int(row["generation"]) for row in history]
    best = [row["best"] for row in history]
    average = [row["average"] for row in history]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(generations, best, linewidth=2.4, label="Best fitness")
    ax.plot(generations, average, linewidth=1.8, label="Average fitness")
    ax.set_title("GA Convergence: Best vs Average Fitness")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness score")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_trajectories(best: UAVChromosome, env: GridEnvironment, output: Path, show: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(env.hazard_map, cmap="magma", alpha=0.82, cbar_kws={"label": "Gaussian hazard intensity"},
                ax=ax, square=True, xticklabels=5, yticklabels=5)
    obs_y, obs_x = np.where(env.blocked)
    ax.scatter(obs_x + 0.5, obs_y + 0.5, marker="s", s=18, c="black", label="No-fly cells", linewidths=0)
    target_x = [target.x for target in env.targets]
    target_y = [target.y for target in env.targets]
    target_sizes = [20 + 11 * target.priority for target in env.targets]
    ax.scatter(target_x, target_y, marker="*", s=target_sizes, facecolors="white", edgecolors="black",
               linewidths=0.7, label="Disaster targets")
    for drone, route in enumerate(chromosome_routes(best, env)):
        ax.plot(route[:, 0], route[:, 1], marker="o", markersize=3.8, linewidth=2.0,
                label=f"UAV {drone + 1} (v={best.speeds[drone]:.1f} m/s, P={best.sensor_power[drone]:.2f})")
        ax.scatter(route[0, 0], route[0, 1], marker="D", s=65, edgecolor="black", zorder=5)
    ax.set_title("Evolved UAV Routes over Hazard and No-Fly Map")
    ax.set_xlabel("Grid x coordinate")
    ax.set_ylabel("Grid y coordinate")
    ax.set_xlim(0, env.size)
    ax.set_ylim(env.size, 0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.30, 1.0), borderaxespad=0.0, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_coverage_snapshots(history_best: list[UAVChromosome | None], env: GridEnvironment,
                            generation_labels: list[str], output: Path, show: bool = True) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes_flat = axes.ravel()
    for ax, chromosome, label in zip(axes_flat, history_best, generation_labels):
        if chromosome is None:
            ax.axis("off")
            continue
        field = coverage_field(chromosome, env)
        sns.heatmap(field, cmap="viridis", ax=ax, cbar=False, xticklabels=False, yticklabels=False, square=True)
        ax.set_title(label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle("Coverage Footprint Snapshots Across GA Evolution", y=1.01)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def best_by_checkpoint(env: GridEnvironment, rng: np.random.Generator
                       ) -> tuple[UAVChromosome, list[UAVChromosome | None]]:
    population = [UAVChromosome.random(env, rng) for _ in range(POPULATION_SIZE)]
    for item in population:
        compute_fitness(item, env)
    checkpoints = {1, 25, 50, 75}
    saved: dict[int, UAVChromosome] = {}
    for generation in range(1, GENERATIONS + 1):
        population.sort(key=lambda candidate: float(candidate.fitness), reverse=True)
        if generation in checkpoints:
            saved[generation] = population[0].copy()
        if generation == GENERATIONS:
            break
        next_population = [population[i].copy() for i in range(ELITISM)]
        while len(next_population) < POPULATION_SIZE:
            a = tournament_selection(population, rng)
            b = tournament_selection(population, rng)
            c1, c2 = custom_segment_crossover(a, b, env, rng)
            for child in (c1, c2):
                spatial_mutation(child, env, rng)
                if len(next_population) >= POPULATION_SIZE:
                    break
                compute_fitness(child, env)
                next_population.append(child)
        population = next_population
    return saved[75], [saved.get(g) for g in (1, 25, 50, 75)]


def write_results_json(output: Path, data: dict[str, Any]) -> None:
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_simulation(seed: int = 42, output_dir: str | Path = "outputs",
                   show_plots: bool = True) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    env = GridEnvironment(seed=seed)
    ga_rng = np.random.default_rng(seed + 100)
    baseline_rng = np.random.default_rng(seed + 200)
    checkpoint_rng = np.random.default_rng(seed + 300)

    best, history, ga_evaluations = run_genetic_algorithm(env, ga_rng)
    best_score, best_details = compute_fitness(best, env, return_details=True)
    baseline_chromosome, baseline_score = run_random_search(env, baseline_rng, BASELINE_EVALUATIONS)
    _, baseline_details = compute_fitness(baseline_chromosome, env, return_details=True)

    checkpoint_best, snapshots = best_by_checkpoint(env, checkpoint_rng)
    final_history_best = history[-1]["best"]
    observed_improvement = ((final_history_best - baseline_score) / abs(baseline_score) * 100.0
                            if baseline_score else 0.0)

    plot_convergence(history, output_path / "fitness_convergence.png", show=show_plots)
    plot_trajectories(best, env, output_path / "uav_trajectories_hazard_map.png", show=show_plots)
    plot_coverage_snapshots(snapshots, env, ["Generation 1", "Generation 25", "Generation 50", "Generation 75"],
                            output_path / "coverage_heatmap_generations.png", show=show_plots)

    history_rows = [{key: float(value) for key, value in row.items()} for row in history]
    summary: dict[str, Any] = {
        "seed": seed,
        "grid": {"size": GRID_SIZE, "obstacles": N_OBSTACLES,
                 "gaussian_hazard_pockets": N_HAZARD_POCKETS, "targets": N_TARGETS},
        "fleet": {"drones": N_DRONES, "waypoints_per_drone": WAYPOINTS_PER_DRONE,
                  "speed_range_m_per_s": list(SPEED_RANGE),
                  "sensor_power_range": list(SENSOR_POWER_RANGE)},
        "ga": {"population_size": POPULATION_SIZE, "generations": GENERATIONS,
               "mutation_rate": MUTATION_RATE, "elitism": ELITISM,
               "tournament_k": TOURNAMENT_K, "objective_evaluations": ga_evaluations},
        "weights": {"coverage": W_COV, "energy": W_ENG, "time": W_TIME, "hazard": W_HAZ},
        "observed_run": {
            "generation_1_best": history_rows[0]["best"],
            "generation_75_best": history_rows[-1]["best"],
            "random_search_best": float(baseline_score),
            "random_search_evaluations": BASELINE_EVALUATIONS,
            "ga_improvement_over_random_percent": float(observed_improvement),
            "best_candidate_details": best_details,
            "baseline_candidate_details": baseline_details,
            "history": history_rows,
        },
        "provided_reference_results": REFERENCE_RESULTS,
        "figures": ["fitness_convergence.png", "uav_trajectories_hazard_map.png",
                    "coverage_heatmap_generations.png"],
    }
    write_results_json(output_path / "results.json", summary)

    print("\n=== Run configuration ===")
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE}; obstacles={N_OBSTACLES}; hazards={N_HAZARD_POCKETS}; targets={N_TARGETS}")
    print(f"GA: population={POPULATION_SIZE}; generations={GENERATIONS}; mutation={MUTATION_RATE}; elitism={ELITISM}")
    print("\n=== Observed results for this execution ===")
    print(f"Generation 1 best:  {history_rows[0]['best']:.2f}")
    print(f"Generation 75 best: {history_rows[-1]['best']:.2f}")
    print(f"Random search best ({BASELINE_EVALUATIONS} evaluations): {baseline_score:.2f}")
    print(f"Observed GA improvement over random search: {observed_improvement:.2f}%")
    print(f"Best candidate details: detected={int(best_details['targets_detected'])}/{N_TARGETS}, "
          f"energy={best_details['energy']:.2f}, time={best_details['flight_time']:.2f}, "
          f"hazard integral={best_details['hazard_exposure']:.2f}, "
          f"collided segments={int(best_details['collided_segments'])}")
    print("\n=== Supplied reference results (kept separate from this run) ===")
    print(f"Generation 1 best:  {REFERENCE_RESULTS['generation_1_best_fitness']:.2f}")
    print(f"Generation 75 best: {REFERENCE_RESULTS['generation_75_best_fitness']:.2f}")
    print(f"Random search ({BASELINE_EVALUATIONS} evaluations): {REFERENCE_RESULTS['random_search_baseline_score']:.2f}")
    print(f"Reported GA improvement: ~{REFERENCE_RESULTS['reported_ga_improvement_percent']:.0f}%")
    print(f"\nFigures and results written to: {output_path.resolve()}")
    return summary


if __name__ == "__main__":
    run_simulation(seed=42, output_dir="outputs", show_plots=True)
