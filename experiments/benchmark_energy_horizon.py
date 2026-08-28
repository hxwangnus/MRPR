#!/usr/bin/env python3
"""Compare repository routing policies under one shared energy-budget model.

This benchmark is deliberately small enough for the repository's exact
Pareto-frontier dynamic program.  The exact method is a restricted-model
repository oracle (routes may be at most two switches longer than nearest), not
a literature baseline and not an electrical-network oracle.

No third-party packages are required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from power_routing_balanced import ModularRobotGraph as BalancedGraph
from power_routing_exact import ExactModularRobotGraph


DEFAULT_SEED = 21160720
DEFAULT_TRIALS = 200
GRID_SIZE = 3
BATTERY_COUNT = 2
CAPACITY_VALUES = tuple(range(30, 61, 5))
DEMAND_VALUES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
BASE_MODULE_COST = 1.0
SWITCH_LOSS_FACTOR = 1.0
MAX_EXTRA_COST = 2.0
MAX_SIMULATION_ROUNDS = 50


def _make_grid_spec(
    battery_positions: Sequence[Tuple[int, int]],
) -> Dict[str, Dict[str, Any]]:
    battery_at = {
        position: f"B{index + 1}"
        for index, position in enumerate(battery_positions)
    }
    spec: Dict[str, Dict[str, Any]] = {}
    module_index = 1
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            position = (x, y)
            if position in battery_at:
                spec[battery_at[position]] = {
                    "type": "battery",
                    "pos": position,
                }
            else:
                spec[f"M{module_index:02d}"] = {
                    "type": "action",
                    "pos": position,
                }
                module_index += 1
    return spec


def _count_load_objective(
    assignments: Mapping[str, Any],
    batteries: Sequence[str],
) -> Tuple[int, float, Tuple[int, ...]]:
    loads = {battery: 0 for battery in batteries}
    for assignment in assignments.values():
        if assignment.battery is not None:
            loads[assignment.battery] += 1
    values = list(loads.values())
    load_mean = sum(values) / len(values)
    return (
        max(values) - min(values),
        sum((value - load_mean) ** 2 for value in values),
        tuple(sorted(values, reverse=True)),
    )


def _static_horizon(
    assignments: Mapping[str, Any],
    batteries: Sequence[str],
    capacities: Mapping[str, float],
    demands: Mapping[str, float],
) -> Tuple[int, Dict[str, float]]:
    drain = {battery: 0.0 for battery in batteries}
    for module, assignment in assignments.items():
        if assignment.battery is None or math.isinf(assignment.distance):
            return 0, drain
        drain[assignment.battery] += demands[module] * (
            BASE_MODULE_COST + SWITCH_LOSS_FACTOR * assignment.distance
        )

    active_horizons = [
        math.floor((capacities[battery] + 1e-9) / drain[battery])
        for battery in batteries
        if drain[battery] > 0
    ]
    return (min(active_horizons) if active_horizons else 0), drain


def _method_summary(
    method: str,
    rows: Sequence[Mapping[str, int]],
) -> Dict[str, Any]:
    values = [row[method] for row in rows]
    return {
        "method": method,
        "mean_feasible_rounds": statistics.mean(values),
        "median_feasible_rounds": statistics.median(values),
        "minimum_feasible_rounds": min(values),
        "maximum_feasible_rounds": max(values),
        "mean_gap_to_exact": statistics.mean(
            row["exact_restricted_repository_oracle"] - row[method]
            for row in rows
        ),
        "exact_match_rate_percent": round(
            100.0 * statistics.mean(
                row[method] == row["exact_restricted_repository_oracle"]
                for row in rows
            ),
            10,
        ),
    }


def run_benchmark(*, seed: int, trials: int) -> Dict[str, Any]:
    positions = [
        (x, y)
        for y in range(GRID_SIZE)
        for x in range(GRID_SIZE)
    ]
    rows: List[Dict[str, Any]] = []

    for trial in range(trials):
        rng = random.Random(seed + trial)
        battery_positions = rng.sample(positions, BATTERY_COUNT)
        spec = _make_grid_spec(battery_positions)
        balanced_graph = BalancedGraph.from_grid_layout(spec)
        exact_graph = ExactModularRobotGraph.from_grid_layout(spec)
        batteries = balanced_graph.get_batteries()
        modules = balanced_graph.get_actions()

        capacities = {
            battery: float(rng.choice(CAPACITY_VALUES))
            for battery in batteries
        }
        demands = {
            module: rng.choice(DEMAND_VALUES)
            for module in modules
        }

        nearest = balanced_graph.nearest_battery_assignment(
            batteries=batteries,
            modules=modules,
            respect_switch_state=False,
        )
        balanced, _ = balanced_graph.rebalanced_nearest_battery_assignment(
            batteries=batteries,
            modules=modules,
            respect_switch_state=False,
        )
        nearest_rounds, nearest_drain = _static_horizon(
            nearest,
            batteries,
            capacities,
            demands,
        )
        balanced_rounds, balanced_drain = _static_horizon(
            balanced,
            batteries,
            capacities,
            demands,
        )

        nearest_simulation = exact_graph.simulate_until_failure(
            capacities,
            strategy="nearest",
            batteries=batteries,
            modules=modules,
            module_demand=demands,
            respect_switch_state=False,
            base_module_cost=BASE_MODULE_COST,
            switch_loss_factor=SWITCH_LOSS_FACTOR,
            max_rounds=MAX_SIMULATION_ROUNDS,
        )
        greedy_lifetime = exact_graph.simulate_until_failure(
            capacities,
            strategy="lifetime",
            batteries=batteries,
            modules=modules,
            module_demand=demands,
            respect_switch_state=False,
            max_extra_cost=MAX_EXTRA_COST,
            base_module_cost=BASE_MODULE_COST,
            switch_loss_factor=SWITCH_LOSS_FACTOR,
            max_rounds=MAX_SIMULATION_ROUNDS,
        )
        exact = exact_graph.maximize_total_rounds(
            capacities,
            batteries=batteries,
            modules=modules,
            module_demand=demands,
            respect_switch_state=False,
            max_extra_cost=MAX_EXTRA_COST,
            base_module_cost=BASE_MODULE_COST,
            switch_loss_factor=SWITCH_LOSS_FACTOR,
        )

        assert nearest_rounds == nearest_simulation.completed_rounds
        assert all(
            math.isclose(nearest[module].distance, balanced[module].distance)
            for module in modules
        )
        assert math.isclose(
            sum(nearest_drain.values()),
            sum(balanced_drain.values()),
        )
        assert _count_load_objective(balanced, batteries) <= _count_load_objective(
            nearest,
            batteries,
        )
        assert greedy_lifetime.completed_rounds <= exact.max_rounds

        rows.append({
            "dual_nearest_static": nearest_rounds,
            "balanced_static": balanced_rounds,
            "greedy_lifetime": greedy_lifetime.completed_rounds,
            "exact_restricted_repository_oracle": exact.max_rounds,
            "total_energy_difference_balanced_minus_nearest": (
                sum(balanced_drain.values()) - sum(nearest_drain.values())
            ),
        })

    methods = (
        "dual_nearest_static",
        "balanced_static",
        "greedy_lifetime",
        "exact_restricted_repository_oracle",
    )
    summaries = [_method_summary(method, rows) for method in methods]

    return {
        "configuration": {
            "seed": seed,
            "trials": trials,
            "grid": "3x3",
            "battery_count": BATTERY_COUNT,
            "action_module_count": GRID_SIZE * GRID_SIZE - BATTERY_COUNT,
            "switch_state": "all available; planning mode",
            "edge_cost": 1.0,
            "battery_capacity_distribution": "discrete uniform {30,35,...,60}",
            "module_demand_distribution": (
                "discrete uniform {0.5,0.75,1,1.25,1.5,2}"
            ),
            "energy_model": "demand * (1 + route_distance)",
            "maximum_extra_route_cost_for_greedy_and_exact": MAX_EXTRA_COST,
            "exact_method_label": (
                "restricted-model repository oracle; not a literature baseline"
            ),
            "greedy_method_label": "one-step greedy lifetime heuristic",
            "electrical_caveat": (
                "No current/voltage constraints, switching penalty, or "
                "multi-battery-domain isolation constraint"
            ),
        },
        "method_summary": summaries,
        "paired_comparisons": {
            "balanced_vs_dual_nearest": {
                "better_rate_percent": round(
                    100.0 * statistics.mean(
                        row["balanced_static"] > row["dual_nearest_static"]
                        for row in rows
                    ),
                    10,
                ),
                "equal_rate_percent": round(
                    100.0 * statistics.mean(
                        row["balanced_static"] == row["dual_nearest_static"]
                        for row in rows
                    ),
                    10,
                ),
                "worse_rate_percent": round(
                    100.0 * statistics.mean(
                        row["balanced_static"] < row["dual_nearest_static"]
                        for row in rows
                    ),
                    10,
                ),
                "mean_total_energy_difference_per_round": statistics.mean(
                    row["total_energy_difference_balanced_minus_nearest"]
                    for row in rows
                ),
            }
        },
        "sanity_assertions": [
            "Static dual-nearest horizon equals the nearest lifetime simulator",
            "Balanced preserves every module's nearest route distance",
            "Balanced preserves total per-round route energy",
            "Balanced never worsens its count-load objective",
            "Greedy lifetime horizon never exceeds the restricted exact horizon",
        ],
    }


def _write_json(result: Mapping[str, Any], output_path: Path) -> None:
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(result: Mapping[str, Any], output_path: Path) -> None:
    columns = [
        "method",
        "mean_feasible_rounds",
        "median_feasible_rounds",
        "minimum_feasible_rounds",
        "maximum_feasible_rounds",
        "mean_gap_to_exact",
        "exact_match_rate_percent",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(result["method_summary"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    args = parser.parse_args()

    result = run_benchmark(seed=args.seed, trials=args.trials)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "energy_horizon_summary.json"
    csv_path = args.output_dir / "energy_horizon_summary.csv"
    _write_json(result, json_path)
    _write_csv(result, csv_path)
    print(json.dumps({
        "json": str(json_path),
        "csv": str(csv_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
