#!/usr/bin/env python3
"""Reproducible nearest-vs-balanced routing and switch-safety benchmark.

All action modules are active. The benchmark still measures the raw union of
``path_switches`` returned by ``rebalanced_nearest_battery_assignment`` so the
historical safety gap remains visible. It separately checks the fail-closed
``recommend_balanced_switch_plan`` topology gate.

No third-party packages are required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from power_routing_balanced import ModularRobotGraph


DEFAULT_SEED = 20260720
DEFAULT_TRIALS = 200
DEFAULT_GRID_SIZE = 6
DEFAULT_BATTERIES = 4
DEFAULT_OUTAGE_RATES = (0.0, 0.10, 0.25)


def _scenario_seed(base_seed: int, outage_rate: float, trial: int) -> int:
    """Give every outage-rate/trial pair its own deterministic random stream."""
    return base_seed + int(outage_rate * 10_000) * 1_000 + trial


def _build_random_grid(
    *,
    grid_size: int,
    battery_count: int,
    rng: random.Random,
    outage_rate: float,
) -> ModularRobotGraph:
    positions = [
        (x, y)
        for x in range(grid_size)
        for y in range(grid_size)
    ]
    battery_positions = rng.sample(positions, battery_count)
    battery_at = {
        position: f"B{index + 1}"
        for index, position in enumerate(battery_positions)
    }

    modules: Dict[str, Dict[str, Any]] = {}
    module_index = 1
    for position in positions:
        if position in battery_at:
            modules[battery_at[position]] = {
                "type": "battery",
                "pos": position,
            }
        else:
            modules[f"M{module_index:02d}"] = {
                "type": "action",
                "pos": position,
            }
            module_index += 1

    graph = ModularRobotGraph.from_grid_layout(modules)
    for switch_id in sorted(graph.switch_open):
        graph.set_switch_state(
            switch_id,
            rng.random() >= outage_rate,
        )
    return graph


def _load_metrics(
    assignments: Mapping[str, Any],
    batteries: Sequence[str],
) -> Dict[str, Any]:
    loads = {battery: 0 for battery in batteries}
    total_route_distance = 0.0
    unreachable = 0

    for assignment in assignments.values():
        if assignment.battery is None or math.isinf(assignment.distance):
            unreachable += 1
            continue
        loads[assignment.battery] += 1
        total_route_distance += assignment.distance

    values = list(loads.values())
    load_mean = sum(values) / len(values)
    return {
        "loads": loads,
        "load_spread": max(values) - min(values),
        # This is the exact unnormalized variance term used by
        # ModularRobotGraph._load_balance_objective.
        "load_variance_sse": sum((value - load_mean) ** 2 for value in values),
        "total_route_distance": total_route_distance,
        "unreachable_modules": unreachable,
    }


def _load_objective(load_tuple: Tuple[int, ...]) -> Tuple[int, float, Tuple[int, ...]]:
    load_mean = sum(load_tuple) / len(load_tuple)
    return (
        max(load_tuple) - min(load_tuple),
        sum((value - load_mean) ** 2 for value in load_tuple),
        tuple(sorted(load_tuple, reverse=True)),
    )


def _equal_shortest_load_oracle(
    graph: ModularRobotGraph,
    batteries: Sequence[str],
    modules: Sequence[str],
) -> Tuple[int, ...]:
    """Globally minimize the count-load objective within shortest-route ties.

    This is an internal dynamic-programming oracle for the current abstraction,
    not a literature baseline.  A state is the battery load vector after a
    prefix of modules.  Unreachable modules add no load, matching the heuristic.
    """
    distance_table = graph.all_battery_distance_table(
        batteries=batteries,
        modules=modules,
        respect_switch_state=True,
    )
    states: Set[Tuple[int, ...]] = {(0,) * len(batteries)}

    for module in modules:
        minimum_distance = min(distance_table[module].values())
        if math.isinf(minimum_distance):
            continue
        candidate_indices = [
            index
            for index, battery in enumerate(batteries)
            if math.isclose(distance_table[module][battery], minimum_distance)
        ]
        next_states: Set[Tuple[int, ...]] = set()
        for state in states:
            for index in candidate_indices:
                next_state = list(state)
                next_state[index] += 1
                next_states.add(tuple(next_state))
        states = next_states

    return min(states, key=_load_objective)


def _switch_union(assignments: Mapping[str, Any]) -> Set[str]:
    return {
        switch_id
        for assignment in assignments.values()
        for switch_id in (assignment.path_switches or [])
    }


def _switch_safety_metrics(
    graph: ModularRobotGraph,
    switch_ids: Iterable[str],
    batteries: Sequence[str],
) -> Dict[str, Any]:
    switch_set = set(switch_ids)
    vertices: Set[str] = set()
    edges: List[Tuple[str, str]] = []
    for switch_id in switch_set:
        u, v = graph.switch_endpoints[switch_id]
        vertices.update((u, v))
        edges.append((u, v))

    parent = {vertex: vertex for vertex in vertices}

    def find(vertex: str) -> str:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    for u, v in edges:
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            parent[root_u] = root_v

    batteries_by_component: Dict[str, List[str]] = {}
    for battery in batteries:
        if battery in vertices:
            batteries_by_component.setdefault(find(battery), []).append(battery)

    multi_battery_component = any(
        len(component_batteries) > 1
        for component_batteries in batteries_by_component.values()
    )
    component_count = len({find(vertex) for vertex in vertices}) if vertices else 0
    cycle_rank = len(edges) - len(vertices) + component_count

    return {
        "switch_count": len(switch_set),
        "multi_battery_component": multi_battery_component,
        "cycle_rank": cycle_rank,
        "has_cycle": cycle_rank > 0,
    }


def _targeted_diagnostics() -> Dict[str, Any]:
    diagonal_spec: Dict[str, Dict[str, Any]] = {}
    for y in range(3):
        for x in range(3):
            node_id = (
                "B1" if (x, y) == (0, 0)
                else "B2" if (x, y) == (2, 2)
                else f"M{x}{y}"
            )
            diagonal_spec[node_id] = {
                "type": "battery" if node_id.startswith("B") else "action",
                "pos": (x, y),
            }

    tie_graph = ModularRobotGraph.from_grid_layout(diagonal_spec)
    sorted_modules = tie_graph.get_actions()
    reversed_modules = list(reversed(sorted_modules))

    def changed_modules(module_order: Sequence[str]) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
        nearest = tie_graph.nearest_battery_assignment(
            modules=module_order,
            respect_switch_state=False,
        )
        balanced, loads = tie_graph.rebalanced_nearest_battery_assignment(
            modules=module_order,
            respect_switch_state=False,
        )
        changed = [
            {
                "module": module,
                "from": nearest[module].battery,
                "to": balanced[module].battery,
            }
            for module in module_order
            if nearest[module].battery != balanced[module].battery
        ]
        return changed, loads

    sorted_changes, sorted_loads = changed_modules(sorted_modules)
    reversed_changes, reversed_loads = changed_modules(reversed_modules)

    bottom_corner_spec: Dict[str, Dict[str, Any]] = {}
    module_index = 1
    for y in range(3):
        for x in range(3):
            if (x, y) == (0, 0):
                node_id = "B1"
            elif (x, y) == (2, 0):
                node_id = "B2"
            else:
                node_id = f"M{module_index:02d}"
                module_index += 1
            bottom_corner_spec[node_id] = {
                "type": "battery" if node_id.startswith("B") else "action",
                "pos": (x, y),
            }

    switch_graph = ModularRobotGraph.from_grid_layout(bottom_corner_spec)
    batteries = switch_graph.get_batteries()
    modules = switch_graph.get_actions()
    nearest = switch_graph.nearest_battery_assignment(
        modules=modules,
        respect_switch_state=False,
    )
    balanced, _ = switch_graph.rebalanced_nearest_battery_assignment(
        modules=modules,
        respect_switch_state=False,
    )
    api_plan = switch_graph.recommend_switch_plan(
        active_modules=modules,
        respect_switch_state=False,
    )
    balanced_api_plan = switch_graph.recommend_balanced_switch_plan(
        active_modules=modules,
        respect_switch_state=False,
    )
    nearest_safety = _switch_safety_metrics(
        switch_graph,
        _switch_union(nearest),
        batteries,
    )
    balanced_safety = _switch_safety_metrics(
        switch_graph,
        _switch_union(balanced),
        batteries,
    )

    def same_assignments(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        return all(
            left[module].battery == right[module].battery
            and math.isclose(left[module].distance, right[module].distance)
            for module in modules
        )

    switch_changes = [
        {
            "module": module,
            "from": nearest[module].battery,
            "to": balanced[module].battery,
            "balanced_path": balanced[module].path_nodes,
        }
        for module in modules
        if nearest[module].battery != balanced[module].battery
    ]

    zero_cost_graph = ModularRobotGraph()
    for node_id, node_type in (
        ("B1", "battery"),
        ("B2", "battery"),
        ("X", "action"),
        ("M", "action"),
    ):
        zero_cost_graph.add_node(node_id, node_type)
    zero_cost_graph.add_switch_edge("B1", "X", "S1", cost=0.0)
    zero_cost_graph.add_switch_edge("X", "B2", "S2", cost=0.0)
    zero_cost_graph.add_switch_edge("B2", "M", "S3", cost=1.0)
    _, zero_parent, zero_owner = zero_cost_graph.multi_source_shortest_paths(
        ["B1", "B2"],
        weighted=True,
        respect_switch_state=False,
    )
    zero_plan = zero_cost_graph.recommend_switch_plan(
        active_modules=["M"],
        batteries=["B1", "B2"],
        weighted=True,
        respect_switch_state=False,
    )
    zero_safety = _switch_safety_metrics(
        zero_cost_graph,
        zero_plan["required_open_switches"],
        ["B1", "B2"],
    )

    return {
        "tie_order_sensitivity": {
            "topology": "3x3 grid; B1=(0,0), B2=(2,2)",
            "sorted_module_order_changes": sorted_changes,
            "reversed_module_order_changes": reversed_changes,
            "sorted_loads": sorted_loads,
            "reversed_loads": reversed_loads,
            "same_final_load_vector": sorted_loads == reversed_loads,
        },
        "balanced_switch_plan_integration": {
            "topology": "3x3 grid; B1=(0,0), B2=(2,0)",
            "recommend_switch_plan_equals_nearest": same_assignments(
                api_plan["assignments"],
                nearest,
            ),
            "recommend_switch_plan_equals_balanced": same_assignments(
                api_plan["assignments"],
                balanced,
            ),
            "balanced_api_equals_balanced": same_assignments(
                balanced_api_plan["assignments"],
                balanced,
            ),
            "balanced_api_safe_to_apply": balanced_api_plan["safe_to_apply"],
            "balanced_api_battery_conflicts": balanced_api_plan["battery_conflicts"],
            "balanced_api_emitted_switch_count": len(
                balanced_api_plan["required_closed_switches"]
            ),
            "balanced_api_candidate_switch_count": len(
                balanced_api_plan["candidate_required_closed_switches"]
            ),
            "balanced_changes": switch_changes,
            "nearest_switch_count": nearest_safety["switch_count"],
            "balanced_manual_switch_count": balanced_safety["switch_count"],
            "nearest_multi_battery_component": nearest_safety[
                "multi_battery_component"
            ],
            "balanced_manual_multi_battery_component": balanced_safety[
                "multi_battery_component"
            ],
        },
        "dual_weighted_zero_cost_edge_case": {
            "topology": "B1 --0-- X --0-- B2 --1-- M",
            "owner_of_source_B2_after_search": zero_owner["B2"],
            "parent_of_source_B2_after_search": (
                list(zero_parent["B2"])
                if zero_parent["B2"] is not None
                else None
            ),
            "assigned_path_to_M": zero_plan["assignments"]["M"].path_nodes,
            "required_open_switches": zero_plan["required_open_switches"],
            "multi_battery_component": zero_safety["multi_battery_component"],
            "scope_note": (
                "Balanced mode now keeps every selected battery immutable as a "
                "source root, including when zero-cost weighted edges are allowed"
            ),
        },
    }


def run_benchmark(
    *,
    base_seed: int,
    trials: int,
    grid_size: int,
    battery_count: int,
    outage_rates: Sequence[float],
) -> Dict[str, Any]:
    summaries: List[Dict[str, Any]] = []

    for outage_rate in outage_rates:
        rows: List[Dict[str, Any]] = []
        for trial in range(trials):
            rng = random.Random(_scenario_seed(base_seed, outage_rate, trial))
            graph = _build_random_grid(
                grid_size=grid_size,
                battery_count=battery_count,
                rng=rng,
                outage_rate=outage_rate,
            )
            batteries = graph.get_batteries()
            modules = graph.get_actions()

            nearest = graph.nearest_battery_assignment(
                batteries=batteries,
                modules=modules,
                respect_switch_state=True,
            )
            balanced, _ = graph.rebalanced_nearest_battery_assignment(
                batteries=batteries,
                modules=modules,
                respect_switch_state=True,
            )

            oracle_load_tuple = _equal_shortest_load_oracle(
                graph,
                batteries,
                modules,
            )

            nearest_load = _load_metrics(nearest, batteries)
            balanced_load = _load_metrics(balanced, batteries)
            nearest_safety = _switch_safety_metrics(
                graph,
                _switch_union(nearest),
                batteries,
            )
            balanced_safety = _switch_safety_metrics(
                graph,
                _switch_union(balanced),
                batteries,
            )

            nearest_unreachable = {
                module
                for module, assignment in nearest.items()
                if assignment.battery is None
            }
            balanced_unreachable = {
                module
                for module, assignment in balanced.items()
                if assignment.battery is None
            }
            assert nearest_unreachable == balanced_unreachable
            assert all(
                (
                    math.isinf(nearest[module].distance)
                    and math.isinf(balanced[module].distance)
                )
                or math.isclose(
                    nearest[module].distance,
                    balanced[module].distance,
                )
                for module in modules
            )
            assert (
                balanced_load["load_spread"],
                balanced_load["load_variance_sse"],
            ) <= (
                nearest_load["load_spread"],
                nearest_load["load_variance_sse"],
            )

            balanced_load_tuple = tuple(
                balanced_load["loads"][battery]
                for battery in batteries
            )
            balanced_objective = _load_objective(balanced_load_tuple)
            oracle_objective = _load_objective(oracle_load_tuple)
            assert oracle_objective <= balanced_objective

            rows.append({
                "nearest_load": nearest_load,
                "balanced_load": balanced_load,
                "nearest_safety": nearest_safety,
                "balanced_safety": balanced_safety,
                "balanced_objective": balanced_objective,
                "oracle_objective": oracle_objective,
            })

        summaries.append({
            "outage_rate_percent": int(round(outage_rate * 100)),
            "trials": trials,
            "mean_unreachable_modules": mean(
                row["nearest_load"]["unreachable_modules"]
                for row in rows
            ),
            "nearest": {
                "mean_load_spread": mean(
                    row["nearest_load"]["load_spread"]
                    for row in rows
                ),
                "mean_load_variance_sse": mean(
                    row["nearest_load"]["load_variance_sse"]
                    for row in rows
                ),
                "mean_total_route_distance": mean(
                    row["nearest_load"]["total_route_distance"]
                    for row in rows
                ),
                "mean_switch_count": mean(
                    row["nearest_safety"]["switch_count"]
                    for row in rows
                ),
                "multi_battery_component_rate_percent": round(
                    100.0 * mean(
                        row["nearest_safety"]["multi_battery_component"]
                        for row in rows
                    ),
                    10,
                ),
                "cycle_rate_percent": round(
                    100.0 * mean(
                        row["nearest_safety"]["has_cycle"]
                        for row in rows
                    ),
                    10,
                ),
            },
            "balanced": {
                "mean_load_spread": mean(
                    row["balanced_load"]["load_spread"]
                    for row in rows
                ),
                "mean_load_variance_sse": mean(
                    row["balanced_load"]["load_variance_sse"]
                    for row in rows
                ),
                "mean_total_route_distance": mean(
                    row["balanced_load"]["total_route_distance"]
                    for row in rows
                ),
                "mean_switch_count": mean(
                    row["balanced_safety"]["switch_count"]
                    for row in rows
                ),
                "multi_battery_component_rate_percent": round(
                    100.0 * mean(
                        row["balanced_safety"]["multi_battery_component"]
                        for row in rows
                    ),
                    10,
                ),
                "cycle_rate_percent": round(
                    100.0 * mean(
                        row["balanced_safety"]["has_cycle"]
                        for row in rows
                    ),
                    10,
                ),
            },
            "internal_equal_shortest_dp_oracle": {
                "description": (
                    "Global count-load objective over equal-shortest-distance "
                    "battery candidates; internal oracle, not a literature baseline"
                ),
                "balanced_objective_match_rate_percent": round(
                    100.0 * mean(
                        row["balanced_objective"] == row["oracle_objective"]
                        for row in rows
                    ),
                    10,
                ),
                "mean_balanced_load_spread_gap": mean(
                    row["balanced_objective"][0] - row["oracle_objective"][0]
                    for row in rows
                ),
                "mean_balanced_load_variance_sse_gap": mean(
                    row["balanced_objective"][1] - row["oracle_objective"][1]
                    for row in rows
                ),
            },
        })

    return {
        "configuration": {
            "base_seed": base_seed,
            "trials_per_outage_rate": trials,
            "grid": f"{grid_size}x{grid_size}",
            "battery_count": battery_count,
            "action_modules": grid_size * grid_size - battery_count,
            "outage_rates_percent": [
                int(round(rate * 100))
                for rate in outage_rates
            ],
            "edge_cost": 1.0,
            "all_action_modules_active": True,
            "switch_plan_derivation": "union of assignment path_switches",
            "load_variance_definition": "sum((load - mean_load)^2)",
            "internal_oracle": (
                "dynamic programming over load-vector states and "
                "equal-shortest-distance battery candidates"
            ),
        },
        "summary": summaries,
        "targeted_diagnostics": _targeted_diagnostics(),
    }


def _write_json(result: Mapping[str, Any], output_path: Path) -> None:
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(result: Mapping[str, Any], output_path: Path) -> None:
    columns = [
        "outage_rate_percent",
        "trials",
        "method",
        "mean_unreachable_modules",
        "mean_load_spread",
        "mean_load_variance_sse",
        "mean_total_route_distance",
        "mean_switch_count",
        "multi_battery_component_rate_percent",
        "cycle_rate_percent",
        "balanced_internal_oracle_match_rate_percent",
        "balanced_mean_load_spread_gap_to_internal_oracle",
        "balanced_mean_load_variance_sse_gap_to_internal_oracle",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for summary in result["summary"]:
            for method in ("nearest", "balanced"):
                writer.writerow({
                    "outage_rate_percent": summary["outage_rate_percent"],
                    "trials": summary["trials"],
                    "method": method,
                    "mean_unreachable_modules": summary["mean_unreachable_modules"],
                    **summary[method],
                    "balanced_internal_oracle_match_rate_percent": (
                        summary["internal_equal_shortest_dp_oracle"]
                        ["balanced_objective_match_rate_percent"]
                        if method == "balanced"
                        else ""
                    ),
                    "balanced_mean_load_spread_gap_to_internal_oracle": (
                        summary["internal_equal_shortest_dp_oracle"]
                        ["mean_balanced_load_spread_gap"]
                        if method == "balanced"
                        else ""
                    ),
                    "balanced_mean_load_variance_sse_gap_to_internal_oracle": (
                        summary["internal_equal_shortest_dp_oracle"]
                        ["mean_balanced_load_variance_sse_gap"]
                        if method == "balanced"
                        else ""
                    ),
                })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--batteries", type=int, default=DEFAULT_BATTERIES)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    args = parser.parse_args()

    result = run_benchmark(
        base_seed=args.seed,
        trials=args.trials,
        grid_size=args.grid_size,
        battery_count=args.batteries,
        outage_rates=DEFAULT_OUTAGE_RATES,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "balanced_safety_summary.json"
    csv_path = args.output_dir / "balanced_safety_summary.csv"
    _write_json(result, json_path)
    _write_csv(result, csv_path)
    print(json.dumps({
        "json": str(json_path),
        "csv": str(csv_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
