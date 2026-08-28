#!/usr/bin/env python3
"""Deterministic 10x10 benchmark for duo-mode versus balanced assignment.

The suite contains six connected layout families for each battery count from
one through five (30 cases total).  Both algorithms run on the same physical
layout in unweighted planning mode, with every action module active.

The duo result is produced by ``power_routing_duo_mode.ModularRobotGraph``.
The improved result is produced by
``power_routing_balanced.ModularRobotGraph.rebalanced_nearest_battery_assignment``.
The balanced file's nearest-only implementation is also evaluated as a parity
check against the duo implementation.

No third-party packages are required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import deque
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from power_routing_balanced import ModularRobotGraph as BalancedGraph
from power_routing_duo_mode import ModularRobotGraph as DuoGraph


GRID_SIZE = 10
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "experiments" / "results"
Position = Tuple[int, int]


def _curated_serpentine(_: int) -> Set[Position]:
    cells = {(x, 1) for x in range(10)}
    cells.update({(9, 2)})
    cells.update({(x, 3) for x in range(10)})
    cells.update({(0, 4)})
    cells.update({(x, 5) for x in range(10)})
    return cells


def _curated_dense_rectangle(_: int) -> Set[Position]:
    return {(x, y) for x in range(1, 9) for y in range(1, 6)}


def _curated_cross_hub(_: int) -> Set[Position]:
    return (
        {(x, 4) for x in range(10)}
        | {(4, y) for y in range(10)}
    )


def _curated_perimeter_ring(_: int) -> Set[Position]:
    return {
        (x, y)
        for x in range(1, 9)
        for y in range(1, 8)
        if x in (1, 8) or y in (1, 7)
    }


def _curated_h_frame(_: int) -> Set[Position]:
    return (
        {(1, y) for y in range(9)}
        | {(8, y) for y in range(9)}
        | {(x, 4) for x in range(1, 9)}
    )


def _curated_twin_rooms(_: int) -> Set[Position]:
    left_room = {(x, y) for x in range(0, 4) for y in range(1, 6)}
    right_room = {(x, y) for x in range(6, 10) for y in range(1, 6)}
    return left_room | {(4, 3), (5, 3)} | right_room


LAYOUT_BUILDERS = [
    ("serpentine", _curated_serpentine),
    ("dense_rectangle", _curated_dense_rectangle),
    ("cross_hub", _curated_cross_hub),
    ("perimeter_ring", _curated_perimeter_ring),
    ("h_frame", _curated_h_frame),
    ("twin_rooms", _curated_twin_rooms),
]


BATTERY_POSITION_TABLE: Dict[int, Dict[str, Sequence[Position]]] = {
    1: {
        "serpentine": [(0, 1)],
        "dense_rectangle": [(4, 3)],
        "cross_hub": [(4, 4)],
        "perimeter_ring": [(1, 1)],
        "h_frame": [(1, 4)],
        "twin_rooms": [(4, 3)],
    },
    2: {
        "serpentine": [(2, 1), (4, 5)],
        "dense_rectangle": [(2, 3), (1, 4)],
        "cross_hub": [(0, 4), (8, 4)],
        "perimeter_ring": [(1, 1), (7, 1)],
        "h_frame": [(1, 0), (8, 8)],
        "twin_rooms": [(3, 3), (9, 3)],
    },
    3: {
        "serpentine": [(5, 1), (7, 3), (9, 3)],
        "dense_rectangle": [(1, 1), (8, 1), (4, 5)],
        "cross_hub": [(0, 4), (9, 4), (4, 0)],
        "perimeter_ring": [(7, 1), (5, 7), (7, 7)],
        "h_frame": [(4, 4), (1, 7), (8, 8)],
        "twin_rooms": [(0, 1), (9, 1), (7, 5)],
    },
    4: {
        "serpentine": [(9, 2), (0, 3), (8, 3), (8, 5)],
        "dense_rectangle": [(1, 1), (8, 1), (1, 5), (8, 5)],
        "cross_hub": [(0, 4), (9, 4), (4, 0), (4, 9)],
        "perimeter_ring": [(1, 1), (8, 1), (8, 7), (1, 7)],
        "h_frame": [(1, 0), (8, 0), (1, 8), (8, 8)],
        "twin_rooms": [(2, 3), (3, 4), (9, 4), (8, 5)],
    },
    5: {
        "serpentine": [(2, 1), (1, 3), (7, 3), (8, 3), (9, 5)],
        "dense_rectangle": [(1, 1), (8, 1), (1, 5), (8, 5), (4, 3)],
        "cross_hub": [(0, 4), (9, 4), (4, 0), (4, 9), (4, 4)],
        "perimeter_ring": [(1, 1), (8, 1), (8, 7), (1, 7), (4, 1)],
        "h_frame": [(1, 0), (8, 0), (1, 8), (8, 8), (4, 4)],
        "twin_rooms": [(0, 1), (3, 5), (6, 5), (9, 1), (5, 3)],
    },
}


def _neighbors(position: Position) -> Iterable[Position]:
    x, y = position
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        yield x + dx, y + dy


def _validate_connected_layout(cells: Set[Position]) -> None:
    if not cells:
        raise ValueError("Layout must contain at least one occupied cell.")
    if any(not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE) for x, y in cells):
        raise ValueError("All occupied cells must lie inside the 10x10 grid.")

    start = min(cells)
    reached = {start}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for neighbor in _neighbors(current):
            if neighbor in cells and neighbor not in reached:
                reached.add(neighbor)
                queue.append(neighbor)
    if reached != cells:
        raise ValueError(f"Layout is disconnected: reached {len(reached)} of {len(cells)} cells.")


def _select_battery_positions(
    cells: Set[Position],
    battery_count: int,
    family_name: str,
) -> List[Position]:
    positions = list(BATTERY_POSITION_TABLE[battery_count][family_name])
    if len(positions) != battery_count:
        raise AssertionError("Battery placement count does not match the case.")
    if len(set(positions)) != battery_count:
        raise AssertionError("Battery positions must be unique.")
    if not set(positions).issubset(cells):
        raise AssertionError("Every battery must occupy a cell in its layout.")
    return positions


def _build_module_spec(
    cells: Set[Position],
    battery_positions: Sequence[Position],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[Position, str]]:
    battery_at = {
        position: f"B{index + 1}"
        for index, position in enumerate(battery_positions)
    }
    modules: Dict[str, Dict[str, Any]] = {}
    node_at: Dict[Position, str] = {}
    module_index = 1

    # Row-major by y then x gives a stable adjacency insertion order.
    for position in sorted(cells, key=lambda item: (item[1], item[0])):
        if position in battery_at:
            node_id = battery_at[position]
            node_type = "battery"
        else:
            node_id = f"M{module_index:03d}"
            node_type = "action"
            module_index += 1
        modules[node_id] = {"type": node_type, "pos": position}
        node_at[position] = node_id
    return modules, node_at


def _load_metrics(
    assignments: Mapping[str, Any],
    batteries: Sequence[str],
) -> Dict[str, Any]:
    loads = {battery: 0 for battery in batteries}
    distances: List[float] = []
    unreachable = 0
    for assignment in assignments.values():
        if assignment.battery is None or math.isinf(assignment.distance):
            unreachable += 1
            continue
        loads[assignment.battery] += 1
        distances.append(float(assignment.distance))

    load_values = [loads[battery] for battery in batteries]
    load_mean = sum(load_values) / len(load_values)
    return {
        "loads": loads,
        "load_vector": load_values,
        "max_load": max(load_values),
        "load_spread": max(load_values) - min(load_values),
        "load_sse": sum((value - load_mean) ** 2 for value in load_values),
        "objective": [
            max(load_values) - min(load_values),
            sum((value - load_mean) ** 2 for value in load_values),
            sorted(load_values, reverse=True),
        ],
        "total_route_distance": sum(distances),
        "mean_route_distance": mean(distances) if distances else None,
        "max_route_distance": max(distances) if distances else None,
        "unreachable_modules": unreachable,
    }


def _assignment_payload(
    assignments: Mapping[str, Any],
    graph: Any,
) -> Dict[str, Any]:
    return {
        module: {
            "position": list(graph.node_pos[module]),
            "battery": assignment.battery,
            "distance": None if math.isinf(assignment.distance) else assignment.distance,
            "path_nodes": assignment.path_nodes,
            "path_switches": assignment.path_switches,
        }
        for module, assignment in sorted(assignments.items())
    }


def _run_case(
    battery_count: int,
    family_index: int,
    family_name: str,
    cells: Set[Position],
) -> Dict[str, Any]:
    _validate_connected_layout(cells)
    battery_positions = _select_battery_positions(cells, battery_count, family_name)
    modules, node_at = _build_module_spec(cells, battery_positions)

    duo_graph = DuoGraph.from_grid_layout(modules)
    balanced_graph = BalancedGraph.from_grid_layout(modules)
    batteries = sorted(duo_graph.get_batteries())
    action_modules = sorted(duo_graph.get_actions())

    duo_assignments = duo_graph.nearest_battery_assignment(
        batteries=batteries,
        modules=action_modules,
        weighted=False,
        respect_switch_state=False,
    )
    balanced_nearest = balanced_graph.nearest_battery_assignment(
        batteries=batteries,
        modules=action_modules,
        weighted=False,
        respect_switch_state=False,
    )
    balanced_assignments, returned_loads = (
        balanced_graph.rebalanced_nearest_battery_assignment(
            batteries=batteries,
            modules=action_modules,
            weighted=False,
            respect_switch_state=False,
        )
    )

    # Confirm the baseline duplicated in balanced.py is identical to the actual
    # duo-mode implementation on every generated case.
    for module in action_modules:
        duo_result = duo_assignments[module]
        parity_result = balanced_nearest[module]
        assert duo_result.battery == parity_result.battery
        assert math.isclose(duo_result.distance, parity_result.distance)

    duo_metrics = _load_metrics(duo_assignments, batteries)
    balanced_metrics = _load_metrics(balanced_assignments, batteries)
    assert returned_loads == balanced_metrics["loads"]
    assert duo_metrics["unreachable_modules"] == 0
    assert balanced_metrics["unreachable_modules"] == 0

    changed_modules = []
    for module in action_modules:
        duo_result = duo_assignments[module]
        balanced_result = balanced_assignments[module]
        assert math.isclose(duo_result.distance, balanced_result.distance)
        if duo_result.battery != balanced_result.battery:
            changed_modules.append({
                "module": module,
                "position": list(duo_graph.node_pos[module]),
                "from": duo_result.battery,
                "to": balanced_result.battery,
                "distance": duo_result.distance,
            })

    duo_objective = (
        duo_metrics["load_spread"],
        duo_metrics["load_sse"],
        tuple(sorted(duo_metrics["load_vector"], reverse=True)),
    )
    balanced_objective = (
        balanced_metrics["load_spread"],
        balanced_metrics["load_sse"],
        tuple(sorted(balanced_metrics["load_vector"], reverse=True)),
    )
    assert balanced_objective <= duo_objective
    assert math.isclose(
        duo_metrics["total_route_distance"],
        balanced_metrics["total_route_distance"],
    )

    case_id = f"k{battery_count}_{family_index + 1:02d}_{family_name}"
    battery_payload = {
        battery: list(duo_graph.node_pos[battery])
        for battery in batteries
    }
    occupied_payload = [
        {
            "position": list(position),
            "node": node_at[position],
            "type": modules[node_at[position]]["type"],
        }
        for position in sorted(cells, key=lambda item: (item[1], item[0]))
    ]

    return {
        "case_id": case_id,
        "family": family_name,
        "grid_size": GRID_SIZE,
        "battery_count": battery_count,
        "action_module_count": len(action_modules),
        "occupied_cell_count": len(cells),
        "battery_positions": battery_payload,
        "occupied_cells": occupied_payload,
        "configuration": {
            "weighted": False,
            "respect_switch_state": False,
            "operating_mode": "planning/full-topology",
            "all_action_modules_active": True,
            "adjacency": "4-neighbor",
            "edge_cost": 1.0,
        },
        "duo": {
            "metrics": duo_metrics,
            "assignments": _assignment_payload(duo_assignments, duo_graph),
        },
        "balanced": {
            "metrics": balanced_metrics,
            "assignments": _assignment_payload(balanced_assignments, balanced_graph),
        },
        "comparison": {
            "changed_module_count": len(changed_modules),
            "changed_module_rate_percent": (
                100.0 * len(changed_modules) / len(action_modules)
            ),
            "changed_modules": changed_modules,
            "spread_reduction": (
                duo_metrics["load_spread"] - balanced_metrics["load_spread"]
            ),
            "load_sse_reduction": (
                duo_metrics["load_sse"] - balanced_metrics["load_sse"]
            ),
            "objective_improved": balanced_objective < duo_objective,
            "all_route_distances_preserved": True,
            "duo_file_matches_balanced_nearest_baseline": True,
        },
    }


def _aggregate(cases: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_battery_count: List[Dict[str, Any]] = []
    for battery_count in range(1, 6):
        group = [case for case in cases if case["battery_count"] == battery_count]
        duo_spreads = [case["duo"]["metrics"]["load_spread"] for case in group]
        balanced_spreads = [case["balanced"]["metrics"]["load_spread"] for case in group]
        duo_sses = [case["duo"]["metrics"]["load_sse"] for case in group]
        balanced_sses = [case["balanced"]["metrics"]["load_sse"] for case in group]
        duo_mean_spread = mean(duo_spreads)
        balanced_mean_spread = mean(balanced_spreads)
        duo_mean_sse = mean(duo_sses)
        balanced_mean_sse = mean(balanced_sses)

        by_battery_count.append({
            "battery_count": battery_count,
            "cases": len(group),
            "mean_action_modules": mean(case["action_module_count"] for case in group),
            "duo_mean_load_spread": duo_mean_spread,
            "balanced_mean_load_spread": balanced_mean_spread,
            "mean_spread_reduction_percent": (
                100.0 * (duo_mean_spread - balanced_mean_spread) / duo_mean_spread
                if duo_mean_spread else 0.0
            ),
            "duo_mean_load_sse": duo_mean_sse,
            "balanced_mean_load_sse": balanced_mean_sse,
            "mean_load_sse_reduction_percent": (
                100.0 * (duo_mean_sse - balanced_mean_sse) / duo_mean_sse
                if duo_mean_sse else 0.0
            ),
            "mean_changed_module_rate_percent": mean(
                case["comparison"]["changed_module_rate_percent"] for case in group
            ),
            "objective_improved_cases": sum(
                case["comparison"]["objective_improved"] for case in group
            ),
            "identical_assignment_cases": sum(
                case["comparison"]["changed_module_count"] == 0 for case in group
            ),
            "all_route_distances_preserved": all(
                case["comparison"]["all_route_distances_preserved"] for case in group
            ),
            "total_unreachable_modules": sum(
                case["balanced"]["metrics"]["unreachable_modules"] for case in group
            ),
        })

    multi_source = [case for case in cases if case["battery_count"] >= 2]
    duo_spread_mean = mean(case["duo"]["metrics"]["load_spread"] for case in multi_source)
    balanced_spread_mean = mean(
        case["balanced"]["metrics"]["load_spread"] for case in multi_source
    )
    duo_sse_mean = mean(case["duo"]["metrics"]["load_sse"] for case in multi_source)
    balanced_sse_mean = mean(
        case["balanced"]["metrics"]["load_sse"] for case in multi_source
    )

    return {
        "case_count": len(cases),
        "layouts_per_battery_count": 6,
        "battery_counts": [1, 2, 3, 4, 5],
        "by_battery_count": by_battery_count,
        "multi_source_overall": {
            "case_count": len(multi_source),
            "duo_mean_load_spread": duo_spread_mean,
            "balanced_mean_load_spread": balanced_spread_mean,
            "mean_spread_reduction_percent": (
                100.0 * (duo_spread_mean - balanced_spread_mean) / duo_spread_mean
                if duo_spread_mean else 0.0
            ),
            "duo_mean_load_sse": duo_sse_mean,
            "balanced_mean_load_sse": balanced_sse_mean,
            "mean_load_sse_reduction_percent": (
                100.0 * (duo_sse_mean - balanced_sse_mean) / duo_sse_mean
                if duo_sse_mean else 0.0
            ),
            "objective_improved_cases": sum(
                case["comparison"]["objective_improved"] for case in multi_source
            ),
            "mean_changed_module_rate_percent": mean(
                case["comparison"]["changed_module_rate_percent"]
                for case in multi_source
            ),
        },
        "invariants": {
            "all_duo_file_parity_checks_passed": all(
                case["comparison"]["duo_file_matches_balanced_nearest_baseline"]
                for case in cases
            ),
            "all_route_distances_preserved": all(
                case["comparison"]["all_route_distances_preserved"]
                for case in cases
            ),
            "total_unreachable_modules": sum(
                case["balanced"]["metrics"]["unreachable_modules"] for case in cases
            ),
        },
    }


def _representative_case_ids(cases: Sequence[Mapping[str, Any]]) -> List[str]:
    representative_family = {
        1: "cross_hub",
        2: "cross_hub",
        3: "h_frame",
        4: "dense_rectangle",
        5: "twin_rooms",
    }
    return [
        next(
            case["case_id"]
            for case in cases
            if case["battery_count"] == battery_count
            and case["family"] == representative_family[battery_count]
        )
        for battery_count in range(1, 6)
    ]


def run_suite() -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    for battery_count in range(1, 6):
        for family_index, (family_name, builder) in enumerate(LAYOUT_BUILDERS):
            cells = builder(battery_count)
            cases.append(
                _run_case(
                    battery_count,
                    family_index,
                    family_name,
                    cells,
                )
            )

    return {
        "study": {
            "title": "Duo-mode versus balanced assignment on 30 deterministic 10x10 layouts",
            "algorithm_baseline": "power_routing_duo_mode.nearest_battery_assignment",
            "algorithm_improved": "power_routing_balanced.rebalanced_nearest_battery_assignment",
            "load_definition": "count of assigned action modules per battery",
            "load_sse_definition": "sum((load - mean_load)^2)",
            "balanced_scope": (
                "local post-processing over equal-shortest-distance battery candidates"
            ),
            "safety_note": (
                "Balanced assignments are compared; independently reconstructed path unions "
                "are not treated as electrically validated switch plans."
            ),
        },
        "aggregate": _aggregate(cases),
        "representative_case_ids": _representative_case_ids(cases),
        "cases": cases,
    }


def _write_json(result: Mapping[str, Any], output_path: Path) -> None:
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(result: Mapping[str, Any], output_path: Path) -> None:
    columns = [
        "case_id",
        "family",
        "battery_count",
        "action_module_count",
        "occupied_cell_count",
        "duo_load_vector",
        "balanced_load_vector",
        "duo_load_spread",
        "balanced_load_spread",
        "spread_reduction",
        "duo_load_sse",
        "balanced_load_sse",
        "load_sse_reduction",
        "changed_module_count",
        "changed_module_rate_percent",
        "duo_total_route_distance",
        "balanced_total_route_distance",
        "all_route_distances_preserved",
        "unreachable_modules",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for case in result["cases"]:
            writer.writerow({
                "case_id": case["case_id"],
                "family": case["family"],
                "battery_count": case["battery_count"],
                "action_module_count": case["action_module_count"],
                "occupied_cell_count": case["occupied_cell_count"],
                "duo_load_vector": "/".join(
                    str(value) for value in case["duo"]["metrics"]["load_vector"]
                ),
                "balanced_load_vector": "/".join(
                    str(value) for value in case["balanced"]["metrics"]["load_vector"]
                ),
                "duo_load_spread": case["duo"]["metrics"]["load_spread"],
                "balanced_load_spread": case["balanced"]["metrics"]["load_spread"],
                "spread_reduction": case["comparison"]["spread_reduction"],
                "duo_load_sse": case["duo"]["metrics"]["load_sse"],
                "balanced_load_sse": case["balanced"]["metrics"]["load_sse"],
                "load_sse_reduction": case["comparison"]["load_sse_reduction"],
                "changed_module_count": case["comparison"]["changed_module_count"],
                "changed_module_rate_percent": case["comparison"]["changed_module_rate_percent"],
                "duo_total_route_distance": case["duo"]["metrics"]["total_route_distance"],
                "balanced_total_route_distance": case["balanced"]["metrics"]["total_route_distance"],
                "all_route_distances_preserved": case["comparison"]["all_route_distances_preserved"],
                "unreachable_modules": case["balanced"]["metrics"]["unreachable_modules"],
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for JSON and CSV results.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result = run_suite()
    json_path = args.output_dir / "duo_vs_balanced_10x10_results.json"
    csv_path = args.output_dir / "duo_vs_balanced_10x10_summary.csv"
    _write_json(result, json_path)
    _write_csv(result, csv_path)

    aggregate = result["aggregate"]["multi_source_overall"]
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Cases: {result['aggregate']['case_count']}")
    print(f"Representative cases: {result['representative_case_ids']}")
    print(
        "Multi-source mean spread: "
        f"{aggregate['duo_mean_load_spread']:.3f} -> "
        f"{aggregate['balanced_mean_load_spread']:.3f} "
        f"({aggregate['mean_spread_reduction_percent']:.1f}% lower)"
    )
    print(
        "Multi-source mean load SSE: "
        f"{aggregate['duo_mean_load_sse']:.3f} -> "
        f"{aggregate['balanced_mean_load_sse']:.3f} "
        f"({aggregate['mean_load_sse_reduction_percent']:.1f}% lower)"
    )


if __name__ == "__main__":
    main()
