#!/usr/bin/env python3
"""Run balanced routing for the role-aware JSON example and draw an SVG."""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from power_routing_balanced import ModularRobotGraph


DEFAULT_INPUT = Path(__file__).with_name("sample_role_layout.json")
DEFAULT_OUTPUT = REPOSITORY_ROOT / "output" / "balanced_role_layout_demo.svg"


def _assignment_loads(
    batteries: List[str],
    assignments: Dict[str, Any],
) -> Dict[str, int]:
    loads = {battery: 0 for battery in batteries}
    for assignment in assignments.values():
        if assignment.battery in loads:
            loads[assignment.battery] += 1
    return loads


def _assignment_demands(
    graph: ModularRobotGraph,
    batteries: List[str],
    assignments: Dict[str, Any],
) -> Dict[str, float]:
    demands = {battery: 0.0 for battery in batteries}
    for module, assignment in assignments.items():
        if assignment.battery in demands:
            demands[assignment.battery] += graph.get_module_demand_weight(module)
    return demands


def _svg_text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 16,
    weight: int = 400,
    fill: str = "#172033",
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Inter, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}">{escape(value)}</text>'
    )


def _svg_code_text(x: float, y: float, value: str, *, size: int = 10) -> str:
    return (
        f'<text x="{x}" y="{y}" '
        'font-family="SFMono-Regular, Menlo, Consolas, monospace" '
        f'font-size="{size}" font-weight="500" fill="#344054">'
        f'{escape(value)}</text>'
    )


def _layout_json_lines(graph: ModularRobotGraph) -> List[str]:
    """Reconstruct a compact, exact role-aware JSON representation."""
    if graph.grid_cells is None or graph.grid_rows is None or graph.grid_cols is None:
        raise ValueError("The graph does not contain layout metadata.")

    lines = [
        "{",
        '  "layout": {',
        f'    "cols": {graph.grid_cols},',
        f'    "rows": {graph.grid_rows},',
        '    "cells": [',
    ]
    for row_index, row in enumerate(graph.grid_cells):
        cells: List[str] = []
        for module in row:
            if module is None:
                cells.append("null")
                continue
            metadata = graph.get_module_metadata(module)
            value = (
                f"{module}({graph.module_rotation[module]}d,"
                f"[{metadata.role},{metadata.digital_io},{metadata.unused}])"
            )
            cells.append(json.dumps(value, ensure_ascii=False))
        suffix = "," if row_index < graph.grid_rows - 1 else ""
        lines.append(f"      [{', '.join(cells)}]{suffix}")
    lines.extend(["    ]", "  }", "}"])
    return lines


def _close_switch_json_lines(plan: Dict[str, Any]) -> List[str]:
    payload = {"close_switches": plan["required_closed_switches"]}
    return json.dumps(payload, indent=2, ensure_ascii=False).splitlines()


def render_layout_svg(
    graph: ModularRobotGraph,
    plan: Dict[str, Any],
    output_path: Path,
    *,
    baseline: Optional[Dict[str, Any]] = None,
) -> None:
    """Render the physical layout, selected links, routes, and battery loads."""
    if graph.grid_cells is None or graph.grid_rows is None or graph.grid_cols is None:
        raise ValueError("The graph does not contain layout metadata.")

    cell_width = 190
    cell_height = 158
    grid_x = 44
    grid_y = 100
    panel_x = grid_x + graph.grid_cols * cell_width + 55
    panel_width = 525
    width = panel_x + panel_width + 45
    box_width = 144
    box_height = 106

    battery_order = sorted(plan["battery_loads"])
    demand_mode = plan.get("mode") == "demand_aware"
    baseline_loads = (
        _assignment_loads(battery_order, baseline)
        if baseline is not None
        else None
    )
    baseline_demands = (
        _assignment_demands(graph, battery_order, baseline)
        if baseline is not None
        else None
    )
    moves = [
        (
            module,
            baseline[module].battery,
            plan["assignments"][module].battery,
        )
        for module in sorted(plan["assignments"])
        if baseline is not None
        and baseline[module].battery != plan["assignments"][module].battery
    ]
    json_lines = _layout_json_lines(graph)
    close_switch_json_lines = _close_switch_json_lines(plan)
    grid_bottom = grid_y + graph.grid_rows * cell_height
    json_panel_y = grid_bottom + 170
    json_line_height = 15
    json_panel_height = 62 + json_line_height * len(json_lines)
    height = max(
        730,
        grid_bottom + 135,
        json_panel_y + json_panel_height + 30,
        480
        + 28 * len(battery_order)
        + 51 * len(plan["assignments"])
        + 16 * len(close_switch_json_lines)
        + 26 * len(moves)
        + (55 if demand_mode else 0),
    )

    centers: Dict[str, Tuple[float, float]] = {}
    for row_index, row in enumerate(graph.grid_cells):
        for col_index, module in enumerate(row):
            if module is None:
                continue
            centers[module] = (
                grid_x + col_index * cell_width + cell_width / 2,
                grid_y + row_index * cell_height + cell_height / 2,
            )

    required_links = {tuple(link) for link in plan["required_links"]}
    domain_palette = [
        "#079669",
        "#7C3AED",
        "#DF5A00",
        "#2563EB",
    ]
    domain_colors = {
        battery: domain_palette[index % len(domain_palette)]
        for index, battery in enumerate(battery_order)
    }
    link_domains: Dict[Tuple[str, str], set[str]] = {}
    node_domains: Dict[str, set[str]] = {}
    for assignment in plan["assignments"].values():
        battery = assignment.battery
        if battery is None:
            continue
        path_nodes = assignment.path_nodes or []
        for node in path_nodes:
            node_domains.setdefault(node, set()).add(battery)
        for u, v in zip(path_nodes, path_nodes[1:]):
            link = tuple(sorted((u, v)))
            link_domains.setdefault(link, set()).add(battery)
    for battery in battery_order:
        node_domains.setdefault(battery, set()).add(battery)

    switch_domains: Dict[str, str] = {}
    for link, domains in link_domains.items():
        if len(domains) != 1:
            continue
        domain = next(iter(domains))
        for switch_id in graph.get_link_switches(*link):
            switch_domains[switch_id] = domain

    colors = {
        "battery": ("#FFF1B8", "#B7791F"),
        "action": ("#DCEBFF", "#2563EB"),
        "relay": ("#EEF1F5", "#667085"),
    }
    parts: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#F8FAFC"/>',
        _svg_text(
            44,
            48,
            (
                "MRPR demand-aware routing — role-aware layout"
                if demand_mode
                else "MRPR balanced routing — role-aware layout"
            ),
            size=25,
            weight=700,
        ),
        _svg_text(
            44,
            75,
            "Line colors show battery domains; T/R/B/L edge badges are the local switches to close.",
            size=14,
            fill="#526071",
        ),
    ]

    for row_index in range(graph.grid_rows):
        for col_index in range(graph.grid_cols):
            x = grid_x + col_index * cell_width + 5
            y = grid_y + row_index * cell_height + 5
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_width - 10}" '
                f'height="{cell_height - 10}" rx="16" fill="none" '
                'stroke="#D7DEE8" stroke-width="1.5" stroke-dasharray="5 6"/>'
            )

    for endpoints in sorted(graph.link_to_switches):
        u, v = endpoints
        x1, y1 = centers[u]
        x2, y2 = centers[v]
        selected = endpoints in required_links
        domains = link_domains.get(endpoints, set())
        if len(domains) > 1:
            link_color = "#D92D20"
        elif domains:
            link_color = domain_colors[next(iter(domains))]
        else:
            link_color = "#C8D1DC"
        parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{link_color}" '
            f'stroke-width="{8 if selected else 3}" stroke-linecap="round"/>'
        )

    for module in sorted(centers):
        center_x, center_y = centers[module]
        node_type = graph.node_type[module]
        fill, stroke = colors[node_type]
        domains = node_domains.get(module, set())
        x = center_x - box_width / 2
        y = center_y - box_height / 2
        metadata = graph.get_module_metadata(module)
        role_line = (
            f"role {metadata.role} · {metadata.role_name} "
            f"· w={metadata.demand_weight:g}"
        )
        rotation_line = (
            f"↺ {graph.module_rotation[module]}° · IO {metadata.digital_io_bits}"
        )
        parts.extend(
            [
                f'<rect x="{x}" y="{y}" width="{box_width}" height="{box_height}" '
                f'rx="17" fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>',
                _svg_text(center_x, y + 30, module, size=21, weight=700, anchor="middle"),
                _svg_text(center_x, y + 57, role_line, size=12, weight=600, anchor="middle"),
                _svg_text(
                    center_x,
                    y + 83,
                    rotation_line,
                    size=13,
                    fill="#455468",
                    anchor="middle",
                ),
            ]
        )
        if len(domains) == 1:
            domain = next(iter(domains))
            parts.append(
                f'<circle cx="{x + box_width - 13}" cy="{y + box_height - 13}" '
                f'r="6" fill="{domain_colors[domain]}">'
                f'<title>{escape(domain)} power domain</title></circle>'
            )
        elif len(domains) > 1:
            parts.append(
                f'<circle cx="{x + box_width - 13}" cy="{y + box_height - 13}" '
                'r="6" fill="#D92D20"><title>Conflicting power domains</title></circle>'
            )

    badge_labels = {"top": "T", "right": "R", "bottom": "B", "left": "L"}
    badge_width = 26
    badge_height = 22
    for switch_id in plan["required_closed_switches"]:
        info = graph.switch_info.get(switch_id)
        if info is None or info.module not in centers:
            continue
        center_x, center_y = centers[info.module]
        box_x = center_x - box_width / 2
        box_y = center_y - box_height / 2
        if info.world_direction == "top":
            badge_x = center_x - badge_width / 2
            badge_y = box_y - badge_height / 2
        elif info.world_direction == "right":
            badge_x = box_x + box_width - badge_width / 2
            badge_y = center_y - badge_height / 2
        elif info.world_direction == "bottom":
            badge_x = center_x - badge_width / 2
            badge_y = box_y + box_height - badge_height / 2
        else:
            badge_x = box_x - badge_width / 2
            badge_y = center_y - badge_height / 2

        domain = switch_domains.get(switch_id)
        badge_color = domain_colors.get(domain, "#344054")
        badge_label = badge_labels[info.local_direction]
        parts.extend(
            [
                "<g>",
                f"<title>{escape(switch_id)} — close local {info.local_direction} switch</title>",
                f'<rect x="{badge_x}" y="{badge_y}" width="{badge_width}" '
                f'height="{badge_height}" rx="7" fill="#FFFFFF" '
                f'stroke="{badge_color}" stroke-width="2.5"/>',
                _svg_text(
                    badge_x + badge_width / 2,
                    badge_y + 15,
                    badge_label,
                    size=12,
                    weight=800,
                    fill=badge_color,
                    anchor="middle",
                ),
                "</g>",
            ]
        )

    panel_height = height - 125
    parts.append(
        f'<rect x="{panel_x}" y="100" width="{panel_width}" height="{panel_height}" '
        'rx="20" fill="#FFFFFF" stroke="#D7DEE8" stroke-width="1.5"/>'
    )
    y = 136
    parts.append(_svg_text(panel_x + 24, y, "Optimization result", size=21, weight=700))
    y += 35
    if baseline_loads is not None:
        baseline_metric = (
            baseline_demands
            if demand_mode and baseline_demands is not None
            else baseline_loads
        )
        nearest_summary = " · ".join(
            f"{battery}={baseline_metric[battery]:g}"
            for battery in battery_order
        )
        parts.append(
            _svg_text(
                panel_x + 24,
                y,
                (
                    f"Count-balanced demand: {nearest_summary}"
                    if demand_mode
                    else f"Nearest loads: {nearest_summary}"
                ),
                size=15,
                weight=600,
                fill="#526071",
            )
        )
        y += 28
        parts.append(
            _svg_text(
                panel_x + 24,
                y,
                "Demand-aware battery drain" if demand_mode else "Balanced loads",
                size=17,
                weight=700,
            )
        )
        y += 27

    for battery, load in plan["battery_loads"].items():
        noun = "load module" if load == 1 else "load modules"
        load_text = (
            f"● {battery}: {plan['battery_effective_drain'][battery]:g} units "
            f"· {load} modules"
            if demand_mode
            else f"● {battery} supplies {load} {noun}"
        )
        parts.append(
            _svg_text(
                panel_x + 24,
                y,
                load_text,
                size=17,
                weight=700,
                fill=domain_colors[battery],
            )
        )
        y += 28

    if demand_mode and baseline_demands is not None:
        before_peak = max(baseline_demands.values())
        after_peak = max(plan["battery_effective_drain"].values())
        gain_percent = (
            (before_peak / after_peak - 1.0) * 100.0
            if after_peak > 0
            else 0.0
        )
        parts.append(
            _svg_text(
                panel_x + 24,
                y,
                f"Peak drain {before_peak:g} → {after_peak:g}; "
                f"lifetime ≈ +{gain_percent:.0f}% (equal capacity)",
                size=13,
                weight=700,
                fill="#087A50",
            )
        )
        y += 28
        loss_factor = plan["route_loss_factor"]
        parts.append(
            _svg_text(
                panel_x + 24,
                y,
                (
                    "Switch-loss coefficient unknown; path cost is lexicographic"
                    if loss_factor == 0
                    else f"Switch-loss factor {loss_factor:g} included in effective drain"
                ),
                size=12,
                weight=600,
                fill="#526071",
            )
        )
        y += 24

    if moves:
        y += 8
        parts.append(
            _svg_text(
                panel_x + 24,
                y,
                (
                    "Demand-aware reassignments"
                    if demand_mode
                    else "Equal-distance reassignments"
                ),
                size=17,
                weight=700,
            )
        )
        y += 26
        for module, old_battery, new_battery in moves:
            parts.append(
                _svg_text(
                    panel_x + 24,
                    y,
                    f"{module}: {old_battery} → {new_battery}",
                    size=14,
                    weight=600,
                    fill=domain_colors[new_battery],
                )
            )
            y += 24

    y += 14
    parts.append(_svg_text(panel_x + 24, y, "Shortest routes", size=17, weight=700))
    y += 28
    for module, assignment in sorted(plan["assignments"].items()):
        route = " → ".join(assignment.path_nodes or [])
        route_color = domain_colors.get(assignment.battery, "#172033")
        parts.append(
            _svg_text(
                panel_x + 24,
                y,
                f"{module}: {route}",
                size=15,
                weight=600,
                fill=route_color,
            )
        )
        y += 22
        hop_count = int(assignment.distance)
        switch_count = assignment.switch_count or 0
        parts.append(
            _svg_text(
                panel_x + 42,
                y,
                f"{hop_count} {'hop' if hop_count == 1 else 'hops'} · "
                f"{switch_count} {'switch' if switch_count == 1 else 'switches'}",
                size=13,
                fill="#526071",
            )
        )
        y += 29

    y += 5
    parts.append(
        _svg_text(
            panel_x + 24,
            y,
            "Close switches (JSON)",
            size=17,
            weight=700,
        )
    )
    y += 12
    close_json_line_height = 14
    close_json_box_height = 22 + close_json_line_height * len(close_switch_json_lines)
    parts.append(
        f'<rect x="{panel_x + 20}" y="{y}" width="{panel_width - 40}" '
        f'height="{close_json_box_height}" rx="12" fill="#F8FAFC" '
        'stroke="#CBD5E1" stroke-width="1.25"/>'
    )
    close_code_y = y + 17
    for line in close_switch_json_lines:
        parts.append(_svg_code_text(panel_x + 34, close_code_y, line, size=10))
        close_code_y += close_json_line_height
    y += close_json_box_height + 14

    remaining_open = len(plan["recommended_open_switches"])
    parts.append(
        _svg_text(
            panel_x + 24,
            y,
            f"Keep the other {remaining_open} local switches open",
            size=14,
            weight=600,
            fill="#526071",
        )
    )
    y += 23
    safety = "PASS — battery-isolated" if plan["safe_to_apply"] else "FAIL — battery conflict"
    parts.append(
        _svg_text(
            panel_x + 24,
            y,
            f"Topology gate: {safety}",
            size=14,
            weight=700,
            fill="#087A50" if plan["safe_to_apply"] else "#B42318",
        )
    )

    legend_y = grid_y + graph.grid_rows * cell_height + 44
    legend_items = [
        ("battery", "power source (role 1)"),
        ("action", "powered load (roles 2–16)"),
        ("relay", "routing-only relay (legacy)"),
    ]
    legend_x = 50
    for node_type, label in legend_items:
        fill, stroke = colors[node_type]
        parts.append(
            f'<rect x="{legend_x}" y="{legend_y - 16}" width="18" height="18" '
            f'rx="4" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        )
        parts.append(_svg_text(legend_x + 28, legend_y, label, size=13))
        legend_x += 205

    domain_legend_y = legend_y + 38
    parts.append(_svg_text(50, domain_legend_y, "Power domains:", size=13, weight=700))
    domain_legend_x = 165
    for battery in battery_order:
        color = domain_colors[battery]
        parts.append(
            f'<line x1="{domain_legend_x}" y1="{domain_legend_y - 5}" '
            f'x2="{domain_legend_x + 30}" y2="{domain_legend_y - 5}" '
            f'stroke="{color}" stroke-width="7" stroke-linecap="round"/>'
        )
        parts.append(
            _svg_text(
                domain_legend_x + 40,
                domain_legend_y,
                f"{battery} domain",
                size=13,
                weight=600,
                fill=color,
            )
        )
        domain_legend_x += 150

    parts.append(
        _svg_text(
            50,
            domain_legend_y + 32,
            "Closed-switch badges: T=top · R=right · B=bottom · L=left (module-local)",
            size=12,
            weight=600,
            fill="#526071",
        )
    )
    parts.append(
        _svg_text(
            50,
            domain_legend_y + 56,
            "Demand weights: empty=1 · buffer1–11=2 · CPU=3 · navigation=4 · motor=5",
            size=12,
            weight=700,
            fill="#344054",
        )
    )

    json_panel_x = grid_x + 5
    json_panel_width = graph.grid_cols * cell_width - 10
    parts.append(
        f'<rect x="{json_panel_x}" y="{json_panel_y}" '
        f'width="{json_panel_width}" height="{json_panel_height}" rx="16" '
        'fill="#F1F5F9" stroke="#CBD5E1" stroke-width="1.5"/>'
    )
    parts.append(
        _svg_text(
            json_panel_x + 16,
            json_panel_y + 27,
            "JSON input for this example",
            size=15,
            weight=700,
        )
    )
    code_y = json_panel_y + 50
    for line in json_lines:
        parts.append(_svg_code_text(json_panel_x + 16, code_y, line))
        code_y += json_line_height

    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--optimizer",
        choices=("balanced", "demand"),
        default="balanced",
    )
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    graph = ModularRobotGraph.from_layout_json(payload, default_closed=False)
    if args.optimizer == "demand":
        count_plan = graph.recommend_balanced_switch_plan(
            respect_switch_state=False,
        )
        baseline = count_plan["assignments"]
        plan = graph.recommend_demand_aware_switch_plan(
            respect_switch_state=False,
        )
    else:
        baseline = graph.nearest_battery_assignment(respect_switch_state=False)
        plan = graph.recommend_balanced_switch_plan(respect_switch_state=False)
    if not plan["safe_to_apply"]:
        raise RuntimeError(f"Unsafe battery connection: {plan['battery_conflicts']}")

    baseline_loads = _assignment_loads(graph.get_batteries(), baseline)
    if args.optimizer == "demand":
        baseline_demands = _assignment_demands(
            graph,
            graph.get_batteries(),
            baseline,
        )
        print("Count-balanced module counts:", baseline_loads)
        print("Count-balanced demands:", baseline_demands)
        print("Demand-aware module counts:", plan["battery_loads"])
        print("Demand-aware battery demands:", plan["battery_demands"])
        print("Demand-aware objective:", plan["objective"])
        print(
            "Search:",
            {"complete": plan["search_complete"], "states": plan["states_explored"]},
        )
    else:
        print("Nearest battery loads:", baseline_loads)
        print("Battery loads:", plan["battery_loads"])
    moves = [
        f"{module}: {baseline[module].battery}->{assignment.battery}"
        for module, assignment in sorted(plan["assignments"].items())
        if baseline[module].battery != assignment.battery
    ]
    print("Balanced moves:", moves or "none")
    for module, assignment in sorted(plan["assignments"].items()):
        print(
            f"{module}: weight={graph.get_module_demand_weight(module):g}; "
            f"battery={assignment.battery}; path={assignment.path_nodes}; "
            f"hops={int(assignment.distance)}; switches={assignment.switch_count}"
        )
    print("Close switches JSON:")
    print(json.dumps({"close_switches": plan["required_closed_switches"]}, indent=2))
    render_layout_svg(graph, plan, args.output, baseline=baseline)
    print("Diagram:", args.output)


if __name__ == "__main__":
    main()
