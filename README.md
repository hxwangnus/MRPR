# MRPR: Modular Robot Power Routing

This repository prototypes power routing for a modular robot with multiple batteries and multiple action modules.

The robot is modeled as a graph:

- A node is a module, typically a `battery` or an `action`.
- An edge is a physical connection between two modules.
- Each edge is governed by a switch.
- Power loss is modeled by the number of traversed switches, or more generally by a per-edge cost.

The current role-aware requirement assumes identical switch losses, so its
routing path uses BFS by default. Some legacy prototypes still expose Dijkstra
for backward-compatible weighted-cost experiments; demand-aware balanced mode
does not require it under the current hardware model.

## Current Scope

The repo is a compact research/prototyping codebase with six standalone scripts:

- [power_routing_basic.py](/Users/hongxuan/Documents/MRPR/power_routing_basic.py): smallest illustrative version
- [power_routing_duo_mode.py](/Users/hongxuan/Documents/MRPR/power_routing_duo_mode.py): planning mode + runtime mode
- [power_routing_balanced.py](/Users/hongxuan/Documents/MRPR/power_routing_balanced.py): nearest routing plus load balancing for equal-distance cases
- [power_routing_dynamic.py](/Users/hongxuan/Documents/MRPR/power_routing_dynamic.py): usage-aware dynamic reassignment across time
- [power_routing_lifetime.py](/Users/hongxuan/Documents/MRPR/power_routing_lifetime.py): explicit battery capacity/discharge model with a stronger lifetime-aware heuristic
- [power_routing_exact.py](/Users/hongxuan/Documents/MRPR/power_routing_exact.py): exact total-working-time solver for the current simplified energy-budget model

Everything uses only the Python standard library. There is no `requirements.txt`.

## Algorithm Review and Experiments

- [POWER_ROUTING_STUDY.md](/Users/hongxuan/Documents/MRPR/POWER_ROUTING_STUDY.md) contains code-faithful dual-mode and balanced flowcharts, a primary-literature comparison, theoretical guarantees and limitations, and experimental results.
- [MRPR_Power_Routing_Study_Concise.pdf](/Users/hongxuan/Documents/MRPR/output/pdf/MRPR_Power_Routing_Study_Concise.pdf) is the nine-page concise edition: five content pages plus four strict flowchart sheets.
- [MRPR_Power_Routing_Study.pdf](/Users/hongxuan/Documents/MRPR/output/pdf/MRPR_Power_Routing_Study.pdf) is the polished, print-ready PDF edition with vector flowcharts and mixed portrait/landscape layouts.
- [experiments/benchmark_balanced_safety.py](/Users/hongxuan/Documents/MRPR/experiments/benchmark_balanced_safety.py) reproduces the nearest-vs-balanced load and switch-topology Monte Carlo study.
- [experiments/benchmark_energy_horizon.py](/Users/hongxuan/Documents/MRPR/experiments/benchmark_energy_horizon.py) compares static nearest/balanced routing with the repository's greedy-lifetime and restricted exact methods under one shared energy-budget model.

Important safety boundary: `rebalanced_nearest_battery_assignment()` still returns raw balanced assignments and independently reconstructed paths. Use `recommend_balanced_switch_plan()` before applying them: it validates the path union and fails closed if it would connect two battery domains. This is a topology gate, not a substitute for voltage/current and hardware interlocks.

Dual-mode's battery-isolated-forest guarantee also assumes unweighted routing or strictly positive weighted edge costs. The current graph accepts zero-cost edges; a zero-cost weighted path can reparent a battery source. The study contains the counterexample and recommended guard.

## Shared Modeling Assumptions

Across the repo, the common assumptions are:

- Each action module is assigned to one battery at a time.
- The routing cost is the number of traversed links in unweighted mode. Legacy links use one switch; role-aware rotated-layout links use two endpoint switches per link, so minimizing hops also minimizes traversed switches.
- Legacy weighted mode uses `Edge.cost` and Dijkstra; current role-aware routing
  uses equal-cost BFS.
- A disconnected module is reported as unreachable with infinite distance.
- Switch plans are derived from the union of the selected shortest paths.

In the later files:

- `power_routing_lifetime.py` and `power_routing_exact.py` add explicit battery energy state.
- Per-round module energy can depend on both module demand and route cost.

What is not modeled yet:

- current limits
- voltage constraints
- wire resistance separate from switch loss
- battery chemistry / nonlinear discharge behavior
- switching penalties between rounds
- per-round current or power caps on individual batteries
- globally optimal multi-commodity flow

## File-By-File Overview

### [power_routing_basic.py](/Users/hongxuan/Documents/MRPR/power_routing_basic.py)

This is the minimal educational version.

- Defines a small undirected graph with switch-controlled edges.
- Implements single-source BFS for unit-cost routing.
- Retains single-source Dijkstra for legacy weighted-cost experiments.
- Builds a simple battery-to-module distance table.
- Includes `demo_l_shape()` for a line-shaped example with two batteries.

This file is best viewed as the simplest explanation of the core idea.

### [power_routing_duo_mode.py](/Users/hongxuan/Documents/MRPR/power_routing_duo_mode.py)

This is the main baseline implementation.

It introduces two operating modes:

- Planning mode: ignore current switch states and assume the full physical topology is available
- Runtime mode: respect the current enabled/disabled switches

Main capabilities:

- single-source BFS or Dijkstra
- multi-source BFS or Dijkstra from all batteries at once
- deterministic nearest-battery assignment
- path reconstruction for each module
- switch-plan recommendation for only the active modules
- graph construction from a manual node/switch list
- graph construction from a 4-neighbor grid layout

The key method is `recommend_switch_plan(...)`, which returns:

- assignments
- required switches that must be enabled
- switches that can remain disabled
- unreachable modules

This file is the best reference for the "plan first, then only close what is needed at runtime" idea.

### [power_routing_balanced.py](/Users/hongxuan/Documents/MRPR/power_routing_balanced.py)

This file extends `duo_mode` with battery-load balancing.

There are two related ideas here:

- `dynamic_load_balanced_assignment(...)`: assign modules one by one, breaking ties by current battery load
- `rebalanced_nearest_battery_assignment(...)`: start from the original nearest-battery solution, then move only modules that have another battery at exactly the same shortest distance
- `recommend_balanced_switch_plan(...)`: turn the balanced assignments into endpoint switch states and fail closed on a multi-battery connected component
- `recommend_demand_aware_switch_plan(...)`: use role demand weights, BFS route candidates, and a bounded exact global search to minimize peak battery drain before route cost

The second method is the main one demonstrated in the file. It preserves shortest-path distance (within `math.isclose` tolerance in weighted mode) while applying only strictly improving equal-distance reassignments.

The balancing objective is heuristic but clear:

- reduce load spread across batteries
- reduce load variance
- use deterministic tie-breaking where implemented

Equal-objective moves now use a persistent deterministic comparison key, removing the earlier last-scanned/input-order tie bug documented in the study.

So this file is a good next step if the routing should still be "nearest" but you do not want lexicographic tie-breaking to overload one battery unnecessarily.

Demand-aware mode addresses the stronger lifetime problem. Its default objective
is lexicographic rather than an arbitrary weighted sum: minimize maximum battery
demand, then demand spread, variance, total demand-weighted path cost, and finally
the number of closed switches. Because every physical adjacency closes the same
two endpoint switches, every module-to-battery candidate is a minimum-hop BFS
path. The optional `route_loss_factor` remains zero until
switch-loss measurements are available; when supplied, effective drain becomes
`demand * (1 + route_loss_factor * route_cost)`.

The zero default is not a claim that physical switch loss is zero. It means the
unknown coefficient is not allowed to change the known demand-priority order;
minimum path cost remains a later lexicographic objective. Once measured, the
coefficient can be supplied for sensitivity analysis or unified effective-drain
optimization.

The bounded search reports `search_complete`. A complete result is exact over
one deterministic minimum-cost path per module/battery pair; equal-cost path
variants, nonlinear battery behavior, current/voltage limits, and link
congestion remain outside the model.

### [power_routing_dynamic.py](/Users/hongxuan/Documents/MRPR/power_routing_dynamic.py)

This file goes beyond static routing and tries to optimize long-term battery usage.

Instead of only asking "which battery is nearest right now?", it asks:

- how much has each battery already been used?
- should some modules be moved to slightly longer routes now to extend total operating time later?

Key ideas:

- start from the nearest-battery baseline
- estimate per-step battery usage as `module_demand * path_distance`
- add that to cumulative battery usage
- greedily move modules to other reachable batteries when the projected usage balance improves
- allow bounded detours via `max_extra_switches`

The dynamic objective is evaluated lexicographically using:

- peak projected battery usage
- spread between max and min projected usage
- variance
- total usage in the next step

This is a reasonable prototype for "lifetime-aware routing", but it is still heuristic and local rather than globally optimal.

### [power_routing_lifetime.py](/Users/hongxuan/Documents/MRPR/power_routing_lifetime.py)

This file makes battery state explicit and moves much closer to the real
"maximize working time" goal.

It adds:

- battery capacity per battery
- cumulative battery discharge
- per-module demand
- an energy model of the form `demand * (base_module_cost + switch_loss_factor * route_cost)`
- simulation until the first infeasible round

The main solver starts from the nearest-battery assignment, then greedily moves
modules between reachable batteries when doing so improves a lifetime-oriented
objective based on remaining energy.

This is stronger than `dynamic.py`, because it optimizes against explicit
remaining battery energy rather than only historical usage. However, it is still
a heuristic:

- it optimizes a one-step lifetime proxy
- it uses greedy single-module moves
- it is not guaranteed to maximize total future rounds globally

### [power_routing_exact.py](/Users/hongxuan/Documents/MRPR/power_routing_exact.py)

This file is the first version in the repo that directly optimizes:

- the maximum number of future feasible rounds

Instead of optimizing a local proxy, it solves the horizon-allocation problem:

- how many of the next `T` rounds should each module be powered by each reachable battery
- subject to each battery's remaining energy budget

Implementation highlights:

- reuses the graph and energy model from `power_routing_lifetime.py`
- solves fixed-horizon feasibility exactly with a Pareto-frontier dynamic program
- binary-searches on `T` to find the largest feasible total working time
- can reconstruct one valid round-by-round schedule from the exact allocation counts

Important limitation:

- it is exact only for the current simplified model

That means:

- finite battery energy budgets
- fixed per-module per-route energy per round
- no switching penalty between rounds
- no per-round battery power/current cap

So `power_routing_exact.py` is the right reference if your current research question
is "under this simplified model, what is the maximum total working time?".
It is also more computationally expensive than the heuristic files.

## How The Modes Relate

A simple way to think about the repo is:

1. `basic`: prove the shortest-path idea
2. `duo_mode`: add planning mode and runtime mode
3. `balanced`: keep shortest-path routing, but balance equal-distance assignments
4. `dynamic`: allow controlled detours to improve long-term battery lifetime
5. `lifetime`: add explicit battery capacities and optimize a stronger lifetime-aware heuristic
6. `exact`: directly maximize total feasible future rounds for the current simplified model

## Heuristic vs Exact

The repo now contains both heuristic and exact variants.

- `basic`, `duo_mode`, `balanced`, `dynamic`, and `lifetime` are heuristic or policy-oriented files.
- `exact` is exact only within its stated assumptions.
- If you later add switching penalties, per-round current caps, or richer electrochemical models, the exact file will stop being exact unless the formulation is upgraded accordingly.

## Running The Demos

Run the scripts directly:

```bash
python3 power_routing_basic.py
python3 power_routing_duo_mode.py
python3 power_routing_balanced.py
python3 power_routing_dynamic.py
python3 power_routing_lifetime.py
python3 power_routing_exact.py
python3 -m unittest discover -v
```

What each script demonstrates:

- `power_routing_basic.py`: simple L-shape example
- `power_routing_duo_mode.py`: manual example, auto-built grid example, and larger planning/runtime scenarios
- `power_routing_balanced.py`: comparison between nearest-only and balanced tie-breaking
- `power_routing_dynamic.py`: comparison between nearest-only routing and usage-aware dynamic reassignment over multiple rounds
- `power_routing_lifetime.py`: explicit-capacity lifetime-aware heuristic versus nearest-only routing
- `power_routing_exact.py`: an exact horizon solver, including a counterexample where the greedy lifetime heuristic is not globally optimal

## Quick Usage Pattern

The most reusable entry point is the `ModularRobotGraph` class in the richer scripts.

Typical flow:

1. Add nodes as batteries/actions
2. Add switch-controlled edges
3. Choose planning mode or runtime mode
4. Compute assignments
5. Extract the required switch set

Small example:

```python
from power_routing_duo_mode import ModularRobotGraph

g = ModularRobotGraph()
g.add_node("B1", "battery")
g.add_node("B2", "battery")
g.add_node("M1", "action")
g.add_node("M2", "action")

g.add_switch_edge("B1", "M1", "S_B1_M1", open_=True)
g.add_switch_edge("M1", "M2", "S_M1_M2", open_=True)
g.add_switch_edge("M2", "B2", "S_M2_B2", open_=True)

plan = g.recommend_switch_plan(
    batteries=["B1", "B2"],
    active_modules=["M1", "M2"],
    weighted=False,
    respect_switch_state=False,
)

print(plan["required_open_switches"])
```

## Role-Aware Layout JSON in Balanced and Demand-Aware Modes

`power_routing_balanced.py` accepts the row-major rotated layout format used by
the modular robot controller:

```python
from power_routing_balanced import ModularRobotGraph

payload = {
    "layout": {
        "cols": 3,
        "rows": 3,
        "cells": [
            ["M2(180d,[8,252,0])", "M3(90d,[1,252,0])", None],
            ["M1(0d,[3,252,0])", "M4(90d,[2,252,0])", "M6(180d,[12,252,0])"],
            [None, "M5(270d,[7,252,0])", None],
        ],
    }
}

graph = ModularRobotGraph.from_layout_json(payload, default_closed=False)
plan = graph.recommend_balanced_switch_plan(
    respect_switch_state=False,  # plan over the complete physical topology
)

print(plan["battery_loads"])               # {'M3': 5}; roles 2–16 all consume power
print(plan["required_closed_switches"])
print(plan["safe_to_apply"])               # True
```

The three values after the rotation are `[role, digital_io, unused]`. Role 1 is
a battery. Every role from 2 through 16 is a powered load: motor, CPU,
navigation, and empty are roles 2/3/4/5; roles 6–16 are buffer1–buffer11.
Default demand weights are motor=5, navigation=4, CPU=3, every buffer=2, and
empty=1. The digital-I/O byte is stored and exposed as both the integer value
and an eight-bit string; it does not yet affect optimization.

The first `cells` row is the physical top row. Every module gets `.top`,
`.right`, `.bottom`, and `.left` local switch IDs. Rotation is counter-clockwise.
For example, `M2(180d,...)` has `M2.top` facing grid-bottom and `M2.left` facing
grid-right; `M3(90d,...)` has `M3.top` facing grid-left. The M2—M3 link therefore
requires both `M2.left` and `M3.top`.

At runtime, an adjacency is traversable only when both endpoint switches are
closed. Use `set_switch_closed(...)` for the physical terminology. Balanced planning
output includes assignments, per-battery load counts, required links,
`required_closed_switches`, `recommended_open_switches`, and the topology gate
`safe_to_apply`. The old
`required_open_switches`/`recommended_closed_switches` keys remain as compatibility
aliases because the original prototype used `open=True` to mean traversable.

Empty grid locations can be `null`, `""`, or `"."`. The parser validates declared
dimensions, duplicate module IDs, role range (1–16), digital-I/O byte range
(0–255), and rotations (multiples of 90 degrees). The old `grid` key and cells
without the three-value metadata remain supported for compatibility.

Run the complete example and regenerate its SVG diagram with:

```bash
python3 examples/demo_balanced_role_layout.py
```

The reproducible 4×4 / three-battery example (seed `20260828`) can be run with:

```bash
python3 examples/demo_balanced_role_layout.py \
  --input examples/complex_4x4_role_layout.json \
  --output output/balanced_complex_4x4_demo.svg
```

Run the same topology with demand-aware optimization:

```bash
python3 examples/demo_balanced_role_layout.py \
  --optimizer demand \
  --input examples/complex_4x4_role_layout.json \
  --output output/demand_aware_complex_4x4_demo.svg
```

In the generated SVG, module fill colors encode module type, colored links encode
battery domains, and the `T/R/B/L` badges drawn on module edges identify the
module-local top/right/bottom/left switches that must be closed. The right-hand
close-switch section is valid JSON using the schema
`{"close_switches": ["module.direction", ...]}` for downstream integration.
