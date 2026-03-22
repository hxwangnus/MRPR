# MRPR: Modular Robot Power Routing

This repository prototypes power routing for a modular robot with multiple batteries and multiple action modules.

The robot is modeled as a graph:

- A node is a module, typically a `battery` or an `action`.
- An edge is a physical connection between two modules.
- Each edge is governed by a switch.
- Power loss is modeled by the number of traversed switches, or more generally by a per-edge cost.

With identical switch losses, the unweighted problem is solved with BFS. When different switches or links have different losses, the same framework switches to Dijkstra.

## Current Scope

The repo is a compact research/prototyping codebase with six standalone scripts:

- [power_routing_basic.py](/Users/hongxuan/Documents/MRPR/power_routing_basic.py): smallest illustrative version
- [power_routing_duo_mode.py](/Users/hongxuan/Documents/MRPR/power_routing_duo_mode.py): planning mode + runtime mode
- [power_routing_balanced.py](/Users/hongxuan/Documents/MRPR/power_routing_balanced.py): nearest routing plus load balancing for equal-distance cases
- [power_routing_dynamic.py](/Users/hongxuan/Documents/MRPR/power_routing_dynamic.py): usage-aware dynamic reassignment across time
- [power_routing_lifetime.py](/Users/hongxuan/Documents/MRPR/power_routing_lifetime.py): explicit battery capacity/discharge model with a stronger lifetime-aware heuristic
- [power_routing_exact.py](/Users/hongxuan/Documents/MRPR/power_routing_exact.py): exact total-working-time solver for the current simplified energy-budget model

Everything uses only the Python standard library. There is no `requirements.txt`.

## Shared Modeling Assumptions

Across the repo, the common assumptions are:

- Each action module is assigned to one battery at a time.
- The routing cost is the number of traversed switches in unweighted mode.
- Weighted mode uses `Edge.cost` and Dijkstra.
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
- Implements single-source Dijkstra for future weighted losses.
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

The second method is the main one demonstrated in the file. It preserves shortest-path distance while improving load distribution whenever an equal-distance reassignment exists.

The balancing objective is heuristic but clear:

- reduce load spread across batteries
- reduce load variance
- keep deterministic behavior

So this file is a good next step if the routing should still be "nearest" but you do not want lexicographic tie-breaking to overload one battery unnecessarily.

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
