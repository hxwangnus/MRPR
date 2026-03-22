from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR, getcontext
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Iterable, Any
import math

from power_routing_lifetime import (
    AssignmentResult,
    ModularRobotGraph as LifetimeModularRobotGraph,
)


getcontext().prec = 50


@dataclass(frozen=True)
class RouteCandidate:
    """One reachable battery candidate for one module."""

    battery: str
    distance: float
    energy_per_round: Decimal
    assignment: AssignmentResult


@dataclass
class ExactLifetimePlan:
    """
    Exact horizon plan under the simplified energy-budget model.

    `allocation_counts[module][battery] = n` means:
        across the next `max_rounds` rounds, module `module` should be powered by
        battery `battery` in exactly `n` rounds.

    Since this model has no switching penalty or per-round power-cap constraint,
    any per-round schedule consistent with those counts is feasible.
    """

    max_rounds: int
    battery_list: List[str]
    module_list: List[str]
    remaining_before: Dict[str, float]
    total_energy_by_battery: Dict[str, float]
    remaining_after: Dict[str, float]
    allocation_counts: Dict[str, Dict[str, int]]
    round_assignments: List[Dict[str, AssignmentResult]]


@lru_cache(maxsize=None)
def _compositions(total: int, parts: int) -> Tuple[Tuple[int, ...], ...]:
    if parts <= 0:
        return tuple()
    if parts == 1:
        return ((total,),)

    out: List[Tuple[int, ...]] = []
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            out.append((first,) + rest)
    return tuple(out)


def _to_decimal(value: Any) -> Decimal:
    return Decimal(str(value))


class ExactModularRobotGraph(LifetimeModularRobotGraph):
    """
    Exact lifetime optimizer for the current simplified model.

    This solver directly maximizes the total number of feasible future rounds.
    It is exact for the assumptions below:

    - finite battery energy budgets
    - per-round module demand is known
    - route energy for one module/battery pair is fixed
    - no switching penalty between rounds
    - no per-round current-cap / power-cap constraint

    Under these assumptions, the problem can be written as an integer allocation:

        choose n[m, b] >= 0 integers
        such that sum_b n[m, b] = T for every module m
        and sum_m e[m, b] * n[m, b] <= R[b] for every battery b

    This file solves the feasibility problem for a fixed horizon T exactly with
    a Pareto-frontier dynamic program, then binary-searches on T.
    """

    @staticmethod
    def _dominates(lhs: Tuple[Decimal, ...], rhs: Tuple[Decimal, ...]) -> bool:
        return all(a <= b for a, b in zip(lhs, rhs)) and any(a < b for a, b in zip(lhs, rhs))

    @staticmethod
    def _usage_objective(
        usage: Tuple[Decimal, ...],
        capacities: Tuple[Decimal, ...],
    ) -> Tuple[Tuple[Decimal, ...], Decimal]:
        ratios: List[Decimal] = []
        for used, cap in zip(usage, capacities):
            if cap <= 0:
                ratio = Decimal("Infinity") if used > 0 else Decimal(0)
            else:
                ratio = used / cap
            ratios.append(ratio)
        return (tuple(sorted(ratios, reverse=True)), sum(usage, Decimal(0)))

    def _prune_dominated_states(
        self,
        states: Dict[Tuple[Decimal, ...], Optional[Tuple[Tuple[Decimal, ...], Tuple[int, ...]]]],
    ) -> Dict[Tuple[Decimal, ...], Optional[Tuple[Tuple[Decimal, ...], Tuple[int, ...]]]]:
        items = sorted(states.items(), key=lambda item: (sum(item[0], Decimal(0)), item[0]))
        kept: List[Tuple[Decimal, ...]] = []
        kept_map: Dict[Tuple[Decimal, ...], Optional[Tuple[Tuple[Decimal, ...], Tuple[int, ...]]]] = {}

        for state, info in items:
            dominated = False
            to_remove: List[Tuple[Decimal, ...]] = []

            for other in kept:
                if self._dominates(other, state):
                    dominated = True
                    break
                if self._dominates(state, other):
                    to_remove.append(other)

            if dominated:
                continue

            for other in to_remove:
                kept.remove(other)
                del kept_map[other]

            kept.append(state)
            kept_map[state] = info

        return kept_map

    def _normalize_remaining_energy(
        self,
        battery_list: List[str],
        battery_capacity: Dict[str, float],
        battery_discharge: Optional[Dict[str, float]],
    ) -> Dict[str, Decimal]:
        discharge = self._normalized_battery_discharge(
            battery_list,
            battery_capacity,
            battery_discharge,
        )
        remaining: Dict[str, Decimal] = {}
        for battery in battery_list:
            cap = _to_decimal(battery_capacity[battery])
            used = _to_decimal(discharge[battery])
            rem = cap - used
            remaining[battery] = rem if rem > 0 else Decimal(0)
        return remaining

    def _build_route_candidates(
        self,
        *,
        batteries: Optional[Iterable[str]],
        modules: Optional[Iterable[str]],
        battery_capacity: Dict[str, float],
        battery_discharge: Optional[Dict[str, float]],
        module_demand: Optional[Dict[str, float]],
        weighted: bool,
        respect_switch_state: bool,
        base_module_cost: float,
        switch_loss_factor: float,
        max_extra_cost: Optional[float],
    ) -> Tuple[
        List[str],
        List[str],
        Dict[str, Decimal],
        Dict[str, RouteCandidate],
        Dict[str, Dict[str, RouteCandidate]],
    ]:
        battery_list = list(batteries) if batteries is not None else self.get_batteries()
        module_list = list(modules) if modules is not None else self.get_actions()
        remaining = self._normalize_remaining_energy(
            battery_list,
            battery_capacity,
            battery_discharge,
        )

        demand_float = self._normalized_module_demand(module_list, module_demand)
        demand_dec = {module: _to_decimal(value) for module, value in demand_float.items()}
        base_cost = _to_decimal(base_module_cost)
        switch_factor = _to_decimal(switch_loss_factor)

        shortest_cache: Dict[str, Tuple[Dict[str, float], Dict[str, Optional[Tuple[str, str]]]]] = {}
        for battery in battery_list:
            shortest_cache[battery] = self.shortest_paths_from(
                battery,
                weighted=weighted,
                respect_switch_state=respect_switch_state,
            )

        baseline_distances: Dict[str, float] = {}
        if max_extra_cost is not None:
            baseline = self.nearest_battery_assignment(
                batteries=battery_list,
                modules=module_list,
                weighted=weighted,
                respect_switch_state=respect_switch_state,
            )
            baseline_distances = {module: baseline[module].distance for module in module_list}

        route_lookup: Dict[str, Dict[str, RouteCandidate]] = {}
        flat_lookup: Dict[str, RouteCandidate] = {}

        for module in module_list:
            route_lookup[module] = {}
            base_distance = baseline_distances.get(module, math.inf)

            for battery in battery_list:
                dist_map, parent = shortest_cache[battery]
                if module not in dist_map:
                    continue

                distance = dist_map[module]
                if (
                    max_extra_cost is not None
                    and not math.isinf(base_distance)
                    and distance > base_distance + max_extra_cost
                ):
                    continue

                path = self.reconstruct_path(parent, module)
                nodes_path, switches_path = path if path is not None else (None, None)
                assignment = AssignmentResult(
                    module=module,
                    battery=battery,
                    distance=distance,
                    path_nodes=nodes_path,
                    path_switches=switches_path,
                )
                route_cost = _to_decimal(distance)
                energy = demand_dec[module] * (base_cost + switch_factor * route_cost)

                candidate = RouteCandidate(
                    battery=battery,
                    distance=distance,
                    energy_per_round=energy,
                    assignment=assignment,
                )
                route_lookup[module][battery] = candidate
                flat_lookup[f"{module}:{battery}"] = candidate

        return battery_list, module_list, remaining, flat_lookup, route_lookup

    def _module_horizon_options(
        self,
        *,
        module: str,
        candidates: Dict[str, RouteCandidate],
        battery_list: List[str],
        battery_index: Dict[str, int],
        horizon: int,
        capacities: Tuple[Decimal, ...],
    ) -> List[Tuple[Tuple[int, ...], Tuple[Decimal, ...]]]:
        local_candidates = [candidates[battery] for battery in sorted(candidates)]
        if not local_candidates:
            return []

        raw_options: Dict[Tuple[Decimal, ...], Tuple[int, ...]] = {}
        for local_counts in _compositions(horizon, len(local_candidates)):
            full_counts = [0] * len(battery_list)
            full_energy = [Decimal(0)] * len(battery_list)
            feasible = True

            for count, candidate in zip(local_counts, local_candidates):
                if count <= 0:
                    continue
                idx = battery_index[candidate.battery]
                energy = candidate.energy_per_round * count
                if energy > capacities[idx]:
                    feasible = False
                    break
                full_counts[idx] = count
                full_energy[idx] = energy

            if feasible:
                raw_options[tuple(full_energy)] = tuple(full_counts)

        pruned = self._prune_dominated_states({energy: None for energy in raw_options})
        options = [
            (raw_options[energy], energy)
            for energy in pruned
        ]
        options.sort(key=lambda item: (sum(item[1], Decimal(0)), item[1], item[0]))
        return options

    def _solve_exact_horizon(
        self,
        *,
        horizon: int,
        battery_list: List[str],
        module_order: List[str],
        route_lookup: Dict[str, Dict[str, RouteCandidate]],
        capacities: Tuple[Decimal, ...],
        construct: bool,
    ) -> Optional[Tuple[Tuple[Decimal, ...], List[Tuple[str, Dict[Tuple[Decimal, ...], Tuple[Tuple[Decimal, ...], Tuple[int, ...]]]]]]]:
        if horizon == 0:
            return (tuple(Decimal(0) for _ in battery_list), [])

        battery_index = {battery: idx for idx, battery in enumerate(battery_list)}
        module_options: Dict[str, List[Tuple[Tuple[int, ...], Tuple[Decimal, ...]]]] = {}
        for module in module_order:
            options = self._module_horizon_options(
                module=module,
                candidates=route_lookup[module],
                battery_list=battery_list,
                battery_index=battery_index,
                horizon=horizon,
                capacities=capacities,
            )
            if not options:
                return None
            module_options[module] = options

        zero_state = tuple(Decimal(0) for _ in battery_list)
        frontier: Dict[Tuple[Decimal, ...], Optional[Tuple[Tuple[Decimal, ...], Tuple[int, ...]]]] = {zero_state: None}
        layers: List[Tuple[str, Dict[Tuple[Decimal, ...], Tuple[Tuple[Decimal, ...], Tuple[int, ...]]]]] = []

        for module in module_order:
            next_frontier: Dict[Tuple[Decimal, ...], Optional[Tuple[Tuple[Decimal, ...], Tuple[int, ...]]]] = {}

            for state in frontier:
                for count_tuple, energy_tuple in module_options[module]:
                    next_state = tuple(
                        state[idx] + energy_tuple[idx]
                        for idx in range(len(battery_list))
                    )
                    if any(next_state[idx] > capacities[idx] for idx in range(len(battery_list))):
                        continue
                    if next_state not in next_frontier:
                        next_frontier[next_state] = (state, count_tuple) if construct else None

            if not next_frontier:
                return None

            next_frontier = self._prune_dominated_states(next_frontier)
            if construct:
                layers.append((module, {state: info for state, info in next_frontier.items() if info is not None}))
            frontier = next_frontier

        best_state = min(
            frontier.keys(),
            key=lambda usage: self._usage_objective(usage, capacities),
        )
        return best_state, layers

    def _build_round_assignments(
        self,
        *,
        horizon: int,
        allocation_counts: Dict[str, Dict[str, int]],
        route_lookup: Dict[str, Dict[str, RouteCandidate]],
    ) -> List[Dict[str, AssignmentResult]]:
        if horizon <= 0:
            return []

        rounds: List[Dict[str, AssignmentResult]] = [dict() for _ in range(horizon)]

        for module, counts in allocation_counts.items():
            remaining = {battery: count for battery, count in counts.items() if count > 0}
            sequence: List[str] = []

            while len(sequence) < horizon:
                progressed = False
                for battery in sorted(remaining, key=lambda b: (-remaining[b], b)):
                    if remaining[battery] <= 0:
                        continue
                    sequence.append(battery)
                    remaining[battery] -= 1
                    progressed = True
                    if len(sequence) == horizon:
                        break
                if not progressed:
                    break

            for round_idx, battery in enumerate(sequence):
                rounds[round_idx][module] = route_lookup[module][battery].assignment

        return rounds

    def maximize_total_rounds(
        self,
        battery_capacity: Dict[str, float],
        *,
        battery_discharge: Optional[Dict[str, float]] = None,
        batteries: Optional[Iterable[str]] = None,
        modules: Optional[Iterable[str]] = None,
        module_demand: Optional[Dict[str, float]] = None,
        weighted: bool = False,
        respect_switch_state: bool = True,
        base_module_cost: float = 1.0,
        switch_loss_factor: float = 1.0,
        max_extra_cost: Optional[float] = None,
        max_rounds_upper_bound: Optional[int] = None,
    ) -> ExactLifetimePlan:
        """
        Compute the exact maximum number of future feasible rounds.

        If `max_extra_cost` is None, the result is exact for the full reachable
        topology under this file's simplified assumptions.

        If `max_extra_cost` is not None, the result is exact only inside that
        restricted candidate set.
        """
        (
            battery_list,
            module_list,
            remaining_before_dec,
            _,
            route_lookup,
        ) = self._build_route_candidates(
            batteries=batteries,
            modules=modules,
            battery_capacity=battery_capacity,
            battery_discharge=battery_discharge,
            module_demand=module_demand,
            weighted=weighted,
            respect_switch_state=respect_switch_state,
            base_module_cost=base_module_cost,
            switch_loss_factor=switch_loss_factor,
            max_extra_cost=max_extra_cost,
        )

        remaining_tuple = tuple(remaining_before_dec[battery] for battery in battery_list)
        module_order = sorted(
            module_list,
            key=lambda module: (
                len(route_lookup[module]),
                -min(candidate.energy_per_round for candidate in route_lookup[module].values()),
                -(
                    max(candidate.energy_per_round for candidate in route_lookup[module].values())
                    - min(candidate.energy_per_round for candidate in route_lookup[module].values())
                ),
                module,
            ),
        )

        if any(not route_lookup[module] for module in module_list):
            return ExactLifetimePlan(
                max_rounds=0,
                battery_list=battery_list,
                module_list=module_list,
                remaining_before={battery: float(remaining_before_dec[battery]) for battery in battery_list},
                total_energy_by_battery={battery: 0.0 for battery in battery_list},
                remaining_after={battery: float(remaining_before_dec[battery]) for battery in battery_list},
                allocation_counts={module: {} for module in module_list},
                round_assignments=[],
            )

        if max_rounds_upper_bound is None:
            min_total_energy = sum(
                min(candidate.energy_per_round for candidate in route_lookup[module].values())
                for module in module_list
            )
            if min_total_energy <= 0:
                raise ValueError(
                    "Cannot infer a finite exact horizon bound because the total minimum "
                    "per-round energy is zero. Please pass max_rounds_upper_bound explicitly."
                )
            total_remaining = sum(remaining_tuple, Decimal(0))
            upper_dec = (total_remaining / min_total_energy).to_integral_value(rounding=ROUND_FLOOR)
            upper = int(upper_dec)
        else:
            upper = max_rounds_upper_bound

        best_horizon = 0
        lo = 0
        hi = max(0, upper)

        while lo <= hi:
            mid = (lo + hi) // 2
            feasible = self._solve_exact_horizon(
                horizon=mid,
                battery_list=battery_list,
                module_order=module_order,
                route_lookup=route_lookup,
                capacities=remaining_tuple,
                construct=False,
            )
            if feasible is not None:
                best_horizon = mid
                lo = mid + 1
            else:
                hi = mid - 1

        best_usage, layers = self._solve_exact_horizon(
            horizon=best_horizon,
            battery_list=battery_list,
            module_order=module_order,
            route_lookup=route_lookup,
            capacities=remaining_tuple,
            construct=True,
        ) or (tuple(Decimal(0) for _ in battery_list), [])

        allocation_counts: Dict[str, Dict[str, int]] = {module: {} for module in module_list}
        if best_horizon > 0:
            state = best_usage
            for module, layer in reversed(layers):
                prev_state, count_tuple = layer[state]
                counts = {
                    battery: count_tuple[idx]
                    for idx, battery in enumerate(battery_list)
                    if count_tuple[idx] > 0
                }
                allocation_counts[module] = counts
                state = prev_state

        total_energy_by_battery = {
            battery: float(best_usage[idx])
            for idx, battery in enumerate(battery_list)
        }
        remaining_before = {
            battery: float(remaining_before_dec[battery])
            for battery in battery_list
        }
        remaining_after = {
            battery: float(remaining_before_dec[battery] - best_usage[idx])
            for idx, battery in enumerate(battery_list)
        }
        round_assignments = self._build_round_assignments(
            horizon=best_horizon,
            allocation_counts=allocation_counts,
            route_lookup=route_lookup,
        )

        return ExactLifetimePlan(
            max_rounds=best_horizon,
            battery_list=battery_list,
            module_list=module_list,
            remaining_before=remaining_before,
            total_energy_by_battery=total_energy_by_battery,
            remaining_after=remaining_after,
            allocation_counts=allocation_counts,
            round_assignments=round_assignments,
        )


def print_energy_table(title: str, values: Dict[str, float]) -> None:
    print(title)
    pretty = {battery: round(value, 2) for battery, value in sorted(values.items())}
    print(f"  {pretty}")


def print_allocation_counts(plan: ExactLifetimePlan) -> None:
    print("Allocation counts across the exact horizon:")
    for module in sorted(plan.allocation_counts):
        counts = plan.allocation_counts[module]
        if not counts:
            print(f"  {module}: {{}}")
            continue
        print(f"  {module}: {dict(sorted(counts.items()))}")


def print_schedule(plan: ExactLifetimePlan) -> None:
    print("One valid round-by-round schedule:")
    if not plan.round_assignments:
        print("  None")
        return
    for round_idx, assignments in enumerate(plan.round_assignments, start=1):
        compact = {
            module: info.battery
            for module, info in sorted(assignments.items())
        }
        print(f"  Round {round_idx}: {compact}")


def build_counterexample_graph() -> ExactModularRobotGraph:
    nodes = [
        {"id": "B1", "type": "battery"},
        {"id": "B2", "type": "battery"},
        {"id": "M1", "type": "action"},
        {"id": "M2", "type": "action"},
    ]
    switches = [
        {"u": "B1", "v": "M1", "switch_id": "S_B1_M1", "open": True, "cost": 0.0},
        {"u": "B2", "v": "M1", "switch_id": "S_B2_M1", "open": True, "cost": 0.0},
        {"u": "B1", "v": "M2", "switch_id": "S_B1_M2", "open": True, "cost": 2.0},
        {"u": "B2", "v": "M2", "switch_id": "S_B2_M2", "open": True, "cost": 2.0},
    ]
    return ExactModularRobotGraph.from_manual_spec(nodes, switches)


def build_demo4_graph() -> ExactModularRobotGraph:
    modules = {
        "B1":  {"type": "battery", "pos": (0, 6)},
        "M01": {"type": "action",  "pos": (1, 6)},
        "M02": {"type": "action",  "pos": (2, 6)},
        "M03": {"type": "action",  "pos": (2, 5)},
        "M19": {"type": "action",  "pos": (0, 4)},
        "M18": {"type": "action",  "pos": (1, 4)},
        "M04": {"type": "action",  "pos": (2, 4)},
        "B2":  {"type": "battery", "pos": (3, 4)},
        "M05": {"type": "action",  "pos": (4, 4)},
        "B5":  {"type": "battery", "pos": (0, 3)},
        "M17": {"type": "action",  "pos": (2, 3)},
        "M06": {"type": "action",  "pos": (4, 3)},
        "M20": {"type": "action",  "pos": (5, 3)},
        "M15": {"type": "action",  "pos": (0, 2)},
        "M14": {"type": "action",  "pos": (1, 2)},
        "M16": {"type": "action",  "pos": (2, 2)},
        "M08": {"type": "action",  "pos": (3, 2)},
        "M07": {"type": "action",  "pos": (4, 2)},
        "B3":  {"type": "battery", "pos": (5, 2)},
        "M13": {"type": "action",  "pos": (1, 1)},
        "M09": {"type": "action",  "pos": (3, 1)},
        "B4":  {"type": "battery", "pos": (0, 0)},
        "M12": {"type": "action",  "pos": (1, 0)},
        "M11": {"type": "action",  "pos": (2, 0)},
        "M10": {"type": "action",  "pos": (3, 0)},
    }
    return ExactModularRobotGraph.from_grid_layout(modules)


def build_compact_demo_graph() -> ExactModularRobotGraph:
    modules = {
        "B1":  {"type": "battery", "pos": (0, 2)},
        "M01": {"type": "action",  "pos": (1, 2)},
        "M02": {"type": "action",  "pos": (2, 2)},
        "B2":  {"type": "battery", "pos": (3, 2)},
        "M03": {"type": "action",  "pos": (0, 1)},
        "M04": {"type": "action",  "pos": (1, 1)},
        "M05": {"type": "action",  "pos": (2, 1)},
        "M06": {"type": "action",  "pos": (3, 1)},
        "B3":  {"type": "battery", "pos": (0, 0)},
        "M07": {"type": "action",  "pos": (1, 0)},
        "M08": {"type": "action",  "pos": (2, 0)},
    }
    return ExactModularRobotGraph.from_grid_layout(modules)


def demo_counterexample() -> None:
    """
    This matches the key criticism of the one-step lifetime heuristic:
    the best immediate lifetime proxy is not always the best total horizon.
    """
    g = build_counterexample_graph()
    battery_capacity = {"B1": 3.0, "B2": 5.0}
    modules = ["M1", "M2"]

    nearest = g.simulate_until_failure(
        battery_capacity,
        strategy="nearest",
        modules=modules,
        weighted=True,
        respect_switch_state=True,
        base_module_cost=1.0,
        switch_loss_factor=1.0,
        max_rounds=10,
    )
    lifetime = g.simulate_until_failure(
        battery_capacity,
        strategy="lifetime",
        modules=modules,
        weighted=True,
        respect_switch_state=True,
        base_module_cost=1.0,
        switch_loss_factor=1.0,
        max_rounds=10,
    )
    exact = g.maximize_total_rounds(
        battery_capacity,
        modules=modules,
        weighted=True,
        respect_switch_state=True,
        base_module_cost=1.0,
        switch_loss_factor=1.0,
    )

    print("=" * 76)
    print("DEMO 1: Counterexample where one-step lifetime balancing is not globally optimal")
    print("=" * 76)
    print("Capacities:")
    print(f"  {battery_capacity}")
    print("Per-round route energies:")
    print("  M1 -> B1 = 1, M1 -> B2 = 1")
    print("  M2 -> B1 = 3, M2 -> B2 = 3")
    print(f"Nearest-only feasible rounds: {nearest.completed_rounds}")
    print(f"Greedy lifetime heuristic feasible rounds: {lifetime.completed_rounds}")
    print(f"Exact maximum feasible rounds: {exact.max_rounds}")
    print_energy_table("Remaining energy before the exact horizon:", exact.remaining_before)
    print_energy_table("Total battery energy used by the exact horizon:", exact.total_energy_by_battery)
    print_energy_table("Remaining energy after the exact horizon:", exact.remaining_after)
    print_allocation_counts(exact)
    print_schedule(exact)


def demo_exact_on_robot_layout() -> None:
    """
    Compare nearest, greedy lifetime, and exact total-horizon optimization
    on a compact robot-style layout that is fast enough for a default demo run.
    """
    g = build_compact_demo_graph()
    modules = g.get_actions()
    battery_capacity = {
        "B1": 45.0,
        "B2": 45.0,
        "B3": 45.0,
    }
    module_demand = {
        "M02": 1.3,
        "M04": 1.4,
        "M05": 1.5,
        "M06": 1.2,
        "M08": 1.4,
    }

    nearest = g.simulate_until_failure(
        battery_capacity,
        strategy="nearest",
        modules=modules,
        module_demand=module_demand,
        respect_switch_state=False,
        base_module_cost=1.0,
        switch_loss_factor=1.0,
        max_rounds=20,
    )
    lifetime = g.simulate_until_failure(
        battery_capacity,
        strategy="lifetime",
        modules=modules,
        module_demand=module_demand,
        respect_switch_state=False,
        base_module_cost=1.0,
        switch_loss_factor=1.0,
        max_rounds=20,
    )
    exact = g.maximize_total_rounds(
        battery_capacity,
        modules=modules,
        module_demand=module_demand,
        respect_switch_state=False,
        base_module_cost=1.0,
        switch_loss_factor=1.0,
        max_rounds_upper_bound=10,
    )

    print()
    print("=" * 76)
    print("DEMO 2: Exact total working time on a compact robot-style layout")
    print("=" * 76)
    print(f"Nearest-only feasible rounds: {nearest.completed_rounds}")
    print(f"Greedy lifetime heuristic feasible rounds: {lifetime.completed_rounds}")
    print(f"Exact maximum feasible rounds: {exact.max_rounds}")
    print_energy_table("Remaining energy before the exact horizon:", exact.remaining_before)
    print_energy_table("Total battery energy used by the exact horizon:", exact.total_energy_by_battery)
    print_energy_table("Remaining energy after the exact horizon:", exact.remaining_after)
    print_allocation_counts(exact)
    print_schedule(exact)


if __name__ == "__main__":
    demo_counterexample()
    demo_exact_on_robot_layout()
