from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Iterable, Any
import heapq
import math


@dataclass(frozen=True)
class Edge:
    """One undirected physical link between two modules, governed by one switch."""
    other: str
    switch_id: str
    cost: float = 1.0


@dataclass
class AssignmentResult:
    """Routing result for one action module."""
    module: str
    battery: Optional[str]
    distance: float
    path_nodes: Optional[List[str]]
    path_switches: Optional[List[str]]


@dataclass
class RoutingSnapshot:
    """One dynamic-routing decision for a single time step."""
    assignments: Dict[str, AssignmentResult]
    step_usage: Dict[str, float]
    projected_usage: Dict[str, float]
    moved_modules: List[Tuple[str, str, str]]


class ModularRobotGraph:
    """
    Dynamic power routing for modular robots.

    Key idea:
      - `duo_mode` solves "who is nearest right now?"
      - `balanced` solves "how do we fix equal-distance ties?"
      - this file solves "given cumulative battery usage over time,
        how should we reassign modules now to keep battery lifetimes aligned?"

    The dynamic optimizer starts from the original nearest-battery assignment,
    then moves modules between reachable batteries if doing so improves a
    global lifetime-oriented objective based on cumulative battery usage.
    """

    def __init__(self) -> None:
        self.node_type: Dict[str, str] = {}
        self.node_pos: Dict[str, Tuple[int, int]] = {}
        self.adj: Dict[str, List[Edge]] = defaultdict(list)

        self.switch_open: Dict[str, bool] = {}
        self.switch_endpoints: Dict[str, Tuple[str, str]] = {}
        self.link_to_switch: Dict[Tuple[str, str], str] = {}

    @staticmethod
    def _norm_pair(u: str, v: str) -> Tuple[str, str]:
        if u == v:
            raise ValueError("Self-loop is not allowed.")
        return tuple(sorted((u, v)))

    def _edge_traversable(self, edge: Edge, respect_switch_state: bool) -> bool:
        return (not respect_switch_state) or self.switch_open.get(edge.switch_id, False)

    def add_node(self, node_id: str, node_type: str, pos: Optional[Tuple[int, int]] = None) -> None:
        if node_id in self.node_type and self.node_type[node_id] != node_type:
            raise ValueError(
                f"Node {node_id} already exists with type {self.node_type[node_id]}, "
                f"cannot change to {node_type}."
            )
        self.node_type[node_id] = node_type
        if pos is not None:
            self.node_pos[node_id] = pos
        _ = self.adj[node_id]

    def add_switch_edge(
        self,
        u: str,
        v: str,
        switch_id: str,
        open_: bool = True,
        cost: float = 1.0,
    ) -> None:
        if u not in self.node_type or v not in self.node_type:
            raise KeyError("Both endpoints must be added first via add_node().")
        if cost < 0:
            raise ValueError("Edge cost must be nonnegative.")

        norm = self._norm_pair(u, v)
        if switch_id in self.switch_endpoints:
            old = self.switch_endpoints[switch_id]
            if old != norm:
                raise ValueError(
                    f"switch_id={switch_id} already belongs to endpoints {old}, not {norm}."
                )
            raise ValueError(f"Duplicate switch_id detected: {switch_id}")
        if norm in self.link_to_switch:
            prev_switch = self.link_to_switch[norm]
            raise ValueError(
                f"Physical link {norm} already has switch {prev_switch}. "
                f"Duplicate edge is not allowed in this simplified model."
            )

        self.switch_open[switch_id] = open_
        self.switch_endpoints[switch_id] = norm
        self.link_to_switch[norm] = switch_id
        self.adj[u].append(Edge(other=v, switch_id=switch_id, cost=cost))
        self.adj[v].append(Edge(other=u, switch_id=switch_id, cost=cost))

    def set_switch_state(self, switch_id: str, open_: bool) -> None:
        if switch_id not in self.switch_open:
            raise KeyError(f"Unknown switch_id: {switch_id}")
        self.switch_open[switch_id] = open_

    def get_batteries(self) -> List[str]:
        return sorted(n for n, t in self.node_type.items() if t == "battery")

    def get_actions(self) -> List[str]:
        return sorted(n for n, t in self.node_type.items() if t == "action")

    def shortest_paths_from(
        self,
        source: str,
        *,
        weighted: bool = False,
        respect_switch_state: bool = True,
    ) -> Tuple[Dict[str, float], Dict[str, Optional[Tuple[str, str]]]]:
        if source not in self.node_type:
            raise KeyError(f"Unknown source node: {source}")

        if not weighted:
            dist: Dict[str, float] = {source: 0.0}
            parent: Dict[str, Optional[Tuple[str, str]]] = {source: None}
            q = deque([source])

            while q:
                u = q.popleft()
                for e in self.adj[u]:
                    if not self._edge_traversable(e, respect_switch_state):
                        continue
                    v = e.other
                    if v not in dist:
                        dist[v] = dist[u] + 1.0
                        parent[v] = (u, e.switch_id)
                        q.append(v)

            return dist, parent

        dist = {source: 0.0}
        parent = {source: None}
        pq: List[Tuple[float, str]] = [(0.0, source)]

        while pq:
            d_u, u = heapq.heappop(pq)
            if d_u != dist.get(u, math.inf):
                continue
            for e in self.adj[u]:
                if not self._edge_traversable(e, respect_switch_state):
                    continue
                v = e.other
                nd = d_u + e.cost
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    parent[v] = (u, e.switch_id)
                    heapq.heappush(pq, (nd, v))

        return dist, parent

    @staticmethod
    def reconstruct_path(
        parent: Dict[str, Optional[Tuple[str, str]]],
        target: str,
    ) -> Optional[Tuple[List[str], List[str]]]:
        if target not in parent:
            return None

        nodes: List[str] = []
        switches: List[str] = []
        cur = target

        while True:
            nodes.append(cur)
            p = parent[cur]
            if p is None:
                break
            prev, sw = p
            switches.append(sw)
            cur = prev

        nodes.reverse()
        switches.reverse()
        return nodes, switches

    def nearest_battery_assignment(
        self,
        *,
        batteries: Optional[Iterable[str]] = None,
        modules: Optional[Iterable[str]] = None,
        weighted: bool = False,
        respect_switch_state: bool = True,
    ) -> Dict[str, AssignmentResult]:
        battery_list = list(batteries) if batteries is not None else self.get_batteries()
        module_list = list(modules) if modules is not None else self.get_actions()

        shortest_cache: Dict[str, Tuple[Dict[str, float], Dict[str, Optional[Tuple[str, str]]]]] = {}
        for battery in battery_list:
            shortest_cache[battery] = self.shortest_paths_from(
                battery,
                weighted=weighted,
                respect_switch_state=respect_switch_state,
            )

        result: Dict[str, AssignmentResult] = {}
        for module in module_list:
            best: Optional[Tuple[float, str, Optional[List[str]], Optional[List[str]]]] = None
            for battery in battery_list:
                dist_map, parent = shortest_cache[battery]
                if module not in dist_map:
                    continue
                path = self.reconstruct_path(parent, module)
                nodes_path, switches_path = path if path is not None else (None, None)
                candidate = (dist_map[module], battery, nodes_path, switches_path)
                if best is None or candidate < best:
                    best = candidate

            if best is None:
                result[module] = AssignmentResult(
                    module=module,
                    battery=None,
                    distance=math.inf,
                    path_nodes=None,
                    path_switches=None,
                )
            else:
                distance, battery, nodes_path, switches_path = best
                result[module] = AssignmentResult(
                    module=module,
                    battery=battery,
                    distance=distance,
                    path_nodes=nodes_path,
                    path_switches=switches_path,
                )

        return result

    @classmethod
    def from_grid_layout(
        cls,
        modules: Dict[str, Dict[str, Any]],
        *,
        default_open: bool = True,
        default_cost: float = 1.0,
        switch_prefix: str = "S",
    ) -> "ModularRobotGraph":
        g = cls()
        pos_to_node: Dict[Tuple[int, int], str] = {}

        for node_id, info in modules.items():
            pos = tuple(info["pos"])
            if pos in pos_to_node:
                raise ValueError(
                    f"Duplicate grid position {pos}: {node_id} conflicts with {pos_to_node[pos]}"
                )
            pos_to_node[pos] = node_id
            g.add_node(node_id, info["type"], pos=pos)

        for node_id, info in modules.items():
            x, y = info["pos"]
            for dx, dy in [(1, 0), (0, 1)]:
                nb_pos = (x + dx, y + dy)
                if nb_pos in pos_to_node:
                    other = pos_to_node[nb_pos]
                    g.add_switch_edge(
                        u=node_id,
                        v=other,
                        switch_id=f"{switch_prefix}_{node_id}_{other}",
                        open_=default_open,
                        cost=default_cost,
                    )

        return g

    @staticmethod
    def _usage_objective(
        projected_usage: Dict[str, float],
        step_usage: Dict[str, float],
    ) -> Tuple[float, float, float, float]:
        values = list(projected_usage.values())
        if not values:
            return (0.0, 0.0, 0.0, 0.0)
        peak = max(values)
        spread = max(values) - min(values)
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values)
        total_step = sum(step_usage.values())
        return (peak, spread, variance, total_step)

    @staticmethod
    def _compute_step_usage(
        assignments: Dict[str, AssignmentResult],
        battery_list: List[str],
        module_demand: Dict[str, float],
    ) -> Dict[str, float]:
        usage = {battery: 0.0 for battery in battery_list}
        for module, info in assignments.items():
            if info.battery is None or math.isinf(info.distance):
                continue
            usage[info.battery] += module_demand.get(module, 1.0) * info.distance
        return usage

    def usage_aware_dynamic_assignment(
        self,
        battery_usage: Dict[str, float],
        *,
        batteries: Optional[Iterable[str]] = None,
        modules: Optional[Iterable[str]] = None,
        module_demand: Optional[Dict[str, float]] = None,
        weighted: bool = False,
        respect_switch_state: bool = True,
        max_extra_switches: Optional[float] = 2.0,
    ) -> RoutingSnapshot:
        """
        Compute the next-step routing decision using cumulative battery usage.

        Start from the nearest-battery assignment, then reassign modules when
        doing so improves the projected battery-usage objective for the next
        step. Unlike `balanced.py`, moves are allowed even when the new route
        is longer, as long as the move improves long-term usage balance.
        """
        battery_list = list(batteries) if batteries is not None else self.get_batteries()
        module_list = list(modules) if modules is not None else self.get_actions()
        demand = {module: 1.0 for module in module_list}
        if module_demand is not None:
            demand.update(module_demand)

        shortest_cache: Dict[str, Tuple[Dict[str, float], Dict[str, Optional[Tuple[str, str]]]]] = {}
        for battery in battery_list:
            shortest_cache[battery] = self.shortest_paths_from(
                battery,
                weighted=weighted,
                respect_switch_state=respect_switch_state,
            )

        baseline = self.nearest_battery_assignment(
            batteries=battery_list,
            modules=module_list,
            weighted=weighted,
            respect_switch_state=respect_switch_state,
        )

        current_owner: Dict[str, Optional[str]] = {m: baseline[m].battery for m in module_list}
        current_distance: Dict[str, float] = {m: baseline[m].distance for m in module_list}
        step_usage = self._compute_step_usage(baseline, battery_list, demand)
        projected_usage = {
            battery: battery_usage.get(battery, 0.0) + step_usage[battery]
            for battery in battery_list
        }

        moved_modules: List[Tuple[str, str, str]] = []

        while True:
            current_obj = self._usage_objective(projected_usage, step_usage)
            best_move: Optional[Tuple[str, str, str, float]] = None
            best_obj = current_obj

            for module in module_list:
                owner = current_owner[module]
                if owner is None or math.isinf(current_distance[module]):
                    continue

                base_distance = baseline[module].distance
                for candidate in battery_list:
                    if candidate == owner:
                        continue

                    dist_map, _ = shortest_cache[candidate]
                    candidate_distance = dist_map.get(module, math.inf)
                    if math.isinf(candidate_distance):
                        continue
                    if (
                        max_extra_switches is not None
                        and candidate_distance > base_distance + max_extra_switches
                    ):
                        continue

                    trial_step_usage = dict(step_usage)
                    trial_step_usage[owner] -= demand[module] * current_distance[module]
                    trial_step_usage[candidate] += demand[module] * candidate_distance
                    trial_projected = {
                        battery: battery_usage.get(battery, 0.0) + trial_step_usage[battery]
                        for battery in battery_list
                    }
                    trial_obj = self._usage_objective(trial_projected, trial_step_usage)

                    move_key = (trial_obj, module, owner, candidate)
                    best_key = (best_obj, chr(255) * 20, chr(255) * 20, chr(255) * 20)
                    if move_key < best_key:
                        best_obj = trial_obj
                        best_move = (module, owner, candidate, candidate_distance)

            if best_move is None or best_obj >= current_obj:
                break

            module, old_owner, new_owner, new_distance = best_move
            step_usage[old_owner] -= demand[module] * current_distance[module]
            step_usage[new_owner] += demand[module] * new_distance
            current_owner[module] = new_owner
            current_distance[module] = new_distance
            projected_usage = {
                battery: battery_usage.get(battery, 0.0) + step_usage[battery]
                for battery in battery_list
            }
            moved_modules.append((module, old_owner, new_owner))

        assignments: Dict[str, AssignmentResult] = {}
        for module in module_list:
            owner = current_owner[module]
            if owner is None or math.isinf(current_distance[module]):
                assignments[module] = AssignmentResult(
                    module=module,
                    battery=None,
                    distance=math.inf,
                    path_nodes=None,
                    path_switches=None,
                )
                continue

            dist_map, parent = shortest_cache[owner]
            path = self.reconstruct_path(parent, module)
            nodes_path, switches_path = path if path is not None else (None, None)
            assignments[module] = AssignmentResult(
                module=module,
                battery=owner,
                distance=dist_map[module],
                path_nodes=nodes_path,
                path_switches=switches_path,
            )

        return RoutingSnapshot(
            assignments=assignments,
            step_usage=step_usage,
            projected_usage=projected_usage,
            moved_modules=moved_modules,
        )

    def simulate_over_time(
        self,
        *,
        rounds: int,
        strategy: str,
        initial_usage: Optional[Dict[str, float]] = None,
        batteries: Optional[Iterable[str]] = None,
        modules: Optional[Iterable[str]] = None,
        module_demand: Optional[Dict[str, float]] = None,
        respect_switch_state: bool = True,
        max_extra_switches: Optional[float] = 2.0,
    ) -> List[RoutingSnapshot]:
        battery_list = list(batteries) if batteries is not None else self.get_batteries()
        module_list = list(modules) if modules is not None else self.get_actions()

        usage = {battery: 0.0 for battery in battery_list}
        if initial_usage is not None:
            usage.update(initial_usage)

        history: List[RoutingSnapshot] = []
        for _ in range(rounds):
            if strategy == "nearest":
                assignments = self.nearest_battery_assignment(
                    batteries=battery_list,
                    modules=module_list,
                    respect_switch_state=respect_switch_state,
                )
                step_usage = self._compute_step_usage(
                    assignments,
                    battery_list,
                    module_demand or {},
                )
                usage = {
                    battery: usage[battery] + step_usage[battery]
                    for battery in battery_list
                }
                history.append(RoutingSnapshot(
                    assignments=assignments,
                    step_usage=step_usage,
                    projected_usage=dict(usage),
                    moved_modules=[],
                ))
            elif strategy == "dynamic":
                snapshot = self.usage_aware_dynamic_assignment(
                    usage,
                    batteries=battery_list,
                    modules=module_list,
                    module_demand=module_demand,
                    respect_switch_state=respect_switch_state,
                    max_extra_switches=max_extra_switches,
                )
                usage = dict(snapshot.projected_usage)
                history.append(snapshot)
            else:
                raise ValueError("strategy must be 'nearest' or 'dynamic'")

        return history


def print_usage_table(title: str, usage: Dict[str, float]) -> None:
    print(title)
    pretty = {battery: round(value, 2) for battery, value in sorted(usage.items())}
    print(f"  {pretty}")


def print_compact_assignments(title: str, assignments: Dict[str, AssignmentResult]) -> None:
    print(title)
    for module, info in sorted(assignments.items()):
        distance_str = "inf" if math.isinf(info.distance) else info.distance
        print(f"  {module}: {info.battery} (d={distance_str})")


def print_moved_modules(moves: List[Tuple[str, str, str]]) -> None:
    print("Moved modules:")
    if not moves:
        print("  None")
        return
    for module, old_battery, new_battery in moves:
        print(f"  {module}: {old_battery} -> {new_battery}")


def build_demo4_graph() -> ModularRobotGraph:
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
    return ModularRobotGraph.from_grid_layout(modules)


def demo_dynamic_replan_after_skew() -> None:
    """
    Build usage skew by running the nearest-only strategy for 4 rounds,
    then compare round 5:
      - keep using the original nearest assignment
      - replan with the dynamic lifetime-aware optimizer
    """
    g = build_demo4_graph()
    modules = g.get_actions()
    layout_lines = [
        "B1  M01 M02",
        "        M03",
        "M19 M18 M04 B2 M05",
        "B5     M17     M06 M20",
        "M15 M14 M16 M08 M07 B3",
        "    M13     M09",
        "B4  M12 M11 M10",
    ]

    nearest_history = g.simulate_over_time(
        rounds=4,
        strategy="nearest",
        modules=modules,
        respect_switch_state=False,
    )
    usage_after_4 = nearest_history[-1].projected_usage

    nearest_round_5 = g.simulate_over_time(
        rounds=1,
        strategy="nearest",
        initial_usage=usage_after_4,
        modules=modules,
        respect_switch_state=False,
    )[0]
    dynamic_round_5 = g.usage_aware_dynamic_assignment(
        usage_after_4,
        modules=modules,
        respect_switch_state=False,
        max_extra_switches=2.0,
    )

    print("=" * 76)
    print("DEMO 1: Dynamic replanning after usage has become skewed")
    print("=" * 76)
    print("Layout:")
    for line in layout_lines:
        print(f"  {line}")
    print_usage_table("Cumulative battery usage after 4 nearest-only rounds:", usage_after_4)
    print()
    print_compact_assignments("Round 5 if we keep the original nearest assignment:", nearest_round_5.assignments)
    print_usage_table("Projected cumulative usage after that nearest-only round:", nearest_round_5.projected_usage)
    print()
    print_compact_assignments("Round 5 with dynamic usage-aware reassignment:", dynamic_round_5.assignments)
    print_moved_modules(dynamic_round_5.moved_modules)
    print_usage_table("Projected cumulative usage after the dynamic round:", dynamic_round_5.projected_usage)


def demo_multi_round_comparison() -> None:
    """
    Compare lifetime behavior over multiple rounds:
      - original nearest-only routing
      - dynamic usage-aware routing
    """
    g = build_demo4_graph()
    modules = g.get_actions()

    nearest_history = g.simulate_over_time(
        rounds=6,
        strategy="nearest",
        modules=modules,
        respect_switch_state=False,
    )
    dynamic_history = g.simulate_over_time(
        rounds=6,
        strategy="dynamic",
        modules=modules,
        respect_switch_state=False,
        max_extra_switches=2.0,
    )

    print()
    print("=" * 76)
    print("DEMO 2: Multi-round nearest-only vs real dynamic routing")
    print("=" * 76)
    print("Per-round cumulative battery usage:")
    for round_idx in range(6):
        nearest_usage = nearest_history[round_idx].projected_usage
        dynamic_usage = dynamic_history[round_idx].projected_usage
        print(f"  Round {round_idx + 1}:")
        print(f"    nearest = { {b: round(v, 2) for b, v in sorted(nearest_usage.items())} }")
        print(f"    dynamic = { {b: round(v, 2) for b, v in sorted(dynamic_usage.items())} }")

    print()
    print_compact_assignments(
        "Final dynamic assignments on round 6:",
        dynamic_history[-1].assignments,
    )
    print_moved_modules(dynamic_history[-1].moved_modules)
    print_usage_table("Final nearest-only cumulative usage:", nearest_history[-1].projected_usage)
    print_usage_table("Final dynamic cumulative usage:", dynamic_history[-1].projected_usage)


if __name__ == "__main__":
    demo_dynamic_replan_after_skew()
    demo_multi_round_comparison()
