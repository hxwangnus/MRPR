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
class LifetimeSnapshot:
    """
    One routing decision evaluated against explicit battery energy state.

    `rounds_remaining[b]` estimates how many more identical rounds battery `b`
    could support from the current pre-step energy state:

        remaining_before[b] / step_drain[b]

    when `step_drain[b] > 0`, otherwise `inf`.
    """

    assignments: Dict[str, AssignmentResult]
    step_drain: Dict[str, float]
    projected_discharge: Dict[str, float]
    remaining_before: Dict[str, float]
    remaining_after: Dict[str, float]
    rounds_remaining: Dict[str, float]
    moved_modules: List[Tuple[str, str, str]]
    limiting_battery: Optional[str]
    unpowered_modules: List[str]
    overloaded_batteries: List[str]
    feasible: bool


@dataclass
class SimulationResult:
    """History of feasible rounds until the first failure, if any."""

    strategy: str
    history: List[LifetimeSnapshot]
    completed_rounds: int
    failure_round: Optional[int]
    failure_snapshot: Optional[LifetimeSnapshot]


class ModularRobotGraph:
    """
    Capacity-aware power routing for modular robots.

    Compared with `power_routing_dynamic.py`, this file makes battery state
    explicit:

      - each battery has a finite energy capacity
      - each battery can already have some cumulative discharge
      - each action module has a per-round demand

    The optimizer starts from the nearest-battery assignment, then greedily
    reassigns modules to maximize the lifetime of the most stressed battery.
    The primary objective is:

        maximize time until the first battery can no longer support
        its assigned modules

    For one candidate assignment, if battery `b` has remaining energy `R_b`
    before this round and would spend `D_b` energy during this round, then the
    lifetime proxy for that battery is:

        R_b / D_b

    when `D_b > 0`, otherwise `inf`.

    The optimizer lexicographically improves the vector of per-battery
    utilization fractions `D_b / R_b`, which is equivalent to maximizing the
    worst remaining lifetime first, then the second worst, and so on.
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
    def from_manual_spec(
        cls,
        nodes: Iterable[Dict[str, Any]],
        switches: Iterable[Dict[str, Any]],
    ) -> "ModularRobotGraph":
        g = cls()
        for item in nodes:
            g.add_node(
                node_id=item["id"],
                node_type=item["type"],
                pos=item.get("pos"),
            )
        for item in switches:
            g.add_switch_edge(
                u=item["u"],
                v=item["v"],
                switch_id=item["switch_id"],
                open_=item.get("open", True),
                cost=item.get("cost", 1.0),
            )
        return g

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
    def _normalized_battery_discharge(
        battery_list: List[str],
        battery_capacity: Dict[str, float],
        battery_discharge: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        for battery in battery_list:
            if battery not in battery_capacity:
                raise KeyError(f"Missing capacity for battery {battery}")
            if battery_capacity[battery] < 0:
                raise ValueError(f"Battery capacity must be nonnegative: {battery}")

        discharge = {battery: 0.0 for battery in battery_list}
        if battery_discharge is not None:
            for battery, value in battery_discharge.items():
                if battery in discharge:
                    if value < 0:
                        raise ValueError(f"Battery discharge must be nonnegative: {battery}")
                    discharge[battery] = value
        return discharge

    @staticmethod
    def _normalized_module_demand(
        module_list: List[str],
        module_demand: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        demand = {module: 1.0 for module in module_list}
        if module_demand is not None:
            for module, value in module_demand.items():
                if module in demand:
                    if value < 0:
                        raise ValueError(f"Module demand must be nonnegative: {module}")
                    demand[module] = value
        return demand

    @staticmethod
    def _energy_for_module(
        demand: float,
        route_cost: float,
        *,
        base_module_cost: float,
        switch_loss_factor: float,
    ) -> float:
        return demand * (base_module_cost + switch_loss_factor * route_cost)

    def _compute_step_drain(
        self,
        assignments: Dict[str, AssignmentResult],
        battery_list: List[str],
        module_demand: Dict[str, float],
        *,
        base_module_cost: float,
        switch_loss_factor: float,
    ) -> Dict[str, float]:
        drain = {battery: 0.0 for battery in battery_list}
        for module, info in assignments.items():
            if info.battery is None or math.isinf(info.distance):
                continue
            drain[info.battery] += self._energy_for_module(
                module_demand.get(module, 1.0),
                info.distance,
                base_module_cost=base_module_cost,
                switch_loss_factor=switch_loss_factor,
            )
        return drain

    @staticmethod
    def _rounds_remaining(
        remaining_before: Dict[str, float],
        step_drain: Dict[str, float],
        battery_list: List[str],
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for battery in battery_list:
            drain = step_drain[battery]
            remaining = remaining_before[battery]
            if drain <= 0:
                out[battery] = math.inf
            elif remaining <= 0:
                out[battery] = 0.0
            else:
                out[battery] = remaining / drain
        return out

    @staticmethod
    def _lifetime_objective(
        remaining_before: Dict[str, float],
        step_drain: Dict[str, float],
        battery_list: List[str],
    ) -> Tuple[Tuple[float, ...], float, float]:
        """
        Smaller is better.

        Primary term:
            sorted battery utilization fractions in descending order

                utilization = step_drain / remaining_before

            So minimizing the first element is equivalent to maximizing the
            worst battery lifetime `remaining_before / step_drain`.

        Secondary terms:
            - smaller total energy per round
            - smaller peak drain on any battery
        """
        utilizations: List[float] = []
        for battery in battery_list:
            drain = step_drain[battery]
            remaining = remaining_before[battery]
            if drain <= 0:
                utilization = 0.0
            elif remaining <= 0:
                utilization = math.inf
            else:
                utilization = drain / remaining
            utilizations.append(utilization)

        total_drain = sum(step_drain.values())
        peak_drain = max(step_drain.values()) if step_drain else 0.0
        return (tuple(sorted(utilizations, reverse=True)), total_drain, peak_drain)

    @staticmethod
    def _limiting_battery(
        rounds_remaining: Dict[str, float],
        step_drain: Dict[str, float],
        battery_list: List[str],
    ) -> Optional[str]:
        active = [battery for battery in battery_list if step_drain[battery] > 0]
        if not active:
            return None
        return min(active, key=lambda battery: (rounds_remaining[battery], battery))

    def _build_snapshot(
        self,
        assignments: Dict[str, AssignmentResult],
        *,
        battery_list: List[str],
        battery_capacity: Dict[str, float],
        battery_discharge: Dict[str, float],
        module_demand: Dict[str, float],
        base_module_cost: float,
        switch_loss_factor: float,
        moved_modules: List[Tuple[str, str, str]],
    ) -> LifetimeSnapshot:
        step_drain = self._compute_step_drain(
            assignments,
            battery_list,
            module_demand,
            base_module_cost=base_module_cost,
            switch_loss_factor=switch_loss_factor,
        )
        remaining_before = {
            battery: battery_capacity[battery] - battery_discharge[battery]
            for battery in battery_list
        }
        projected_discharge = {
            battery: battery_discharge[battery] + step_drain[battery]
            for battery in battery_list
        }
        remaining_after = {
            battery: battery_capacity[battery] - projected_discharge[battery]
            for battery in battery_list
        }
        rounds_remaining = self._rounds_remaining(remaining_before, step_drain, battery_list)

        unpowered_modules = sorted(
            module
            for module, info in assignments.items()
            if info.battery is None or math.isinf(info.distance)
        )
        overloaded_batteries = sorted(
            battery
            for battery in battery_list
            if step_drain[battery] > remaining_before[battery] + 1e-9
        )
        feasible = not unpowered_modules and not overloaded_batteries

        return LifetimeSnapshot(
            assignments=assignments,
            step_drain=step_drain,
            projected_discharge=projected_discharge,
            remaining_before=remaining_before,
            remaining_after=remaining_after,
            rounds_remaining=rounds_remaining,
            moved_modules=moved_modules,
            limiting_battery=self._limiting_battery(rounds_remaining, step_drain, battery_list),
            unpowered_modules=unpowered_modules,
            overloaded_batteries=overloaded_batteries,
            feasible=feasible,
        )

    def nearest_lifetime_snapshot(
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
    ) -> LifetimeSnapshot:
        battery_list = list(batteries) if batteries is not None else self.get_batteries()
        module_list = list(modules) if modules is not None else self.get_actions()
        discharge = self._normalized_battery_discharge(
            battery_list,
            battery_capacity,
            battery_discharge,
        )
        demand = self._normalized_module_demand(module_list, module_demand)

        assignments = self.nearest_battery_assignment(
            batteries=battery_list,
            modules=module_list,
            weighted=weighted,
            respect_switch_state=respect_switch_state,
        )
        return self._build_snapshot(
            assignments,
            battery_list=battery_list,
            battery_capacity=battery_capacity,
            battery_discharge=discharge,
            module_demand=demand,
            base_module_cost=base_module_cost,
            switch_loss_factor=switch_loss_factor,
            moved_modules=[],
        )

    def lifetime_maximizing_assignment(
        self,
        battery_capacity: Dict[str, float],
        *,
        battery_discharge: Optional[Dict[str, float]] = None,
        batteries: Optional[Iterable[str]] = None,
        modules: Optional[Iterable[str]] = None,
        module_demand: Optional[Dict[str, float]] = None,
        weighted: bool = False,
        respect_switch_state: bool = True,
        max_extra_cost: Optional[float] = 2.0,
        base_module_cost: float = 1.0,
        switch_loss_factor: float = 1.0,
    ) -> LifetimeSnapshot:
        """
        Compute a routing decision that directly targets battery lifetime.

        Strategy:
          1) start from the original nearest-battery assignment
          2) greedily move modules between reachable batteries
          3) accept a move only if it improves the lifetime objective

        The primary objective is to maximize the number of additional rounds
        before the most stressed battery fails.
        """
        battery_list = list(batteries) if batteries is not None else self.get_batteries()
        module_list = list(modules) if modules is not None else self.get_actions()
        discharge = self._normalized_battery_discharge(
            battery_list,
            battery_capacity,
            battery_discharge,
        )
        demand = self._normalized_module_demand(module_list, module_demand)
        remaining_before = {
            battery: battery_capacity[battery] - discharge[battery]
            for battery in battery_list
        }

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

        current_owner: Dict[str, Optional[str]] = {module: baseline[module].battery for module in module_list}
        current_distance: Dict[str, float] = {module: baseline[module].distance for module in module_list}
        current_assignments: Dict[str, AssignmentResult] = dict(baseline)
        step_drain = self._compute_step_drain(
            current_assignments,
            battery_list,
            demand,
            base_module_cost=base_module_cost,
            switch_loss_factor=switch_loss_factor,
        )
        moved_modules: List[Tuple[str, str, str]] = []

        while True:
            current_obj = self._lifetime_objective(remaining_before, step_drain, battery_list)
            best_move: Optional[Tuple[str, str, str, float, AssignmentResult, float, float]] = None
            best_obj = current_obj

            for module in module_list:
                owner = current_owner[module]
                if owner is None or math.isinf(current_distance[module]):
                    continue

                base_distance = baseline[module].distance
                current_energy = self._energy_for_module(
                    demand[module],
                    current_distance[module],
                    base_module_cost=base_module_cost,
                    switch_loss_factor=switch_loss_factor,
                )

                for candidate in battery_list:
                    if candidate == owner:
                        continue

                    dist_map, parent = shortest_cache[candidate]
                    candidate_distance = dist_map.get(module, math.inf)
                    if math.isinf(candidate_distance):
                        continue
                    if (
                        max_extra_cost is not None
                        and candidate_distance > base_distance + max_extra_cost
                    ):
                        continue

                    path = self.reconstruct_path(parent, module)
                    nodes_path, switches_path = path if path is not None else (None, None)
                    candidate_assignment = AssignmentResult(
                        module=module,
                        battery=candidate,
                        distance=candidate_distance,
                        path_nodes=nodes_path,
                        path_switches=switches_path,
                    )
                    candidate_energy = self._energy_for_module(
                        demand[module],
                        candidate_distance,
                        base_module_cost=base_module_cost,
                        switch_loss_factor=switch_loss_factor,
                    )

                    trial_step_drain = dict(step_drain)
                    trial_step_drain[owner] -= current_energy
                    trial_step_drain[candidate] += candidate_energy
                    trial_obj = self._lifetime_objective(
                        remaining_before,
                        trial_step_drain,
                        battery_list,
                    )

                    move_key = (trial_obj, module, owner, candidate)
                    best_key = (best_obj, chr(255) * 20, chr(255) * 20, chr(255) * 20)
                    if move_key < best_key:
                        best_obj = trial_obj
                        best_move = (
                            module,
                            owner,
                            candidate,
                            candidate_distance,
                            candidate_assignment,
                            current_energy,
                            candidate_energy,
                        )

            if best_move is None or best_obj >= current_obj:
                break

            (
                module,
                old_owner,
                new_owner,
                new_distance,
                candidate_assignment,
                current_energy,
                candidate_energy,
            ) = best_move

            step_drain[old_owner] -= current_energy
            step_drain[new_owner] += candidate_energy
            current_owner[module] = new_owner
            current_distance[module] = new_distance
            current_assignments[module] = candidate_assignment
            moved_modules.append((module, old_owner, new_owner))

        return self._build_snapshot(
            current_assignments,
            battery_list=battery_list,
            battery_capacity=battery_capacity,
            battery_discharge=discharge,
            module_demand=demand,
            base_module_cost=base_module_cost,
            switch_loss_factor=switch_loss_factor,
            moved_modules=moved_modules,
        )

    def simulate_until_failure(
        self,
        battery_capacity: Dict[str, float],
        *,
        strategy: str,
        initial_discharge: Optional[Dict[str, float]] = None,
        batteries: Optional[Iterable[str]] = None,
        modules: Optional[Iterable[str]] = None,
        module_demand: Optional[Dict[str, float]] = None,
        weighted: bool = False,
        respect_switch_state: bool = True,
        max_extra_cost: Optional[float] = 2.0,
        base_module_cost: float = 1.0,
        switch_loss_factor: float = 1.0,
        max_rounds: int = 100,
    ) -> SimulationResult:
        battery_list = list(batteries) if batteries is not None else self.get_batteries()
        module_list = list(modules) if modules is not None else self.get_actions()
        discharge = self._normalized_battery_discharge(
            battery_list,
            battery_capacity,
            initial_discharge,
        )

        history: List[LifetimeSnapshot] = []
        for round_idx in range(1, max_rounds + 1):
            if strategy == "nearest":
                snapshot = self.nearest_lifetime_snapshot(
                    battery_capacity,
                    battery_discharge=discharge,
                    batteries=battery_list,
                    modules=module_list,
                    module_demand=module_demand,
                    weighted=weighted,
                    respect_switch_state=respect_switch_state,
                    base_module_cost=base_module_cost,
                    switch_loss_factor=switch_loss_factor,
                )
            elif strategy == "lifetime":
                snapshot = self.lifetime_maximizing_assignment(
                    battery_capacity,
                    battery_discharge=discharge,
                    batteries=battery_list,
                    modules=module_list,
                    module_demand=module_demand,
                    weighted=weighted,
                    respect_switch_state=respect_switch_state,
                    max_extra_cost=max_extra_cost,
                    base_module_cost=base_module_cost,
                    switch_loss_factor=switch_loss_factor,
                )
            else:
                raise ValueError("strategy must be 'nearest' or 'lifetime'")

            if not snapshot.feasible:
                return SimulationResult(
                    strategy=strategy,
                    history=history,
                    completed_rounds=len(history),
                    failure_round=round_idx,
                    failure_snapshot=snapshot,
                )

            history.append(snapshot)
            discharge = dict(snapshot.projected_discharge)

        return SimulationResult(
            strategy=strategy,
            history=history,
            completed_rounds=len(history),
            failure_round=None,
            failure_snapshot=None,
        )


def format_energy_value(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.2f}"


def print_energy_table(title: str, values: Dict[str, float]) -> None:
    print(title)
    pretty = {
        battery: ("inf" if math.isinf(value) else round(value, 2))
        for battery, value in sorted(values.items())
    }
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


def print_snapshot_summary(title: str, snapshot: LifetimeSnapshot) -> None:
    print(title)
    print_compact_assignments("Assignments:", snapshot.assignments)
    print_moved_modules(snapshot.moved_modules)
    print_energy_table("Remaining energy before this round:", snapshot.remaining_before)
    print_energy_table("Step drain for this round:", snapshot.step_drain)
    print_energy_table("Estimated identical rounds remaining:", snapshot.rounds_remaining)
    print(f"Limiting battery: {snapshot.limiting_battery}")
    print(f"Feasible this round: {snapshot.feasible}")
    if snapshot.unpowered_modules:
        print(f"Unpowered modules: {snapshot.unpowered_modules}")
    if snapshot.overloaded_batteries:
        print(f"Overloaded batteries: {snapshot.overloaded_batteries}")


def print_simulation_result(title: str, result: SimulationResult) -> None:
    print(title)
    print(f"  strategy         = {result.strategy}")
    print(f"  completed_rounds = {result.completed_rounds}")
    if result.failure_round is None:
        print("  failure_round    = none within simulation horizon")
    else:
        print(f"  failure_round    = {result.failure_round}")
    if result.history:
        last = result.history[-1]
        print_energy_table("  cumulative discharge after last feasible round:", last.projected_discharge)
        print_energy_table("  remaining energy after last feasible round:", last.remaining_after)
    if result.failure_snapshot is not None:
        failure = result.failure_snapshot
        print_energy_table("  next-round remaining energy before failure:", failure.remaining_before)
        print_energy_table("  next-round drain request:", failure.step_drain)
        print(f"  next-round overloaded batteries = {failure.overloaded_batteries}")
        print(f"  next-round limiting battery     = {failure.limiting_battery}")


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


def demo_lifetime_replan_from_skewed_state() -> None:
    """
    Compare one next-step decision when one battery is already heavily depleted.
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

    battery_capacity = {
        "B1": 90.0,
        "B2": 90.0,
        "B3": 90.0,
        "B4": 90.0,
        "B5": 90.0,
    }
    initial_discharge = {
        "B1": 18.0,
        "B2": 62.0,
        "B3": 24.0,
        "B4": 26.0,
        "B5": 16.0,
    }
    module_demand = {
        "M03": 1.2,
        "M04": 1.4,
        "M05": 1.6,
        "M06": 1.4,
        "M16": 1.6,
        "M17": 1.5,
        "M18": 1.4,
        "M09": 1.3,
    }

    nearest = g.nearest_lifetime_snapshot(
        battery_capacity,
        battery_discharge=initial_discharge,
        modules=modules,
        module_demand=module_demand,
        respect_switch_state=False,
        base_module_cost=1.0,
        switch_loss_factor=1.0,
    )
    lifetime = g.lifetime_maximizing_assignment(
        battery_capacity,
        battery_discharge=initial_discharge,
        modules=modules,
        module_demand=module_demand,
        respect_switch_state=False,
        max_extra_cost=2.0,
        base_module_cost=1.0,
        switch_loss_factor=1.0,
    )

    print("=" * 78)
    print("DEMO 1: Lifetime-aware replanning from an already skewed battery state")
    print("=" * 78)
    print("Layout:")
    for line in layout_lines:
        print(f"  {line}")
    print_energy_table("Battery capacities:", battery_capacity)
    print_energy_table("Cumulative discharge before this round:", initial_discharge)
    print()
    print_snapshot_summary("Nearest-only decision:", nearest)
    print()
    print_snapshot_summary("Lifetime-aware decision:", lifetime)


def demo_total_working_time_comparison() -> None:
    """
    Compare total feasible rounds until the first battery failure.
    """
    g = build_demo4_graph()
    modules = g.get_actions()

    battery_capacity = {
        "B1": 90.0,
        "B2": 90.0,
        "B3": 90.0,
        "B4": 90.0,
        "B5": 90.0,
    }
    module_demand = {
        "M03": 1.2,
        "M04": 1.4,
        "M05": 1.6,
        "M06": 1.4,
        "M09": 1.3,
        "M16": 1.6,
        "M17": 1.5,
        "M18": 1.4,
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
        max_extra_cost=2.0,
        base_module_cost=1.0,
        switch_loss_factor=1.0,
        max_rounds=20,
    )

    print()
    print("=" * 78)
    print("DEMO 2: Total working time until the first battery can no longer cope")
    print("=" * 78)
    print_simulation_result("Nearest-only routing:", nearest)
    print()
    print_simulation_result("Lifetime-aware routing:", lifetime)
    print()
    print(f"Rounds gained by lifetime-aware routing: {lifetime.completed_rounds - nearest.completed_rounds}")


if __name__ == "__main__":
    demo_lifetime_replan_from_skewed_state()
    demo_total_working_time_comparison()
