from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Iterable, Set, Any
import math
import heapq


@dataclass(frozen=True)
class Edge:
    """One undirected physical link between two modules, governed by one switch."""
    other: str
    switch_id: str
    cost: float = 1.0


@dataclass
class AssignmentResult:
    """
    Final routing result for one action module.

    `load_before` and `load_after` are used by the balanced assignment logic.
    For plain nearest-battery assignment they remain None.
    """
    module: str
    battery: Optional[str]
    distance: float
    path_nodes: Optional[List[str]]
    path_switches: Optional[List[str]]
    load_before: Optional[int] = None
    load_after: Optional[int] = None


class ModularRobotGraph:
    """
    Graph model for modular robot power routing.

    This file keeps the same capabilities as `power_routing_duo_mode.py`
    and adds load-aware tie-breaking for dynamic assignment.
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

    def get_switch_state(self, switch_id: str) -> bool:
        if switch_id not in self.switch_open:
            raise KeyError(f"Unknown switch_id: {switch_id}")
        return self.switch_open[switch_id]

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

    def multi_source_shortest_paths(
        self,
        sources: Iterable[str],
        *,
        weighted: bool = False,
        respect_switch_state: bool = True,
    ) -> Tuple[
        Dict[str, float],
        Dict[str, Optional[Tuple[str, str]]],
        Dict[str, str],
    ]:
        src_list = sorted(set(sources))
        if not src_list:
            raise ValueError("sources must be non-empty")

        for s in src_list:
            if s not in self.node_type:
                raise KeyError(f"Unknown source node: {s}")

        if not weighted:
            dist: Dict[str, float] = {}
            parent: Dict[str, Optional[Tuple[str, str]]] = {}
            owner: Dict[str, str] = {}
            q = deque()

            for s in src_list:
                dist[s] = 0.0
                parent[s] = None
                owner[s] = s
                q.append(s)

            while q:
                u = q.popleft()
                for e in self.adj[u]:
                    if not self._edge_traversable(e, respect_switch_state):
                        continue
                    v = e.other
                    if v not in dist:
                        dist[v] = dist[u] + 1.0
                        parent[v] = (u, e.switch_id)
                        owner[v] = owner[u]
                        q.append(v)

            return dist, parent, owner

        dist = {}
        parent = {}
        owner = {}
        pq: List[Tuple[float, str, str]] = []

        for s in src_list:
            dist[s] = 0.0
            parent[s] = None
            owner[s] = s
            heapq.heappush(pq, (0.0, s, s))

        while pq:
            d_u, owner_u, u = heapq.heappop(pq)
            current_key = (dist.get(u, math.inf), owner.get(u, chr(255) * 20))
            if (d_u, owner_u) != current_key:
                continue

            for e in self.adj[u]:
                if not self._edge_traversable(e, respect_switch_state):
                    continue
                v = e.other
                nd = d_u + e.cost

                old_d = dist.get(v, math.inf)
                old_owner = owner.get(v, chr(255) * 20)
                if (nd < old_d) or (math.isclose(nd, old_d) and owner_u < old_owner):
                    dist[v] = nd
                    parent[v] = (u, e.switch_id)
                    owner[v] = owner_u
                    heapq.heappush(pq, (nd, owner_u, v))

        return dist, parent, owner

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

    def all_battery_distance_table(
        self,
        *,
        batteries: Optional[Iterable[str]] = None,
        modules: Optional[Iterable[str]] = None,
        weighted: bool = False,
        respect_switch_state: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        batteries = list(batteries) if batteries is not None else self.get_batteries()
        modules = list(modules) if modules is not None else self.get_actions()

        table: Dict[str, Dict[str, float]] = {m: {} for m in modules}
        for b in batteries:
            dist, _ = self.shortest_paths_from(
                b,
                weighted=weighted,
                respect_switch_state=respect_switch_state,
            )
            for m in modules:
                table[m][b] = dist.get(m, math.inf)
        return table

    def nearest_battery_assignment(
        self,
        *,
        batteries: Optional[Iterable[str]] = None,
        modules: Optional[Iterable[str]] = None,
        weighted: bool = False,
        respect_switch_state: bool = True,
    ) -> Dict[str, AssignmentResult]:
        batteries = list(batteries) if batteries is not None else self.get_batteries()
        modules = list(modules) if modules is not None else self.get_actions()

        dist, parent, owner = self.multi_source_shortest_paths(
            batteries,
            weighted=weighted,
            respect_switch_state=respect_switch_state,
        )

        result: Dict[str, AssignmentResult] = {}
        for m in modules:
            if m not in dist:
                result[m] = AssignmentResult(
                    module=m,
                    battery=None,
                    distance=math.inf,
                    path_nodes=None,
                    path_switches=None,
                )
            else:
                path = self.reconstruct_path(parent, m)
                nodes_path, switches_path = path if path is not None else (None, None)
                result[m] = AssignmentResult(
                    module=m,
                    battery=owner[m],
                    distance=dist[m],
                    path_nodes=nodes_path,
                    path_switches=switches_path,
                )

        return result

    def recommend_switch_plan(
        self,
        *,
        active_modules: Optional[Iterable[str]] = None,
        batteries: Optional[Iterable[str]] = None,
        weighted: bool = False,
        respect_switch_state: bool = False,
    ) -> Dict[str, Any]:
        modules = list(active_modules) if active_modules is not None else self.get_actions()
        assignments = self.nearest_battery_assignment(
            batteries=batteries,
            modules=modules,
            weighted=weighted,
            respect_switch_state=respect_switch_state,
        )

        required_open_switches: Set[str] = set()
        unreachable_modules: List[str] = []

        for _, info in assignments.items():
            if info.battery is None:
                unreachable_modules.append(info.module)
                continue
            required_open_switches.update(info.path_switches or [])

        all_switches = set(self.switch_open.keys())
        return {
            "assignments": assignments,
            "required_open_switches": sorted(required_open_switches),
            "recommended_closed_switches": sorted(all_switches - required_open_switches),
            "unreachable_modules": sorted(unreachable_modules),
        }

    def dynamic_load_balanced_assignment(
        self,
        modules: Iterable[str],
        *,
        batteries: Optional[Iterable[str]] = None,
        existing_assignments: Optional[Dict[str, str]] = None,
        weighted: bool = False,
        respect_switch_state: bool = True,
    ) -> Tuple[Dict[str, AssignmentResult], Dict[str, int]]:
        """
        Assign modules one-by-one with load-aware tie-breaking.

        Priority:
          1) shorter distance
          2) lighter current load
          3) lexicographically smaller battery ID
        """
        battery_list = list(batteries) if batteries is not None else self.get_batteries()
        module_list = list(modules)
        existing = dict(existing_assignments or {})

        loads: Dict[str, int] = {b: 0 for b in battery_list}
        for _, battery in existing.items():
            if battery in loads:
                loads[battery] += 1

        shortest_cache: Dict[str, Tuple[Dict[str, float], Dict[str, Optional[Tuple[str, str]]]]] = {}
        for battery in battery_list:
            shortest_cache[battery] = self.shortest_paths_from(
                battery,
                weighted=weighted,
                respect_switch_state=respect_switch_state,
            )

        results: Dict[str, AssignmentResult] = {}
        for module in module_list:
            candidates: List[Tuple[float, int, str, Optional[List[str]], Optional[List[str]]]] = []
            for battery in battery_list:
                dist_map, parent = shortest_cache[battery]
                if module not in dist_map:
                    continue
                path = self.reconstruct_path(parent, module)
                nodes_path, switches_path = path if path is not None else (None, None)
                candidates.append((
                    dist_map[module],
                    loads[battery],
                    battery,
                    nodes_path,
                    switches_path,
                ))

            if not candidates:
                results[module] = AssignmentResult(
                    module=module,
                    battery=None,
                    distance=math.inf,
                    path_nodes=None,
                    path_switches=None,
                )
                continue

            best_distance, load_before, battery, nodes_path, switches_path = min(candidates)
            loads[battery] += 1
            existing[module] = battery
            results[module] = AssignmentResult(
                module=module,
                battery=battery,
                distance=best_distance,
                path_nodes=nodes_path,
                path_switches=switches_path,
                load_before=load_before,
                load_after=loads[battery],
            )

        return results, loads

    @staticmethod
    def _load_balance_objective(loads: Dict[str, int]) -> Tuple[int, float, Tuple[int, ...]]:
        values = list(loads.values())
        if not values:
            return (0, 0.0, ())
        mean = sum(values) / len(values)
        spread = max(values) - min(values)
        variance = sum((v - mean) ** 2 for v in values)
        return (spread, variance, tuple(sorted(values, reverse=True)))

    def rebalanced_nearest_battery_assignment(
        self,
        *,
        batteries: Optional[Iterable[str]] = None,
        modules: Optional[Iterable[str]] = None,
        weighted: bool = False,
        respect_switch_state: bool = True,
    ) -> Tuple[Dict[str, AssignmentResult], Dict[str, int]]:
        """
        Start from the original nearest-battery assignment, then rebalance
        only modules that have another battery at the exact same shortest
        distance. Moves are applied only when they strictly improve the
        global load-balance objective.
        """
        battery_list = list(batteries) if batteries is not None else self.get_batteries()
        module_list = list(modules) if modules is not None else self.get_actions()

        baseline = self.nearest_battery_assignment(
            batteries=battery_list,
            modules=module_list,
            weighted=weighted,
            respect_switch_state=respect_switch_state,
        )

        loads: Dict[str, int] = {b: 0 for b in battery_list}
        for module in module_list:
            battery = baseline[module].battery
            if battery in loads:
                loads[battery] += 1

        shortest_cache: Dict[str, Tuple[Dict[str, float], Dict[str, Optional[Tuple[str, str]]]]] = {}
        for battery in battery_list:
            shortest_cache[battery] = self.shortest_paths_from(
                battery,
                weighted=weighted,
                respect_switch_state=respect_switch_state,
            )

        current_owner: Dict[str, Optional[str]] = {m: baseline[m].battery for m in module_list}

        while True:
            current_obj = self._load_balance_objective(loads)
            best_move: Optional[Tuple[str, str, str]] = None
            best_obj = current_obj

            for module in module_list:
                owner = current_owner[module]
                if owner is None:
                    continue

                base_distance = baseline[module].distance
                for candidate in battery_list:
                    if candidate == owner:
                        continue

                    dist_map, _ = shortest_cache[candidate]
                    candidate_distance = dist_map.get(module, math.inf)
                    if not math.isclose(candidate_distance, base_distance):
                        continue

                    trial_loads = dict(loads)
                    trial_loads[owner] -= 1
                    trial_loads[candidate] += 1
                    trial_obj = self._load_balance_objective(trial_loads)

                    move_key = (trial_obj, module, owner, candidate)
                    best_key = (best_obj, chr(255) * 20, chr(255) * 20, chr(255) * 20)
                    if move_key < best_key:
                        best_obj = trial_obj
                        best_move = (module, owner, candidate)

            if best_move is None or best_obj >= current_obj:
                break

            module, old_owner, new_owner = best_move
            current_owner[module] = new_owner
            loads[old_owner] -= 1
            loads[new_owner] += 1

        result: Dict[str, AssignmentResult] = {}
        for module in module_list:
            owner = current_owner[module]
            if owner is None:
                result[module] = AssignmentResult(
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
            result[module] = AssignmentResult(
                module=module,
                battery=owner,
                distance=dist_map[module],
                path_nodes=nodes_path,
                path_switches=switches_path,
            )

        return result, loads

    def assign_one_module(
        self,
        module: str,
        *,
        batteries: Optional[Iterable[str]] = None,
        existing_assignments: Optional[Dict[str, str]] = None,
        weighted: bool = False,
        respect_switch_state: bool = True,
    ) -> Tuple[AssignmentResult, Dict[str, int]]:
        results, loads = self.dynamic_load_balanced_assignment(
            [module],
            batteries=batteries,
            existing_assignments=existing_assignments,
            weighted=weighted,
            respect_switch_state=respect_switch_state,
        )
        return results[module], loads

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


def print_distance_table(table: Dict[str, Dict[str, float]]) -> None:
    print("Distance table:")
    for module, row in sorted(table.items()):
        pretty = {b: ("inf" if math.isinf(d) else d) for b, d in sorted(row.items())}
        print(f"  {module}: {pretty}")


def print_assignments(assignments: Dict[str, AssignmentResult]) -> None:
    print("Assignments:")
    for module, info in sorted(assignments.items()):
        distance_str = "inf" if math.isinf(info.distance) else info.distance
        print(f"  {module}:")
        print(f"    battery      = {info.battery}")
        print(f"    distance     = {distance_str}")
        print(f"    path_nodes   = {info.path_nodes}")
        print(f"    path_switches= {info.path_switches}")
        if info.load_before is not None or info.load_after is not None:
            print(f"    load_before  = {info.load_before}")
            print(f"    load_after   = {info.load_after}")


def print_loads(loads: Dict[str, int]) -> None:
    print("Battery loads:", dict(sorted(loads.items())))


def print_compact_assignments(
    title: str,
    assignments: Dict[str, AssignmentResult],
    *,
    show_loads: bool = False,
) -> None:
    print(title)
    for module, info in sorted(assignments.items()):
        distance_str = "inf" if math.isinf(info.distance) else info.distance
        suffix = ""
        if show_loads:
            suffix = f", load {info.load_before}->{info.load_after}"
        print(f"  {module}: {info.battery} (d={distance_str}{suffix})")


def print_changed_modules(
    baseline: Dict[str, AssignmentResult],
    balanced: Dict[str, AssignmentResult],
) -> None:
    changed = [
        module for module in sorted(baseline)
        if baseline[module].battery != balanced[module].battery
    ]
    print("Modules that changed because of balanced tie-breaking:")
    if not changed:
        print("  None")
        return
    for module in changed:
        print(f"  {module}: {baseline[module].battery} -> {balanced[module].battery}")


def build_demo3_graph() -> ModularRobotGraph:
    modules = {
        "B1":  {"type": "battery", "pos": (0, 3)},
        "M01": {"type": "action",  "pos": (1, 3)},
        "M02": {"type": "action",  "pos": (2, 3)},
        "M03": {"type": "action",  "pos": (3, 3)},
        "B2":  {"type": "battery", "pos": (4, 3)},
        "M04": {"type": "action",  "pos": (0, 2)},
        "M05": {"type": "action",  "pos": (1, 2)},
        "M06": {"type": "action",  "pos": (2, 2)},
        "M07": {"type": "action",  "pos": (3, 2)},
        "M08": {"type": "action",  "pos": (4, 2)},
        "M09": {"type": "action",  "pos": (0, 1)},
        "M10": {"type": "action",  "pos": (1, 1)},
        "M11": {"type": "action",  "pos": (3, 1)},
        "M12": {"type": "action",  "pos": (4, 1)},
        "B3":  {"type": "battery", "pos": (0, 0)},
        "M13": {"type": "action",  "pos": (1, 0)},
        "M14": {"type": "action",  "pos": (2, 0)},
        "M15": {"type": "action",  "pos": (3, 0)},
        "B4":  {"type": "battery", "pos": (4, 0)},
    }
    return ModularRobotGraph.from_grid_layout(modules)


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


def compare_modes(
    g: ModularRobotGraph,
    *,
    title: str,
    layout_lines: List[str],
    batteries: Optional[Iterable[str]] = None,
    modules: Optional[Iterable[str]] = None,
    respect_switch_state: bool,
) -> None:
    module_list = list(modules) if modules is not None else g.get_actions()

    baseline = g.nearest_battery_assignment(
        batteries=batteries,
        modules=module_list,
        weighted=False,
        respect_switch_state=respect_switch_state,
    )
    balanced, loads = g.rebalanced_nearest_battery_assignment(
        batteries=batteries,
        modules=module_list,
        weighted=False,
        respect_switch_state=respect_switch_state,
    )

    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
    print("Layout:")
    for line in layout_lines:
        print(f"  {line}")
    print_compact_assignments("Without balanced mode (original duo-mode behavior):", baseline)
    print_compact_assignments("With balanced mode (post-processed rebalance):", balanced)
    print_changed_modules(baseline, balanced)
    print_loads(loads)


def demo_compare_large_quad() -> None:
    layout_lines = [
        "B1  M01 M02 M03 B2",
        "M04 M05 M06 M07 M08",
        "M09 M10 .   M11 M12",
        "B3  M13 M14 M15 B4",
    ]

    compare_modes(
        build_demo3_graph(),
        title="DEMO 3A: Large 4-battery / 15-action topology on full topology",
        layout_lines=layout_lines,
        respect_switch_state=False,
    )

    g_runtime = build_demo3_graph()
    g_runtime.set_switch_state("S_B1_M01", False)
    g_runtime.set_switch_state("S_M06_M07", False)
    g_runtime.set_switch_state("S_M13_M10", False)
    compare_modes(
        g_runtime,
        title="DEMO 3B: Same topology in runtime mode after outages",
        layout_lines=layout_lines,
        batteries=["B2", "B3", "B4"],
        respect_switch_state=True,
    )


def demo_compare_articulated_arm() -> None:
    layout_lines = [
        "B1  M01 M02",
        "        M03",
        "M19 M18 M04 B2 M05",
        "B5     M17     M06 M20",
        "M15 M14 M16 M08 M07 B3",
        "    M13     M09",
        "B4  M12 M11 M10",
    ]

    compare_modes(
        build_demo4_graph(),
        title="DEMO 4A: 5-battery articulated arm on full topology",
        layout_lines=layout_lines,
        respect_switch_state=False,
    )

    g_runtime = build_demo4_graph()
    g_runtime.set_switch_state("S_M04_B2", False)
    g_runtime.set_switch_state("S_M16_M17", False)
    g_runtime.set_switch_state("S_M09_M08", False)
    g_runtime.set_switch_state("S_M06_M20", False)
    compare_modes(
        g_runtime,
        title="DEMO 4B: Articulated arm in runtime mode after battery/joint faults",
        layout_lines=layout_lines,
        batteries=["B1", "B3", "B4", "B5"],
        respect_switch_state=True,
    )


if __name__ == "__main__":
    demo_compare_large_quad()
    demo_compare_articulated_arm()
