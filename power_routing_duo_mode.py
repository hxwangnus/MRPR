from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Iterable, Set, Any
import math
import heapq


# =========================
# Data classes
# =========================

@dataclass(frozen=True)
class Edge:
    """
    One undirected physical link between two modules, governed by one switch.
    """
    other: str
    switch_id: str
    cost: float = 1.0


@dataclass
class AssignmentResult:
    """
    Final routing result for one action module.
    """
    module: str
    battery: Optional[str]
    distance: float
    path_nodes: Optional[List[str]]
    path_switches: Optional[List[str]]


# =========================
# Core graph class
# =========================

class ModularRobotGraph:
    """
    Graph model for modular robot power routing.

    Node:
        - battery module
        - action module
        - other types if needed later

    Edge:
        - physical connection between two modules
        - controlled by a switch
        - can be traversable or blocked depending on switch state

    Key features:
        - manual graph construction
        - optional grid-layout auto construction
        - single-source shortest path
        - multi-source shortest path
        - nearest battery assignment
        - recommended switch plan
    """

    def __init__(self) -> None:
        self.node_type: Dict[str, str] = {}                 # node_id -> "battery" / "action" / ...
        self.node_pos: Dict[str, Tuple[int, int]] = {}      # optional geometric position
        self.adj: Dict[str, List[Edge]] = defaultdict(list)

        self.switch_open: Dict[str, bool] = {}              # switch_id -> current state
        self.switch_endpoints: Dict[str, Tuple[str, str]] = {}   # switch_id -> normalized endpoints
        self.link_to_switch: Dict[Tuple[str, str], str] = {}     # normalized endpoints -> switch_id

    # -------------------------
    # Internal helpers
    # -------------------------

    @staticmethod
    def _norm_pair(u: str, v: str) -> Tuple[str, str]:
        if u == v:
            raise ValueError("Self-loop is not allowed.")
        return tuple(sorted((u, v)))

    def _edge_traversable(self, edge: Edge, respect_switch_state: bool) -> bool:
        """
        If respect_switch_state=False:
            treat all physical links as available (planning mode)
        If respect_switch_state=True:
            only open switches are traversable (runtime mode)
        """
        return (not respect_switch_state) or self.switch_open.get(edge.switch_id, False)

    # -------------------------
    # Graph construction
    # -------------------------

    def add_node(self, node_id: str, node_type: str, pos: Optional[Tuple[int, int]] = None) -> None:
        if node_id in self.node_type and self.node_type[node_id] != node_type:
            raise ValueError(
                f"Node {node_id} already exists with type {self.node_type[node_id]}, "
                f"cannot change to {node_type}."
            )
        self.node_type[node_id] = node_type
        if pos is not None:
            self.node_pos[node_id] = pos
        _ = self.adj[node_id]  # ensure key exists

    def add_switch_edge(
        self,
        u: str,
        v: str,
        switch_id: str,
        open_: bool = True,
        cost: float = 1.0,
    ) -> None:
        """
        Add one physical undirected connection between u and v.
        """
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

    # -------------------------
    # Node query helpers
    # -------------------------

    def get_batteries(self) -> List[str]:
        return sorted(n for n, t in self.node_type.items() if t == "battery")

    def get_actions(self) -> List[str]:
        return sorted(n for n, t in self.node_type.items() if t == "action")

    # -------------------------
    # Single-source shortest paths
    # -------------------------

    def shortest_paths_from(
        self,
        source: str,
        *,
        weighted: bool = False,
        respect_switch_state: bool = True,
    ) -> Tuple[Dict[str, float], Dict[str, Optional[Tuple[str, str]]]]:
        """
        Single-source shortest paths.

        Returns:
            dist[node]   = shortest distance from source
            parent[node] = (prev_node, switch_id) or None for source

        If weighted=False:
            BFS, each traversed switch counts as 1

        If weighted=True:
            Dijkstra, edge cost = switch loss / custom cost
        """
        if source not in self.node_type:
            raise KeyError(f"Unknown source node: {source}")

        # ---- Unweighted BFS ----
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

        # ---- Weighted Dijkstra ----
        dist: Dict[str, float] = {source: 0.0}
        parent: Dict[str, Optional[Tuple[str, str]]] = {source: None}
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

    # -------------------------
    # Multi-source shortest paths
    # -------------------------

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
        """
        Multi-source shortest paths.

        Returns:
            dist[node]   = shortest distance from nearest source
            parent[node] = (prev_node, switch_id) or None if node itself is a source
            owner[node]  = which source owns this shortest path

        This is the key function for:
            "Which battery should power each module?"
        """
        src_list = sorted(set(sources))
        if not src_list:
            raise ValueError("sources must be non-empty")

        for s in src_list:
            if s not in self.node_type:
                raise KeyError(f"Unknown source node: {s}")

        # ---- Unweighted multi-source BFS ----
        if not weighted:
            dist: Dict[str, float] = {}
            parent: Dict[str, Optional[Tuple[str, str]]] = {}
            owner: Dict[str, str] = {}
            q = deque()

            # Initialize all sources at distance 0
            # Sorted order gives deterministic tie-breaking
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

        # ---- Weighted multi-source Dijkstra ----
        dist: Dict[str, float] = {}
        parent: Dict[str, Optional[Tuple[str, str]]] = {}
        owner: Dict[str, str] = {}
        pq: List[Tuple[float, str, str]] = []   # (distance, owner_id, node)

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

                # Tie-breaking:
                #   1) smaller distance wins
                #   2) if equal distance, lexicographically smaller battery ID wins
                if (nd < old_d) or (math.isclose(nd, old_d) and owner_u < old_owner):
                    dist[v] = nd
                    parent[v] = (u, e.switch_id)
                    owner[v] = owner_u
                    heapq.heappush(pq, (nd, owner_u, v))

        return dist, parent, owner

    # -------------------------
    # Path reconstruction
    # -------------------------

    @staticmethod
    def reconstruct_path(
        parent: Dict[str, Optional[Tuple[str, str]]],
        target: str,
    ) -> Optional[Tuple[List[str], List[str]]]:
        """
        Recover:
            nodes_path    = [source, ..., target]
            switches_path = [switch along each step]
        """
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

    # -------------------------
    # Distance table
    # -------------------------

    def all_battery_distance_table(
        self,
        *,
        batteries: Optional[Iterable[str]] = None,
        modules: Optional[Iterable[str]] = None,
        weighted: bool = False,
        respect_switch_state: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Returns:
            table[module][battery] = shortest distance
        """
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

    # -------------------------
    # Final assignment
    # -------------------------

    def nearest_battery_assignment(
        self,
        *,
        batteries: Optional[Iterable[str]] = None,
        modules: Optional[Iterable[str]] = None,
        weighted: bool = False,
        respect_switch_state: bool = True,
    ) -> Dict[str, AssignmentResult]:
        """
        Final assignment:
            for each action module, choose the nearest battery

        Returns:
            result[module] = AssignmentResult(...)
        """
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

    # -------------------------
    # Switch plan recommendation
    # -------------------------

    def recommend_switch_plan(
        self,
        *,
        active_modules: Optional[Iterable[str]] = None,
        batteries: Optional[Iterable[str]] = None,
        weighted: bool = False,
        respect_switch_state: bool = False,
    ) -> Dict[str, Any]:
        """
        Recommend a switch-opening plan for active modules.

        Default:
            respect_switch_state=False
        means:
            we are planning over the physical topology, not limited by current switch states.

        Returns:
            {
                "assignments": ...,
                "required_open_switches": ...,
                "recommended_closed_switches": ...,
                "unreachable_modules": ...
            }

        Interpretation:
            - required_open_switches:
                switches needed by the chosen shortest-path forest
            - recommended_closed_switches:
                all other switches can stay closed
              (this also helps avoid unnecessary cross-links between battery domains)
        """
        modules = list(active_modules) if active_modules is not None else self.get_actions()

        assignments = self.nearest_battery_assignment(
            batteries=batteries,
            modules=modules,
            weighted=weighted,
            respect_switch_state=respect_switch_state,
        )

        required_open_switches: Set[str] = set()
        unreachable_modules: List[str] = []

        for m, info in assignments.items():
            if info.battery is None:
                unreachable_modules.append(m)
                continue
            required_open_switches.update(info.path_switches or [])

        all_switches = set(self.switch_open.keys())

        return {
            "assignments": assignments,
            "required_open_switches": sorted(required_open_switches),
            "recommended_closed_switches": sorted(all_switches - required_open_switches),
            "unreachable_modules": sorted(unreachable_modules),
        }

    # -------------------------
    # Construction from manual spec
    # -------------------------

    @classmethod
    def from_manual_spec(
        cls,
        nodes: Iterable[Dict[str, Any]],
        switches: Iterable[Dict[str, Any]],
    ) -> "ModularRobotGraph":
        """
        Example:

        nodes = [
            {"id": "B1", "type": "battery", "pos": (0, 2)},
            {"id": "M1", "type": "action",  "pos": (0, 1)},
        ]

        switches = [
            {"u": "B1", "v": "M1", "switch_id": "S_B1_M1", "open": True, "cost": 1.0},
        ]
        """
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

    # -------------------------
    # Construction from grid layout
    # -------------------------

    @classmethod
    def from_grid_layout(
        cls,
        modules: Dict[str, Dict[str, Any]],
        *,
        default_open: bool = True,
        default_cost: float = 1.0,
        switch_prefix: str = "S",
    ) -> "ModularRobotGraph":
        """
        Automatically build graph from grid-like geometry.

        Input format example:
        modules = {
            "B1": {"type": "battery", "pos": (0, 3)},
            "M1": {"type": "action",  "pos": (0, 2)},
            "M2": {"type": "action",  "pos": (0, 1)},
            "M3": {"type": "action",  "pos": (1, 1)},
            "M4": {"type": "action",  "pos": (1, 0)},
            "B2": {"type": "battery", "pos": (2, 1)},
        }

        Rule:
            if two modules are 4-neighbor adjacent on the grid,
            automatically create one switch edge between them.

        This is very useful when your robot layout is "Tetris-like".
        """
        g = cls()
        pos_to_node: Dict[Tuple[int, int], str] = {}

        # Add nodes
        for node_id, info in modules.items():
            pos = tuple(info["pos"])
            if pos in pos_to_node:
                raise ValueError(
                    f"Duplicate grid position {pos}: {node_id} conflicts with {pos_to_node[pos]}"
                )
            pos_to_node[pos] = node_id
            g.add_node(node_id, info["type"], pos=pos)

        # Add edges by 4-neighbor adjacency
        # Only check right and up to avoid duplicate addition
        directions = [(1, 0), (0, 1)]

        for node_id, info in modules.items():
            x, y = info["pos"]
            for dx, dy in directions:
                nb_pos = (x + dx, y + dy)
                if nb_pos in pos_to_node:
                    other = pos_to_node[nb_pos]
                    switch_id = f"{switch_prefix}_{node_id}_{other}"
                    g.add_switch_edge(
                        u=node_id,
                        v=other,
                        switch_id=switch_id,
                        open_=default_open,
                        cost=default_cost,
                    )

        return g


# =========================
# Pretty print helpers
# =========================

def print_distance_table(table: Dict[str, Dict[str, float]]) -> None:
    print("Distance table:")
    for module, row in sorted(table.items()):
        pretty = {
            b: ("inf" if math.isinf(d) else d)
            for b, d in sorted(row.items())
        }
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


# =========================
# Demo 1: manual example
# =========================

def demo_manual_l_shape() -> None:
    """
    Example topology:
        B1 -- M1 -- M2 -- M3 -- B2
                   |
                  (can be extended later)

    This demo shows:
        1) theoretical planning on full physical topology
        2) runtime routing under current switch states
    """
    nodes = [
        {"id": "B1", "type": "battery", "pos": (0, 3)},
        {"id": "M1", "type": "action",  "pos": (0, 2)},
        {"id": "M2", "type": "action",  "pos": (0, 1)},
        {"id": "M3", "type": "action",  "pos": (1, 1)},
        {"id": "B2", "type": "battery", "pos": (2, 1)},
    ]

    switches = [
        {"u": "B1", "v": "M1", "switch_id": "S_B1_M1", "open": True, "cost": 1.0},
        {"u": "M1", "v": "M2", "switch_id": "S_M1_M2", "open": True, "cost": 1.0},
        {"u": "M2", "v": "M3", "switch_id": "S_M2_M3", "open": True, "cost": 1.0},
        {"u": "M3", "v": "B2", "switch_id": "S_M3_B2", "open": True, "cost": 1.0},
    ]

    g = ModularRobotGraph.from_manual_spec(nodes, switches)

    print("=" * 70)
    print("DEMO 1A: Planning mode (ignore current switch states)")
    print("=" * 70)

    table_plan = g.all_battery_distance_table(
        weighted=False,
        respect_switch_state=False,   # ignore current states, use full topology
    )
    print_distance_table(table_plan)

    assignments_plan = g.nearest_battery_assignment(
        weighted=False,
        respect_switch_state=False,
    )
    print_assignments(assignments_plan)

    plan = g.recommend_switch_plan(
        weighted=False,
        respect_switch_state=False,
    )
    print("Recommended switch plan:")
    print("  required_open_switches   =", plan["required_open_switches"])
    print("  recommended_closed_switches =", plan["recommended_closed_switches"])
    print("  unreachable_modules      =", plan["unreachable_modules"])

    print()
    print("=" * 70)
    print("DEMO 1B: Runtime mode")
    print("Case: close M2-M3, others remain open")
    print("=" * 70)

    g.set_switch_state("S_M2_M3", False)

    table_runtime = g.all_battery_distance_table(
        weighted=False,
        respect_switch_state=True,    # now respect actual switch states
    )
    print_distance_table(table_runtime)

    assignments_runtime = g.nearest_battery_assignment(
        weighted=False,
        respect_switch_state=True,
    )
    print_assignments(assignments_runtime)


# =========================
# Demo 2: build from grid layout
# =========================

def demo_grid_layout() -> None:
    """
    Example geometry exactly matching a 'Tetris-like' layout.

    Layout:
        B1
        M1
        M2 M3 B2
           M4
    """
    modules = {
        "B1": {"type": "battery", "pos": (0, 3)},
        "M1": {"type": "action",  "pos": (0, 2)},
        "M2": {"type": "action",  "pos": (0, 1)},
        "M3": {"type": "action",  "pos": (1, 1)},
        "B2": {"type": "battery", "pos": (2, 1)},
        "M4": {"type": "action",  "pos": (1, 0)},
    }

    g = ModularRobotGraph.from_grid_layout(modules)

    print()
    print("=" * 70)
    print("DEMO 2: Auto-build from grid layout")
    print("=" * 70)

    table = g.all_battery_distance_table(
        weighted=False,
        respect_switch_state=False,
    )
    print_distance_table(table)

    assignments = g.nearest_battery_assignment(
        weighted=False,
        respect_switch_state=False,
    )
    print_assignments(assignments)

    plan = g.recommend_switch_plan(
        weighted=False,
        respect_switch_state=False,
    )
    print("Recommended switch plan:")
    print("  required_open_switches   =", plan["required_open_switches"])
    print("  recommended_closed_switches =", plan["recommended_closed_switches"])
    print("  unreachable_modules      =", plan["unreachable_modules"])


# =========================
# Demo 3: larger 4-battery / 15-action example
# =========================

def demo_large_quad_battery_layout() -> None:
    """
    A larger example with 4 batteries and 15 action modules.

    Grid layout (x increases to the right, y increases upward):

        B1  M01 M02 M03 B2
        M04 M05 M06 M07 M08
        M09 M10 .   M11 M12
        B3  M13 M14 M15 B4

    Notes:
        - Batteries sit at the four corners.
        - One center cell is intentionally left empty to create a slightly
          asymmetric routing problem.
        - Edges are auto-created between 4-neighbor adjacent modules.
    """
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

    g = ModularRobotGraph.from_grid_layout(modules)

    print()
    print("=" * 70)
    print("DEMO 3A: Large 4-battery / 15-action planning example")
    print("=" * 70)
    print("Layout:")
    print("  B1  M01 M02 M03 B2")
    print("  M04 M05 M06 M07 M08")
    print("  M09 M10 .   M11 M12")
    print("  B3  M13 M14 M15 B4")

    table = g.all_battery_distance_table(
        weighted=False,
        respect_switch_state=False,
    )
    print_distance_table(table)

    assignments = g.nearest_battery_assignment(
        weighted=False,
        respect_switch_state=False,
    )
    print_assignments(assignments)

    plan = g.recommend_switch_plan(
        weighted=False,
        respect_switch_state=False,
    )
    print("Recommended switch plan:")
    print("  required_open_switches   =", plan["required_open_switches"])
    print("  recommended_closed_switches =", plan["recommended_closed_switches"])
    print("  unreachable_modules      =", plan["unreachable_modules"])

    print()
    print("=" * 70)
    print("DEMO 3B: Runtime mode with B1 unavailable and two links closed")
    print("=" * 70)
    print("Scenario:")
    print("  - B1 is treated as unavailable")
    print("  - close S_B1_M01 to disconnect B1 physically")
    print("  - close S_M06_M07 and S_M13_M10 to create local detours")

    g.set_switch_state("S_B1_M01", False)
    g.set_switch_state("S_M06_M07", False)
    g.set_switch_state("S_M13_M10", False)

    runtime_batteries = ["B2", "B3", "B4"]

    runtime_table = g.all_battery_distance_table(
        batteries=runtime_batteries,
        weighted=False,
        respect_switch_state=True,
    )
    print_distance_table(runtime_table)

    runtime_assignments = g.nearest_battery_assignment(
        batteries=runtime_batteries,
        weighted=False,
        respect_switch_state=True,
    )
    print_assignments(runtime_assignments)

    runtime_plan = g.recommend_switch_plan(
        active_modules=["M01", "M02", "M03", "M06", "M07", "M10", "M11", "M14", "M15"],
        batteries=runtime_batteries,
        weighted=False,
        respect_switch_state=True,
    )
    print("Runtime active-module switch summary:")
    print("  required_open_switches   =", runtime_plan["required_open_switches"])
    print("  recommended_closed_switches =", runtime_plan["recommended_closed_switches"])
    print("  unreachable_modules      =", runtime_plan["unreachable_modules"])


# =========================
# Demo 4: articulated arm-like layout
# =========================

def demo_articulated_arm_layout() -> None:
    """
    A more irregular example with 5 batteries and 20 action modules.

    The shape is intentionally "arm-like": multiple L-shaped bends,
    branch points, and a few local cross-links that allow rerouting.

    Grid layout:

        B1  M01 M02
                M03
        M19 M18 M04 B2 M05
        B5     M17     M06 M20
        M15 M14 M16 M08 M07 B3
            M13     M09
        B4  M12 M11 M10
    """
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

    g = ModularRobotGraph.from_grid_layout(modules)

    print()
    print("=" * 70)
    print("DEMO 4A: 5-battery articulated arm planning example")
    print("=" * 70)
    print("Layout:")
    print("  B1  M01 M02")
    print("          M03")
    print("  M19 M18 M04 B2 M05")
    print("  B5     M17     M06 M20")
    print("  M15 M14 M16 M08 M07 B3")
    print("      M13     M09")
    print("  B4  M12 M11 M10")

    table = g.all_battery_distance_table(
        weighted=False,
        respect_switch_state=False,
    )
    print_distance_table(table)

    assignments = g.nearest_battery_assignment(
        weighted=False,
        respect_switch_state=False,
    )
    print_assignments(assignments)

    plan = g.recommend_switch_plan(
        weighted=False,
        respect_switch_state=False,
    )
    print("Recommended switch plan:")
    print("  required_open_switches   =", plan["required_open_switches"])
    print("  recommended_closed_switches =", plan["recommended_closed_switches"])
    print("  unreachable_modules      =", plan["unreachable_modules"])

    print()
    print("=" * 70)
    print("DEMO 4B: Runtime mode with joint failures and one dead battery")
    print("=" * 70)
    print("Scenario:")
    print("  - B2 is unavailable and excluded from the active battery list")
    print("  - close S_M04_B2 to isolate the dead battery physically")
    print("  - close S_M16_M17 and S_M09_M08 to simulate elbow joint faults")
    print("  - close S_M06_M20 to cut one distal branch")

    g.set_switch_state("S_M04_B2", False)
    g.set_switch_state("S_M16_M17", False)
    g.set_switch_state("S_M09_M08", False)
    g.set_switch_state("S_M06_M20", False)

    runtime_batteries = ["B1", "B3", "B4", "B5"]

    runtime_table = g.all_battery_distance_table(
        batteries=runtime_batteries,
        weighted=False,
        respect_switch_state=True,
    )
    print_distance_table(runtime_table)

    runtime_assignments = g.nearest_battery_assignment(
        batteries=runtime_batteries,
        weighted=False,
        respect_switch_state=True,
    )
    print_assignments(runtime_assignments)

    runtime_plan = g.recommend_switch_plan(
        active_modules=["M02", "M03", "M06", "M07", "M08", "M10", "M13", "M16", "M17", "M20"],
        batteries=runtime_batteries,
        weighted=False,
        respect_switch_state=True,
    )
    print("Runtime active-module switch summary:")
    print("  required_open_switches   =", runtime_plan["required_open_switches"])
    print("  recommended_closed_switches =", runtime_plan["recommended_closed_switches"])
    print("  unreachable_modules      =", runtime_plan["unreachable_modules"])


# =========================
# Main
# =========================

if __name__ == "__main__":
    demo_manual_l_shape()
    demo_grid_layout()
    demo_large_quad_battery_layout()
    demo_articulated_arm_layout()
