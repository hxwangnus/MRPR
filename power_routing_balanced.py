from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Iterable, Set, Any, Mapping, Union
import json
import math
import heapq
import re


LOCAL_DIRECTIONS: Tuple[str, ...] = ("top", "right", "bottom", "left")
ROLE_NAMES: Dict[int, str] = {
    1: "battery",
    2: "motor",
    3: "cpu",
    4: "navigation",
    5: "empty",
    **{role: f"buffer{role - 5}" for role in range(6, 17)},
}
ROLE_DEMAND_WEIGHTS: Dict[int, float] = {
    1: 0.0,
    2: 5.0,
    3: 3.0,
    4: 4.0,
    5: 1.0,
    **{role: 2.0 for role in range(6, 17)},
}
BATTERY_ROLES = frozenset({1})
LOAD_ROLES = frozenset(range(2, 17))
_GRID_CELL_PATTERN = re.compile(
    r"""
    ^\s*
    (?P<module>[^()\[\],]+?)\s*
    \(\s*
    (?P<rotation>-?\d+)\s*d
    (?:\s*,\s*\[\s*
        (?P<role>\d+)\s*,\s*
        (?P<digital_io>\d+)\s*,\s*
        (?P<unused>-?\d+)\s*
    \])?
    \s*\)\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)
ParentSwitches = Union[str, Tuple[str, ...]]
ParentMap = Dict[str, Optional[Tuple[str, ParentSwitches]]]


@dataclass(frozen=True)
class Edge:
    """One undirected physical link governed by one or two physical switches."""
    other: str
    switch_id: str
    cost: float = 1.0
    switch_ids: Tuple[str, ...] = ()

    @property
    def required_switches(self) -> Tuple[str, ...]:
        """All switches that must conduct before this edge can be traversed."""
        return self.switch_ids or (self.switch_id,)

    @property
    def parent_switches(self) -> ParentSwitches:
        """Preserve the legacy scalar parent value for one-switch edges."""
        switches = self.required_switches
        return switches[0] if len(switches) == 1 else switches


@dataclass
class ModuleSwitchInfo:
    """Physical orientation and connection metadata for one module-local switch."""
    module: str
    local_direction: str
    world_direction: str
    neighbor: Optional[str] = None


@dataclass(frozen=True)
class ModuleMetadata:
    """Role and the two currently opaque values encoded in a layout cell."""

    role: int
    digital_io: int
    unused: int

    @property
    def role_name(self) -> str:
        return ROLE_NAMES[self.role]

    @property
    def digital_io_bits(self) -> str:
        return f"{self.digital_io:08b}"

    @property
    def is_battery(self) -> bool:
        return self.role in BATTERY_ROLES

    @property
    def is_load(self) -> bool:
        return self.role in LOAD_ROLES

    @property
    def demand_weight(self) -> float:
        return ROLE_DEMAND_WEIGHTS[self.role]


@dataclass
class AssignmentResult:
    """
    Final routing result for one action module.

    `load_before` and `load_after` are used by the balanced assignment logic.
    For plain nearest-battery assignment they remain None.

    For rotated grids, `path_switches` contains two consecutive switch IDs per
    traversed module adjacency: source-side first, then destination-side.
    """
    module: str
    battery: Optional[str]
    distance: float
    path_nodes: Optional[List[str]]
    path_switches: Optional[List[str]]
    load_before: Optional[int] = None
    load_after: Optional[int] = None

    @property
    def switch_count(self) -> Optional[int]:
        """Number of physical endpoint switches traversed by this route."""
        return None if self.path_switches is None else len(self.path_switches)


@dataclass(frozen=True)
class DemandRouteCandidate:
    """One module-to-battery route considered by demand-aware optimization."""

    module: str
    battery: str
    demand_weight: float
    distance: float
    path_nodes: Tuple[str, ...]
    path_switches: Tuple[str, ...]
    effective_drain: float
    weighted_route_cost: float


class ModularRobotGraph:
    """
    Graph model for modular robot power routing.

    This file keeps the same capabilities as `power_routing_duo_mode.py`
    and adds load-aware tie-breaking for dynamic assignment. Rotated JSON grids
    additionally model the local switch at both ends of every adjacency.
    """

    def __init__(self) -> None:
        self.node_type: Dict[str, str] = {}
        self.node_pos: Dict[str, Tuple[int, int]] = {}
        self.adj: Dict[str, List[Edge]] = defaultdict(list)

        # True means electrically conducting (the physical switch is closed).
        # `switch_open` is retained as a compatibility alias for the original API,
        # whose `open_=True` parameter also meant "traversable".
        self.switch_closed: Dict[str, bool] = {}
        self.switch_open = self.switch_closed
        self.switch_endpoints: Dict[str, Tuple[str, str]] = {}
        self.link_to_switch: Dict[Tuple[str, str], str] = {}
        self.link_to_switches: Dict[Tuple[str, str], Tuple[str, ...]] = {}
        self.link_endpoint_order: Dict[Tuple[str, str], Tuple[str, str]] = {}

        # Populated by from_grid_json(). Every rotated module has four switches,
        # including switches on sides with no neighboring module.
        self.module_rotation: Dict[str, int] = {}
        self.module_metadata: Dict[str, ModuleMetadata] = {}
        self.module_switches: Dict[str, Dict[str, str]] = {}
        self.switch_info: Dict[str, ModuleSwitchInfo] = {}
        self.grid_rows: Optional[int] = None
        self.grid_cols: Optional[int] = None
        self.grid_cells: Optional[List[List[Optional[str]]]] = None

    @staticmethod
    def _norm_pair(u: str, v: str) -> Tuple[str, str]:
        if u == v:
            raise ValueError("Self-loop is not allowed.")
        return tuple(sorted((u, v)))

    def _edge_traversable(self, edge: Edge, respect_switch_state: bool) -> bool:
        return (not respect_switch_state) or all(
            self.switch_closed.get(switch_id, False)
            for switch_id in edge.required_switches
        )

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
        """Add a legacy link controlled by one traversable/closed switch."""
        self._add_edge_with_switches(
            u,
            v,
            switch_ids=(switch_id,),
            switch_states=(open_,),
            cost=cost,
        )

    def add_dual_switch_edge(
        self,
        u: str,
        v: str,
        u_switch_id: str,
        v_switch_id: str,
        *,
        closed_: bool = True,
        cost: float = 1.0,
    ) -> None:
        """
        Add a physical adjacency controlled by one local switch at each end.

        Traversal is possible only when both switches are closed. In a path,
        the two IDs are returned in traversal order: source-side first.
        """
        self._add_edge_with_switches(
            u,
            v,
            switch_ids=(u_switch_id, v_switch_id),
            switch_states=(closed_, closed_),
            cost=cost,
        )

    def _add_edge_with_switches(
        self,
        u: str,
        v: str,
        *,
        switch_ids: Tuple[str, ...],
        switch_states: Tuple[bool, ...],
        cost: float,
    ) -> None:
        if u not in self.node_type or v not in self.node_type:
            raise KeyError("Both endpoints must be added first via add_node().")
        if cost < 0:
            raise ValueError("Edge cost must be nonnegative.")
        if not switch_ids or len(switch_ids) != len(switch_states):
            raise ValueError("Each edge switch must have one corresponding state.")
        if len(set(switch_ids)) != len(switch_ids):
            raise ValueError("An edge cannot use the same physical switch twice.")

        norm = self._norm_pair(u, v)

        if norm in self.link_to_switches:
            prev_switches = self.link_to_switches[norm]
            raise ValueError(
                f"Physical link {norm} already has switches {prev_switches}. "
                f"Duplicate edge is not allowed in this simplified model."
            )

        for switch_id in switch_ids:
            if switch_id in self.switch_endpoints:
                old = self.switch_endpoints[switch_id]
                if old != norm:
                    raise ValueError(
                        f"switch_id={switch_id} already belongs to endpoints {old}, not {norm}."
                    )
                raise ValueError(f"Duplicate switch_id detected: {switch_id}")

        if len(switch_ids) == 2:
            for switch_id, expected_module in zip(switch_ids, (u, v)):
                info = self.switch_info.get(switch_id)
                if info is not None and info.module != expected_module:
                    raise ValueError(
                        f"Switch {switch_id!r} belongs to {info.module!r}, "
                        f"not endpoint {expected_module!r}."
                    )

        for switch_id, state in zip(switch_ids, switch_states):
            # A rotated-grid builder pre-registers all four local switches. Do
            # not overwrite such a switch's configured default state here.
            self.switch_closed.setdefault(switch_id, state)
            self.switch_endpoints[switch_id] = norm

        self.link_to_switches[norm] = switch_ids
        self.link_endpoint_order[norm] = (u, v)
        # Compatibility view: legacy callers see the first switch. New code
        # should use get_link_switches() / link_to_switches for dual links.
        self.link_to_switch[norm] = switch_ids[0]

        self.adj[u].append(Edge(
            other=v,
            switch_id=switch_ids[0],
            cost=cost,
            switch_ids=switch_ids,
        ))
        reverse_switch_ids = tuple(reversed(switch_ids))
        self.adj[v].append(Edge(
            other=u,
            switch_id=reverse_switch_ids[0],
            cost=cost,
            switch_ids=reverse_switch_ids,
        ))

        for switch_id in switch_ids:
            info = self.switch_info.get(switch_id)
            if info is not None:
                info.neighbor = v if info.module == u else u

    def set_switch_state(self, switch_id: str, open_: bool) -> None:
        """Compatibility API: True means conducting (physically closed)."""
        self.set_switch_closed(switch_id, open_)

    def get_switch_state(self, switch_id: str) -> bool:
        """Compatibility API: return whether the switch is conducting."""
        return self.is_switch_closed(switch_id)

    def set_switch_closed(self, switch_id: str, closed: bool) -> None:
        if switch_id not in self.switch_closed:
            raise KeyError(f"Unknown switch_id: {switch_id}")
        self.switch_closed[switch_id] = closed

    def is_switch_closed(self, switch_id: str) -> bool:
        if switch_id not in self.switch_closed:
            raise KeyError(f"Unknown switch_id: {switch_id}")
        return self.switch_closed[switch_id]

    def get_link_switches(self, u: str, v: str) -> Tuple[str, ...]:
        """Return the one or two physical switch IDs controlling a link."""
        norm = self._norm_pair(u, v)
        if norm not in self.link_to_switches:
            raise KeyError(f"Modules {u} and {v} are not physically adjacent.")

        switch_ids = self.link_to_switches[norm]
        endpoint_order = self.link_endpoint_order[norm]
        return switch_ids if endpoint_order == (u, v) else tuple(reversed(switch_ids))

    @staticmethod
    def _world_direction_for_local(local_direction: str, rotation: int) -> str:
        """
        Map a module-local switch to the side it faces in the JSON grid.

        This intentionally follows the supplied grid convention: at 270d the
        module's local top switch faces grid-right, and its local right switch
        faces grid-bottom.
        """
        if local_direction not in LOCAL_DIRECTIONS:
            raise ValueError(
                f"Unknown local direction {local_direction!r}; "
                f"expected one of {LOCAL_DIRECTIONS}."
            )
        if rotation % 90 != 0:
            raise ValueError("Module rotation must be a multiple of 90 degrees.")
        local_index = LOCAL_DIRECTIONS.index(local_direction)
        return LOCAL_DIRECTIONS[(local_index - rotation // 90) % 4]

    def get_local_switch_facing(self, module: str, world_direction: str) -> str:
        """Return the switch ID on `module` that faces one side of the grid."""
        if module not in self.module_rotation or module not in self.module_switches:
            raise KeyError(f"Module {module!r} has no rotated-grid switch metadata.")
        if world_direction not in LOCAL_DIRECTIONS:
            raise ValueError(
                f"Unknown world direction {world_direction!r}; "
                f"expected one of {LOCAL_DIRECTIONS}."
            )

        rotation = self.module_rotation[module]
        world_index = LOCAL_DIRECTIONS.index(world_direction)
        local_direction = LOCAL_DIRECTIONS[(world_index + rotation // 90) % 4]
        return self.module_switches[module][local_direction]

    def get_batteries(self) -> List[str]:
        return sorted(n for n, t in self.node_type.items() if t == "battery")

    def get_actions(self) -> List[str]:
        return sorted(n for n, t in self.node_type.items() if t == "action")

    def get_relays(self) -> List[str]:
        """Return non-load, non-battery modules that may still carry power."""
        return sorted(n for n, t in self.node_type.items() if t == "relay")

    def get_module_metadata(self, module: str) -> ModuleMetadata:
        if module not in self.module_metadata:
            raise KeyError(f"Module {module!r} has no role metadata.")
        return self.module_metadata[module]

    def get_module_demand_weight(self, module: str) -> float:
        """Return the role-derived per-round demand for a role-aware module."""
        return self.get_module_metadata(module).demand_weight

    def shortest_paths_from(
        self,
        source: str,
        *,
        weighted: bool = False,
        respect_switch_state: bool = True,
        blocked_nodes: Optional[Iterable[str]] = None,
    ) -> Tuple[Dict[str, float], ParentMap]:
        if source not in self.node_type:
            raise KeyError(f"Unknown source node: {source}")
        blocked = set(blocked_nodes or ())
        blocked.discard(source)

        if not weighted:
            dist: Dict[str, float] = {source: 0.0}
            parent: ParentMap = {source: None}
            q = deque([source])

            while q:
                u = q.popleft()
                for e in self.adj[u]:
                    if not self._edge_traversable(e, respect_switch_state):
                        continue
                    v = e.other
                    if v in blocked:
                        continue
                    if v not in dist:
                        dist[v] = dist[u] + 1.0
                        parent[v] = (u, e.parent_switches)
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
                if v in blocked:
                    continue
                nd = d_u + e.cost
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    parent[v] = (u, e.parent_switches)
                    heapq.heappush(pq, (nd, v))

        return dist, parent

    def multi_source_shortest_paths(
        self,
        sources: Iterable[str],
        *,
        weighted: bool = False,
        respect_switch_state: bool = True,
        blocked_nodes: Optional[Iterable[str]] = None,
    ) -> Tuple[
        Dict[str, float],
        ParentMap,
        Dict[str, str],
    ]:
        src_list = sorted(set(sources))
        if not src_list:
            raise ValueError("sources must be non-empty")

        for s in src_list:
            if s not in self.node_type:
                raise KeyError(f"Unknown source node: {s}")
        source_set = set(src_list)
        blocked = set(blocked_nodes or ()) - source_set

        if not weighted:
            dist: Dict[str, float] = {}
            parent: ParentMap = {}
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
                    if v in blocked:
                        continue
                    if v not in dist:
                        dist[v] = dist[u] + 1.0
                        parent[v] = (u, e.parent_switches)
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
                if v in blocked or v in source_set:
                    # Selected batteries remain immutable roots, including on
                    # zero-cost weighted links.
                    continue
                nd = d_u + e.cost

                old_d = dist.get(v, math.inf)
                old_owner = owner.get(v, chr(255) * 20)
                if (nd < old_d) or (math.isclose(nd, old_d) and owner_u < old_owner):
                    dist[v] = nd
                    parent[v] = (u, e.parent_switches)
                    owner[v] = owner_u
                    heapq.heappush(pq, (nd, owner_u, v))

        return dist, parent, owner

    @staticmethod
    def reconstruct_path(
        parent: ParentMap,
        target: str,
    ) -> Optional[Tuple[List[str], List[str]]]:
        if target not in parent:
            return None

        nodes: List[str] = []
        switch_groups: List[Tuple[str, ...]] = []
        cur = target

        while True:
            nodes.append(cur)
            p = parent[cur]
            if p is None:
                break
            prev, raw_switches = p
            edge_switches = (
                (raw_switches,)
                if isinstance(raw_switches, str)
                else tuple(raw_switches)
            )
            switch_groups.append(edge_switches)
            cur = prev

        nodes.reverse()
        switch_groups.reverse()
        switches = [
            switch_id
            for edge_switches in switch_groups
            for switch_id in edge_switches
        ]
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
                blocked_nodes=set(self.get_batteries()) - {b},
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
            blocked_nodes=set(self.get_batteries()) - set(batteries),
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
        """
        Build the original nearest-battery switch plan.

        For a rotated grid, every traversed adjacency contributes both local
        endpoint switches. `required_closed_switches` and
        `recommended_open_switches` use physical electrical terminology; the
        original prototype's inverse-named keys remain as aliases.

        This method intentionally retains nearest-mode behavior. Use
        rebalanced_nearest_battery_assignment() for balanced assignments and
        inspect each result's path_switches. Independent balanced paths must be
        electrically validated before their union is applied to hardware.
        """
        modules = list(active_modules) if active_modules is not None else self.get_actions()
        assignments = self.nearest_battery_assignment(
            batteries=batteries,
            modules=modules,
            weighted=weighted,
            respect_switch_state=respect_switch_state,
        )

        required_switches: Set[str] = set()
        unreachable_modules: List[str] = []

        for _, info in assignments.items():
            if info.battery is None:
                unreachable_modules.append(info.module)
                continue
            required_switches.update(info.path_switches or [])

        all_switches = set(self.switch_closed.keys())
        required_closed_switches = sorted(required_switches)
        recommended_open_switches = sorted(all_switches - required_switches)
        return {
            "assignments": assignments,
            # Physical terminology for the rotated-module model.
            "required_closed_switches": required_closed_switches,
            "recommended_open_switches": recommended_open_switches,
            # Backward-compatible aliases from the original prototype, where
            # "open=True" meant enabled/traversable rather than electrically open.
            "required_open_switches": required_closed_switches,
            "recommended_closed_switches": recommended_open_switches,
            "unreachable_modules": sorted(unreachable_modules),
        }

    def battery_conflicts_for_switches(
        self,
        closed_switches: Iterable[str],
    ) -> List[List[str]]:
        """
        Return battery groups that would become electrically connected.

        A physical adjacency conducts only if all of its endpoint switches are
        in `closed_switches`. An empty result means every conducting connected
        component contains at most one known battery module.
        """
        closed = set(closed_switches)
        conducting: Dict[str, Set[str]] = defaultdict(set)
        for endpoints, switch_ids in self.link_to_switches.items():
            if not all(switch_id in closed for switch_id in switch_ids):
                continue
            u, v = endpoints
            conducting[u].add(v)
            conducting[v].add(u)

        batteries = set(self.get_batteries())
        conflicts: List[List[str]] = []
        visited: Set[str] = set()
        for start in sorted(conducting):
            if start in visited:
                continue
            component: Set[str] = set()
            queue = deque([start])
            visited.add(start)
            while queue:
                node = queue.popleft()
                component.add(node)
                for neighbor in conducting[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            component_batteries = sorted(component & batteries)
            if len(component_batteries) > 1:
                conflicts.append(component_batteries)

        return conflicts

    def recommend_balanced_switch_plan(
        self,
        *,
        active_modules: Optional[Iterable[str]] = None,
        batteries: Optional[Iterable[str]] = None,
        weighted: bool = False,
        respect_switch_state: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute balanced assignments and a fail-closed physical switch plan.

        Unlike `recommend_switch_plan()`, this method uses the final balanced
        owners. It validates the union of their independently reconstructed
        shortest paths. If that union would join two battery modules, the
        candidate is reported for diagnosis but `required_closed_switches` is
        empty and `safe_to_apply` is False.
        """
        module_list = (
            list(active_modules)
            if active_modules is not None
            else self.get_actions()
        )
        battery_list = (
            list(batteries)
            if batteries is not None
            else self.get_batteries()
        )
        assignments, loads = self.rebalanced_nearest_battery_assignment(
            batteries=battery_list,
            modules=module_list,
            weighted=weighted,
            respect_switch_state=respect_switch_state,
        )

        candidate_switches: Set[str] = set()
        required_links: Set[Tuple[str, str]] = set()
        unreachable_modules: List[str] = []
        for info in assignments.values():
            if info.battery is None:
                unreachable_modules.append(info.module)
                continue
            candidate_switches.update(info.path_switches or [])
            nodes = info.path_nodes or []
            for u, v in zip(nodes, nodes[1:]):
                required_links.add(self._norm_pair(u, v))

        candidate_required = sorted(candidate_switches)
        conflicts = self.battery_conflicts_for_switches(candidate_required)
        safe_to_apply = not conflicts
        required_closed = candidate_required if safe_to_apply else []
        all_switches = set(self.switch_closed)
        recommended_open = sorted(all_switches - set(required_closed))

        return {
            "mode": "balanced",
            "assignments": assignments,
            "battery_loads": dict(sorted(loads.items())),
            "required_links": sorted(required_links),
            "required_closed_switches": required_closed,
            "recommended_open_switches": recommended_open,
            "candidate_required_closed_switches": candidate_required,
            "safe_to_apply": safe_to_apply,
            "battery_conflicts": conflicts,
            "unreachable_modules": sorted(unreachable_modules),
            # Compatibility aliases retained for callers of the old prototype.
            "required_open_switches": required_closed,
            "recommended_closed_switches": recommended_open,
        }

    def recommend_demand_aware_switch_plan(
        self,
        *,
        active_modules: Optional[Iterable[str]] = None,
        batteries: Optional[Iterable[str]] = None,
        module_weights: Optional[Mapping[str, float]] = None,
        weighted: bool = False,
        respect_switch_state: bool = False,
        route_loss_factor: float = 0.0,
        max_extra_cost: Optional[float] = None,
        require_battery_isolation: bool = True,
        max_search_states: int = 250_000,
    ) -> Dict[str, Any]:
        """
        Optimize first-depletion lifetime for unequal module demands.

        BFS first builds one minimum-hop route from every battery to every
        reachable load by default. The legacy `weighted=True` option remains
        available for callers that explicitly provide unequal edge costs. A
        bounded exact branch-and-bound search then assigns loads globally with
        the lexicographic objective below:

        1. minimize maximum per-battery effective drain
        2. minimize drain spread and variance
        3. minimize total demand-weighted route cost
        4. minimize the number of distinct closed switches

        `route_loss_factor` models proportional transmission loss:

            effective_drain = demand * (1 + route_loss_factor * route_cost)

        It defaults to zero until switch-loss measurements are available. That
        zero preserves the lexicographic demand-first policy; it does not claim
        that physical switch loss is actually zero.
        With equal battery capacities this primary objective maximizes the time
        until the first battery is depleted under the current static model.

        The search is exact when `search_complete` is True, over one
        deterministic minimum-cost path for each module/battery pair. It does
        not yet enumerate alternative equal-cost paths or model nonlinear
        battery behavior, voltage/current limits, or shared-link congestion.
        """
        battery_list = sorted(dict.fromkeys(
            list(batteries) if batteries is not None else self.get_batteries()
        ))
        module_list = list(dict.fromkeys(
            list(active_modules) if active_modules is not None else self.get_actions()
        ))
        if not battery_list:
            raise ValueError("Demand-aware optimization requires at least one battery.")
        for battery in battery_list:
            if battery not in self.node_type:
                raise KeyError(f"Unknown battery node: {battery}")
        for module in module_list:
            if module not in self.node_type:
                raise KeyError(f"Unknown active module: {module}")

        if (
            isinstance(route_loss_factor, bool)
            or not isinstance(route_loss_factor, (int, float))
            or not math.isfinite(route_loss_factor)
            or route_loss_factor < 0
        ):
            raise ValueError("route_loss_factor must be a finite nonnegative number.")
        if (
            max_extra_cost is not None
            and (
                isinstance(max_extra_cost, bool)
                or not isinstance(max_extra_cost, (int, float))
                or not math.isfinite(max_extra_cost)
                or max_extra_cost < 0
            )
        ):
            raise ValueError("max_extra_cost must be None or a finite nonnegative number.")
        if isinstance(max_search_states, bool) or not isinstance(max_search_states, int):
            raise TypeError("max_search_states must be an integer.")
        if max_search_states <= len(module_list):
            raise ValueError(
                "max_search_states must exceed the number of active modules."
            )

        overrides = dict(module_weights or {})
        unknown_overrides = set(overrides) - set(module_list)
        if unknown_overrides:
            raise ValueError(
                f"module_weights contains inactive/unknown IDs: {sorted(unknown_overrides)}."
            )
        demands: Dict[str, float] = {}
        for module in module_list:
            if module in overrides:
                value = overrides[module]
            elif module in self.module_metadata:
                value = self.module_metadata[module].demand_weight
            else:
                # Legacy manually-typed action nodes retain unit demand.
                value = 1.0
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    f"Demand weight for {module!r} must be a finite positive number."
                )
            demands[module] = float(value)

        all_known_batteries = set(self.get_batteries())
        shortest_cache: Dict[str, Tuple[Dict[str, float], ParentMap]] = {}
        for battery in battery_list:
            shortest_cache[battery] = self.shortest_paths_from(
                battery,
                weighted=weighted,
                respect_switch_state=respect_switch_state,
                blocked_nodes=all_known_batteries - {battery},
            )

        candidates_by_module: Dict[str, List[DemandRouteCandidate]] = {}
        unreachable_modules: List[str] = []
        for module in module_list:
            raw: List[DemandRouteCandidate] = []
            for battery in battery_list:
                dist_map, parent = shortest_cache[battery]
                if module not in dist_map:
                    continue
                path = self.reconstruct_path(parent, module)
                if path is None:
                    continue
                path_nodes, path_switches = path
                distance = dist_map[module]
                demand = demands[module]
                raw.append(DemandRouteCandidate(
                    module=module,
                    battery=battery,
                    demand_weight=demand,
                    distance=distance,
                    path_nodes=tuple(path_nodes),
                    path_switches=tuple(path_switches),
                    effective_drain=demand * (
                        1.0 + float(route_loss_factor) * distance
                    ),
                    weighted_route_cost=demand * distance,
                ))

            if raw and max_extra_cost is not None:
                shortest_distance = min(candidate.distance for candidate in raw)
                raw = [
                    candidate
                    for candidate in raw
                    if candidate.distance <= shortest_distance + max_extra_cost
                ]
            raw.sort(key=lambda candidate: (
                candidate.distance,
                candidate.battery,
            ))
            candidates_by_module[module] = raw
            if not raw:
                unreachable_modules.append(module)

        search_modules = sorted(
            (module for module in module_list if candidates_by_module[module]),
            key=lambda module: (
                -demands[module],
                len(candidates_by_module[module]),
                module,
            ),
        )
        battery_index = {
            battery: index for index, battery in enumerate(battery_list)
        }
        suffix_min_drain = [0.0] * (len(search_modules) + 1)
        for index in range(len(search_modules) - 1, -1, -1):
            module = search_modules[index]
            suffix_min_drain[index] = (
                suffix_min_drain[index + 1]
                + min(
                    candidate.effective_drain
                    for candidate in candidates_by_module[module]
                )
            )

        best_key: Optional[Tuple[Any, ...]] = None
        best_assignment: Optional[Dict[str, DemandRouteCandidate]] = None
        best_switches: frozenset[str] = frozenset()
        states_explored = 0
        search_complete = True
        safe_cache: Dict[frozenset[str], bool] = {frozenset(): True}
        selected: Dict[str, DemandRouteCandidate] = {}

        def switch_union_is_safe(switches: frozenset[str]) -> bool:
            if not require_battery_isolation:
                return True
            cached = safe_cache.get(switches)
            if cached is not None:
                return cached
            safe = not self.battery_conflicts_for_switches(switches)
            safe_cache[switches] = safe
            return safe

        def objective_key(
            drains: Tuple[float, ...],
            route_cost: float,
            switches: frozenset[str],
        ) -> Tuple[Any, ...]:
            values = list(drains)
            mean = sum(values) / len(values)
            spread = max(values) - min(values)
            variance = sum((value - mean) ** 2 for value in values)
            owner_key = tuple(
                selected[module].battery
                for module in sorted(selected)
            )
            return (
                max(values),
                spread,
                variance,
                route_cost,
                len(switches),
                owner_key,
            )

        def search(
            index: int,
            drains: Tuple[float, ...],
            route_cost: float,
            switches: frozenset[str],
        ) -> None:
            nonlocal best_key, best_assignment, best_switches
            nonlocal states_explored, search_complete

            if states_explored >= max_search_states:
                search_complete = False
                return
            states_explored += 1

            if best_key is not None:
                total_lower_bound = sum(drains) + suffix_min_drain[index]
                peak_lower_bound = max(
                    max(drains),
                    total_lower_bound / len(battery_list),
                )
                if peak_lower_bound > best_key[0] + 1e-12:
                    return

            if index == len(search_modules):
                key = objective_key(drains, route_cost, switches)
                if best_key is None or key < best_key:
                    best_key = key
                    best_assignment = dict(selected)
                    best_switches = switches
                return

            module = search_modules[index]
            ordered_candidates = sorted(
                candidates_by_module[module],
                key=lambda candidate: (
                    max(
                        drains[battery_index[candidate.battery]]
                        + candidate.effective_drain,
                        max(drains),
                    ),
                    candidate.weighted_route_cost,
                    candidate.battery,
                ),
            )
            for candidate in ordered_candidates:
                battery_pos = battery_index[candidate.battery]
                next_drains = list(drains)
                next_drains[battery_pos] += candidate.effective_drain
                if best_key is not None and max(next_drains) > best_key[0] + 1e-12:
                    continue

                next_switches = switches.union(candidate.path_switches)
                if not switch_union_is_safe(next_switches):
                    continue

                selected[module] = candidate
                search(
                    index + 1,
                    tuple(next_drains),
                    route_cost + candidate.weighted_route_cost,
                    next_switches,
                )
                del selected[module]

        zero_drains = tuple(0.0 for _ in battery_list)
        search(0, zero_drains, 0.0, frozenset())
        if best_assignment is None or best_key is None:
            qualifier = (
                " within max_search_states"
                if not search_complete
                else ""
            )
            raise ValueError(
                "No battery-isolated demand-aware assignment was found"
                + qualifier
                + "."
            )

        assignments: Dict[str, AssignmentResult] = {}
        battery_loads = {battery: 0 for battery in battery_list}
        battery_demands = {battery: 0.0 for battery in battery_list}
        battery_effective_drain = {battery: 0.0 for battery in battery_list}
        required_links: Set[Tuple[str, str]] = set()
        for module in module_list:
            candidate = best_assignment.get(module)
            if candidate is None:
                assignments[module] = AssignmentResult(
                    module=module,
                    battery=None,
                    distance=math.inf,
                    path_nodes=None,
                    path_switches=None,
                )
                continue
            assignments[module] = AssignmentResult(
                module=module,
                battery=candidate.battery,
                distance=candidate.distance,
                path_nodes=list(candidate.path_nodes),
                path_switches=list(candidate.path_switches),
            )
            battery_loads[candidate.battery] += 1
            battery_demands[candidate.battery] += candidate.demand_weight
            battery_effective_drain[candidate.battery] += candidate.effective_drain
            for u, v in zip(candidate.path_nodes, candidate.path_nodes[1:]):
                required_links.add(self._norm_pair(u, v))

        candidate_required = sorted(best_switches)
        conflicts = self.battery_conflicts_for_switches(candidate_required)
        safe_to_apply = not conflicts
        required_closed = candidate_required if safe_to_apply else []
        recommended_open = sorted(
            set(self.switch_closed) - set(required_closed)
        )
        peak_drain, spread, variance, total_route_cost, _, _ = best_key
        return {
            "mode": "demand_aware",
            "assignments": assignments,
            "module_weights": dict(sorted(demands.items())),
            "battery_loads": dict(sorted(battery_loads.items())),
            "battery_demands": dict(sorted(battery_demands.items())),
            "battery_effective_drain": dict(sorted(battery_effective_drain.items())),
            "route_loss_factor": float(route_loss_factor),
            "path_solver": "dijkstra" if weighted else "bfs",
            "objective_order": [
                "peak_effective_drain",
                "drain_spread",
                "drain_variance",
                "total_demand_weighted_route_cost",
                "closed_switch_count",
            ],
            "objective": {
                "peak_effective_drain": peak_drain,
                "drain_spread": spread,
                "drain_variance": variance,
                "total_demand_weighted_route_cost": total_route_cost,
                "closed_switch_count": len(candidate_required),
            },
            "required_links": sorted(required_links),
            "required_closed_switches": required_closed,
            "close_switches": required_closed,
            "recommended_open_switches": recommended_open,
            "candidate_required_closed_switches": candidate_required,
            "safe_to_apply": safe_to_apply,
            "battery_conflicts": conflicts,
            "unreachable_modules": sorted(unreachable_modules),
            "search_complete": search_complete,
            "states_explored": states_explored,
            "max_search_states": max_search_states,
            # Compatibility aliases retained for callers of the old prototype.
            "required_open_switches": required_closed,
            "recommended_closed_switches": recommended_open,
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

        shortest_cache: Dict[str, Tuple[Dict[str, float], ParentMap]] = {}
        for battery in battery_list:
            shortest_cache[battery] = self.shortest_paths_from(
                battery,
                weighted=weighted,
                respect_switch_state=respect_switch_state,
                blocked_nodes=set(self.get_batteries()) - {battery},
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

        shortest_cache: Dict[str, Tuple[Dict[str, float], ParentMap]] = {}
        for battery in battery_list:
            shortest_cache[battery] = self.shortest_paths_from(
                battery,
                weighted=weighted,
                respect_switch_state=respect_switch_state,
                blocked_nodes=set(self.get_batteries()) - {battery},
            )

        current_owner: Dict[str, Optional[str]] = {m: baseline[m].battery for m in module_list}

        while True:
            current_obj = self._load_balance_objective(loads)
            best_move: Optional[Tuple[str, str, str]] = None
            best_move_key: Optional[
                Tuple[Tuple[int, float, Tuple[int, ...]], str, str, str]
            ] = None
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
                    if best_move_key is None or move_key < best_move_key:
                        best_obj = trial_obj
                        best_move_key = move_key
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
    def from_layout_json(
        cls,
        data: Union[str, Mapping[str, Any]],
        *,
        node_types: Optional[Mapping[str, str]] = None,
        battery_ids: Optional[Iterable[str]] = None,
        default_closed: bool = True,
        default_cost: float = 1.0,
        switch_separator: str = ".",
    ) -> "ModularRobotGraph":
        """Preferred name for `from_grid_json()` with the new `layout` schema."""
        return cls.from_grid_json(
            data,
            node_types=node_types,
            battery_ids=battery_ids,
            default_closed=default_closed,
            default_cost=default_cost,
            switch_separator=switch_separator,
        )

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

    @classmethod
    def from_grid_json(
        cls,
        data: Union[str, Mapping[str, Any]],
        *,
        node_types: Optional[Mapping[str, str]] = None,
        battery_ids: Optional[Iterable[str]] = None,
        default_closed: bool = True,
        default_cost: float = 1.0,
        switch_separator: str = ".",
    ) -> "ModularRobotGraph":
        """
        Build a graph from a row-major rotated-module layout.

        Accepted shape (either a mapping or a JSON string)::

            {
                "layout": {
                    "cols": 3,
                    "rows": 3,
                    "cells": [
                        ["M2(180d,[8,252,0])", "M3(90d,[1,252,0])", null],
                        ["M1(0d,[3,252,0])", "M4(90d,[2,252,0])", "M6(180d,[12,252,0])"],
                        [null, "M5(270d,[7,252,0])", null]
                    ]
                }
            }

        `cells` is ordered top-to-bottom, then left-to-right. Empty cells may
        be null, "", or ".". Rotation must be a multiple of 90 degrees and is
        normalized to 0/90/180/270. The preferred cell syntax includes
        `[role, digital_io, unused]`. Role 1 is a battery. Every role from 2
        through 16 is a powered load: motor/cpu/navigation/empty are roles
        2/3/4/5, while roles 6 through 16 are buffer1 through buffer11. Their
        default demand weights are 5/3/4/1 and 2 respectively.

        The legacy `grid` key and `<id>(<rotation>d)` cell syntax remain
        supported. For legacy cells, node types can be supplied with
        `node_types`, or battery IDs can be
        supplied with `battery_ids` (all remaining modules become actions).
        The same values may instead be embedded at the top level as
        `node_types`/`module_types` or `battery_ids`/`batteries`. Without either,
        IDs beginning with "B" are inferred as batteries for compatibility.

        Each module gets four physical switches named, for example,
        "T2.top" and "T2.right". Every adjacent link is traversable only when
        the correctly rotated switch at both endpoints is closed.
        """
        if isinstance(data, str):
            try:
                parsed = json.loads(data)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid grid JSON: {exc.msg}.") from exc
        else:
            parsed = data

        if not isinstance(parsed, Mapping):
            raise TypeError("Grid JSON must decode to an object.")

        root = parsed
        container_keys = [key for key in ("layout", "grid") if key in root]
        if len(container_keys) > 1:
            raise ValueError("Provide either 'layout' or legacy 'grid', not both.")
        grid = root[container_keys[0]] if container_keys else root
        if not isinstance(grid, Mapping):
            raise TypeError("'layout'/'grid' must be an object.")

        cols = grid.get("cols")
        rows = grid.get("rows")
        for name, value in (("cols", cols), ("rows", rows)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"'{name}' must be a positive integer.")

        cells = grid.get("cells")
        if not isinstance(cells, (list, tuple)):
            raise TypeError("'cells' must be a two-dimensional array.")
        if len(cells) != rows:
            raise ValueError(
                f"'cells' has {len(cells)} rows, but 'rows' declares {rows}."
            )

        parsed_cells: List[
            List[Optional[Tuple[str, int, Optional[ModuleMetadata]]]]
        ] = []
        seen_modules: Set[str] = set()
        metadata_presence: Set[bool] = set()
        for row_index, row in enumerate(cells):
            if not isinstance(row, (list, tuple)):
                raise TypeError(f"cells[{row_index}] must be an array.")
            if len(row) != cols:
                raise ValueError(
                    f"cells[{row_index}] has {len(row)} columns, "
                    f"but 'cols' declares {cols}."
                )

            parsed_row: List[
                Optional[Tuple[str, int, Optional[ModuleMetadata]]]
            ] = []
            for col_index, cell in enumerate(row):
                if cell is None or (
                    isinstance(cell, str) and cell.strip() in {"", "."}
                ):
                    parsed_row.append(None)
                    continue
                if not isinstance(cell, str):
                    raise TypeError(
                        f"cells[{row_index}][{col_index}] must be a string or empty."
                    )

                match = _GRID_CELL_PATTERN.fullmatch(cell)
                if match is None:
                    raise ValueError(
                        f"Invalid cell {cell!r} at ({row_index}, {col_index}); "
                        "expected '<module_id>(<rotation>d,[role,digital_io,unused])' "
                        "or the legacy '<module_id>(<rotation>d)'."
                    )

                module = match.group("module").strip()
                rotation = int(match.group("rotation"))
                if not module:
                    raise ValueError(
                        f"Empty module ID at cells[{row_index}][{col_index}]."
                    )
                if rotation % 90 != 0:
                    raise ValueError(
                        f"Rotation for {module!r} must be a multiple of 90 degrees."
                    )
                if module in seen_modules:
                    raise ValueError(f"Duplicate module ID in grid: {module!r}.")

                role_raw = match.group("role")
                metadata: Optional[ModuleMetadata]
                if role_raw is None:
                    metadata = None
                    metadata_presence.add(False)
                else:
                    role = int(role_raw)
                    digital_io = int(match.group("digital_io"))
                    unused = int(match.group("unused"))
                    if not 1 <= role <= 16:
                        raise ValueError(
                            f"Role for {module!r} must be between 1 and 16."
                        )
                    if not 0 <= digital_io <= 255:
                        raise ValueError(
                            f"digital_io for {module!r} must be between 0 and 255."
                        )
                    metadata = ModuleMetadata(
                        role=role,
                        digital_io=digital_io,
                        unused=unused,
                    )
                    metadata_presence.add(True)

                seen_modules.add(module)
                parsed_row.append((module, rotation % 360, metadata))
            parsed_cells.append(parsed_row)

        if not seen_modules:
            raise ValueError("'cells' must contain at least one module.")
        if len(metadata_presence) > 1:
            raise ValueError(
                "Do not mix role-aware and legacy cells in one layout; "
                "either include [role,digital_io,unused] for every module or none."
            )
        if not isinstance(switch_separator, str) or not switch_separator:
            raise ValueError("switch_separator must be a non-empty string.")
        if not isinstance(default_closed, bool):
            raise TypeError("default_closed must be a boolean.")
        if (
            isinstance(default_cost, bool)
            or not isinstance(default_cost, (int, float))
            or not math.isfinite(default_cost)
            or default_cost < 0
        ):
            raise ValueError("default_cost must be a finite nonnegative number.")

        embedded_node_types = root.get("node_types", root.get("module_types"))
        embedded_battery_ids = root.get("battery_ids", root.get("batteries"))
        has_role_metadata = metadata_presence == {True}
        if has_role_metadata:
            if any(
                value is not None
                for value in (
                    node_types,
                    battery_ids,
                    embedded_node_types,
                    embedded_battery_ids,
                )
            ):
                raise ValueError(
                    "Role-aware cells already define batteries and loads; "
                    "do not also provide node_types or battery_ids."
                )
            metadata_by_module = {
                module: metadata
                for row in parsed_cells
                for parsed_cell in row
                if parsed_cell is not None
                for module, _, metadata in (parsed_cell,)
                if metadata is not None
            }
            resolved_types = {
                module: (
                    "battery"
                    if metadata.is_battery
                    else "action"
                    if metadata.is_load
                    else "relay"
                )
                for module, metadata in metadata_by_module.items()
            }
        else:
            if node_types is None and battery_ids is None:
                if embedded_node_types is not None and embedded_battery_ids is not None:
                    raise ValueError(
                        "Provide either node types or battery IDs, not both."
                    )
                node_types = embedded_node_types
                battery_ids = embedded_battery_ids
            elif node_types is not None and battery_ids is not None:
                raise ValueError("Provide either node_types or battery_ids, not both.")

            resolved_types: Dict[str, str]
            metadata_by_module: Dict[str, ModuleMetadata] = {}

        if not has_role_metadata and node_types is not None:
            if not isinstance(node_types, Mapping):
                raise TypeError("node_types must be an object mapping IDs to types.")
            provided_ids = set(node_types)
            missing_ids = seen_modules - provided_ids
            extra_ids = provided_ids - seen_modules
            if missing_ids or extra_ids:
                details = []
                if missing_ids:
                    details.append(f"missing {sorted(missing_ids)}")
                if extra_ids:
                    details.append(f"unknown {sorted(extra_ids, key=str)}")
                raise ValueError(
                    "node_types IDs do not match grid: "
                    + ", ".join(details)
                    + "."
                )

            resolved_types = {}
            for module in seen_modules:
                node_type = node_types[module]
                if not isinstance(node_type, str) or node_type.lower() not in {
                    "battery",
                    "action",
                    "relay",
                }:
                    raise ValueError(
                        f"Invalid type for {module!r}: {node_type!r}; "
                        "expected 'battery', 'action', or 'relay'."
                    )
                resolved_types[module] = node_type.lower()
        elif not has_role_metadata:
            if battery_ids is None:
                battery_set = {
                    module for module in seen_modules
                    if module.upper().startswith("B")
                }
            else:
                if isinstance(battery_ids, (str, bytes)):
                    raise TypeError("battery_ids must be an iterable of module IDs.")
                battery_set = set(battery_ids)
                unknown_batteries = battery_set - seen_modules
                if unknown_batteries:
                    raise ValueError(
                        f"Unknown battery IDs: {sorted(unknown_batteries, key=str)}."
                    )
            resolved_types = {
                module: ("battery" if module in battery_set else "action")
                for module in seen_modules
            }

        g = cls()
        g.grid_rows = rows
        g.grid_cols = cols
        g.grid_cells = [[None for _ in range(cols)] for _ in range(rows)]

        # Register nodes and all four switches before creating physical links.
        for row_index, row in enumerate(parsed_cells):
            for col_index, parsed_cell in enumerate(row):
                if parsed_cell is None:
                    continue
                module, rotation, metadata = parsed_cell
                position = (col_index, rows - 1 - row_index)
                g.add_node(module, resolved_types[module], pos=position)
                g.module_rotation[module] = rotation
                if metadata is not None:
                    g.module_metadata[module] = metadata
                g.grid_cells[row_index][col_index] = module
                g.module_switches[module] = {}

                for local_direction in LOCAL_DIRECTIONS:
                    switch_id = f"{module}{switch_separator}{local_direction}"
                    if switch_id in g.switch_closed:
                        raise ValueError(f"Generated duplicate switch ID: {switch_id!r}.")
                    world_direction = g._world_direction_for_local(
                        local_direction,
                        rotation,
                    )
                    g.module_switches[module][local_direction] = switch_id
                    g.switch_closed[switch_id] = default_closed
                    g.switch_info[switch_id] = ModuleSwitchInfo(
                        module=module,
                        local_direction=local_direction,
                        world_direction=world_direction,
                    )

        # Check grid-right and grid-bottom only, so each adjacency is added once.
        neighbor_specs = (
            (0, 1, "right", "left"),
            (1, 0, "bottom", "top"),
        )
        for row_index in range(rows):
            for col_index in range(cols):
                module = g.grid_cells[row_index][col_index]
                if module is None:
                    continue
                for d_row, d_col, module_side, neighbor_side in neighbor_specs:
                    neighbor_row = row_index + d_row
                    neighbor_col = col_index + d_col
                    if neighbor_row >= rows or neighbor_col >= cols:
                        continue
                    neighbor = g.grid_cells[neighbor_row][neighbor_col]
                    if neighbor is None:
                        continue

                    module_switch = g.get_local_switch_facing(module, module_side)
                    neighbor_switch = g.get_local_switch_facing(neighbor, neighbor_side)
                    g.add_dual_switch_edge(
                        module,
                        neighbor,
                        module_switch,
                        neighbor_switch,
                        closed_=default_closed,
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
        print(f"    switch_count = {info.switch_count}")
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


def build_cartesian_layout_array(
    node_pos: Dict[str, Tuple[int, int]],
    *,
    grid_size: int = 10,
) -> List[List[str]]:
    """
    Build a Cartesian occupancy grid addressed as layout[x][y].

    This keeps the module coordinates unchanged while embedding the demo inside
    a fixed-size 2D plane, e.g. B4 at layout[0][0] and B1 at layout[0][6].
    """
    layout = [["." for _ in range(grid_size)] for _ in range(grid_size)]

    for node_id, (x, y) in node_pos.items():
        if not (0 <= x < grid_size and 0 <= y < grid_size):
            raise ValueError(
                f"Node {node_id} at {(x, y)} is outside the {grid_size}x{grid_size} layout."
            )
        if layout[x][y] != ".":
            raise ValueError(
                f"Duplicate node placement at {(x, y)}: {node_id} conflicts with {layout[x][y]}."
            )
        layout[x][y] = node_id

    return layout


def print_cartesian_layout(layout: List[List[str]]) -> None:
    """
    Print the full 2D Cartesian layout.

    The storage convention is layout[x][y], with x increasing to the right and
    y increasing upward.
    """
    if not layout:
        print("  <empty layout>")
        return

    grid_size = len(layout)
    if any(len(column) != grid_size for column in layout):
        raise ValueError("layout must be a square grid stored as layout[x][y].")

    cell_width = max(
        3,
        max(len(cell) for column in layout for cell in column),
    )

    print(
        f"  Cartesian {grid_size}x{grid_size} grid "
        "(layout[x][y], x -> right, y -> up, '.' = empty):"
    )
    x_axis = " ".join(f"{x:>{cell_width}}" for x in range(grid_size))
    print(f"  y\\x | {x_axis}")
    for y in range(grid_size - 1, -1, -1):
        row = " ".join(f"{layout[x][y]:>{cell_width}}" for x in range(grid_size))
        print(f"  {y:>3} | {row}")


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
    layout_lines: Optional[List[str]] = None,
    layout_grid_size: Optional[int] = None,
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
    if layout_grid_size is not None:
        print_cartesian_layout(
            build_cartesian_layout_array(g.node_pos, grid_size=layout_grid_size)
        )
    elif layout_lines is not None:
        for line in layout_lines:
            print(f"  {line}")
    else:
        print("  <layout unavailable>")
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
    compare_modes(
        build_demo4_graph(),
        title="DEMO 4A: 5-battery articulated arm on full topology",
        layout_grid_size=10,
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
        layout_grid_size=10,
        batteries=["B1", "B3", "B4", "B5"],
        respect_switch_state=True,
    )


if __name__ == "__main__":
    demo_compare_large_quad()
    demo_compare_articulated_arm()
