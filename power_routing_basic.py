from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Iterable, Set, Any
import math
import heapq


@dataclass(frozen=True)
class Edge:
    """One undirected physical link between two modules, governed by a switch."""
    other: str
    switch_id: str
    cost: float = 1.0  # default: unit cost per switch traversal


class ModularRobotGraph:
    """
    Graph model:
      - vertex: module (battery or action)
      - edge: a connection governed by a switch
      - open switch => traversable edge
      - closed switch => blocked edge
    """

    def __init__(self) -> None:
        self.node_type: Dict[str, str] = {}          # node -> "battery" | "action" | ...
        self.adj: Dict[str, List[Edge]] = defaultdict(list)
        self.switch_open: Dict[str, bool] = {}       # switch_id -> True/False

    def add_node(self, node_id: str, node_type: str) -> None:
        if node_id in self.node_type and self.node_type[node_id] != node_type:
            raise ValueError(f"Node {node_id} already exists with type {self.node_type[node_id]}")
        self.node_type[node_id] = node_type
        _ = self.adj[node_id]  # ensure key exists

    def add_switch_edge(self, u: str, v: str, switch_id: str, open_: bool = True, cost: float = 1.0) -> None:
        if u not in self.node_type or v not in self.node_type:
            raise KeyError("Both endpoints must be added as nodes before adding an edge.")
        if switch_id in self.switch_open and self.switch_open[switch_id] != open_:
            # Allow changing later via set_switch_state; here we just keep the original.
            pass
        self.switch_open[switch_id] = open_
        # Undirected: store in both adjacency lists
        self.adj[u].append(Edge(other=v, switch_id=switch_id, cost=cost))
        self.adj[v].append(Edge(other=u, switch_id=switch_id, cost=cost))

    def set_switch_state(self, switch_id: str, open_: bool) -> None:
        if switch_id not in self.switch_open:
            raise KeyError(f"Unknown switch_id: {switch_id}")
        self.switch_open[switch_id] = open_

    def get_batteries(self) -> List[str]:
        return [n for n, t in self.node_type.items() if t == "battery"]

    def get_actions(self) -> List[str]:
        return [n for n, t in self.node_type.items() if t == "action"]

    # --------- Core algorithms ---------

    def bfs_shortest_switches(self, source: str) -> Tuple[Dict[str, int], Dict[str, Optional[Tuple[str, str]]]]:
        """
        Unweighted shortest paths (# of switches) from source using BFS.
        Returns:
          dist[node] = minimum number of traversed open switches (edges)
          parent[node] = (prev_node, switch_id) for path reconstruction, or None for source
        """
        if source not in self.node_type:
            raise KeyError(f"Unknown source node: {source}")

        dist: Dict[str, int] = {source: 0}
        parent: Dict[str, Optional[Tuple[str, str]]] = {source: None}
        q = deque([source])

        while q:
            u = q.popleft()
            for e in self.adj[u]:
                if not self.switch_open.get(e.switch_id, False):
                    continue  # switch closed => edge blocked
                v = e.other
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = (u, e.switch_id)
                    q.append(v)

        return dist, parent

    def dijkstra_weighted(self, source: str) -> Tuple[Dict[str, float], Dict[str, Optional[Tuple[str, str]]]]:
        """
        Weighted shortest paths for nonnegative costs (generalization).
        Useful if later you assign different losses per switch/connection.
        """
        if source not in self.node_type:
            raise KeyError(f"Unknown source node: {source}")

        dist: Dict[str, float] = {source: 0.0}
        parent: Dict[str, Optional[Tuple[str, str]]] = {source: None}
        pq: List[Tuple[float, str]] = [(0.0, source)]

        while pq:
            d_u, u = heapq.heappop(pq)
            if d_u != dist.get(u, math.inf):
                continue

            for e in self.adj[u]:
                if not self.switch_open.get(e.switch_id, False):
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
        target: str
    ) -> Optional[Tuple[List[str], List[str]]]:
        """
        Reconstructs (nodes_path, switches_path) from a parent map.
        nodes_path: [source, ..., target]
        switches_path: [switch used between consecutive nodes], length = len(nodes_path) - 1
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

    # --------- Multi-battery helpers ---------

    def battery_to_all_distances(self, batteries: Iterable[str]) -> Dict[str, Dict[str, int]]:
        """
        Returns nested dict:
          dist_by_battery[b][v] = #switches on shortest path from b to v (if reachable)
        """
        out: Dict[str, Dict[str, int]] = {}
        for b in batteries:
            dist_b, _ = self.bfs_shortest_switches(b)
            out[b] = dist_b
        return out

    def per_module_battery_distances(
        self,
        batteries: Iterable[str],
        modules: Iterable[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Returns:
          table[module][battery] = distance (#switches), or +inf if unreachable
        """
        dist_by_b = self.battery_to_all_distances(batteries)
        table: Dict[str, Dict[str, float]] = {}
        for m in modules:
            row: Dict[str, float] = {}
            for b, distmap in dist_by_b.items():
                row[b] = float(distmap.get(m, math.inf))
            table[m] = row
        return table

    @staticmethod
    def argmin_battery(distance_row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        """
        Given row[battery] = distance, return (best_battery, best_distance),
        ignoring +inf. Returns None if all are unreachable.
        """
        best_b: Optional[str] = None
        best_d: float = math.inf
        for b, d in distance_row.items():
            if d < best_d:
                best_b, best_d = b, d
        if best_b is None or math.isinf(best_d):
            return None
        return best_b, best_d


def demo_l_shape() -> None:
    """
    Your example:
      B1 --S1-- M1 --S2-- M2 --S3-- M3 --S4-- B2
    """

    g = ModularRobotGraph()
    g.add_node("B1", "battery")
    g.add_node("B2", "battery")
    g.add_node("M1", "action")
    g.add_node("M2", "action")
    g.add_node("M3", "action")

    g.add_switch_edge("B1", "M1", "S_B1_M1", open_=True)
    g.add_switch_edge("M1", "M2", "S_M1_M2", open_=True)
    g.add_switch_edge("M2", "M3", "S_M2_M3", open_=True)
    g.add_switch_edge("M3", "B2", "S_M3_B2", open_=True)

    batteries = ["B1", "B2"]
    modules = ["M1", "M2", "M3"]

    # Case 1: Close B2-M3, open others => B1 powers all reachable
    g.set_switch_state("S_M3_B2", False)
    table1 = g.per_module_battery_distances(batteries, modules)
    assignment1 = {m: g.argmin_battery(table1[m]) for m in modules}

    print("Case 1 distances:", table1)
    print("Case 1 best battery per module:", assignment1)

    # Case 2: Open B1-M1, M1-M2, and M3-B2; close M2-M3
    g.set_switch_state("S_M3_B2", True)
    g.set_switch_state("S_M2_M3", False)
    table2 = g.per_module_battery_distances(batteries, modules)
    assignment2 = {m: g.argmin_battery(table2[m]) for m in modules}

    print("Case 2 distances:", table2)
    print("Case 2 best battery per module:", assignment2)


if __name__ == "__main__":
    demo_l_shape()
