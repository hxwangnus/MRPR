import json
from pathlib import Path
import unittest

from power_routing_balanced import ModularRobotGraph


SAMPLE_GRID = {
    "grid": {
        "cols": 2,
        "rows": 3,
        "cells": [
            ["T2(270d)", "T3(180d)"],
            ["T1(0d)", "T4(90d)"],
            ["T6(180d)", "T5(0d)"],
        ],
    }
}

SAMPLE_ROLE_LAYOUT = {
    "layout": {
        "cols": 3,
        "rows": 3,
        "cells": [
            ["M2(180d,[8,252,0])", "M3(90d,[1,252,0])", None],
            [
                "M1(0d,[3,252,0])",
                "M4(90d,[2,252,0])",
                "M6(180d,[12,252,0])",
            ],
            [None, "M5(270d,[7,252,0])", None],
        ],
    }
}


class RotatedGridTests(unittest.TestCase):
    def test_complex_4x4_example_balances_three_batteries(self):
        example_path = (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "complex_4x4_role_layout.json"
        )
        payload = json.loads(example_path.read_text(encoding="utf-8"))
        graph = ModularRobotGraph.from_layout_json(payload, default_closed=False)

        roles = [metadata.role for metadata in graph.module_metadata.values()]
        null_count = sum(
            cell is None
            for row in payload["layout"]["cells"]
            for cell in row
        )
        self.assertEqual((graph.grid_rows, graph.grid_cols), (4, 4))
        self.assertEqual(roles.count(1), 3)
        self.assertEqual(roles.count(2), 2)
        self.assertEqual(len(roles) - roles.count(1) - roles.count(2), 7)
        self.assertEqual(null_count, 4)

        baseline = graph.nearest_battery_assignment(respect_switch_state=False)
        baseline_loads = {battery: 0 for battery in graph.get_batteries()}
        for assignment in baseline.values():
            baseline_loads[assignment.battery] += 1
        plan = graph.recommend_balanced_switch_plan(respect_switch_state=False)

        self.assertEqual(baseline_loads, {"M06": 4, "M09": 0, "M11": 2})
        self.assertEqual(plan["battery_loads"], {"M06": 2, "M09": 2, "M11": 2})
        self.assertEqual(
            {
                module
                for module, assignment in plan["assignments"].items()
                if baseline[module].battery != assignment.battery
            },
            {"M03", "M04"},
        )
        self.assertTrue(plan["safe_to_apply"])
        self.assertEqual(plan["unreachable_modules"], [])
        self.assertEqual(len(plan["required_closed_switches"]), 14)
        for module, assignment in plan["assignments"].items():
            self.assertEqual(assignment.distance, baseline[module].distance)

    def test_role_layout_drives_battery_load_and_relay_classification(self):
        graph = ModularRobotGraph.from_layout_json(SAMPLE_ROLE_LAYOUT)

        self.assertEqual(graph.get_batteries(), ["M3"])
        self.assertEqual(graph.get_actions(), ["M1", "M4"])
        self.assertEqual(graph.get_relays(), ["M2", "M5", "M6"])
        self.assertEqual(
            graph.node_type,
            {
                "M2": "relay",
                "M3": "battery",
                "M1": "action",
                "M4": "action",
                "M6": "relay",
                "M5": "relay",
            },
        )

        m3 = graph.get_module_metadata("M3")
        self.assertEqual(m3.role, 1)
        self.assertEqual(m3.role_name, "battery")
        self.assertEqual(m3.digital_io, 252)
        self.assertEqual(m3.digital_io_bits, "11111100")
        self.assertEqual(m3.unused, 0)
        self.assertEqual(graph.get_module_metadata("M2").role_name, "unused")
        self.assertEqual(graph.get_module_metadata("M6").role_name, "unused")

    def test_user_rotation_example_maps_local_switches_to_expected_neighbors(self):
        graph = ModularRobotGraph.from_grid_json(SAMPLE_ROLE_LAYOUT)

        self.assertEqual(
            graph.get_link_switches("M2", "M3"),
            ("M2.left", "M3.top"),
        )
        self.assertEqual(
            graph.get_link_switches("M2", "M1"),
            ("M2.top", "M1.top"),
        )
        self.assertEqual(graph.switch_info["M2.top"].world_direction, "bottom")
        self.assertEqual(graph.switch_info["M2.left"].world_direction, "right")
        self.assertEqual(graph.switch_info["M3.top"].world_direction, "left")
        self.assertIsNone(graph.switch_info["M2.bottom"].neighbor)
        self.assertIsNone(graph.switch_info["M2.right"].neighbor)

    def test_sample_balanced_plan_uses_unused_relay_and_counts_only_loads(self):
        graph = ModularRobotGraph.from_grid_json(
            SAMPLE_ROLE_LAYOUT,
            default_closed=False,
        )
        plan = graph.recommend_balanced_switch_plan(
            respect_switch_state=False,
        )

        self.assertTrue(plan["safe_to_apply"])
        self.assertEqual(plan["battery_conflicts"], [])
        self.assertEqual(plan["battery_loads"], {"M3": 2})
        self.assertEqual(plan["unreachable_modules"], [])
        self.assertEqual(
            plan["required_links"],
            [("M1", "M2"), ("M2", "M3"), ("M3", "M4")],
        )
        self.assertEqual(
            plan["required_closed_switches"],
            [
                "M1.top",
                "M2.left",
                "M2.top",
                "M3.left",
                "M3.top",
                "M4.right",
            ],
        )

        m1 = plan["assignments"]["M1"]
        self.assertEqual(m1.path_nodes, ["M3", "M2", "M1"])
        self.assertEqual(
            m1.path_switches,
            ["M3.top", "M2.left", "M2.top", "M1.top"],
        )
        self.assertEqual(m1.distance, 2.0)
        self.assertEqual(m1.switch_count, 4)

        m4 = plan["assignments"]["M4"]
        self.assertEqual(m4.path_nodes, ["M3", "M4"])
        self.assertEqual(m4.path_switches, ["M3.left", "M4.right"])
        self.assertEqual(m4.distance, 1.0)
        self.assertEqual(m4.switch_count, 2)

    def test_unsafe_balanced_switch_union_fails_closed(self):
        modules = {}
        module_index = 1
        for y in range(3):
            for x in range(3):
                if (x, y) == (0, 0):
                    node_id = "B1"
                elif (x, y) == (2, 0):
                    node_id = "B2"
                else:
                    node_id = f"M{module_index:02d}"
                    module_index += 1
                modules[node_id] = {
                    "type": "battery" if node_id.startswith("B") else "action",
                    "pos": (x, y),
                }

        graph = ModularRobotGraph.from_grid_layout(modules)
        plan = graph.recommend_balanced_switch_plan(
            respect_switch_state=False,
        )

        self.assertFalse(plan["safe_to_apply"])
        self.assertEqual(plan["battery_conflicts"], [["B1", "B2"]])
        self.assertEqual(plan["required_closed_switches"], [])
        self.assertTrue(plan["candidate_required_closed_switches"])

    def test_rebalanced_equal_objective_ties_are_input_order_independent(self):
        modules = {}
        for y in range(3):
            for x in range(3):
                node_id = (
                    "B1"
                    if (x, y) == (0, 0)
                    else "B2"
                    if (x, y) == (2, 2)
                    else f"M{x}{y}"
                )
                modules[node_id] = {
                    "type": "battery" if node_id.startswith("B") else "action",
                    "pos": (x, y),
                }

        graph = ModularRobotGraph.from_grid_layout(modules)
        module_order = graph.get_actions()
        forward, forward_loads = graph.rebalanced_nearest_battery_assignment(
            modules=module_order,
            respect_switch_state=False,
        )
        reverse, reverse_loads = graph.rebalanced_nearest_battery_assignment(
            modules=reversed(module_order),
            respect_switch_state=False,
        )

        self.assertEqual(forward_loads, reverse_loads)
        self.assertEqual(
            {module: info.battery for module, info in forward.items()},
            {module: info.battery for module, info in reverse.items()},
        )

    def test_sample_grid_builds_expected_positions_and_switch_pairs(self):
        graph = ModularRobotGraph.from_grid_json(SAMPLE_GRID)

        self.assertEqual(
            graph.node_pos,
            {
                "T2": (0, 2),
                "T3": (1, 2),
                "T1": (0, 1),
                "T4": (1, 1),
                "T6": (0, 0),
                "T5": (1, 0),
            },
        )
        self.assertEqual(
            graph.module_rotation,
            {"T2": 270, "T3": 180, "T1": 0, "T4": 90, "T6": 180, "T5": 0},
        )

        expected_pairs = {
            ("T2", "T3"): ("T2.top", "T3.right"),
            ("T2", "T1"): ("T2.right", "T1.top"),
            ("T3", "T4"): ("T3.top", "T4.right"),
            ("T1", "T4"): ("T1.right", "T4.top"),
            ("T1", "T6"): ("T1.bottom", "T6.bottom"),
            ("T4", "T5"): ("T4.left", "T5.top"),
            ("T6", "T5"): ("T6.left", "T5.left"),
        }
        for endpoints, switches in expected_pairs.items():
            with self.subTest(endpoints=endpoints):
                self.assertEqual(graph.get_link_switches(*endpoints), switches)
                self.assertEqual(
                    graph.get_link_switches(*reversed(endpoints)),
                    tuple(reversed(switches)),
                )

        self.assertEqual(len(graph.link_to_switches), 7)
        self.assertEqual(len(graph.switch_closed), 24)
        self.assertEqual(len(graph.switch_endpoints), 14)

    def test_unused_local_switches_exist_but_have_no_neighbor(self):
        graph = ModularRobotGraph.from_grid_json(SAMPLE_GRID)

        self.assertIsNone(graph.switch_info["T2.bottom"].neighbor)
        self.assertIsNone(graph.switch_info["T2.left"].neighbor)
        self.assertEqual(graph.switch_info["T2.top"].neighbor, "T3")
        self.assertEqual(graph.switch_info["T2.right"].neighbor, "T1")
        self.assertEqual(graph.switch_info["T2.top"].world_direction, "right")
        self.assertEqual(graph.switch_info["T2.right"].world_direction, "bottom")

    def test_runtime_link_requires_both_endpoint_switches(self):
        graph = ModularRobotGraph.from_grid_json(
            {
                "grid": {
                    "cols": 2,
                    "rows": 1,
                    "cells": [["B1(270d)", "M1(180d)"]],
                }
            },
            default_closed=False,
        )
        left_switch, right_switch = graph.get_link_switches("B1", "M1")
        self.assertEqual((left_switch, right_switch), ("B1.top", "M1.right"))

        graph.set_switch_closed(left_switch, True)
        distance, _ = graph.shortest_paths_from(
            "B1",
            respect_switch_state=True,
        )
        self.assertNotIn("M1", distance)

        graph.set_switch_closed(right_switch, True)
        distance, parent = graph.shortest_paths_from(
            "B1",
            respect_switch_state=True,
        )
        self.assertEqual(distance["M1"], 1.0)
        self.assertEqual(
            graph.reconstruct_path(parent, "M1"),
            (["B1", "M1"], ["B1.top", "M1.right"]),
        )

        graph.set_switch_closed(left_switch, False)
        distance, _ = graph.shortest_paths_from(
            "B1",
            respect_switch_state=True,
        )
        self.assertNotIn("M1", distance)

    def test_planning_plan_uses_physical_close_open_terminology(self):
        graph = ModularRobotGraph.from_grid_json(
            {
                "grid": {
                    "cols": 2,
                    "rows": 1,
                    "cells": [["B1(270d)", "M1(180d)"]],
                }
            },
            default_closed=False,
        )
        plan = graph.recommend_switch_plan(
            active_modules=["M1"],
            batteries=["B1"],
            respect_switch_state=False,
        )

        self.assertEqual(
            plan["required_closed_switches"],
            ["B1.top", "M1.right"],
        )
        self.assertEqual(len(plan["recommended_open_switches"]), 6)
        self.assertEqual(
            plan["required_open_switches"],
            plan["required_closed_switches"],
        )
        self.assertEqual(
            plan["recommended_closed_switches"],
            plan["recommended_open_switches"],
        )

    def test_json_string_and_embedded_batteries_are_supported(self):
        payload = dict(SAMPLE_GRID)
        payload["batteries"] = ["T2", "T5"]
        graph = ModularRobotGraph.from_grid_json(json.dumps(payload))

        self.assertEqual(graph.get_batteries(), ["T2", "T5"])
        self.assertEqual(graph.get_actions(), ["T1", "T3", "T4", "T6"])
        balanced, loads = graph.rebalanced_nearest_battery_assignment(
            respect_switch_state=False,
        )
        self.assertEqual(loads, {"T2": 2, "T5": 2})
        self.assertEqual(
            balanced["T3"].path_switches,
            ["T2.top", "T3.right"],
        )
        self.assertEqual(
            balanced["T1"].path_switches,
            ["T2.right", "T1.top"],
        )

    def test_balanced_assignment_keeps_shortest_distances_with_dual_switch_paths(self):
        graph = ModularRobotGraph.from_grid_json(
            {
                "grid": {
                    "cols": 3,
                    "rows": 2,
                    "cells": [
                        ["B1(0d)", "M1(90d)", "B2(180d)"],
                        [None, "M2(270d)", None],
                    ],
                }
            }
        )

        baseline = graph.nearest_battery_assignment(respect_switch_state=False)
        balanced, loads = graph.rebalanced_nearest_battery_assignment(
            respect_switch_state=False,
        )

        self.assertEqual(loads, {"B1": 1, "B2": 1})
        self.assertEqual(
            {info.battery for info in balanced.values()},
            {"B1", "B2"},
        )
        for module, info in balanced.items():
            with self.subTest(module=module):
                self.assertEqual(info.distance, baseline[module].distance)
                self.assertEqual(
                    len(info.path_switches or []),
                    2 * int(info.distance),
                )

    def test_legacy_single_switch_grid_api_remains_compatible(self):
        graph = ModularRobotGraph.from_grid_layout(
            {
                "B1": {"type": "battery", "pos": (0, 0)},
                "M1": {"type": "action", "pos": (1, 0)},
            }
        )
        switch_id = graph.get_link_switches("B1", "M1")[0]

        _, parent = graph.shortest_paths_from("B1")
        self.assertEqual(parent["M1"], ("B1", switch_id))
        assignment = graph.nearest_battery_assignment()["M1"]
        self.assertEqual(assignment.path_switches, [switch_id])
        graph.set_switch_state(switch_id, False)
        assignment = graph.nearest_battery_assignment()["M1"]
        self.assertIsNone(assignment.battery)

    def test_manual_dual_switch_edge_tracks_endpoint_order(self):
        graph = ModularRobotGraph()
        graph.add_node("A", "battery")
        graph.add_node("B", "action")
        graph.add_dual_switch_edge("A", "B", "A.right", "B.left")

        self.assertEqual(
            graph.get_link_switches("A", "B"),
            ("A.right", "B.left"),
        )
        self.assertEqual(
            graph.get_link_switches("B", "A"),
            ("B.left", "A.right"),
        )
        assignment = graph.nearest_battery_assignment()["B"]
        self.assertEqual(assignment.path_switches, ["A.right", "B.left"])

    def test_invalid_grid_inputs_fail_early(self):
        cases = [
            (
                {"grid": {"cols": 2, "rows": 2, "cells": [["T1(0d)", None]]}},
                ValueError,
            ),
            (
                {"grid": {"cols": 1, "rows": 1, "cells": [["T1(45d)"]]}},
                ValueError,
            ),
            (
                {
                    "grid": {
                        "cols": 2,
                        "rows": 1,
                        "cells": [["T1(0d)", "T1(90d)"]],
                    }
                },
                ValueError,
            ),
        ]
        for payload, error in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(error):
                    ModularRobotGraph.from_grid_json(payload)

        with self.assertRaisesRegex(ValueError, "node_types IDs do not match"):
            ModularRobotGraph.from_grid_json(
                SAMPLE_GRID,
                node_types={"T1": "battery"},
            )

        with self.assertRaisesRegex(ValueError, "Unknown battery IDs"):
            ModularRobotGraph.from_grid_json(
                SAMPLE_GRID,
                battery_ids=["not-in-grid"],
            )

    def test_invalid_role_metadata_fails_early(self):
        invalid_cells = [
            "M1(0d,[0,252,0])",
            "M1(0d,[17,252,0])",
            "M1(0d,[1,256,0])",
            "M1(0d,[1,252])",
        ]
        for cell in invalid_cells:
            with self.subTest(cell=cell):
                with self.assertRaises(ValueError):
                    ModularRobotGraph.from_grid_json(
                        {
                            "layout": {
                                "cols": 1,
                                "rows": 1,
                                "cells": [[cell]],
                            }
                        }
                    )

        with self.assertRaisesRegex(ValueError, "Do not mix"):
            ModularRobotGraph.from_grid_json(
                {
                    "layout": {
                        "cols": 2,
                        "rows": 1,
                        "cells": [["M1(0d,[1,252,0])", "M2(0d)"]],
                    }
                }
            )

        with self.assertRaisesRegex(ValueError, "already define"):
            ModularRobotGraph.from_grid_json(
                SAMPLE_ROLE_LAYOUT,
                battery_ids=["M3"],
            )

        with self.assertRaisesRegex(ValueError, "not both"):
            ModularRobotGraph.from_grid_json(
                {
                    "layout": SAMPLE_ROLE_LAYOUT["layout"],
                    "grid": SAMPLE_GRID["grid"],
                }
            )


if __name__ == "__main__":
    unittest.main()
