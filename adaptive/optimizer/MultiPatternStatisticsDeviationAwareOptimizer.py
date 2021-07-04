from queue import Queue
from typing import List

from adaptive.optimizer.StatisticsDeviationAwareOptimizer import StatisticsDeviationAwareOptimizer
from adaptive.statistics.StatisticsCollector import StatisticsCollector
from base.Pattern import Pattern
from plan.TreePlan import TreePlan, TreePlanNode, TreePlanLeafNode, TreePlanNestedNode, TreePlanUnaryNode, \
    TreePlanBinaryNode
from plan.TreePlanBuilder import TreePlanBuilder
from plan.multi.TreePlanMerger import TreePlanMerger


class MultiPatternStatisticsDeviationAwareOptimizer(StatisticsDeviationAwareOptimizer):
    """
    Represents the Multi Pattern optimizer.
    For every pattern(single tree in the forest), monitors statistics deviations from their latest observed values.
    """
    def __init__(self, patterns: List[Pattern], recursive_traversal_tree_plan_merger: TreePlanMerger,
                 tree_plan_builder: TreePlanBuilder, is_adaptivity_enabled: bool,
                 type_to_deviation_aware_functions_map, statistics_collector: StatisticsCollector,
                 patterns_changed_threshold: float):
        super().__init__(patterns, tree_plan_builder, is_adaptivity_enabled, type_to_deviation_aware_functions_map,
                         statistics_collector)
        self.__connected_graph_to_pattern_ids_map = {}
        self.__pattern_id_to_connected_graph_map = {}
        self.__connected_graph_to_changed_patterns_counter = {}
        self.__recursive_traversal_tree_plan_merger = recursive_traversal_tree_plan_merger
        self.__changed_patterns = set()
        self.__pattern_to_tree_plan_map = None
        self.__pattern_id_to_pattern_map = None
        self.__max_changed_patterns_threshold = patterns_changed_threshold
        self.__connected_graph_to_known_unique_tree_plan_nodes = {}

    def try_optimize(self):
        changed_pattern_to_tree_plan_map = {}
        for pattern in self._patterns:
            if pattern.id not in self.__changed_patterns:
                new_statistics = self._statistics_collector.get_pattern_statistics(pattern)

                if self._should_optimize(new_statistics, pattern):
                    self.__optimize(pattern, new_statistics)

        for pattern_id in self.__changed_patterns:
            pattern = self.__pattern_id_to_pattern_map[pattern_id]
            changed_pattern_to_tree_plan_map[pattern] = self.__pattern_to_tree_plan_map[pattern]

        # prepares for the next optimization
        self.__changed_patterns = set()
        self.__init_connected_graph_maps()

        return changed_pattern_to_tree_plan_map

    def __optimize(self, pattern: Pattern, new_statistics: dict):
        """
        creates a new (single)tree for the pattern
        """

        if pattern.id in self.__changed_patterns:
            # if the tree that belongs to the pattern has changed then return
            return

        # check if violated maximum allowed patterns
        connected_graph_id = self.__pattern_id_to_connected_graph_map[pattern.id]
        if self.__should_rebuild_connected_graph(connected_graph_id):
            return

        self.__connected_graph_to_changed_patterns_counter[connected_graph_id] += 1

        # Before creating new trees, find intersections with other trees
        intersections_pattern_ids_before, known_unique_tree_plan_nodes = self.__find_intersections(pattern)

        self.__save_known_unique_tree_plan_nodes(known_unique_tree_plan_nodes, pattern)

        # Reconstruct the tree corresponding to the current single pattern
        tree_plan = self._build_new_plan(new_statistics, pattern)

        tree_plan_node = tree_plan.root
        still_merged = self.__merge(tree_plan_node, intersections_pattern_ids_before,
                                    known_unique_tree_plan_nodes)

        new_tree_plan = TreePlan(tree_plan_node)
        self.__propagate_pattern_id(pattern.id, new_tree_plan)
        self.__pattern_to_tree_plan_map[pattern] = new_tree_plan

        self.__changed_patterns.add(pattern.id)

        to_rebuild = intersections_pattern_ids_before - still_merged - self.__changed_patterns
        for pattern_id_to_rebuild in to_rebuild:
            pattern_to_rebuild = self.__pattern_id_to_pattern_map[pattern_id_to_rebuild]
            pattern_to_rebuild_statistics = self._statistics_collector.get_pattern_statistics(pattern_to_rebuild)
            self.__optimize(pattern_to_rebuild, pattern_to_rebuild_statistics)

    def __merge(self, tree_plan_node: TreePlanNode, intersection_pattern_ids_before: set,
                known_unique_tree_plan_nodes: dict):
        """
        Merge tree plan to nodes in known_unique_tree_plan_nodes dictionary.
        Returns patterns ids that merged with the tree plan.
        Currently, use the traverse_tree_plan_function to try and merge,
        it could be any merge algorithm.
        """
        still_merged = set()
        for intersection_pattern_id in intersection_pattern_ids_before:
            is_merged = [False]
            _ = self.__recursive_traversal_tree_plan_merger.traverse_tree_plan(tree_plan_node,
                                                                               known_unique_tree_plan_nodes[
                                                                                intersection_pattern_id],
                                                                               is_merged)
            if is_merged:
                still_merged.add(intersection_pattern_id)

        return still_merged

    def __should_rebuild_connected_graph(self, connected_graph_id: int):
        changed_patterns_num = self.__connected_graph_to_changed_patterns_counter[connected_graph_id]
        if changed_patterns_num > self.__max_changed_patterns_threshold * \
                len(self.__connected_graph_to_pattern_ids_map[connected_graph_id]):
            self.__rebuild_connected_graph(connected_graph_id)
            return True
        return False

    def __save_known_unique_tree_plan_nodes(self, known_unique_tree_plan_nodes: dict, pattern: Pattern):
        """
        Saves all the nodes of tree that were already built.
        """
        connected_graph = self.__pattern_id_to_connected_graph_map[pattern.id]
        self.__connected_graph_to_known_unique_tree_plan_nodes[connected_graph].union(known_unique_tree_plan_nodes)

    def __rebuild_connected_graph(self, connected_graph_id: int):
        """
        Rebuild entire connected graph
        """
        pattern_to_change_to_tree_plan_map = {}
        for pattern_id in self.__connected_graph_to_pattern_ids_map[connected_graph_id]:
            if pattern_id not in self.__changed_patterns:
                pattern = self.__pattern_id_to_pattern_map[pattern_id]
                new_statistics = self._statistics_collector.get_pattern_statistics(pattern)
                new_tree_plan = self._build_new_plan(new_statistics, pattern)
                pattern_to_change_to_tree_plan_map[pattern] = new_tree_plan
                self.__propagate_pattern_id(pattern_id, new_tree_plan)
                self.__changed_patterns.add(pattern_id)

        known_unique_tree_plan_nodes = self.__connected_graph_to_known_unique_tree_plan_nodes[connected_graph_id]
        new_pattern_to_tree_plan_merge = \
            self.__recursive_traversal_tree_plan_merger.merge_tree_plans(pattern_to_change_to_tree_plan_map,
                                                                         known_unique_tree_plan_nodes)

        for pattern, tree_plan in new_pattern_to_tree_plan_merge.items():
            self.__pattern_to_tree_plan_map[pattern] = tree_plan

    def __find_intersections(self, pattern: Pattern):
        """
        Find intersections of the tree plan correspond to pattern with other tree plans.
        """
        tree_plan = self.__pattern_to_tree_plan_map[pattern]
        root = tree_plan.root
        pattern_ids_intersections = set()
        known_unique_tree_plan_nodes = {}
        self.__find_intersections_aux(root, known_unique_tree_plan_nodes, pattern_ids_intersections, pattern.id)
        return pattern_ids_intersections, known_unique_tree_plan_nodes

    def __find_intersections_aux(self, current: TreePlanNode, known_unique_tree_plan_nodes: dict,
                                 pattern_ids_intersections: set, pattern_id: int):

        pattern_intersections = current.get_pattern_ids()

        self.__update_intersections(pattern_intersections, pattern_id, known_unique_tree_plan_nodes,
                                    pattern_ids_intersections, current)

        if isinstance(current, TreePlanLeafNode):
            return
        if isinstance(current, TreePlanNestedNode):
            self.__find_intersections_aux(current.sub_tree_plan, known_unique_tree_plan_nodes,
                                          pattern_ids_intersections, pattern_id)
            return
        if isinstance(current, TreePlanUnaryNode):
            self.__find_intersections_aux(current.child, known_unique_tree_plan_nodes,
                                          pattern_ids_intersections, pattern_id)
            return
        if isinstance(current, TreePlanBinaryNode):
            self.__find_intersections_aux(current.left_child, known_unique_tree_plan_nodes,
                                          pattern_ids_intersections, pattern_id)
            self.__find_intersections_aux(current.right_child, known_unique_tree_plan_nodes,
                                          pattern_ids_intersections, pattern_id)
            return
        raise Exception("Unexpected node type: %s" % (type(current),))

    def __update_intersections(self, pattern_intersections, pattern_id, known_unique_tree_plan_nodes,pattern_ids_intersections,
                               current):

        if pattern_intersections:
            # Use of remove so that we will be left with nodes that contain only patterns that didnt change
            pattern_intersections.remove(pattern_id)

            for pattern_intersection_id in pattern_intersections:
                if pattern_intersection_id not in known_unique_tree_plan_nodes:
                    known_unique_tree_plan_nodes[pattern_intersection_id] = set()
                known_unique_tree_plan_nodes[pattern_intersection_id].add(current)

                pattern_ids_intersections.add(pattern_intersection_id)

    def build_initial_pattern_to_tree_plan_map(self, patterns: list, cost_model_type):
        """
        Creates an initial mapping from patterns to tree plan
        """
        pattern_to_tree_plan_map = super().build_initial_pattern_to_tree_plan_map(patterns, cost_model_type)

        self.__pattern_to_tree_plan_map = \
            self.__recursive_traversal_tree_plan_merger.merge_tree_plans(pattern_to_tree_plan_map, set())

        self.__pattern_id_to_pattern_map = {}

        for pattern, tree_plan in self.__pattern_to_tree_plan_map.items():
            self.__pattern_id_to_pattern_map[pattern.id] = pattern
            self.__propagate_pattern_id(pattern.id, tree_plan)

        # For know the Related components in all graph
        self.__init_connected_graph_maps()

        return self.__pattern_to_tree_plan_map

    def __init_connected_graph_maps(self):
        pattern_id_to_neighbors_map = {}
        for pattern, tree_plan in self.__pattern_to_tree_plan_map.items():
            neighbors_ids = set()
            tree_plan_leaves = tree_plan.root.get_leaves()
            for leaf in tree_plan_leaves:
                leaf_patterns_ids = leaf.get_pattern_ids()
                neighbors_ids = neighbors_ids.union(leaf_patterns_ids)
            neighbors_ids.remove(pattern.id)
            pattern_id_to_neighbors_map[pattern.id] = neighbors_ids

        self.__connected_graph_to_pattern_ids_map = \
            self.__get_connected_graph_to_pattern_ids_map(pattern_id_to_neighbors_map)

        self.__connected_graph_to_changed_patterns_counter = {}
        self.__connected_graph_to_known_unique_tree_plan_nodes = {}
        for connected_graph in self.__connected_graph_to_pattern_ids_map.keys():
            self.__connected_graph_to_changed_patterns_counter[connected_graph] = 0
            self.__connected_graph_to_known_unique_tree_plan_nodes[connected_graph] = set()

    def __get_connected_graph_to_pattern_ids_map(self, pattern_id_to_neighbors_map: dict):
        """
        Return all connected graphs in the forrest
        """
        remaining_pattern_ids = set(pattern_id_to_neighbors_map.keys())
        connected_graph_id = 0
        connected_graph_to_pattern_ids_map = {}
        while remaining_pattern_ids:
            q = Queue()
            pattern_id = remaining_pattern_ids.pop()
            q.put(pattern_id)
            connected_graph_to_pattern_ids_map[connected_graph_id] = self.__bfs_like(pattern_id_to_neighbors_map, q,
                                                                                     connected_graph_id,
                                                                                     remaining_pattern_ids)

            connected_graph_id += 1

        return connected_graph_to_pattern_ids_map

    def __bfs_like(self, pattern_id_to_neighbors_map, q, connected_graph_id, remaining_pattern_ids):
        """
        Returns all pattern ids that belong to the connected graph
        """
        connected_graph = set()
        while not q.empty():
            pattern_id = q.get()
            self.__pattern_id_to_connected_graph_map[pattern_id] = connected_graph_id
            neighbors_pattern_id = pattern_id_to_neighbors_map[pattern_id]
            for neighbor_pattern_id in neighbors_pattern_id:
                if neighbor_pattern_id in remaining_pattern_ids:
                    q.put(neighbor_pattern_id)
                    remaining_pattern_ids.remove(neighbor_pattern_id)

            connected_graph.add(pattern_id)

        return connected_graph

    @staticmethod
    def __propagate_pattern_id(pattern_id: int, tree_plan: TreePlan):
        """
        Propagates the given pattern ID down the tree plan.
        """
        root = tree_plan.root
        root.propagate_pattern_id(pattern_id)
