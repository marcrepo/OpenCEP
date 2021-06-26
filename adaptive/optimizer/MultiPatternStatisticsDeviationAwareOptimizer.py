from queue import Queue
from typing import List

from adaptive.optimizer.StatisticsDeviationAwareOptimizer import StatisticsDeviationAwareOptimizer
from adaptive.statistics.StatisticsCollector import StatisticsCollector
from base.Pattern import Pattern
from plan.TreePlan import TreePlan, TreePlanNode, TreePlanLeafNode, TreePlanNestedNode, TreePlanUnaryNode, \
    TreePlanBinaryNode
from plan.TreePlanBuilder import TreePlanBuilder


class MultiPatternStatisticsDeviationAwareOptimizer(StatisticsDeviationAwareOptimizer):

    def __init__(self, patterns: List[Pattern], recursive_traversal_tree_plan_merger, tree_plan_builder: TreePlanBuilder,
                 is_adaptivity_enabled: bool,
                 type_to_deviation_aware_functions_map, statistics_collector: StatisticsCollector,
                 patterns_changed_threshold: float):
        super().__init__(patterns, tree_plan_builder, is_adaptivity_enabled, type_to_deviation_aware_functions_map,
                         statistics_collector)
        self.__connected_graph_to_pattern_ids_map = {}
        self.__pattern_id_to_connected_graph_map = {}
        self.__connected_graph_to_changed_patterns_number = {}
        self.__recursive_traversal_tree_plan_merger = recursive_traversal_tree_plan_merger
        self.__pattern_was_changed = set()
        self.__pattern_to_tree_plan_map = None
        self.__pattern_id_to_pattern_map = None
        self.__patterns_changed_threshold = patterns_changed_threshold
        self.__connected_graph_to_known_unique_tree_plan_nodes = {}

    def try_optimize(self):
        changed_pattern_to_tree_plan_map = {}
        for pattern in self._patterns:
            if pattern.id not in self.__pattern_was_changed:
                new_statistics = self._statistics_collector.get_specific_statistics(pattern)

                if self._should_optimize(new_statistics, pattern):
                    self.__optimize(pattern, new_statistics)

        for pattern_id in self.__pattern_was_changed:
            pattern = self.__pattern_id_to_pattern_map[pattern_id]
            changed_pattern_to_tree_plan_map[pattern] = self.__pattern_to_tree_plan_map[pattern]

        # initialize for next time
        self.__pattern_was_changed = set()
        self.__init_connected_graph_map()

        return changed_pattern_to_tree_plan_map

    def __optimize(self, pattern: Pattern, new_statistics: dict):

        if pattern.id in self.__pattern_was_changed:
            return

        connected_graph_id = self.__pattern_id_to_connected_graph_map[pattern.id]
        if self.__should_rebuild_connected_graph(connected_graph_id):
            return

        self.__connected_graph_to_changed_patterns_number[connected_graph_id] += 1
        # First, before create new trees, we want to find all intersection of each
        # single pattern (which need to change) with other single patterns
        pattern_ids_intersections, known_unique_tree_plan_nodes = self.__find_intersections(pattern)

        self.__save_known_unique_tree_plan_nodes(known_unique_tree_plan_nodes, pattern)

        # Now, we can reconstruct the tree corresponding to the current single pattern
        tree_plan = self._build_new_plan(new_statistics, pattern)

        # check if the trees was intersect this tree before reconstruct are still intersect the new tree
        tree_plan_node = tree_plan.root

        still_merge_with = self.__get_still_merged(tree_plan_node, pattern_ids_intersections,
                                                   known_unique_tree_plan_nodes)

        new_tree_plan = TreePlan(tree_plan_node)
        self.propagate_pattern_id(pattern.id, new_tree_plan)
        self.__pattern_to_tree_plan_map[pattern] = new_tree_plan

        self.__pattern_was_changed.add(pattern.id)

        needs_rebuild = pattern_ids_intersections - still_merge_with - self.__pattern_was_changed
        for pattern_need_rebuild_id in needs_rebuild:
            pattern_need_rebuild = self.__pattern_id_to_pattern_map[pattern_need_rebuild_id]
            new_need_statistics = self._statistics_collector.get_specific_statistics(pattern_need_rebuild)
            self.__optimize(pattern_need_rebuild, new_need_statistics)

    def __get_still_merged(self, tree_plan_node: TreePlanNode, pattern_ids_intersections: set,
                           known_unique_tree_plan_nodes: dict):
        still_merge_with = set()
        for pattern_intersection_id in pattern_ids_intersections:
            is_merged = [False]
            tree_plan_node = self.__recursive_traversal_tree_plan_merger.traverse_tree_plan(tree_plan_node,
                                                                                            known_unique_tree_plan_nodes[
                                                                                                pattern_intersection_id],
                                                                                            is_merged)
            if is_merged:
                still_merge_with.add(pattern_intersection_id)

        return still_merge_with

    def __should_rebuild_connected_graph(self, connected_graph_id: int):
        number_patterns_was_changed = self.__connected_graph_to_changed_patterns_number[connected_graph_id]
        if number_patterns_was_changed > self.__patterns_changed_threshold * \
                len(self.__connected_graph_to_pattern_ids_map[connected_graph_id]):
            self.__rebuild_connected_graph(connected_graph_id)
            return True
        return False

    def __save_known_unique_tree_plan_nodes(self, known_unique_tree_plan_nodes: dict, pattern: Pattern):
        """
        If we will pass the threshold and rebuild all graph so we want save all the nodes of trees that
        was already build
        """
        connected_graph = self.__pattern_id_to_connected_graph_map[pattern.id]
        self.__connected_graph_to_known_unique_tree_plan_nodes[connected_graph].union(known_unique_tree_plan_nodes)

    def __rebuild_connected_graph(self, connected_graph_id: int):
        need_changed_pattern_to_tree_plan_map = {}
        for pattern_id in self.__connected_graph_to_pattern_ids_map[connected_graph_id]:
            if pattern_id not in self.__pattern_was_changed:
                pattern = self.__pattern_id_to_pattern_map[pattern_id]
                new_statistics = self._statistics_collector.get_specific_statistics(pattern)
                new_tree_plan = self._build_new_plan(new_statistics, pattern)
                need_changed_pattern_to_tree_plan_map[pattern] = new_tree_plan
                self.propagate_pattern_id(pattern_id, new_tree_plan)
                self.__pattern_was_changed.add(pattern_id)

        known_unique_tree_plan_nodes = self.__connected_graph_to_known_unique_tree_plan_nodes[connected_graph_id]
        new_pattern_to_tree_plan_merge = \
            self.__recursive_traversal_tree_plan_merger.merge_tree_plans(need_changed_pattern_to_tree_plan_map,
                                                                         known_unique_tree_plan_nodes)

        for pattern, tree_plan in new_pattern_to_tree_plan_merge.items():
            self.__pattern_to_tree_plan_map[pattern] = tree_plan

    def __find_intersections(self, pattern: Pattern):
        tree_plan = self.__pattern_to_tree_plan_map[pattern]
        root = tree_plan.root
        pattern_ids_intersections = set()
        known_unique_tree_plan_nodes = {}
        self.__find_intersections_aux(root, known_unique_tree_plan_nodes, pattern_ids_intersections, pattern.id)
        return pattern_ids_intersections, known_unique_tree_plan_nodes

    def __find_intersections_aux(self, current: TreePlanNode, known_unique_tree_plan_nodes: dict,
                                 pattern_ids_intersections: set, pattern_id: int):
        """
        Recursively traverses a tree plan and attempts to merge it with previously traversed subtrees.
        """
        pattern_intersections = current.get_pattern_ids()
        # Because this remove, in the end of the day every node contain just id patterns that
        # dont changed.

        if pattern_intersections:
            pattern_intersections.remove(pattern_id)

            for pattern_intersection_id in pattern_intersections:
                if pattern_intersection_id not in known_unique_tree_plan_nodes:
                    known_unique_tree_plan_nodes[pattern_intersection_id] = set()
                known_unique_tree_plan_nodes[pattern_intersection_id].add(current)

            for pattern_intersection in pattern_intersections:
                if pattern_intersection not in pattern_ids_intersections:
                    pattern_ids_intersections.add(pattern_intersection)

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

    def build_initial_pattern_to_tree_plan_map(self, patterns: list, cost_model_type):
        pattern_to_tree_plan_map = super().build_initial_pattern_to_tree_plan_map(patterns, cost_model_type)

        self.__pattern_to_tree_plan_map = \
            self.__recursive_traversal_tree_plan_merger.merge_tree_plans(pattern_to_tree_plan_map, set())

        self.__pattern_id_to_pattern_map = {}

        for pattern, tree_plan in self.__pattern_to_tree_plan_map.items():
            self.__pattern_id_to_pattern_map[pattern.id] = pattern
            self.propagate_pattern_id(pattern.id, tree_plan)

        # For know the Related components in all graph
        self.__init_connected_graph_map()

        return self.__pattern_to_tree_plan_map

    def __init_connected_graph_map(self):
        pattern_id_to_neighbors_map = {}
        for pattern, tree_plan in self.__pattern_to_tree_plan_map.items():
            neighbors_ids = set()
            tree_plan_leaves = tree_plan.root.get_leaves()
            for leaf in tree_plan_leaves:
                leaf_patterns_ids = leaf.get_pattern_ids()
                neighbors_ids = neighbors_ids.union(leaf_patterns_ids)
            neighbors_ids.remove(pattern.id)
            pattern_id_to_neighbors_map[pattern.id] = neighbors_ids

        self.__connected_graph_to_pattern_ids_map = self.__find_connected_graphs(pattern_id_to_neighbors_map)
        self.__connected_graph_to_changed_patterns_number = {connected_graph: 0 for connected_graph in
                                                             self.__connected_graph_to_pattern_ids_map.keys()}

        for connected_graph in self.__connected_graph_to_pattern_ids_map.keys():
            self.__connected_graph_to_changed_patterns_number[connected_graph] = 0
            self.__connected_graph_to_known_unique_tree_plan_nodes[connected_graph] = set()

    def __find_connected_graphs(self, pattern_id_to_neighbors_map: dict):
        patterns_ids_remains = set(pattern_id_to_neighbors_map.keys())
        connected_graph_id = 0
        connected_graphs = {}
        while patterns_ids_remains:
            q = Queue()
            pattern_id = patterns_ids_remains.pop()
            q.put(pattern_id)
            connected_graphs[connected_graph_id] = self.__bfs_like(pattern_id_to_neighbors_map, q, connected_graph_id,
                                                                   patterns_ids_remains)

            connected_graph_id += 1

        return connected_graphs

    def __bfs_like(self, pattern_id_to_neighbors_map, q, connected_graph_id, patterns_ids_remains):
        connected_graph = set()
        while not q.empty():
            pattern_id = q.get()
            self.__pattern_id_to_connected_graph_map[pattern_id] = connected_graph_id
            pattern_ids_neighbors = pattern_id_to_neighbors_map[pattern_id]
            for pattern_id_neighbor in pattern_ids_neighbors:
                if pattern_id_neighbor in patterns_ids_remains:
                    q.put(pattern_id_neighbor)
                    patterns_ids_remains.remove(pattern_id_neighbor)

            connected_graph.add(pattern_id)

        return connected_graph

    @staticmethod
    def propagate_pattern_id(pattern_id: int, tree_plan: TreePlan):
        root = tree_plan.root
        root.propagate_pattern_id(pattern_id)
