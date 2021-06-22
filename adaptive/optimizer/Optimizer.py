import copy
from abc import ABC, abstractmethod
from queue import Queue
from typing import List

from adaptive.statistics.StatisticsCollector import StatisticsCollector
from base import Pattern
from misc import DefaultConfig
from plan import TreePlanBuilder
from plan.LeftDeepTreeBuilders import TrivialLeftDeepTreeBuilder
from plan.TreeCostModels import TreeCostModels
from plan.TreePlan import TreePlanBinaryNode, TreePlanUnaryNode, TreePlanNestedNode, TreePlanLeafNode, TreePlanNode, \
    TreePlan
from plan.TreePlanBuilderTypes import TreePlanBuilderTypes
from plan.multi.RecursiveTraversalTreePlanMerger import RecursiveTraversalTreePlanMerger
from tree.MultiPatternTree import MultiPatternTree


class Optimizer(ABC):
    """
    The base class for the optimizers that decide when to invoke plan reconstruction.
    """

    def __init__(self, patterns, tree_plan_builder: TreePlanBuilder, is_adaptivity_enabled: bool,
                 statistics_collector: StatisticsCollector):
        self._tree_plan_builder = tree_plan_builder
        self.__is_adaptivity_enabled = is_adaptivity_enabled
        self._statistics_collector = statistics_collector
        self.count = 0
        self._patterns = patterns

    @abstractmethod
    def try_optimize(self):
        raise NotImplementedError()

    @abstractmethod
    def _should_optimize(self, new_statistics: dict, pattern):
        """
        Returns True if it is necessary to attempt a reoptimization at this time, and False otherwise.
        """
        raise NotImplementedError()

    @abstractmethod
    def _build_new_plan(self, new_statistics: dict, pattern: Pattern):
        """
        Builds and returns a new evaluation plan based on the given statistics.
        """
        raise NotImplementedError()

    def is_adaptivity_enabled(self):
        """
        Returns True if adaptive functionality is enabled and False otherwise.
        """
        return self.__is_adaptivity_enabled

    def build_initial_pattern_to_tree_plan_map(self, patterns, cost_model_type):
        pattern_to_tree_plan_map = {
            pattern: self.build_initial_plan(self._statistics_collector.get_specific_statistics(pattern),
                                             cost_model_type, pattern)
            for pattern in patterns}

        return pattern_to_tree_plan_map

    def build_initial_plan(self, initial_statistics: dict, cost_model_type: TreeCostModels,
                           pattern: Pattern):
        """
        initializes the Statistic objects with initial statistics if such statistics exists,
        else, applies the default algorithm that does not require statistics.
        Note: right now only the TrivialLeftDeepTreeBuilder algorithm does not require statistics.
        """
        non_prior_tree_plan_builder = self._build_non_prior_tree_plan_builder(cost_model_type, pattern)
        if non_prior_tree_plan_builder is not None:
            self._tree_plan_builder, temp_tree_plan_builder = non_prior_tree_plan_builder, self._tree_plan_builder
            initial_tree_plan = self._build_new_plan(initial_statistics, pattern)
            self._tree_plan_builder = temp_tree_plan_builder
        else:
            initial_tree_plan = self._build_new_plan(initial_statistics, pattern)
        return initial_tree_plan

    @staticmethod
    def _build_non_prior_tree_plan_builder(cost_model_type: TreeCostModels, pattern: Pattern):
        """
        Attempts to create a tree builder for initializing the run. This only works when no a priori statistics are
        specified in the beginning of the run.
        """
        non_prior_tree_plan_builder = None
        if pattern.prior_statistics_exist is False:
            if DefaultConfig.DEFAULT_INIT_TREE_PLAN_BUILDER == TreePlanBuilderTypes.TRIVIAL_LEFT_DEEP_TREE:
                non_prior_tree_plan_builder = TrivialLeftDeepTreeBuilder(cost_model_type,
                                                                         DefaultConfig.DEFAULT_NEGATION_ALGORITHM)
            else:
                raise Exception("Unknown tree plan builder type: %s" % (DefaultConfig.DEFAULT_INIT_TREE_PLAN_BUILDER,))
        return non_prior_tree_plan_builder


class TrivialOptimizer(Optimizer):
    """
    Represents the trivial optimizer that always initiates plan reconstruction ignoring the statistics.
    """

    def try_optimize(self):
        new_statistics = self._statistics_collector.get_specific_statistics(self._patterns[0])
        if self._should_optimize(new_statistics, self._patterns[0]):
            self.count += 1
            new_tree_plan = self._build_new_plan(new_statistics, self._patterns[0])
            return new_tree_plan
        return None

    def _should_optimize(self, new_statistics: dict, pattern):
        return True

    def _build_new_plan(self, new_statistics: dict, pattern: Pattern):
        tree_plan = self._tree_plan_builder.build_tree_plan(pattern, new_statistics)
        return tree_plan


class StatisticsDeviationAwareOptimizer(Optimizer):
    """
    Represents an optimizer that monitors statistics deviations from their latest observed values.
    """

    def __init__(self, patterns, tree_plan_builder: TreePlanBuilder, is_adaptivity_enabled: bool,
                 type_to_deviation_aware_functions_map: dict, statistics_collector):
        super().__init__(patterns, tree_plan_builder, is_adaptivity_enabled, statistics_collector)
        self._prev_statistics = {pattern: statistics_collector.get_specific_statistics(pattern) for pattern in patterns}
        self.__type_to_deviation_aware_tester_map = type_to_deviation_aware_functions_map

    def try_optimize(self):
        new_statistics = self._statistics_collector.get_specific_statistics(self._patterns[0])
        if self._should_optimize(new_statistics, self._patterns[0]):
            new_tree_plan = self._build_new_plan(new_statistics, self._patterns[0])
            return new_tree_plan
        return None

    def _should_optimize(self, new_statistics: dict, pattern):
        prev_stats = self._prev_statistics[pattern]
        for new_stats_type, new_stats in new_statistics.items():
            if self.__type_to_deviation_aware_tester_map[new_stats_type].is_deviated_by_t(new_stats,
                                                                                          prev_stats[new_stats_type]):
                self._prev_statistics[pattern] = prev_stats
                return True
        return False

    def _build_new_plan(self, new_statistics: dict, pattern: Pattern):
        tree_plan = self._tree_plan_builder.build_tree_plan(pattern, new_statistics)
        return tree_plan


class MultiPatternStatisticsDeviationAwareOptimizer(StatisticsDeviationAwareOptimizer):

    def __init__(self, patterns, recursive_traversal_tree_plan_merger, tree_plan_builder: TreePlanBuilder,
                 is_adaptivity_enabled: bool,
                 type_to_deviation_aware_functions_map, statistics_collector: StatisticsCollector,
                 patterns_changed_threshold: float):
        super().__init__(patterns, tree_plan_builder, is_adaptivity_enabled, type_to_deviation_aware_functions_map,
                         statistics_collector)
        self.__tree = None
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

        self.__pattern_was_changed = set()

        self.__init_connected_graph_map()

        return changed_pattern_to_tree_plan_map

    def __optimize(self, pattern: Pattern, new_statistics):

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

    def __get_still_merged(self, tree_plan_node, pattern_ids_intersections, known_unique_tree_plan_nodes):
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

    def __should_rebuild_connected_graph(self, connected_graph_id):
        number_patterns_was_changed = self.__connected_graph_to_changed_patterns_number[connected_graph_id]
        if number_patterns_was_changed > self.__patterns_changed_threshold * \
                len(self.__connected_graph_to_pattern_ids_map[connected_graph_id]):
            self.__rebuild_connected_graph(connected_graph_id)
            return True
        return False

    def __save_known_unique_tree_plan_nodes(self, known_unique_tree_plan_nodes, pattern):
        """
        If we will pass the threshold and rebuild all graph so we want save all the nodes of trees that
        was already build
        """
        connected_graph = self.__pattern_id_to_connected_graph_map[pattern.id]
        self.__connected_graph_to_known_unique_tree_plan_nodes[connected_graph].union(known_unique_tree_plan_nodes)

    def __rebuild_connected_graph(self, connected_graph_id):
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

    def __find_intersections(self, pattern):
        tree_plan = self.__pattern_to_tree_plan_map[pattern]
        root = tree_plan.root
        pattern_ids_intersections = set()
        known_unique_tree_plan_nodes = {}
        self.__find_intersections_aux(root, known_unique_tree_plan_nodes, pattern_ids_intersections, pattern.id)
        return pattern_ids_intersections, known_unique_tree_plan_nodes

    def __find_intersections_aux(self, current: TreePlanNode, known_unique_tree_plan_nodes: dict,
                                 pattern_ids_intersections: set, pattern_id):
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

    def build_initial_pattern_to_tree_plan_map(self, patterns, cost_model_type):
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


class InvariantsAwareOptimizer(Optimizer):
    """
    Represents the invariant-aware optimizer. A reoptimization attempt is made when at least one of the precalculated
    invariants is violated.
    """

    def __init__(self, patterns, tree_plan_builder: TreePlanBuilder, is_adaptivity_enabled: bool,
                 statistics_collector: StatisticsCollector):
        super().__init__(patterns, tree_plan_builder, is_adaptivity_enabled, statistics_collector)
        self._invariants = None

    def try_optimize(self):
        new_statistics = self._statistics_collector.get_specific_statistics(self._patterns[0])
        if self._should_optimize(new_statistics, self._patterns[0]):
            new_tree_plan = self._build_new_plan(new_statistics, self._patterns[0])
            return new_tree_plan
        return None

    def _should_optimize(self, new_statistics: dict, pattern):
        return self._invariants is None or self._invariants.is_invariants_violated(new_statistics, self._patterns[0])

    def _build_new_plan(self, new_statistics: dict, pattern: Pattern):
        tree_plan, self._invariants = self._tree_plan_builder.build_tree_plan(pattern, new_statistics)
        return tree_plan

    def build_initial_plan(self, new_statistics: dict, cost_model_type: TreeCostModels,
                           pattern: Pattern):
        non_prior_tree_plan_builder = self._build_non_prior_tree_plan_builder(cost_model_type, pattern)
        if non_prior_tree_plan_builder is not None:
            return non_prior_tree_plan_builder.build_tree_plan(pattern, new_statistics)
        return self._build_new_plan(new_statistics, pattern)
