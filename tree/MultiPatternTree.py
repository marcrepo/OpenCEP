from typing import Dict

from base.Pattern import Pattern
from plan.TreePlan import TreePlan
from tree.PatternMatchStorage import TreeStorageParameters
from base.PatternMatch import PatternMatch
from tree.Tree import Tree
from tree.nodes.NegationNode import NegationNode
from tree.nodes.Node import Node


class MultiPatternTree:
    """
    Represents a multi-pattern evaluation tree.
    """

    def __init__(self, pattern_to_tree_plan_map: Dict[Pattern, TreePlan],
                 storage_params: TreeStorageParameters,
                 statistics_collector):

        self.__statistics_collector = statistics_collector
        self.__all_patterns_ids = set()
        self.__id_to_pattern_map = {}

        for pattern in pattern_to_tree_plan_map.keys():
            self.__id_to_pattern_map[pattern.id] = pattern
            self.__all_patterns_ids.add(pattern.id)

        self.__id_to_output_node_map = {}
        self.__storage_params = storage_params
        self.__plan_nodes_to_nodes_map = {}  # a cache for already created subtrees

        self.__construct_multi_pattern_tree(pattern_to_tree_plan_map)

    def rebuild_multi_pattern_tree(self, pattern_to_tree_plan_map: Dict[Pattern, TreePlan]):
        changed_pattern = {pattern.id for pattern in pattern_to_tree_plan_map.keys()}
        not_changed_patterns_id = self.__all_patterns_ids - changed_pattern
        self.__plan_nodes_to_nodes_map = self.__update_plan_nodes_to_nodes_map(not_changed_patterns_id)
        self.__construct_multi_pattern_tree(pattern_to_tree_plan_map)
        # rebuild_multi_pattern_tree function invoke just in adaptivity mode, hence its ok to
        # set statistics collector to conditions
        self.__set_statistics_collector(not_changed_patterns_id)

    def __set_statistics_collector(self, not_changed_patterns_id):
        """
        Unlike single pattern, in multi pattern its important to set the same statistics collector to all patterns.
        Trees may contains old nodes and new nodes but they all should share the same(old) statistics collector.
        Therefore new nodes need to be set with the old statistics collector.
        """
        patterns_not_changed = {self.__id_to_pattern_map[pattern_id] for pattern_id in not_changed_patterns_id}
        for pattern in patterns_not_changed:
            condition = pattern.condition
            for atomic_condition in condition.extract_atomic_conditions():
                # set the statistics collector reference to every atomic condition
                atomic_condition.set_statistics_collector(self.__statistics_collector)

    def __construct_multi_pattern_tree(self, pattern_to_tree_plan_map: Dict[Pattern, TreePlan]):
        """
        Constructs a multi-pattern evaluation tree.
        It is assumed that each pattern appears only once in patterns (which is a legitimate assumption).
        """
        for pattern, tree_plan in pattern_to_tree_plan_map.items():
            new_tree_root = Tree(tree_plan, pattern, self.__storage_params, self.__plan_nodes_to_nodes_map).get_root()
            self.__id_to_output_node_map[pattern.id] = new_tree_root

    def __update_plan_nodes_to_nodes_map(self, not_changed_patterns_id):
        new_plan_nodes_to_nodes_map = {}
        for plan_node, node in self.__plan_nodes_to_nodes_map.items():
            for pattern_id in plan_node.get_pattern_ids():
                if pattern_id in not_changed_patterns_id:
                    new_plan_nodes_to_nodes_map[plan_node] = node
                    break
        return new_plan_nodes_to_nodes_map

    def get_leaves(self):
        """
        Returns all leaves in this multi-pattern-tree.
        """
        leaves = set()
        for output_node in self.__id_to_output_node_map.values():
            leaves |= set(output_node.get_leaves())
        return leaves

    def __should_attach_match_to_pattern(self, match: PatternMatch, pattern: Pattern):
        """
        Returns True if the given match satisfies the window/confidence constraints of the given pattern
        and False otherwise.
        """
        if match.last_timestamp - match.first_timestamp > pattern.window:
            return False
        return pattern.confidence is None or match.probability is None or match.probability >= pattern.confidence

    def get_matches_from_output_node(self,  output_node: Node):
        """
        Returns the matches from specific output nodes.
        """
        matches = []
        while output_node.has_unreported_matches():
            match = output_node.get_next_unreported_match()
            pattern_ids = output_node.get_pattern_ids()
            for pattern_id in pattern_ids:
                if self.__id_to_output_node_map[pattern_id] != output_node:
                    # the current output node is an internal node in pattern #idx, but not it's root.
                    # we don't need to check if there are any matches for this pattern id.
                    continue
                # check if timestamp is correct for this pattern id.
                # the pattern indices start from 1.
                if self.__should_attach_match_to_pattern(match, self.__id_to_pattern_map[pattern_id]):
                    match.add_pattern_id(pattern_id)
            matches.append(match)

        return matches

    def get_matches(self):
        """
        Returns the matches from all of the output nodes.
        """
        matches = []
        for output_node in self.__id_to_output_node_map.values():
            matches.extend(self.get_matches_from_output_node(output_node))
        return matches

    def get_last_matches(self):
        """
        This method is similar to the method- get_last_matches in a Tree.
        """
        for output_node in self.__id_to_output_node_map.values():
            if not isinstance(output_node, NegationNode):
                continue
                # this is the node that contains the pending matches
            first_unbounded_negative_node = output_node.get_first_unbounded_negative_node()
            if first_unbounded_negative_node is None:
                continue
            first_unbounded_negative_node.flush_pending_matches()
            # the pending matches were released and have hopefully reached the roots
        return self.get_matches()

    def get_output_nodes(self, patterns_ids: set):
        return [self.__id_to_output_node_map[pattern_id] for pattern_id in patterns_ids]
