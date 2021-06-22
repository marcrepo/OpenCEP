from typing import Dict

from base.Pattern import Pattern
from plan.TreePlan import TreePlan
from tree.PatternMatchStorage import TreeStorageParameters
from base.PatternMatch import PatternMatch
from tree.Tree import Tree
from tree.nodes.NegationNode import NegationNode


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
        In oposite to single pattern, in multi pattern its importent to set the same statistics collector to all patterns.
        For some tree, he could be contains old nodes and new nodes. so we need the new nodes fave the same referance
        to statistics collector as the old nodes.
        """
        # set the statistics collector reference to every atomic condition
        patterns_not_changed = {self.__id_to_pattern_map[pattern_id] for pattern_id in not_changed_patterns_id}
        for pattern in patterns_not_changed:
            condition = pattern.condition
            for atomic_condition in condition.extract_atomic_conditions():
                atomic_condition.set_statistics_collector(self.__statistics_collector)

    def __construct_multi_pattern_tree(self, pattern_to_tree_plan_map: Dict[Pattern, TreePlan]):
        """
        Constructs a multi-pattern evaluation tree.
        It is assumed that each pattern appears only once in patterns (which is a legitimate assumption).
        """
        # self.f += 1

        # for pattern, tree_plan in pattern_to_tree_plan_map.items():
        #
        #     # self.propagate_pattern_id(pattern.id, tree_plan)
        #
        #     new_tree_root = Tree(tree_plan, pattern, self.__storage_params, self.__plan_nodes_to_nodes_map).get_root()
        #     # print(self.f)
        #     self.__id_to_output_node_map[pattern.id] = new_tree_root
        # print(self.f)

        c = {pattern.id: pattern for pattern in pattern_to_tree_plan_map.keys()}
        patterns_id = [pattern.id for pattern in pattern_to_tree_plan_map.keys()]
        a = sorted(patterns_id)
        b = [pattern_to_tree_plan_map[c[i]] for i in a]

        for i in range(len(a)):

            new_tree_root = Tree(b[i], c[a[i]], self.__storage_params, self.__plan_nodes_to_nodes_map).get_root()
            # print(self.f)
            self.__id_to_output_node_map[a[i]] = new_tree_root

    def __update_plan_nodes_to_nodes_map(self, not_changed_patterns_id):
        new_plan_nodes_to_nodes_map = {}
        for plan_node, node in self.__plan_nodes_to_nodes_map.items():
            for pattern_id in plan_node.get_pattern_ids():
                # if in plan_node there is at least one id its mean that this node
                # belong to some pattern that his tree doesnt changed
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

    def get_matches(self):
        """
        Returns the matches from all of the output nodes.
        """
        matches = []
        for output_node in self.__id_to_output_node_map.values():
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

    def get_specific_output_node(self, pattern: Pattern):
        return self.__id_to_output_node_map[pattern.id]

    def get_output_nodes(self, patterns_ids):
        return [self.__id_to_output_node_map[pattern_id] for pattern_id in patterns_ids]
