from datetime import timedelta
from typing import Dict, Set
from adaptive.optimizer.Optimizer import Optimizer
from adaptive.statistics.StatisticsCollector import StatisticsCollector
from base.Event import Event
from base.Pattern import Pattern
from plan.TreePlan import TreePlan
from stream.Stream import OutputStream
from tree.MultiPatternTree import MultiPatternTree
from tree.PatternMatchStorage import TreeStorageParameters
from tree.evaluation.TreeBasedEvaluationMechanism import TreeBasedEvaluationMechanism
from tree.nodes import Node
from tree.nodes.BinaryNode import BinaryNode
from tree.nodes.LeafNode import LeafNode
from tree.nodes.UnaryNode import UnaryNode


class MultiPatternTreeBasedEvaluationMechanism(TreeBasedEvaluationMechanism):

    def __init__(self, tree: MultiPatternTree, pattern_to_tree_plan_map: Dict[Pattern, TreePlan],
                 storage_params: TreeStorageParameters,
                 statistics_collector: StatisticsCollector = None,
                 optimizer: Optimizer = None,
                 statistics_update_time_window: timedelta = None):
        super().__init__(tree, storage_params, statistics_collector,
                         optimizer, statistics_update_time_window)
        self.__all_patterns_ids = {pattern.id for pattern in pattern_to_tree_plan_map.keys()}

    def _reoptimize(self, last_event: Event):
        changed_pattern_to_new_tree_plan_map = self._optimizer.try_optimize()
        if changed_pattern_to_new_tree_plan_map:
            changed_patterns_ids = {pattern.id for pattern in changed_pattern_to_new_tree_plan_map.keys()}
            old_leaves = self._tree.get_leaves()

            self._tree.rebuild_multi_pattern_tree(changed_pattern_to_new_tree_plan_map)
            self._tree_update(changed_patterns_ids, old_leaves)

    @staticmethod
    def _get_all_old_events(leaves: set):
        """
        Get a list of all relevant old events that were played on the old tree.
        """
        events = set()
        for leaf in leaves:
            partial_matches = leaf.get_storage_unit()
            events |= {pm.events[0] for pm in partial_matches}
        list_events = sorted(list(events), key=lambda event: event.timestamp)

        return list_events

    def _tree_update(self, changed_patterns_ids: Set[int], old_leaves: Set[LeafNode]):
        """
        Find the intersection of the new_tree and the old tree.
        The intersection remains and updates the parts of the forest that are actually new
        """
        self._event_types_listeners = self._register_event_listeners(self._tree)

        changed_output_nodes = self._tree.get_output_nodes(changed_patterns_ids)
        self.__propagate_relevant_data(changed_output_nodes, changed_patterns_ids, old_leaves)

        self.__flush_duplicate_matches(changed_output_nodes)

    def __propagate_relevant_data(self, changed_output_nodes: list, changed_patterns_ids: Set[int], old_leaves: Set[LeafNode]):
        """
        Propagates events and partial matches on changed (single)trees
        """
        not_changed_patterns = self.__all_patterns_ids - changed_patterns_ids

        detected_nodes = set()
        leaves_to_save = set()
        new_event_types_listeners = {}

        for changed_output_node in changed_output_nodes:
            self.__propagate_existing_partial_matches(changed_output_node, not_changed_patterns, detected_nodes,
                                                      leaves_to_save, new_event_types_listeners)

        self.__propagate_events_from_saved_leaves(leaves_to_save, old_leaves, new_event_types_listeners)

    def __propagate_events_from_saved_leaves(self, leaves_to_save: Set[LeafNode], old_leaves: Set[LeafNode],
                                             new_event_types_listeners: dict):
        old_event_types = {old_leaf.get_event_type() for old_leaf in leaves_to_save}
        old_relevant_leaves = {leaf for leaf in old_leaves if leaf.get_event_type() in old_event_types}
        old_events = self._get_all_old_events(old_relevant_leaves)

        self._play_old_events_on_tree(old_events, new_event_types_listeners)

    def __flush_duplicate_matches(self, changed_output_nodes: list):
        """
        To avoid duplicate matches, flushes the matches from the new tree that have already been written.
        """
        for changed_output_node in changed_output_nodes:
            for _ in self._tree.get_matches_from_output_node(changed_output_node):
                pass

    def _play_new_event_on_tree(self, event: Event, matches: OutputStream):
        self._play_new_event(event, self._event_types_listeners)

    def _play_new_event(self, event: Event, event_types_listeners: dict):
        """
        Lets the tree handle the event
        """
        for leaf in event_types_listeners[event.type]:
            leaf.handle_event(event)

    def _get_matches(self, matches: OutputStream):
        """
        Collects the ready matches from the tree and adds them to the evaluation matches.
        """
        for match in self._tree.get_matches():
            matches.add_item(match)

    def __propagate_existing_partial_matches(self, node: Node, unchanged_patterns: set, detected_nodes: set,
                                             leaves_to_save: set,
                                             new_event_types_listeners: dict):
        """
        Propagates the existent partial matches and saves the relevant leaves.
        relevant leaves are leaves that have old events that should be played later on the new tree.
        """
        if node in detected_nodes:
            return

        detected_nodes.add(node)

        patterns_ids = node.get_pattern_ids()
        if unchanged_patterns.intersection(patterns_ids):
            # the node belongs to at least one unchanged (single)tree
            for parent in node.get_parents():
                parent_ids = parent.get_pattern_ids()
                if not unchanged_patterns.intersection(parent_ids):
                    # the parent node does not belong to any unchanged (single)trees
                    partial_matches = node.get_partial_matches()
                    for pm in partial_matches:
                        node.propagate_partial_matches_to_parent(parent, pm)
            return

        if isinstance(node, LeafNode):
            self._register_leaf(node, new_event_types_listeners)
            leaves_to_save.add(node)
            return

        if isinstance(node, UnaryNode):
            self.__propagate_existing_partial_matches(node.get_child(), unchanged_patterns, detected_nodes,
                                                      leaves_to_save,
                                                      new_event_types_listeners)

        if isinstance(node, BinaryNode):
            self.__propagate_existing_partial_matches(node.get_left_subtree(), unchanged_patterns, detected_nodes,
                                                      leaves_to_save,
                                                      new_event_types_listeners)
            self.__propagate_existing_partial_matches(node.get_right_subtree(), unchanged_patterns, detected_nodes,
                                                      leaves_to_save,
                                                      new_event_types_listeners)
