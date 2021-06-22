from datetime import timedelta
from typing import Dict
from adaptive.optimizer.Optimizer import Optimizer
from adaptive.statistics.StatisticsCollector import StatisticsCollector
from base.Event import Event
from base.Pattern import Pattern
from plan.TreePlan import TreePlan
from stream.Stream import OutputStream
from tree.PatternMatchStorage import TreeStorageParameters
from tree.evaluation.TreeBasedEvaluationMechanism import TreeBasedEvaluationMechanism
from tree.nodes import Node


class MultiPatternTreeBasedEvaluationMechanism(TreeBasedEvaluationMechanism):

    def __init__(self, tree, pattern_to_tree_plan_map: Dict[Pattern, TreePlan],
                 storage_params: TreeStorageParameters,
                 statistics_collector: StatisticsCollector = None,
                 optimizer: Optimizer = None,
                 statistics_update_time_window: timedelta = None):
        super().__init__(tree, pattern_to_tree_plan_map, storage_params, statistics_collector,
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
    def _get_all_old_events(leaves):
        """
        get list of all old events that already played on the old tree
        Now, in contrast to trivial tree based evaluation mechanism, this method Works a bit differently.
        In multi pattern can be situation that same event inject into 2 trees and in arbitrary tree the time window is
        1 minute and on the other tree the time widnow is 2 minutes. So if we take just event from one leaf
        (for example, the first tree, and after that we ignore the leaf with the same type in the second tree we
        loose some events (the events occure after the 1 minute)
        """
        events = set()
        for leaf in leaves:
            partial_matches = leaf.get_storage_unit()
            events |= {pm.events[0] for pm in partial_matches}
        list_events = sorted(list(events), key=lambda event: event.timestamp)

        return list_events

    def _tree_update(self, changed_patterns_ids, old_leaves):
        """
        Find intersection of both new_tree and old tree.
        The intersection remains and updates the parts of the forest that are actually new
        """
        self._event_types_listeners = self._register_event_listeners(self._tree)

        not_changed_patterns = self.__all_patterns_ids - changed_patterns_ids

        output_nodes = self._tree.get_output_nodes(changed_patterns_ids)

        count = 0
        new_event_types_listeners = {}
        leaves_need_save_their_events = set()
        for output_node in output_nodes:
            leaves = output_node.get_leaves()
            for leaf in leaves:
                count += 1
                leaf_patterns_ids = leaf.get_pattern_ids()
                if len(not_changed_patterns.intersection(leaf_patterns_ids)) > 0:
                    # in this case its mean that the leaf belong to not change pattern
                    # Todo need to implement this feature
                    firs_node = self.__find_first_node_to_propagate(leaf, changed_patterns_ids, not_changed_patterns)
                else:
                    event_type = leaf.get_event_type()
                    leaves_need_save_their_events = leaves_need_save_their_events.union(
                        {old_leaf for old_leaf in old_leaves if old_leaf.get_event_type() == event_type})

                    self._register_leaf(leaf, new_event_types_listeners)

        old_events = self._get_all_old_events(leaves_need_save_their_events)
        self._play_old_events_on_tree(old_events, new_event_types_listeners)

        for _ in self._tree.get_matches():
            pass

    def __find_first_node_to_propagate(self, node: Node, changed_patterns_ids, not_changed_patterns):

        parents = node.get_parents()
        for parent in parents:
            parent_patterns_ids = parent.get_pattern_ids()
            if len(changed_patterns_ids.intersection(parent_patterns_ids)) > 0:
                if len(not_changed_patterns.intersection(parent_patterns_ids)) > 0:
                    self.__find_first_node_to_propagate(parent, changed_patterns_ids, not_changed_patterns)

        return node

    def _play_new_event_on_tree(self, event: Event, matches: OutputStream):
        self._play_new_event(event, self._event_types_listeners)

    def _play_new_event(self, event: Event, event_types_listeners):
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

