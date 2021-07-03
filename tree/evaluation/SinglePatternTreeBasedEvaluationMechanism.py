from abc import ABC
from stream.Stream import OutputStream
from tree.Tree import Tree
from datetime import timedelta
from typing import Dict, List
from adaptive.optimizer.Optimizer import Optimizer
from adaptive.statistics.StatisticsCollector import StatisticsCollector
from base.Event import Event
from base.Pattern import Pattern
from plan.TreePlan import TreePlan
from tree.PatternMatchStorage import TreeStorageParameters
from tree.evaluation.TreeBasedEvaluationMechanism import TreeBasedEvaluationMechanism
from tree.nodes.LeafNode import LeafNode


class SinglePatternTreeBasedEvaluationMechanism(TreeBasedEvaluationMechanism, ABC):

    def __init__(self, tree: Tree, pattern_to_tree_plan_map: Dict[Pattern, TreePlan],
                 storage_params: TreeStorageParameters,
                 statistics_collector: StatisticsCollector = None,
                 optimizer: Optimizer = None,
                 statistics_update_time_window: timedelta = None):
        super().__init__(tree, storage_params, statistics_collector,
                         optimizer, statistics_update_time_window)

        # The remainder of the initialization process is only relevant for the freeze map feature. This feature can
        # only be enabled in single-pattern mode.
        self._pattern = list(pattern_to_tree_plan_map)[0]
        self.__freeze_map = {}
        self.__active_freezers = []

        if self._pattern.consumption_policy is not None and \
                self._pattern.consumption_policy.freeze_names is not None:
            self.__init_freeze_map()

    def _reoptimize(self, last_event: Event):
        new_tree_plan = self._optimizer.try_optimize()

        if new_tree_plan is not None:
            new_tree = Tree(new_tree_plan, self._pattern, self._storage_params)
            self._tree_update(new_tree, last_event.timestamp)
            self.__update_statistics_collector(new_tree)

    def __update_statistics_collector(self, new_tree):
        """
        set the new statistics collector of the tree to be
        the statistics collector of the TreeBasedEvaluationMechanism
        """
        atomic_conditions = new_tree.get_root().get_condition().extract_atomic_conditions()
        if atomic_conditions:
            # extract the statistics collector from an arbitrary condition
            # because each condition stores a reference to the statistics collector
            first_atomic_conditions = atomic_conditions[0]
            self.__statistics_collector = first_atomic_conditions.get_statistics_collector()

    def _play_new_event(self, event: Event, event_types_listeners):
        """
        Lets the tree handle the event
        """
        for leaf in event_types_listeners[event.type]:
            if self._should_ignore_events_on_leaf(leaf, event_types_listeners):
                continue
            self.__try_register_freezer(event, leaf)
            leaf.handle_event(event)

    def __init_freeze_map(self):
        """
        For each event type specified by the user to be a 'freezer', that is, an event type whose appearance blocks
        initialization of new sequences until it is either matched or expires, this method calculates the list of
        leaves to be disabled.
        """
        sequences = self._pattern.extract_flat_sequences()
        for freezer_event_name in self._pattern.consumption_policy.freeze_names:
            current_event_name_set = set()
            for sequence in sequences:
                if freezer_event_name not in sequence:
                    continue
                for name in sequence:
                    current_event_name_set.add(name)
                    if name == freezer_event_name:
                        break
            if len(current_event_name_set) > 0:
                self.__freeze_map[freezer_event_name] = current_event_name_set

    def _should_ignore_events_on_leaf(self, leaf: LeafNode, event_types_listeners):
        """
        If the 'freeze' consumption policy is enabled, checks whether the given event should be dropped based on it.
        """
        if len(self.__freeze_map) == 0:
            # freeze option disabled
            return False
        for freezer in self.__active_freezers:
            for freezer_leaf in event_types_listeners[freezer.type]:
                if freezer_leaf.get_event_name() not in self.__freeze_map:
                    continue
                if leaf.get_event_name() in self.__freeze_map[freezer_leaf.get_event_name()]:
                    return True
        return False

    def __try_register_freezer(self, event: Event, leaf: LeafNode):
        """
        Check whether the current event is a freezer event, and, if positive, register it.
        """
        if leaf.get_event_name() in self.__freeze_map.keys():
            self.__active_freezers.append(event)

    def _remove_matched_freezers(self, match_events: List[Event]):
        """
        Removes the freezers that have been matched.
        """
        if len(self.__freeze_map) == 0:
            # freeze option disabled
            return False
        self.__active_freezers = [freezer for freezer in self.__active_freezers if freezer not in match_events]

    def __remove_expired_freezers(self, event: Event):
        """
        Removes the freezers that have been expired.
        """
        if len(self.__freeze_map) == 0:
            # freeze option disabled
            return False
        self.__active_freezers = [freezer for freezer in self.__active_freezers
                                  if event.timestamp - freezer.timestamp <= self._pattern.window]

    def _get_matches(self, matches: OutputStream):
        """
        Collects the ready matches from the tree and adds them to the evaluation matches.
        """
        for match in self._tree.get_matches():
            matches.add_item(match)
            self._remove_matched_freezers(match.events)

    def _play_new_event_on_tree(self, event: Event, matches: OutputStream):
        """
        Lets the tree handle the new event
        """
        self.__remove_expired_freezers(event)
        self._play_new_event_on_tree_aux(event, matches)

    def _play_new_event_on_tree_aux(self, event: Event, matches: OutputStream):
        raise NotImplementedError()
