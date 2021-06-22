from abc import ABC, abstractmethod
from typing import Dict
from base.DataFormatter import DataFormatter
from base.Event import Event
from plan.TreePlan import TreePlan
from stream.Stream import InputStream, OutputStream
from misc.Utils import *
from tree.PatternMatchStorage import TreeStorageParameters
from evaluation.EvaluationMechanism import EvaluationMechanism
from tree.MultiPatternTree import MultiPatternTree
from adaptive.statistics import StatisticsCollector
from tree.Tree import Tree
from datetime import timedelta
from adaptive.optimizer import Optimizer


class TreeBasedEvaluationMechanism(EvaluationMechanism, ABC):
    """
    An implementation of the tree-based evaluation mechanism.
    """

    def __init__(self, tree, pattern_to_tree_plan_map: Dict[Pattern, TreePlan],
                 storage_params: TreeStorageParameters,
                 statistics_collector: StatisticsCollector = None,
                 optimizer: Optimizer = None,
                 statistics_update_time_window: timedelta = None):

        self._tree = tree
        self._storage_params = storage_params
        self._event_types_listeners = {}

        # Adaptive
        self._statistics_collector = statistics_collector
        self._optimizer = optimizer
        self.__statistics_update_time_window = statistics_update_time_window

    def eval(self, events: InputStream, matches: OutputStream, data_formatter: DataFormatter):
        """
        Activates the tree evaluation mechanism on the input event stream and reports all found pattern matches to the
        given output stream.
        """
        self._event_types_listeners = self._register_event_listeners(self._tree)
        last_statistics_refresh_time = None

        for raw_event in events:
            event = Event(raw_event, data_formatter)
            if event.type not in self._event_types_listeners.keys():
                continue

            if self._statistics_collector is not None:
                last_statistics_refresh_time = self.__perform_reoptimization(last_statistics_refresh_time, event)

            self._play_new_event_on_tree(event, matches)
            self._get_matches(matches)

        # Now that we finished the input stream, if there were some pending matches somewhere in the tree, we will
        # collect them now
        self._get_last_pending_matches(matches)
        matches.close()

    def __perform_reoptimization(self, last_statistics_refresh_time: timedelta, last_event: Event):
        """
        If needed, reoptimizes the evaluation mechanism to reflect the current statistical properties of the
        input event stream.
        """
        self._statistics_collector.handle_event(last_event)
        if not self._should_try_reoptimize(last_statistics_refresh_time, last_event):
            # it is not yet time to recalculate the statistics
            return last_statistics_refresh_time

        self._reoptimize(last_event)

        return last_event.timestamp

    @abstractmethod
    def _reoptimize(self, last_event: Event):
        raise NotImplementedError()

    def _should_try_reoptimize(self, last_statistics_refresh_time: timedelta, last_event: Event):
        """
        Returns True if statistic recalculation and a reoptimization attempt can now be performed and False otherwise.
        The default implementation merely checks whether enough time has passed since the last reoptimization attempt.
        """
        if last_statistics_refresh_time is None:
            return True
        return last_event.timestamp - last_statistics_refresh_time > self.__statistics_update_time_window

    def _get_last_pending_matches(self, matches):
        """
        Collects the pending matches from the tree
        """
        for match in self._tree.get_last_matches():
            matches.add_item(match)

    def _get_matches(self, matches: OutputStream):
        """
        Collects the ready matches from the tree and adds them to the evaluation matches.
        """
        raise NotImplementedError()

    def _register_event_listeners(self, tree: Tree or MultiPatternTree):
        """
        Given tree, register leaf listeners for event types.
        """
        event_types_listeners = {}
        for leaf in tree.get_leaves():
            self._register_leaf(leaf, event_types_listeners)
        return event_types_listeners

    @staticmethod
    def _register_leaf(leaf, event_types_listeners):
        event_type = leaf.get_event_type()
        if event_type in event_types_listeners.keys():
            event_types_listeners[event_type].append(leaf)
        else:
            event_types_listeners[event_type] = [leaf]

    def get_structure_summary(self):
        return self._tree.get_structure_summary()

    def __repr__(self):
        return self.get_structure_summary()

    def _play_new_event_on_tree(self, event: Event, matches: OutputStream):
        """
        Lets the tree handle the event.
        """
        raise NotImplementedError()

    @staticmethod
    def _play_old_events_on_tree(events, event_types_listeners):
        """
        These events dont need to ask about freeze
        """
        for event in events:
            for leaf in event_types_listeners[event.type]:
                leaf.handle_event(event)

    def _tree_update(self, new_tree: Tree or None, event: Event):
        """
        Registers a new tree in the evaluation mechanism.
        """
        raise NotImplementedError()
