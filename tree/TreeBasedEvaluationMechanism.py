from typing import Dict, Set
from base.DataFormatter import DataFormatter
from base.Event import Event
from misc.Statistics import calculate_selectivity_matrix
from plan.TreePlan import TreePlan
from stream.Stream import InputStream, OutputStream
from misc.Utils import *
from tree.nodes.LeafNode import LeafNode
from tree.PatternMatchStorage import TreeStorageParameters
from evaluation.EvaluationMechanism import EvaluationMechanism
from misc.ConsumptionPolicy import *
from plan.multi.MultiPatternEvaluationParameters import MultiPatternEvaluationParameters
from tree.MultiPatternTree import MultiPatternTree
from statistics_collector import StatisticsCollector
from tree.Tree import Tree
from datetime import timedelta
from optimizer import Optimizer


class TreeBasedEvaluationMechanism(EvaluationMechanism):
    """
    An implementation of the tree-based evaluation mechanism.
    """

    def __init__(self, pattern_to_tree_plan_map: Dict[Pattern, TreePlan],
                 storage_params: TreeStorageParameters,
                 statistics_collector: StatisticsCollector,
                 optimizer: Optimizer,
                 statistics_update_time_window: timedelta = timedelta(seconds=30),
                 multi_pattern_eval_params: MultiPatternEvaluationParameters = MultiPatternEvaluationParameters()):

        self.is_multi_pattern_mode = len(pattern_to_tree_plan_map) > 1
        if self.is_multi_pattern_mode:
            self._tree = MultiPatternTree(pattern_to_tree_plan_map, storage_params, multi_pattern_eval_params)
        else:
            self._tree = Tree(list(pattern_to_tree_plan_map.values())[0],
                              list(pattern_to_tree_plan_map)[0], storage_params)

        self.__storage_params = storage_params
        self.__statistics_collector = statistics_collector
        self.__optimizer = optimizer

        self._event_types_listeners = {}
        self.statistics_update_time_window = statistics_update_time_window

        # The remainder of the initialization process is only relevant for the freeze map feature. This feature can
        # only be enabled in single-pattern mode.
        self._pattern = list(pattern_to_tree_plan_map)[0] if not self.is_multi_pattern_mode else None
        self.__freeze_map = {}
        self.__active_freezers = []

        if not self.is_multi_pattern_mode and self._pattern.consumption_policy is not None and \
                self._pattern.consumption_policy.freeze_names is not None:
            self.__init_freeze_map()

    def eval(self, events: InputStream, matches: OutputStream, data_formatter: DataFormatter):
        """
        Activates the tree evaluation mechanism on the input event stream and reports all found pattern matches to the
        given output stream.
        """
        self._event_types_listeners = self.register_event_listeners(self._tree)
        statistics_update_start_time = datetime.now()
        is_first_time = True


        # selectivity = calculate_selectivity_matrix(self._pattern, events.duplicate())
        for raw_event in events:
            event = Event(raw_event, data_formatter)
            event.arrival_time = datetime.now()
            if event.type not in self._event_types_listeners.keys():
                continue
            self.__remove_expired_freezers(event)
            if not self.is_multi_pattern_mode:
                self.__statistics_collector.event_handler(event)
                if datetime.now() - statistics_update_start_time >= self.statistics_update_time_window or is_first_time:
                    is_first_time = False
                    if self.is_need_new_statistics():
                        new_statistics = self.__statistics_collector.get_statistics()
                        if self.__optimizer.is_need_optimize(new_statistics, self._pattern):
                            new_tree_plan = self.__optimizer.build_new_tree_plan(new_statistics, self._pattern)
                            new_tree = Tree(new_tree_plan, self._pattern, self.__storage_params)
                            self.tree_update(new_tree)
                            event.arrival_time = datetime.now()
                    # re-initialize statistics window start time
                    statistics_update_start_time = datetime.now()

            self.play_new_event_on_tree(event, matches)
            self.get_matches(matches)
        # Now that we finished the input stream, if there were some pending matches somewhere in the tree, we will
        # collect them now
        for match in self._tree.get_last_matches():
            matches.add_item(match)
        my_selectivity = self.__statistics_collector.get_statistics()
        matches.close()

    def play_new_event(self, event: Event, event_types_listeners):
        """
        Lets the tree handle the event
        """
        for leaf in event_types_listeners[event.type]:
            if self.__should_ignore_events_on_leaf(leaf, event_types_listeners):
                continue
            self.__try_register_freezer(event, leaf)
            leaf.handle_event(event)

    def get_matches(self, matches: OutputStream):
        for match in self._tree.get_matches():
            matches.add_item(match)
            self._remove_matched_freezers(match.events)

    def tree_update(self, new_tree: Tree):
        raise NotImplementedError("Must override")

    def is_need_new_statistics(self):
        raise NotImplementedError("Must override")

    def play_new_event_on_tree(self, event: Event, matches: OutputStream):
        raise NotImplementedError("Must override")

    @staticmethod
    def register_event_listeners(tree: Tree):
        """
        Given tree, register leaf listeners for event types.
        """
        event_types_listeners = {}
        for leaf in tree.get_leaves():
            event_type = leaf.get_event_type()
            if event_type in event_types_listeners.keys():
                event_types_listeners[event_type].append(leaf)
            else:
                event_types_listeners[event_type] = [leaf]

        return event_types_listeners

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

    def __should_ignore_events_on_leaf(self, leaf: LeafNode, event_types_listeners):
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

    def get_structure_summary(self):
        return self._tree.get_structure_summary()

    def __repr__(self):
        return self.get_structure_summary()


class TrivialEvaluation(TreeBasedEvaluationMechanism):

    def tree_update(self, new_tree: Tree):
        old_events = self.get_all_old_events()
        self._tree = new_tree
        self._event_types_listeners = self.register_event_listeners(new_tree)
        self.play_old_events_on_tree(old_events)
        """
        If there are new matches that have already been written 
        to a file then all we want is to avoid duplications.
        the method self._tree.get_matches() actually takes out the duplicate matches.
        """
        self._tree.get_matches()

    def get_all_old_events(self):
        old_events = []
        leaf_types = set()
        leaves = self._tree.get_leaves()
        for leaf in leaves:
            leaf_type = leaf.get_event_type()
            if leaf_type not in leaf_types:
                leaf_types.add(leaf_type)
                partial_matches = leaf.get_storage_unit()
                old_events.extend([pm.events[0] for pm in partial_matches])
        return old_events

    def play_old_events_on_tree(self, events):
        """
        These events dont need to ask about freeze
        """
        for event in events:
            for leaf in self._event_types_listeners[event.type]:
                leaf.handle_event(event)

    def is_need_new_statistics(self):
        """
        The Definition of trivial evaluation
        """
        return True

    def play_new_event_on_tree(self, event: Event, matches: OutputStream):
        self.play_new_event(event, self._event_types_listeners)


class SimultaneousEvaluation(TreeBasedEvaluationMechanism):

    def __init__(self, pattern_to_tree_plan_map: Dict[Pattern, TreePlan],
                 storage_params: TreeStorageParameters,
                 statistics_collector: StatisticsCollector,
                 optimizer: Optimizer,
                 statistics_update_time_window: timedelta,
                 multi_pattern_eval_params: MultiPatternEvaluationParameters = MultiPatternEvaluationParameters()):
        super().__init__(pattern_to_tree_plan_map, storage_params,
                         statistics_collector,
                         optimizer,
                         statistics_update_time_window,
                         multi_pattern_eval_params)
        self.new_tree = None
        self.new_event_types_listeners = None
        self.is_simultaneous_state = False
        self.tree_update_time = None

    def tree_update(self, new_tree: Tree):
        self.tree_update_time = datetime.now()
        self.new_tree = new_tree
        self.new_event_types_listeners = self.register_event_listeners(self.new_tree)
        self.is_simultaneous_state = True

    def is_need_new_statistics(self):
        return not self.is_simultaneous_state

    def play_new_event_on_tree(self, event: Event, matches: OutputStream):
        self.play_new_event(event, self._event_types_listeners)
        if self.is_simultaneous_state:
            self.play_new_event(event, self.new_event_types_listeners)
            """
            After this round we ask if we are in a simultaneous state. 
            If the time window is over then we want to return to a state that is not simultaneous, 
            i.e. a single tree
            """
            if datetime.now() - self.tree_update_time > self._pattern.window:
                self._tree, self.new_tree = self.new_tree, None
                self._event_types_listeners, self.new_event_types_listeners = self.new_event_types_listeners, None
                self.is_simultaneous_state = False

    def get_matches(self, matches: OutputStream):
        if self.is_simultaneous_state:
            for match in self._tree.get_matches():
                if not self.is_all_new_event(match):
                    matches.add_item(match)
                    self._remove_matched_freezers(match.events)
            for match in self.new_tree.get_matches():
                matches.add_item(match)
                self._remove_matched_freezers(match.events)
        else:
            super().get_matches(matches)

    def is_all_new_event(self, match: PatternMatch):
        """
        Checks whether all the events in the match are new
        """
        for event in match.events:
            if event.arrival_time < self.tree_update_time:
                return False
        return True
