import heapq
from datetime import datetime
from base.Event import Event
from stream.Stream import OutputStream
from tree.Tree import Tree
from tree.evaluation.SinglePatternTreeBasedEvaluationMechanism import SinglePatternTreeBasedEvaluationMechanism


class TrivialTreeBasedEvaluationMechanism(SinglePatternTreeBasedEvaluationMechanism):
    """
    Represent the trivial tree based evaluation mechanism.
    Whenever a new tree is given, replaces the old tree with the new one.
    """

    def _tree_update(self, new_tree: Tree, tree_update_time: datetime):
        """
        Directly replaces the old tree with the new tree.
        """
        leaves = self._tree.get_leaves()
        old_events = self._get_all_old_events(leaves)

        self._tree = new_tree

        self._event_types_listeners = self._register_event_listeners(new_tree)
        self._play_old_events_on_tree(old_events, self._event_types_listeners)

        # To avoid duplicate matches, flushes the matches from the new tree that have already been written.
        for _ in self._tree.get_matches():
            pass

    @staticmethod
    def _get_all_old_events(leaves):
        """
        get list of all old events that already played on the old tree
        """
        old_pattern_matches_events = []
        leaf_types = set()
        for leaf in leaves:
            leaf_type = leaf.get_event_type()
            if leaf_type not in leaf_types:
                leaf_types.add(leaf_type)
                partial_matches = leaf.get_storage_unit()
                old_pattern_matches_events.append([pm.events[0] for pm in partial_matches])

        # using heap for fast sort of sorted lists
        old_events = list(heapq.merge(*old_pattern_matches_events, key=lambda event: event.timestamp))
        return old_events

    def _play_new_event_on_tree_aux(self, event: Event, matches: OutputStream):
        self._play_new_event(event, self._event_types_listeners)

