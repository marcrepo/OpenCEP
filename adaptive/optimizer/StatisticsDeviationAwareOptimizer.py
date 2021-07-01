from typing import List

from adaptive.optimizer.Optimizer import Optimizer
from adaptive.statistics.StatisticsCollector import StatisticsCollector
from base.Pattern import Pattern
from plan.TreePlanBuilder import TreePlanBuilder


class StatisticsDeviationAwareOptimizer(Optimizer):
    """
    Represents an optimizer that monitors statistics deviations from their latest observed values.
    """

    def __init__(self, patterns: List[Pattern], tree_plan_builder: TreePlanBuilder, is_adaptivity_enabled: bool,
                 type_to_deviation_aware_functions_map: dict, statistics_collector: StatisticsCollector):
        super().__init__(patterns, tree_plan_builder, is_adaptivity_enabled, statistics_collector)
        self._prev_statistics = {pattern: statistics_collector.get_specific_statistics(pattern) for pattern in patterns}
        self.__type_to_deviation_aware_tester_map = type_to_deviation_aware_functions_map

    def _should_optimize(self, new_statistics: dict, pattern: Pattern):
        prev_stats = self._prev_statistics[pattern]
        for new_stats_type, new_stats in new_statistics.items():
            if self.__type_to_deviation_aware_tester_map[new_stats_type].is_deviated_by_t(new_stats,
                                                                                          prev_stats[new_stats_type]):
                return True
        return False

    def _build_new_plan(self, new_statistics: dict, pattern: Pattern):
        tree_plan = self._tree_plan_builder.build_tree_plan(pattern, new_statistics)
        self._prev_statistics[pattern] = new_statistics
        return tree_plan
