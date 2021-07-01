from abc import ABC, abstractmethod

from adaptive.statistics.StatisticsCollector import StatisticsCollector
from base import Pattern
from misc import DefaultConfig
from plan import TreePlanBuilder
from plan.LeftDeepTreeBuilders import TrivialLeftDeepTreeBuilder
from plan.TreeCostModels import TreeCostModels
from plan.TreePlanBuilderTypes import TreePlanBuilderTypes


class Optimizer(ABC):
    """
    The base class for the optimizers that decide when to invoke plan reconstruction.
    """

    def __init__(self, patterns, tree_plan_builder: TreePlanBuilder, is_adaptivity_enabled: bool,
                 statistics_collector: StatisticsCollector):
        self._tree_plan_builder = tree_plan_builder
        self.__is_adaptivity_enabled = is_adaptivity_enabled
        self._statistics_collector = statistics_collector
        self._patterns = patterns

    def try_optimize(self):
        new_statistics = self._statistics_collector.get_specific_statistics(self._patterns[0])
        if self._should_optimize(new_statistics, self._patterns[0]):
            new_tree_plan = self._build_new_plan(new_statistics, self._patterns[0])
            return new_tree_plan
        return None

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

    # def _build_new_plan(self, new_statistics: dict, pattern: Pattern):
    #     tree_plan = self._tree_plan_builder.build_tree_plan(pattern, new_statistics)
    #     return tree_plan

    def is_adaptivity_enabled(self):
        """
        Returns True if adaptive functionality is enabled and False otherwise.
        """
        return self.__is_adaptivity_enabled

    def build_initial_pattern_to_tree_plan_map(self, patterns: list, cost_model_type):
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
        if pattern.statistics is None:
            if DefaultConfig.DEFAULT_INIT_TREE_PLAN_BUILDER == TreePlanBuilderTypes.TRIVIAL_LEFT_DEEP_TREE:
                non_prior_tree_plan_builder = TrivialLeftDeepTreeBuilder(cost_model_type,
                                                                         DefaultConfig.DEFAULT_NEGATION_ALGORITHM)
            else:
                raise Exception("Unknown tree plan builder type: %s" % (DefaultConfig.DEFAULT_INIT_TREE_PLAN_BUILDER,))
        return non_prior_tree_plan_builder