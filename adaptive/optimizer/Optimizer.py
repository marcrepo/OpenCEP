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
    def __init__(self, tree_plan_builder: TreePlanBuilder, is_adaptivity_enabled: bool, statistics_collector: StatisticsCollector):
        self._tree_plan_builder = tree_plan_builder
        self.__is_adaptivity_enabled = is_adaptivity_enabled
        self._statistics_collector = statistics_collector
        # self.count = 0

    @abstractmethod
    def try_optimize(self, pattern: Pattern):
        raise NotImplementedError()

    @abstractmethod
    def _should_optimize(self, new_statistics: dict, pattern: Pattern):
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

    def is_adaptivity_enabled(self):
        """
        Returns True if adaptive functionality is enabled and False otherwise.
        """
        return self.__is_adaptivity_enabled

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
        if pattern.prior_statistics_exist is False:
            if DefaultConfig.DEFAULT_INIT_TREE_PLAN_BUILDER == TreePlanBuilderTypes.TRIVIAL_LEFT_DEEP_TREE:
                non_prior_tree_plan_builder = TrivialLeftDeepTreeBuilder(cost_model_type,
                                                                         DefaultConfig.DEFAULT_NEGATION_ALGORITHM)
            else:
                raise Exception("Unknown tree plan builder type: %s" % (DefaultConfig.DEFAULT_INIT_TREE_PLAN_BUILDER,))
        return non_prior_tree_plan_builder


class TrivialOptimizer(Optimizer):
    """
    Represents the trivial optimizer that always initiates plan reconstruction ignoring the statistics.
    """

    def try_optimize(self, pattern: Pattern):
        new_statistics = self._statistics_collector.get_specific_statistics(pattern)
        if self._should_optimize(new_statistics, pattern):
            # self.count += 1
            new_tree_plan = self._build_new_plan(new_statistics, pattern)
            return new_tree_plan
        return None

    def _should_optimize(self, new_statistics: dict, pattern: Pattern):
        return True

    def _build_new_plan(self, new_statistics: dict, pattern: Pattern):
        tree_plan = self._tree_plan_builder.build_tree_plan(pattern, new_statistics)
        return tree_plan


class StatisticsDeviationAwareOptimizer(Optimizer):
    """
    Represents an optimizer that monitors statistics deviations from their latest observed values.
    """
    def __init__(self, tree_plan_builder: TreePlanBuilder, is_adaptivity_enabled: bool,
                 deviation_threshold, statistics_collector: StatisticsCollector):
        super().__init__(tree_plan_builder, is_adaptivity_enabled, statistics_collector)
        self.__prev_statistics = statistics_collector.get_all_statistics()
        # self.__prev_statistics = None
        self.__deviation_threshold = deviation_threshold

    def try_optimize(self, pattern: Pattern):
        new_statistics = self._statistics_collector.get_all_statistics()
        if self._should_optimize(new_statistics, pattern):
            # self.count += 1
            self.__prev_statistics = new_statistics
            statistics_for_builder = self._statistics_collector.get_specific_statistics(pattern)
            new_tree_plan = self._build_new_plan(statistics_for_builder, pattern)
            return new_tree_plan
        return None

    def _should_optimize(self, new_statistics: dict, pattern: Pattern):
        for new_stats_type, new_stats in new_statistics.items():
            prev_stats = self.__prev_statistics[new_stats_type]
            if self.__is_deviated_by_t(new_stats, prev_stats):
                return True
        return False

    def __is_deviated_by_t(self, new_statistics: dict, prev_statistics: dict):
        for item in new_statistics:
            if prev_statistics[item] * (1 + self.__deviation_threshold) < new_statistics[item] or \
                    prev_statistics[item] * (1 - self.__deviation_threshold) > new_statistics[item]:
                return True
        return False

    def _build_new_plan(self, new_statistics: dict, pattern: Pattern):
        tree_plan = self._tree_plan_builder.build_tree_plan(pattern, new_statistics)
        return tree_plan


class InvariantsAwareOptimizer(Optimizer):
    """
    Represents the invariant-aware optimizer. A reoptimization attempt is made when at least one of the precalculated
    invariants is violated.
    """
    def __init__(self, tree_plan_builder: TreePlanBuilder, is_adaptivity_enabled: bool, statistics_collector: StatisticsCollector):
        super().__init__(tree_plan_builder, is_adaptivity_enabled, statistics_collector)
        self._invariants = None

    def try_optimize(self, pattern: Pattern):
        new_statistics = self._statistics_collector.get_specific_statistics(pattern)
        if self._should_optimize(new_statistics, pattern):
            # self.count += 1
            new_tree_plan = self._build_new_plan(new_statistics, pattern)
            return new_tree_plan
        return None

    def _should_optimize(self, new_statistics: dict, pattern: Pattern):
        return self._invariants is None or self._invariants.is_invariants_violated(new_statistics, pattern)

    def _build_new_plan(self, new_statistics: dict, pattern: Pattern):
        tree_plan, self._invariants = self._tree_plan_builder.build_tree_plan(pattern, new_statistics)
        return tree_plan

    def build_initial_plan(self, new_statistics: dict, cost_model_type: TreeCostModels,
                           pattern: Pattern):
        non_prior_tree_plan_builder = self._build_non_prior_tree_plan_builder(cost_model_type, pattern)
        if non_prior_tree_plan_builder is not None:
            return non_prior_tree_plan_builder.build_tree_plan(pattern, new_statistics)
        return self._build_new_plan(new_statistics, pattern)
