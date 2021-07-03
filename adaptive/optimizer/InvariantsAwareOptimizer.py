from adaptive.optimizer.Optimizer import Optimizer
from adaptive.statistics.StatisticsCollector import StatisticsCollector
from base.Pattern import Pattern
from plan.TreeCostModels import TreeCostModels
from plan.TreePlanBuilder import TreePlanBuilder


class InvariantsAwareOptimizer(Optimizer):
    """
    Represents the invariant-aware optimizer. A reoptimization attempt is made when at least one of the precalculated
    invariants is violated.
    """

    def __init__(self, patterns, tree_plan_builder: TreePlanBuilder, is_adaptivity_enabled: bool,
                 statistics_collector: StatisticsCollector):
        super().__init__(patterns, tree_plan_builder, is_adaptivity_enabled, statistics_collector)
        self._invariants = None

    def _should_optimize(self, new_statistics: dict, pattern: Pattern):
        return self._invariants is None or self._invariants.is_invariants_violated(new_statistics, self._patterns[0])

    def _build_new_plan(self, new_statistics: dict, pattern: Pattern):
        tree_plan, self._invariants = self._tree_plan_builder.build_tree_plan(pattern, new_statistics)
        return tree_plan

    def build_initial_plan(self, new_statistics: dict, cost_model_type: TreeCostModels,
                           pattern: Pattern):
        non_prior_tree_plan_builder = self._build_non_prior_tree_plan_builder(cost_model_type, pattern)
        if non_prior_tree_plan_builder is not None:
            return non_prior_tree_plan_builder.build_tree_plan(pattern, new_statistics)
        return self._build_new_plan(new_statistics, pattern)
