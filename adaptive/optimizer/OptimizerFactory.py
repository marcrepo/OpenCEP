from datetime import timedelta

from adaptive.optimizer.StatisticsDeviationAwareOptimizer import StatisticsDeviationAwareOptimizer
from adaptive.optimizer.TrivialOptimizer import TrivialOptimizer
from adaptive.optimizer.InvariantsAwareOptimizer import InvariantsAwareOptimizer
from adaptive.optimizer.OptimizerTypes import OptimizerTypes
from adaptive.statistics.StatisticsCollector import StatisticsCollector
from adaptive.statistics.StatisticsCollectorFactory import StatisticsCollectorParameters
from misc import DefaultConfig
from adaptive.statistics.StatisticsTypes import StatisticsTypes
from adaptive.optimizer.MultiPatternStatisticsDeviationAwareOptimizer import \
    MultiPatternStatisticsDeviationAwareOptimizer
from adaptive.optimizer.DeviationAwareTesterFactory import DeviationAwareTesterFactory
from plan.invariant.InvariantTreePlanBuilder import InvariantTreePlanBuilder
from plan.TreePlanBuilderFactory import TreePlanBuilderParameters, TreePlanBuilderFactory
from plan.TreePlanBuilderTypes import TreePlanBuilderTypes
from plan.multi.TreePlanMergerFactory import TreePlanMergerFactory, TreePlanMergerParameters


class OptimizerParameters:
    """
    Parameters required for optimizer creation.
    """

    def __init__(self, opt_type: OptimizerTypes = DefaultConfig.DEFAULT_OPTIMIZER_TYPE,
                 tree_plan_params: TreePlanBuilderParameters = TreePlanBuilderParameters(),
                 statistics_collector_params: StatisticsCollectorParameters = StatisticsCollectorParameters(),
                 statistics_updates_wait_time: timedelta = DefaultConfig.STATISTICS_UPDATES_WAIT_TIME):
        self.type = opt_type
        self.tree_plan_params = tree_plan_params
        self.statistics_collector_params = statistics_collector_params
        self.statistics_updates_time_window = statistics_updates_wait_time  # None disabled any adaptive functionality


class TrivialOptimizerParameters(OptimizerParameters):
    """
    Parameters for the creation of the trivial optimizer class.
    """

    def __init__(self, tree_plan_params: TreePlanBuilderParameters = TreePlanBuilderParameters(),
                 statistics_collector_params: StatisticsCollectorParameters = StatisticsCollectorParameters(),
                 statistics_updates_wait_time: timedelta = DefaultConfig.STATISTICS_UPDATES_WAIT_TIME):
        super().__init__(OptimizerTypes.TRIVIAL_OPTIMIZER, tree_plan_params,
                         statistics_collector_params, statistics_updates_wait_time)


class StatisticsDeviationAwareOptimizerParameters(OptimizerParameters):
    """
    Parameters for the creation of StatisticDeviationAwareOptimizer class.
    """

    def __init__(self, tree_plan_params: TreePlanBuilderParameters = TreePlanBuilderParameters(),
                 statistics_collector_params: StatisticsCollectorParameters = StatisticsCollectorParameters(),
                 statistics_updates_wait_time: timedelta = DefaultConfig.STATISTICS_UPDATES_WAIT_TIME,
                 deviation_threshold: float = DefaultConfig.DEVIATION_OPTIMIZER_THRESHOLD,
                 is_multi_pattern: bool = DefaultConfig.IS_MULTI_PATTERN):
        super().__init__(OptimizerTypes.STATISTICS_DEVIATION_AWARE_OPTIMIZER, tree_plan_params,
                         statistics_collector_params, statistics_updates_wait_time)
        statistics_types = statistics_collector_params.statistics_types
        if isinstance(statistics_types, StatisticsTypes):
            statistics_types = [statistics_types]
        self.statistics_types = statistics_types
        self.deviation_threshold = deviation_threshold
        self.is_multi_pattern = is_multi_pattern


class MultiPatternStatisticsDeviationAwareOptimizerParameters(StatisticsDeviationAwareOptimizerParameters):
    """
    Parameters required for multi pattern optimizer creation.
    """

    def __init__(self, tree_merger_params: TreePlanMergerParameters = TreePlanMergerParameters(),
                 tree_plan_params: TreePlanBuilderParameters = TreePlanBuilderParameters(),
                 statistics_collector_params: StatisticsCollectorParameters = StatisticsCollectorParameters(),
                 statistics_updates_wait_time: timedelta = DefaultConfig.STATISTICS_UPDATES_WAIT_TIME,
                 deviation_threshold: float = DefaultConfig.DEVIATION_OPTIMIZER_THRESHOLD,
                 patterns_changed_threshold: float = DefaultConfig.PATTERNS_CHANGED_THRESHOLD):
        super().__init__(tree_plan_params, statistics_collector_params, statistics_updates_wait_time,
                         deviation_threshold, True)
        self.tree_merger_params = tree_merger_params
        self.patterns_changed_threshold = patterns_changed_threshold


class InvariantsAwareOptimizerParameters(OptimizerParameters):
    """
    Parameters for the creation of InvariantsAwareOptimizer class.
    """

    def __init__(self, tree_plan_params: TreePlanBuilderParameters = TreePlanBuilderParameters(
        TreePlanBuilderTypes.INVARIANT_AWARE_GREEDY_LEFT_DEEP_TREE),
                 statistics_collector_params: StatisticsCollectorParameters = StatisticsCollectorParameters(),
                 statistics_updates_wait_time: timedelta = DefaultConfig.STATISTICS_UPDATES_WAIT_TIME):
        super().__init__(OptimizerTypes.INVARIANT_AWARE_OPTIMIZER, tree_plan_params,
                         statistics_collector_params, statistics_updates_wait_time)


class OptimizerFactory:
    """
    Creates an optimizer given its specification.
    """

    @staticmethod
    def build_optimizer(optimizer_parameters: OptimizerParameters, statistics_collector: StatisticsCollector, patterns):
        if optimizer_parameters is None:
            optimizer_parameters = OptimizerFactory.__create_default_optimizer_parameters()
        return OptimizerFactory.__create_optimizer(optimizer_parameters, statistics_collector, patterns)

    @staticmethod
    def __create_optimizer(optimizer_parameters: OptimizerParameters, statistics_collector: StatisticsCollector,
                           patterns):
        tree_plan_builder = TreePlanBuilderFactory.create_tree_plan_builder(optimizer_parameters.tree_plan_params)
        is_adaptivity_enabled = optimizer_parameters.statistics_updates_time_window is not None
        if optimizer_parameters.type == OptimizerTypes.TRIVIAL_OPTIMIZER:
            return TrivialOptimizer(patterns, tree_plan_builder, is_adaptivity_enabled, statistics_collector)

        if optimizer_parameters.type == OptimizerTypes.STATISTICS_DEVIATION_AWARE_OPTIMIZER:
            deviation_threshold = optimizer_parameters.deviation_threshold
            is_multi_pattern = optimizer_parameters.is_multi_pattern

            type_to_deviation_aware_tester_map = {}
            for stat_type in optimizer_parameters.statistics_types:
                deviation_aware_tester = DeviationAwareTesterFactory.create_deviation_aware_tester(stat_type,
                                                                                                   deviation_threshold)
                type_to_deviation_aware_tester_map[stat_type] = deviation_aware_tester

            if is_multi_pattern:
                tree_plan_merger = TreePlanMergerFactory.create_tree_plan_merger(
                    optimizer_parameters.tree_merger_params)
                patterns_changed_threshold = optimizer_parameters.patterns_changed_threshold
                return MultiPatternStatisticsDeviationAwareOptimizer(patterns, tree_plan_merger, tree_plan_builder,
                                                                     is_adaptivity_enabled,
                                                                     type_to_deviation_aware_tester_map,
                                                                     statistics_collector,
                                                                     patterns_changed_threshold)
            else:
                return StatisticsDeviationAwareOptimizer(patterns, tree_plan_builder, is_adaptivity_enabled,
                                                         type_to_deviation_aware_tester_map,
                                                         statistics_collector)

        if optimizer_parameters.type == OptimizerTypes.INVARIANT_AWARE_OPTIMIZER:
            if isinstance(tree_plan_builder, InvariantTreePlanBuilder):
                return InvariantsAwareOptimizer(patterns, tree_plan_builder, is_adaptivity_enabled,
                                                statistics_collector)
            else:
                raise Exception("Tree plan builder must be invariant aware")

        raise Exception("Unknown optimizer type specified")

    @staticmethod
    def __create_default_optimizer_parameters():
        """
        Uses default configurations to create optimizer parameters.
        """
        if DefaultConfig.DEFAULT_OPTIMIZER_TYPE == OptimizerTypes.TRIVIAL_OPTIMIZER:
            return TrivialOptimizerParameters()
        if DefaultConfig.DEFAULT_OPTIMIZER_TYPE == OptimizerTypes.STATISTICS_DEVIATION_AWARE_OPTIMIZER:
            return StatisticsDeviationAwareOptimizerParameters()
        if DefaultConfig.DEFAULT_OPTIMIZER_TYPE == OptimizerTypes.INVARIANT_AWARE_OPTIMIZER:
            return InvariantsAwareOptimizerParameters()
        raise Exception("Unknown optimizer type: %s" % (DefaultConfig.DEFAULT_OPTIMIZER_TYPE,))
