
from misc.OptimizerTypes import OptimizerTypes
from misc import DefaultConfig
from misc.StatisticsTypes import StatisticsTypes
from optimizer import Optimizer
from plan.InvariantTreePlanBuilder import InvariantTreePlanBuilder
from plan.TreePlanBuilderFactory import TreePlanBuilderParameters, TreePlanBuilderFactory
from plan.TreePlanBuilderTypes import TreePlanBuilderTypes


class OptimizerParameters:
    """
    Parameters required for optimizer creation.
    """

    def __init__(self, opt_type: OptimizerTypes = DefaultConfig.DEFAULT_OPTIMIZER_TYPE,
                 tree_plan_params: TreePlanBuilderParameters = TreePlanBuilderParameters()):
        self.type = opt_type
        self.tree_plan_params = tree_plan_params


class TrivialOptimizerParameters(OptimizerParameters):
    """
    Parameters for the creation of the trivial optimizer class.
    """

    def __init__(self, tree_plan_params: TreePlanBuilderParameters):
        super().__init__(OptimizerTypes.TRIVIAL, tree_plan_params)


class StatisticChangesAwareOptimizerParameters(OptimizerParameters):
    """
    Parameters for the creation of StatisticChangesAwareOptimizer class.
    """

    def __init__(self, tree_plan_params: TreePlanBuilderParameters,
                 t: float, statistics_types: List[StatisticsTypes] or StatisticsTypes):
        super().__init__(OptimizerTypes.CHANGES_AWARE, tree_plan_params)
        if isinstance(statistics_types, StatisticsTypes):
            statistics_types = [statistics_types]
        self.t = t
        self.statistics_types = statistics_types


class InvariantsAwareOptimizerParameters(OptimizerParameters):
    """
    Parameters for the creation of InvariantsAwareOptimizer class.
    """

    def __init__(self, tree_plan_params: TreePlanBuilderParameters):
        super().__init__(OptimizerTypes.USING_INVARIANT, tree_plan_params)


class OptimizerFactory:
    """
    Creates an optimizer given its specification.
    """

    @staticmethod
    def build_optimizer(optimizer_parameters: OptimizerParameters):
        if optimizer_parameters is None:
            optimizer_parameters = OptimizerFactory.__create_default_optimizer_parameters()
        return OptimizerFactory.__create_optimizer(optimizer_parameters)

    @staticmethod
    def __create_optimizer(optimizer_parameters: OptimizerParameters):

        tree_plan_builder = TreePlanBuilderFactory.create_tree_plan_builder(optimizer_parameters.tree_plan_params)
        if optimizer_parameters.type == OptimizerTypes.TRIVIAL:
            return Optimizer.TrivialOptimizer(tree_plan_builder)

        if optimizer_parameters.type == OptimizerTypes.CHANGES_AWARE:
            type_to_changes_aware_functions_map = {}
            for stat_type in optimizer_parameters.statistics_types:
                if stat_type == StatisticsTypes.ARRIVAL_RATES:
                    type_to_changes_aware_functions_map[stat_type] = Optimizer.ArrivalRatesChangesAwareOptimizer()
                if stat_type == StatisticsTypes.SELECTIVITY_MATRIX:
                    type_to_changes_aware_functions_map[stat_type] = Optimizer.SelectivityChangesAwareOptimizer()
            return Optimizer.StatisticChangesAwareOptimizer(tree_plan_builder, optimizer_parameters.t)

        if optimizer_parameters.type == OptimizerTypes.USING_INVARIANT:
            if isinstance(tree_plan_builder, InvariantTreePlanBuilder):
                return Optimizer.InvariantsAwareOptimizer(tree_plan_builder)
            else:
                raise Exception("Tree plan builder must be invariant aware")

    @staticmethod
    def __create_default_optimizer_parameters():
        """
        Uses default configurations to create optimizer parameters.
        """
        if DefaultConfig.DEFAULT_OPTIMIZER_TYPE == OptimizerTypes.TRIVIAL:
            return OptimizerParameters()

        raise Exception("Unknown optimizer type: %s" % (DefaultConfig.DEFAULT_OPTIMIZER_TYPE,))
