from abc import ABC
from typing import List

from base.Pattern import Pattern
from misc.LegacyStatistics import MissingStatisticsException
from adaptive.statistics.StatisticsTypes import StatisticsTypes
from plan.TreeCostModels import TreeCostModels
from plan.TreePlan import TreePlanNode, TreePlanLeafNode, TreePlanNestedNode, TreePlanUnaryNode


class TreeCostModel(ABC):
    """
    An abstract class for the cost model used by cost-based tree-structured evaluation plan generation algorithms.
    """
    def get_plan_cost(self, pattern: Pattern, plan: TreePlanNode, statistics: dict):
        """
        Returns the cost of a given plan for a given pattern provided the relevant data characteristics (statistics).
        """
        raise NotImplementedError()


class IntermediateResultsTreeCostModel(TreeCostModel):
    """
    Calculates the plan cost based on the expected size of intermediate results (partial matches).
    Creates an invariant matrix for an arrival rates only case, so that we can still use it in the cost algorithms.
    """
    def get_plan_cost(self, pattern: Pattern, plan: TreePlanNode, statistics: dict):
        if StatisticsTypes.ARRIVAL_RATES not in statistics:
            raise MissingStatisticsException()
        arrival_rates = statistics[StatisticsTypes.ARRIVAL_RATES]
        if StatisticsTypes.SELECTIVITY_MATRIX in statistics:
            selectivity_matrix = statistics[StatisticsTypes.SELECTIVITY_MATRIX]
        else:
            selectivity_matrix = [[1.0 for x in range(len(arrival_rates))] for y in range(len(arrival_rates))]

        already_computed_costs = []
        _, _, cost = IntermediateResultsTreeCostModel.__get_plan_cost_aux(plan, selectivity_matrix,
                                                                          arrival_rates, pattern.window.total_seconds(),already_computed_costs)
        return cost - sum(already_computed_costs)

    @staticmethod
    def __get_plan_cost_aux(tree: TreePlanNode, selectivity_matrix: List[List[float]],
                            arrival_rates: List[int], time_window: float, already_computed_costs: List[float]):
        """
        A helper function for calculating the cost function of the given tree.
        If a subtree cost is already computed inorder of subtree sharing we wont calculate its cost again and recuce the
        cost of that node of the tree.
        """
        # calculate base case: tree is a leaf.

        if isinstance(tree, TreePlanLeafNode):
            if tree.cost_calculation_information:
                already_computed_costs.append(tree.cost_calculation_information[2])
                return tree.cost_calculation_information

            cost = pm = time_window * arrival_rates[tree.event_index] * \
                        selectivity_matrix[tree.event_index][tree.event_index]

            tree.cost_calculation_information = [tree.event_index], pm, cost
            return [tree.event_index], pm, cost

        if isinstance(tree, TreePlanNestedNode):
            if tree.cost_calculation_information:
                already_computed_costs.append(tree.cost_calculation_information[2])
                return tree.cost_calculation_information

            tree.cost_calculation_information = [tree.nested_event_index], tree.cost, tree.cost
            return [tree.nested_event_index], tree.cost, tree.cost

        if isinstance(tree, TreePlanUnaryNode):
            return IntermediateResultsTreeCostModel.__get_plan_cost_aux(tree.child,
                                                                        selectivity_matrix,
                                                                        arrival_rates,
                                                                        time_window, already_computed_costs)
        #binary node case
        if tree.cost_calculation_information:
            #for a binary tree node cost will be in index one
            already_computed_costs.append(tree.cost_calculation_information[1])
            #todo: how to return right args for computation

        # calculate for left subtree
        left_args, left_pm, left_cost = IntermediateResultsTreeCostModel.__get_plan_cost_aux(tree.left_child,
                                                                                             selectivity_matrix,
                                                                                             arrival_rates,
                                                                                             time_window,already_computed_costs)
        # calculate for right subtree
        right_args, right_pm, right_cost = IntermediateResultsTreeCostModel.__get_plan_cost_aux(tree.right_child,
                                                                                                selectivity_matrix,
                                                                                                arrival_rates,
                                                                                                time_window,already_computed_costs)
        # calculate from left and right subtrees for this subtree.
        pm = left_pm * right_pm
        for left_arg in left_args:
            for right_arg in right_args:
                pm *= selectivity_matrix[left_arg][right_arg]
        cost = left_cost + right_cost + pm

        tree.cost_calculation_information = pm, cost

        return left_args + right_args, pm, cost


class TreeCostModelFactory:
    """
    A factory for instantiating the cost model object.
    """
    @staticmethod
    def create_cost_model(cost_model_type: TreeCostModels):
        """
        Returns a cost model of the specified type.
        """
        if cost_model_type == TreeCostModels.INTERMEDIATE_RESULTS_TREE_COST_MODEL:
            return IntermediateResultsTreeCostModel()
        raise Exception("Unknown cost model type: %s" % (cost_model_type,))
