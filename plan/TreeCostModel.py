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
    def get_plan_cost(self, pattern: Pattern, plan: TreePlanNode, statistics: dict, is_local_search=False, event_fixing_mapping=None):
        if StatisticsTypes.ARRIVAL_RATES not in statistics:
            raise MissingStatisticsException()
        arrival_rates = statistics[StatisticsTypes.ARRIVAL_RATES]
        if StatisticsTypes.SELECTIVITY_MATRIX in statistics:
            selectivity_matrix = statistics[StatisticsTypes.SELECTIVITY_MATRIX]
        else:
            selectivity_matrix = [[1.0 for x in range(len(arrival_rates))] for y in range(len(arrival_rates))]
        cost_to_remove = 0
        cost = 0
        if is_local_search:
             cost = IntermediateResultsTreeCostModel.__get_plan_cost_local_search_aux(plan, selectivity_matrix,
                                                                          arrival_rates, pattern.window.total_seconds(),
                                                                                                pattern, cost_to_remove,event_fixing_mapping)[2]
        else:
             cost = IntermediateResultsTreeCostModel.__get_plan_cost_aux(plan, selectivity_matrix,
                                                                                               arrival_rates,
                                                                                             pattern.window.total_seconds())[2]
        return cost-cost_to_remove


    @staticmethod
    def __get_plan_cost_aux(tree: TreePlanNode, selectivity_matrix: List[List[float]],
                            arrival_rates: List[int], time_window: float):
        """
        A helper function for calculating the cost function of the given tree.
        """
        # calculate base case: tree is a leaf.
        if isinstance(tree, TreePlanLeafNode):
            cost = pm = time_window * arrival_rates[tree.event_index] * \
                        selectivity_matrix[tree.event_index][tree.event_index]
            return [tree.event_index], pm, cost

        if isinstance(tree, TreePlanNestedNode):
            return [tree.nested_event_index], tree.cost, tree.cost

        if isinstance(tree, TreePlanUnaryNode):
            return IntermediateResultsTreeCostModel.__get_plan_cost_aux(tree.child,
                                                                        selectivity_matrix,
                                                                        arrival_rates,
                                                                        time_window)

        # calculate for left subtree
        left_args, left_pm, left_cost = IntermediateResultsTreeCostModel.__get_plan_cost_aux(tree.left_child,
                                                                                             selectivity_matrix,
                                                                                             arrival_rates,
                                                                                             time_window)
        # calculate for right subtree
        right_args, right_pm, right_cost = IntermediateResultsTreeCostModel.__get_plan_cost_aux(tree.right_child,
                                                                                                selectivity_matrix,
                                                                                                arrival_rates,
                                                                                                time_window)
        # calculate from left and right subtrees for this subtree.
        pm = left_pm * right_pm
        for left_arg in left_args:
            for right_arg in right_args:
                pm *= selectivity_matrix[left_arg][right_arg]
        cost = left_cost + right_cost + pm
        return left_args + right_args, pm, cost

    @staticmethod
    def __get_plan_cost_local_search_aux(tree: TreePlanNode, selectivity_matrix: List[List[float]],
                            arrival_rates: List[int], time_window: float, pattern, cost_to_remove, event_fixing_mapping):
        """
        A helper function for calculating the cost function of the given tree.
        """
        # calculate base case: tree is a leaf.
        if isinstance(tree, TreePlanLeafNode):
            event_index = tree.event_index
            if tree.is_shared:
                event_index = event_fixing_mapping[pattern][tree.event_name]

            if tree.cost:
                cost_to_remove += tree.cost
                return event_index, tree.cost, tree.cost

            tree.cost = time_window * arrival_rates[event_index] * \
                        selectivity_matrix[event_index][event_index]

            return event_index, tree.cost, tree.cost


        if isinstance(tree, TreePlanNestedNode):
            tree.sub_tree_plan.cost = tree.cost
            return IntermediateResultsTreeCostModel.__get_plan_cost_local_search_aux(tree.sub_tree_plan,
                                                                        selectivity_matrix,
                                                                        arrival_rates,
                                                                        time_window,pattern, cost_to_remove, event_fixing_mapping)

        if isinstance(tree, TreePlanUnaryNode):
            return IntermediateResultsTreeCostModel.__get_plan_cost_local_search_aux(tree.child,
                                                                        selectivity_matrix,
                                                                        arrival_rates,
                                                                        time_window,pattern, cost_to_remove, event_fixing_mapping)

        #binary tree case
        if tree.cost:
            pass

        #todo if binary is allready shared...


        # calculate for left subtree
        left_args, left_pm, left_cost = IntermediateResultsTreeCostModel.__get_plan_cost_local_search_aux(tree.left_child,
                                                                                             selectivity_matrix,
                                                                                             arrival_rates,
                                                                                             time_window,pattern, cost_to_remove, event_fixing_mapping)
        # calculate for right subtree
        right_args, right_pm, right_cost = IntermediateResultsTreeCostModel.__get_plan_cost_local_search_aux(tree.right_child,
                                                                                                selectivity_matrix,
                                                                                                arrival_rates,
                                                                                                time_window,pattern, cost_to_remove, event_fixing_mapping)
        # calculate from left and right subtrees for this subtree.
        pm = left_pm * right_pm
        for left_arg in left_args:
            for right_arg in right_args:
                pm *= selectivity_matrix[left_arg][right_arg]
        cost = left_cost + right_cost + pm
        return left_args + right_args, pm, cost

def get_real_nested_event_index(pattern, tree):
    pass

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
