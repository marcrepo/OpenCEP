import copy

from plan.TreePlan import TreePlanNode
from plan.multi.SubTreeSharingTreePlanMerger import SubTreeSharingTreePlanMerger
from plan.multi.RecursiveTraversalTreePlanMerger import RecursiveTraversalTreePlanMerger
from datetime import timedelta, datetime
from typing import List, Dict
from plan.multi.MultiPatternTreePlanMergeApproaches import MultiPatternTreePlanMergeApproaches
from adaptive.optimizer.Optimizer import Optimizer
from adaptive.statistics.StatisticsCollector import StatisticsCollector
from base.Pattern import Pattern
from evaluation.EvaluationMechanismTypes import EvaluationMechanismTypes
from misc import DefaultConfig
from tree.evaluation.TreeEvaluationMechanismUpdateTypes import TreeEvaluationMechanismUpdateTypes
from adaptive.optimizer.OptimizerFactory import OptimizerParameters, OptimizerFactory, \
    StatisticsDeviationAwareOptimizerParameters
from adaptive.statistics.StatisticsCollectorFactory import StatisticsCollectorFactory
from plan.multi.ShareLeavesTreePlanMerger import ShareLeavesTreePlanMerger
from plan.multi.SubTreeSharingTreePlanMerger import SubTreeSharingTreePlanMerger
from plan.TreePlan import TreePlan
from plan.TreePlanBuilderFactory import TreePlanBuilderFactory
from plan.multi.MultiPatternTreePlanMergeApproaches import MultiPatternTreePlanMergeApproaches
from tree.PatternMatchStorage import TreeStorageParameters
from tree.evaluation.SimultaneousTreeBasedEvaluationMechanism import SimultaneousTreeBasedEvaluationMechanism
from tree.evaluation.TrivialTreeBasedEvaluationMechnism import TrivialTreeBasedEvaluationMechanism
from condition.CompositeCondition import *
from base.PatternStructure import *
from condition.BaseRelationCondition import GreaterThanCondition, SmallerThanCondition, GreaterThanEqCondition, \
    SmallerThanEqCondition
from condition.CompositeCondition import AndCondition
from condition.Condition import Variable
from datetime import timedelta
from typing import List, Dict

from adaptive.optimizer.Optimizer import Optimizer
from adaptive.statistics.StatisticsCollector import StatisticsCollector
from base.Pattern import Pattern
from evaluation.EvaluationMechanismTypes import EvaluationMechanismTypes
from misc import DefaultConfig
from tree.evaluation.TreeEvaluationMechanismUpdateTypes import TreeEvaluationMechanismUpdateTypes
from adaptive.optimizer.OptimizerFactory import OptimizerParameters, OptimizerFactory, \
    StatisticsDeviationAwareOptimizerParameters
from adaptive.statistics.StatisticsCollectorFactory import StatisticsCollectorFactory
from plan.multi.ShareLeavesTreePlanMerger import ShareLeavesTreePlanMerger
from plan.multi.SubTreeSharingTreePlanMerger import SubTreeSharingTreePlanMerger
from plan.TreePlan import TreePlan
from plan.TreePlanBuilderFactory import TreePlanBuilderFactory
from plan.multi.MultiPatternTreePlanMergeApproaches import MultiPatternTreePlanMergeApproaches
from tree.PatternMatchStorage import TreeStorageParameters
from tree.evaluation.SimultaneousTreeBasedEvaluationMechanism import SimultaneousTreeBasedEvaluationMechanism
from tree.evaluation.TrivialTreeBasedEvaluationMechnism import TrivialTreeBasedEvaluationMechanism

from datetime import timedelta

from adaptive.optimizer.OptimizerFactory import OptimizerParameters
from adaptive.optimizer.OptimizerTypes import OptimizerTypes
from base.Pattern import Pattern
from base.PatternStructure import AndOperator, SeqOperator, PrimitiveEventStructure, NegationOperator
from condition.BaseRelationCondition import GreaterThanCondition, SmallerThanCondition, GreaterThanEqCondition, \
    SmallerThanEqCondition
from condition.CompositeCondition import AndCondition
from condition.Condition import Variable
from plan.multi.MultiPatternTreePlanMergeApproaches import MultiPatternTreePlanMergeApproaches
from test.testUtils import *
from tree.nodes.SeqNode import *
from plan.TreeCostModel import *
import random



class MPG:
    def __init__(self, patterns: Pattern or List[Pattern]):
        """
        This is the class representing a mapping between a patterns and there maximal common sub patterns
        and  a mapping between maximal common sub pattern to the patterns which contains it.
        """
        self.patterns = patterns
        self.mcs_to_patterns = {}
        self.pattern_to_various_mcs = {p: set() for p in patterns}
        self.__create_mpg(patterns if isinstance(patterns, List) else list(patterns))

    def __create_mpg(self, patterns):
        """i"""
        for p1 in range(len(patterns)):
            for p2 in range(p1 + 1, len(patterns)):
                maximal_common_subpatterns = self.__find_maximal_common_subpatterns(p1, p2)
                # todo: make every mcs in the list not empty
                for mcs in maximal_common_subpatterns:
                    if mcs == frozenset():
                        break
                    self.pattern_to_various_mcs[patterns[p1]].add(mcs)
                    self.pattern_to_various_mcs[patterns[p2]].add(mcs)
                    if mcs in self.mcs_to_patterns.keys():
                        self.mcs_to_patterns[mcs].add(p1)
                        self.mcs_to_patterns[mcs].add(p2)
                    else:
                        self.mcs_to_patterns[mcs] = {p1, p2}

    def __find_maximal_common_subpatterns(self, p1_idx: int, p2_idx: int) -> Pattern:
        """
        p1_idx, p2_idx: indices in patterns list
        :return:  maximal common subpatterns(mcs) of pattern1 and pattern2
        notes:
        The next are cases of mcs that we will not keep in the mpg and use later in our local search,
        because subtree merger already merges them. When subtree merger merges mcs the pattern has an optimal tree plan,
        and when we share via local search method it has not.
        Thus we won't use the below cases of mcs in our local search, in order to get both sharing and tree plan optimality:
        1) Different operators: Seq(a,b) , And(a,b)
        2) Same operator with one argument: Seq(a,....), seq(a.....) => mcs= seq(a)
        3) One pattern is a nested argument of the other.
        In addition we won't include nested subpattern in our mcs for the same reason above.
        """
        p1 = self.patterns[p1_idx]
        p2 = self.patterns[p2_idx]

        if type(p1.full_structure) is not type(p2.full_structure):
            return None

        return self.__find_maximal_common_subpatterns_helper(p1, p2)


    def find_one_mcs_per_intersection(self, events_intersection, p1, p2):
        events_intersection_names_set = set()
        for event in events_intersection:
            events_intersection_names_set = events_intersection_names_set.union(event.get_all_event_names())

        p1_event_conditions_filtered_by_intersection = p1.condition.get_condition_of(events_intersection_names_set,
                                                                                     get_kleene_closure_conditions=False,
                                                                                     consume_returned_conditions=False).get_conditions_list()
        p2_event_conditions_filtered_by_intersection = p2.condition.get_condition_of(events_intersection_names_set,
                                                                                     get_kleene_closure_conditions=False,
                                                                                     consume_returned_conditions=False).get_conditions_list()

        p1_filtered_only_conditions = [condition for condition in p1_event_conditions_filtered_by_intersection
                                       if condition not in p2_event_conditions_filtered_by_intersection]
        p2_filtered_only_conditions = [condition for condition in p2_event_conditions_filtered_by_intersection
                                       if condition not in p1_event_conditions_filtered_by_intersection]

        p1_filtered_only_conditions_names = set()
        for condition in p1_filtered_only_conditions:
            p1_filtered_only_conditions_names = p1_filtered_only_conditions_names.union(condition.get_event_names())

        p2_filtered_only_conditions_names = set()
        for condition in p2_filtered_only_conditions:
            p2_filtered_only_conditions_names = p2_filtered_only_conditions_names.union(condition.get_event_names())

        events_intersection_after_condition_filtering = copy.deepcopy(events_intersection)
        for event in events_intersection:
            for name in event.get_all_event_names():
                if name in p1_filtered_only_conditions_names or name in p2_filtered_only_conditions_names:
                    events_intersection_after_condition_filtering.remove(event)
                    break

        return frozenset(events_intersection_after_condition_filtering)

    def __find_maximal_common_subpatterns_helper(self, p1: Pattern, p2: Pattern):
        """
        :return: maximal common subpattern for 2 equal operators
        """
        result = []
        events_intersections = p1.full_structure.calc_structure_intersections(p2.full_structure)
        if events_intersections is None:
            return result
        for events_intersection in events_intersections:
            if events_intersection is not None:
                mcs = self.find_one_mcs_per_intersection(events_intersection,p1,p2)
                if len(mcs) > 1:
                    result.append(mcs)
        return result







MAX_NEIGHBOURS_TO_DEVELOP = 1


class LocalSearchTreePlanMerger:
    """
    Merges the given patterns tree by sharing equivalent subtrees found via local search.
    We do a local search for lowest cost pattern to tree plan map
    We starting from initial state that is the pattern to tree plan map after a regular subtree sharing merger run.
    Documentation for how we developing a new neighbour(other pattern to tree plan map) is presented under n-vertex function.
    You can use between two local search heuristics: Tabu-L and Simulated Annealing by give the correct parameters
    to runTest function (see Examples in test).
    """

    def __init__(self, patterns: Pattern or List[Pattern],
                 pattern_to_tree_plan_map: Dict[Pattern, TreePlan],
                 local_search_params: List, initial_statistics, cost_model_type, optimizer):
        self.start_date = datetime.now()
        if (len(local_search_params) < 2):
            raise Exception("local search: not enough parameters")
        self.initial_statistics = initial_statistics
        self.__cost_model = TreeCostModelFactory.create_cost_model(cost_model_type)
        self.cost_model_type = cost_model_type
        #here the different mcs actually calculated
        self.mpg = MPG(patterns)
        self.patterns = patterns
        self.original_tree_plan_event_name_to_event_index_mapping_list = [
                                                        {leaf.event_name: leaf.event_index for leaf in
                                                        tree_plan.root.get_leaves()}
                                                        for pattern, tree_plan in pattern_to_tree_plan_map.items()
                                                        ]
        # a list such that in place i there is a dictionary between pattern i nested args to its real nested indices in pattern
        self.combined_args_to_real_index_map_list = self.init_combined_args_to_real_index_map_list()
        self.sub_tree_sharer = SubTreeSharingTreePlanMerger()
        #the initial pattern to tree plan map of local search is the one merged by subtree sharing
        self.pattern_to_tree_plan_map = self.sub_tree_sharer.merge_tree_plans(pattern_to_tree_plan_map)
        self.optimizer = optimizer
        #we use the data structure below to fix event inidices in case of top operator is Seq
        #Because we always give maximal common subpattern to tree builder algorithems as nested subpattern and therefore
        #cahnging it original indices.

        #self.pattern_to_tree_plan_cost_map = self.set_pattern_to_tree_plan_cost_map(pattern_to_tree_plan_map)
        #initialization of initial solution
        self.initial_solution = self.Solution({}, self.pattern_to_tree_plan_map)
        self.__set_and_init_heuristic(local_search_params[0])
        self.time_limit = self.__set_time_limit(local_search_params[1])


    def init_combined_args_to_real_index_map_list(self):
        combined_list = []
        for i in range(len(self.patterns)):
            pattern = self.patterns[i]
            pattern_dict = {}
            for j, arg in enumerate(pattern.full_structure.args):
                if isinstance(arg, PrimitiveEventStructure):
                    pattern_dict[arg.name] = j
                else:
                    pattern_dict[frozenset(arg.args)] = j
            combined_list.append(pattern_dict)
        return combined_list

    def set_pattern_to_tree_plan_cost_map(self, pattern_to_tree_plan_map):
        """
        Returns a dictionary between pattern to its initial cost before local search starts to run.
        """
        #todo: implement the cost function in a way that if p0 and p1 sharing mcs a than the cost of the p0 will include
        #todo: the mcs and of p1 won't - to write a good decomentaion for that
        dict = {}
        for idx, pair in enumerate(pattern_to_tree_plan_map.items()):
            pattern = pair[0]
            tree_plan = pair[1]
            dict[pattern]=self.__cost_model.get_plan_cost(pattern, tree_plan.root, pattern.statistics, is_local_search=True,
                                                    event_fixing_mapping= self.combined_args_to_real_index_map_list, pattern_idx= idx)

        return dict

    def __set_and_init_heuristic(self, heuristic):
        if heuristic == "TabuSearch":
            self.merge_tree_plans = self.tabu_search_merge_tree_plans
            return

        if heuristic == "SimulatedAnnealing":
            self.merge_tree_plans = self.simulated_annealing_merge_tree_plans()
            return

        raise Exception("local search: Unknown Heuristic")

    def __set_time_limit(self, time_limit):
        if not isinstance(time_limit, timedelta):
            raise Exception("local search: time_delta is not instance of timeDelta class")
        return time_limit

    """------------Functions for user defined local search heuristics------------------------------------------------"""

    class Solution:
        def __init__(self, mcs_to_patterns_sharing, pattern_to_tree_plan_map):
            # a hash between each mcs that is shared inside the solution to the patterns that are sharing it in the solution
            self.mcs_to_patterns_sharing = mcs_to_patterns_sharing
            self.pattern_to_tree_plan_map = pattern_to_tree_plan_map

    def score(self, sol: Solution):
        cost = 0
        for idx, pair in enumerate(sol.pattern_to_tree_plan_map.items()):
            pattern = pair[0]
            tree_plan = pair[1]
            cost += self.__cost_model.get_plan_cost(pattern, tree_plan.root, pattern.statistics,
                                                    is_local_search=True,
                                                    event_fixing_mapping=self.combined_args_to_real_index_map_list,
                                                    pattern_idx=idx)
        return cost


    def find_mcs_to_patterns_that_dont_share_it_in_solution_mapping(self, mcs_to_pattern_dict):
        """
        Returns a dictionary of mcs to patterns that contains them, a key,value of mcs:list[p1,p2,p3....]
        iff for each pattern in the list its tree plan in the current solution does not share mcs
        (we do not share the mcs in this pattern).
        """
        difference_dict = {}
        for mcs, patterns_mcs in self.mpg.mcs_to_patterns.items():
            if mcs not in mcs_to_pattern_dict.keys():
                difference_dict[mcs] = self.mpg.mcs_to_patterns[mcs]
            else:
                patterns_difference = self.mpg.mcs_to_patterns[mcs] - mcs_to_pattern_dict[mcs]
                if len(patterns_difference) > 0:
                    difference_dict[mcs] = patterns_difference
        return difference_dict

    def neighbourhood(self, cur_sol:Solution):
        """
        An implementation to n-vertex neighborhood fucntion presented in article.
        returns a new pattern to tree plan map which created in that way:...............
        """
        num_of_developed_neighbours = 0
        neighbours = []
        #a dictionary that keeps all left possible sharing options that all the neighbors in the neighbourhood do not use
        #see function documentation for better understanding
        left_possible_shares_dict = self.find_mcs_to_patterns_that_dont_share_it_in_solution_mapping (cur_sol.mcs_to_patterns_sharing)
        while left_possible_shares_dict and num_of_developed_neighbours < MAX_NEIGHBOURS_TO_DEVELOP:
            mcs, patterns = random.choice(list(left_possible_shares_dict.items()))
            check = 5


    def stopping_criterion(self):
        return (datetime.now() - self.start_date) >= (self.time_limit - timedelta(seconds=0.1))

    def tabu_search_merge_tree_plans(self):
        """
            Merges the given patterns tree by sharing equivalent subtrees found via local search.
            We do a local search for lowest cost pattern to tree plan map
            We starting from initial state that is the pattern to tree plan map after a regular subtree sharing merger run.
            Documentation for how we developing a new neighbour(other pattern to tree plan map) is presented under n-vertex function.
            You can use between two local search heuristics: Tabu-L and Simulated Annealing by give the correct parameters
            to runTest function (see Examples in test).
        """
        initial_solution = self.initial_solution
        neighbours = self.neighbourhood(initial_solution)
        check=5




    def simulated_annealing_merge_tree_plans(self):
        pass

        while (datetime.now() - self.start_date) < (self.time_limit - timedelta(seconds=0.1)):
            cur_state = self.heuristic.get_best_neighboor()
            if cur_state.cost < best_state.cost:
                best_state = cur_state
        return SubTreeSharingTreePlanMerger().merge_tree_plans(self.__create_pattern_to_tree_plan_map(best_state))

        """
        Code for fixing leaves indices seq operator:
        leaves = pattern_to_tree_plan_map[patterns[1]].root.get_leaves()
        for leaf in leaves:
            leaf.event_index = event_name_to_event_index[leaf.event_name]
        """