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


class MPG:
    def __init__(self, patterns: Pattern or List[Pattern], pattern_to_tree_plan_map: Dict[Pattern, TreePlan],
                 initial_statistics, cost_model_type):
        """
        Hash tables representing the graph
        """
        self.patterns = patterns
        self.pattern_to_tree_plan_map = pattern_to_tree_plan_map
        self.mcs_to_patterns = {}
        self.pattern_to_various_mcs = {p: set() for p in patterns}
        self.__create_mpg(patterns if isinstance(patterns, List) else list(patterns))
        # todo: remove this test function later
        # self.algo_experiment(initial_statistics, cost_model_type)
        # todo:
        # this is the algorithm to find maximal common subpattern
        self.sub_tree_merger = SubTreeSharingTreePlanMerger()

    def algo_experiment1(self):
        p0_left = Pattern(
            SeqOperator(PrimitiveEventStructure("AAPL", "a"),
                        PrimitiveEventStructure("CBRL", "d")),
            AndCondition(),
            timedelta(minutes=2))
        po_right = Pattern(
            SeqOperator(PrimitiveEventStructure("AMZN", "b"),
                        PrimitiveEventStructure("AVID", "c")),
            AndCondition(),
            p0_left_tree=
            timedelta(minutes=2))

    def __create_mpg(self, patterns):
        """i"""
        for p1 in range(len(patterns)):
            for p2 in range(p1 + 1, len(patterns)):
                maximal_common_subpatterns = [self.__find_maximal_common_subpatterns(p1, p2)]
                # todo: make every mcs in the list not empty
                for mcs in maximal_common_subpatterns:
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
        :return:  maximal common subpatterns of pattern1 and pattern2
        """

        p1 = self.patterns[p1_idx]
        p2 = self.patterns[p2_idx]

        if self.__is_first_contained_in_args_of_second(p1, p2):
            return p1
        if self.__is_first_contained_in_args_of_second(p2, p1):
            return p2

        # todo: continue here
        if type(p1.full_structure) is type(p2.full_structure):
            return self.__find_maximal_common_subpatterns_for_equal_operators(p1, p2)

        if type(p1.full_structure) is not type(p2.full_structure):
            return self.__find_maximal_common_subpatterns_for_different_operators(p1, p2)

    """
    #usally it is only one intersection, in seq could be both
    structure_intersections = p1.full_structure.calc_initial_intersection(p2.full_structure)
    maximal_common_subpatterns=[]
    for structure in structure_intersections:
        maximal_common_subpatterns.append(self.__find_maximal_common_subpatterns(structure,p1,p2))
    return maximal_common_subpatterns
    """  # todo: initial

    def remove_mcs_from_pattern(self, pattern, mcs):
        new_positive_structure = set(pattern.positive_structure)
        new_positive_structure.difference(mcs)
        pattern.positive_structure = list(new_positive_structure)

    def __find_mcs_by_structure_intersection(self, structure_intersection, p1, p2):
        p1_conditions = p1.condition.extract_atomic_conditions()
        p2_conditions = p2.condition.extract_atomic_conditions()

        # calc condition intersection
        conditions_intersection = []
        for cond1 in p1_conditions:
            for cond2 in p2_conditions:
                if cond1 == cond2:
                    conditions_intersection.append(cond1)

        # now we check the intersection reffere to same types
        conditions_intersection_after_type_validation = conditions_intersection
        for cond in conditions_intersection:
            event_names = cond.get_event_names()
            # just 2 names
            for name in event_names:
                p1_type = None
                for arg in p1.full_structure.args:
                    if arg.name == name:
                        p1_type = arg.type

                p2_type = None
                for arg in p2.full_structure.args:
                    if arg.name == name:
                        p2_type = arg.type

                if p1_type != p2_type:
                    conditions_intersection_after_type_validation.remove(cond)
                    break

        mcs_condition = None
        mcs_operator = None

        # todo: switch case for all available conditions
        if isinstance(p1.condition, AndCondition):
            mcs_condition = AndCondition(*conditions_intersection_after_type_validation)

        conditions_intersection_after_type_validation_names = set()
        for cond in conditions_intersection_after_type_validation:
            for name in cond.get_event_names():
                conditions_intersection_after_type_validation_names.add(name)

        structure_intersection_after_filtering_by_conditions = []
        for event in structure_intersection:
            if event.name in conditions_intersection_after_type_validation_names:
                structure_intersection_after_filtering_by_conditions.append(event)

        if isinstance(p1.full_structure, AndOperator):
            mcs_operator = AndOperator()
            for event in structure_intersection_after_filtering_by_conditions:
                mcs_operator.args.append(event)

        # todo: change p1.window to needed time_window later

        return Pattern(mcs_operator, mcs_condition, timedelta(minutes=5))

    def __is_first_contained_in_args_of_second(self, p1: Pattern, p2: Pattern):
        """
        :return: True if first pattern is one of the args of the top operator of second else False
        """
        p2_conditions = p2.condition.get_conditions_list()
        for arg in p2.full_structure.args:
            if type(arg) is PrimitiveEventStructure:
                continue
            if arg == p1.full_structure and self.__filter_arg_conditions_pattern_condtions(arg,
                                                                                           p2_conditions) == p1.condition.get_conditions_list():
                return True
        return False

    def __find_maximal_common_subpatterns_for_equal_operators(self, p1: Pattern, p2: Pattern):
        """
        :return: maximal common subpattern for 2 equal operators
        """
        # todo: now funcionts work in case of not nested arguments
        p1_events = set(p1.get_top_level_structure_args())
        p2_events = set(p2.get_top_level_structure_args())
        events_intersection = p1_events.intersection(p2_events)
        events_intersection_names_set = set(event.name for event in events_intersection)
        p1_event_conditions_filtered_by_intersection = p1.condition.get_condition_of(set(events_intersection_names_set),
                                                                                     get_kleene_closure_conditions=False,
                                                                                     consume_returned_conditions=False).get_conditions_list()
        p2_event_conditions_filtered_by_intersection = p2.condition.get_condition_of(set(events_intersection_names_set),
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

        """
        conditions_union = p1_event_conditions_filtered_by_intersection_set.union(p1_event_conditions_filtered_by_intersection_set)
        conditions_intersection = p1_event_conditions_filtered_by_intersection_set.intersection(p1_event_conditions_filtered_by_intersection_set) 
        unmutal_conditions = conditions_union.difference(conditions_intersection)
        unmutal_conditions_names = [condition.get_event_names() for condition in unmutal_conditions]

        """
        events_intersection_after_condition_filtering = copy.deepcopy(events_intersection)
        for event in events_intersection:
            if event.name in p1_filtered_only_conditions_names or event.name in p2_filtered_only_conditions_names:
                events_intersection_after_condition_filtering.remove(event)
        pattern_to_return = copy.deepcopy(p1)
        pattern_to_return.full_structure.args = list(events_intersection_after_condition_filtering)
        pattern_to_return.positive_structure.args = list(events_intersection_after_condition_filtering)
        return pattern_to_return

    def __find_maximal_common_subpatterns_for_different_operators(self, p1: Pattern, p2: Pattern):
        """

        :param p1:
        :param p2:
        :return: maximal common subpatterns for 2 different operators
        """

    def __filter_arg_conditions_pattern_condtions(self, arg, pattern_conditions):
        arg_names = set(arg.get_all_event_names())
        return [condition for condition in pattern_conditions if set(condition.get_event_names()).issubset(arg_names)]

    def __build_tree_that_starts_with_mcs(self, pattern: Pattern, statistics: Dict, mcs):
        p = copy.deepcopy(pattern)
        mcs = copy.deepcopy(mcs)
        statistics = copy.deepcopy(statistics)


class SearchHeuristic(ABC):
    def __init__(self, neighborhood_func):
        self.neighborhood_func = neighborhood_func
        self.stop_local_search = False

    def get_best_neighbor(self, cur_state):
        raise NotImplementedError


class TabuSearch(SearchHeuristic):
    # Tabu search stop if all neighbors are more expensive than current state
    def __init__(self, tabu_list_capacity=1000, neighbors_num_per_iteration=100):
        super().__init__(n_vertex)
        self.tabu_list_capacity = tabu_list_capacity
        self.neighbors_num_per_iteration = neighbors_num_per_iteration
        self.tabu_list = []

    def get_best_neighbor(self, cur_state):
        # tabu doesnt create a new state if  it is in his tabu_list
        # state is identical if for each mcs use in state it is combined for exactly same pattern trees
        best_neighboor = None
        if best_neighboor.cost > cur_state.cost:
            self.stop_local_search = True


class SimulatedAnnealing(SearchHeuristic):
    def __init__(self):
        super().__init__(n_vertex)


def n_vertex():
    """
    :return: a_new_neighboor
    """
    # process:

    # choose v
    # choose mcs of v
    # lambda is a set of patterns of mcs
    # choose min(k, size of lambda) patterns to share
    # a is called to create a plan to each pattern when mcs is shared between all of them

    # notes: i want to create a state that was not before


class state:
    def __init__(self):
        self.cost = None
        # this variable is a hash between each mcs that is shared inside the state to the patterns that are sharing it in the state
        self.mcs_to_patterns_sharing = {}

    def calc_state_cost(self):
        pass


class LocalSearchTreePlanMerger:
    """
    Merges the given patterns by sharing equivalent subtrees found via local search.
    """

    def __init__(self, patterns: Pattern or List[Pattern],
                 pattern_to_tree_plan_map: Dict[Pattern, TreePlan],
                 local_search_params: List, initial_statistics, cost_model_type, optimizer):
        self.initial_statistics = initial_statistics
        self.cost_model_type = cost_model_type
        self.start_date = datetime.now()
        self.mpg = MPG(patterns, pattern_to_tree_plan_map, initial_statistics, cost_model_type)
        if (len(local_search_params) < 2):
            raise Exception("local search: not enough parameters")
        self.heuristic = self.__set_heuristic(local_search_params[0])
        self.time_delta = self.__set_time_delta(local_search_params[1])
        self.optimizer = optimizer
        self.patterns = patterns

    def __set_heuristic(self, heuristic):
        if heuristic == "TabuSearch":
            return TabuSearch()

        if heuristic == "SimulatedAnnealing":
            return SimulatedAnnealing()

        raise Exception("local search: Unknown Heuristic")

    def __set_time_delta(self, time_delta):
        if not isinstance(time_delta, timedelta):
            raise Exception("local search: time_delta is not instance of timeDelta class")
        return time_delta

    def __create_pattern_to_tree_plan_map(self, best_state):
        """returns pattern to tree plan map"""
        pass

    def __set_tree_plan_build_algo(self, algo):
        """switch cases of possible algo choices"""
        pass

    def merge_tree_plans(self):
        # todo: now it is a test for creating a tree that starts with specific common subpattern
        mcs_es = self.mpg.pattern_to_various_mcs[self.patterns[0]]
        mcs = [i for i in mcs_es]
        real_mcs = mcs[0]
        self.optimizer.build_initial_plan(self.initial_statistics, self.cost_model_type, pattern=self.patterns[0],
                                          mcs=real_mcs)

        """
        #creating initial state: initial state is the state where no mcs is shared
        cur_state = state()
        cur_state.calc_state_cost()
        best_state = copy.deepcopy(cur_state)

        while((datetime.now - self.start_date) <= (self.time_delta - timedelta(seconds=0.1)) and not self.heuristic.stop_local_search):
            cur_state = self.heuristic.get_best_neighboor()
            if cur_state.cost < best_state.cost:
                best_state = cur_state
        return SubTreeSharingTreePlanMerger().merge_tree_plans(self.__create_pattern_to_tree_plan_map(best_state))


        """






