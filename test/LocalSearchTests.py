#first commit

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
from plan.multi.LocalSearchTreePlanMerger import *
from adaptive.statistics.StatisticsCollectorFactory import StatisticsCollectorParameters
from misc.DefaultConfig import DEFAULT_TREE_COST_MODEL
from plan.TreePlanBuilderFactory import IterativeImprovementTreePlanBuilderParameters
from test.EvalTestsDefaults import DEFAULT_TESTING_STATISTICS_COLLECTOR_SELECTIVITY_AND_ARRIVAL_RATES_STATISTICS
from test.testUtils import *
from evaluation.EvaluationMechanismFactory import TreeBasedEvaluationMechanismParameters
from adaptive.optimizer.OptimizerFactory import StatisticsDeviationAwareOptimizerParameters
from plan.LeftDeepTreeBuilders import *
from plan.BushyTreeBuilders import *
from datetime import timedelta
from condition.Condition import Variable, TrueCondition, BinaryCondition, SimpleCondition
from condition.CompositeCondition import AndCondition
from condition.BaseRelationCondition import GreaterThanCondition, SmallerThanCondition
from base.PatternStructure import SeqOperator, PrimitiveEventStructure
from base.Pattern import Pattern

currentPath = pathlib.Path(os.path.dirname(__file__))
absolutePath = str(currentPath.parent)
sys.path.append(absolutePath)

bushy_tree_local_search_eval_mechanism_params = TreeBasedEvaluationMechanismParameters(
    optimizer_params=OptimizerParameters(opt_type=OptimizerTypes.TRIVIAL_OPTIMIZER,
                                         tree_plan_params=
                                         TreePlanBuilderParameters(builder_type=TreePlanBuilderTypes.DYNAMIC_PROGRAMMING_BUSHY_TREE,
                              cost_model_type=TreeCostModels.INTERMEDIATE_RESULTS_TREE_COST_MODEL,
                              tree_plan_merger_type=MultiPatternTreePlanMergeApproaches.TREE_PLAN_LOCAL_SEARCH,
                              tree_plan_merger_params=['TabuSearch', timedelta(seconds=30),True])),
    storage_params=TreeStorageParameters(sort_storage=False, clean_up_interval=10, prioritize_sorting_by_timestamp=True))

greedy_left_deep_tree_local_search_eval_mechanism_params = TreeBasedEvaluationMechanismParameters(
    optimizer_params=OptimizerParameters(opt_type=OptimizerTypes.TRIVIAL_OPTIMIZER,
                                         tree_plan_params=
                                         TreePlanBuilderParameters(builder_type=TreePlanBuilderTypes.GREEDY_LEFT_DEEP_TREE,
                              cost_model_type=TreeCostModels.INTERMEDIATE_RESULTS_TREE_COST_MODEL,
                              tree_plan_merger_type=MultiPatternTreePlanMergeApproaches.TREE_PLAN_LOCAL_SEARCH,
                              tree_plan_merger_params=['TabuSearch', timedelta(seconds=30),True])),
    storage_params=TreeStorageParameters(sort_storage=False, clean_up_interval=10, prioritize_sorting_by_timestamp=True))

trivial_left_deep_tree_local_search_eval_mechanism_params = TreeBasedEvaluationMechanismParameters(
    optimizer_params=OptimizerParameters(opt_type=OptimizerTypes.TRIVIAL_OPTIMIZER,
                                         tree_plan_params=
                                         TreePlanBuilderParameters(builder_type=TreePlanBuilderTypes.TRIVIAL_LEFT_DEEP_TREE,
                              cost_model_type=TreeCostModels.INTERMEDIATE_RESULTS_TREE_COST_MODEL,
                              tree_plan_merger_type=MultiPatternTreePlanMergeApproaches.TREE_PLAN_LOCAL_SEARCH,
                              tree_plan_merger_params=['TabuSearch', timedelta(seconds=30),True])),
    storage_params=TreeStorageParameters(sort_storage=False, clean_up_interval=10, prioritize_sorting_by_timestamp=True))

sub_tree_sharing_eval_mechanism_params = TreeBasedEvaluationMechanismParameters(
    optimizer_params=OptimizerParameters(opt_type=OptimizerTypes.TRIVIAL_OPTIMIZER,
                                         tree_plan_params=
                                         TreePlanBuilderParameters(builder_type=TreePlanBuilderTypes.GREEDY_LEFT_DEEP_TREE,
                              cost_model_type=TreeCostModels.INTERMEDIATE_RESULTS_TREE_COST_MODEL,
                              tree_plan_merger_type=MultiPatternTreePlanMergeApproaches.TREE_PLAN_SUBTREES_UNION,
                              tree_plan_merger_params=['TabuSearch', timedelta(seconds=30)])),
    storage_params=TreeStorageParameters(sort_storage=False, clean_up_interval=10, prioritize_sorting_by_timestamp=True))

"""
Correctness tests- Those test check that we got the expected matches like all other tests. 
"""
def leftDeepSimple(createTestFile=False):
    pattern1 = Pattern(
        AndOperator(PrimitiveEventStructure("AAPL", "a1"), PrimitiveEventStructure("AAPL", "a2"),
                    PrimitiveEventStructure("AVID", "c#0"), PrimitiveEventStructure("AMZN", "am#0")
                    ),
        TrueCondition(),
        timedelta(minutes=5)
    )
    pattern2 = Pattern(
        AndOperator(PrimitiveEventStructure("AAPL", "a1"), PrimitiveEventStructure("AAPL", "a2"),
                    PrimitiveEventStructure("AVID", "c1"), PrimitiveEventStructure("AVID", "c2"),
                    ),
        TrueCondition(),
        timedelta(minutes=100)
    )
    pattern3 = Pattern(
        AndOperator(PrimitiveEventStructure("AAPL", "a1"), PrimitiveEventStructure("AAPL", "a2"),
                    PrimitiveEventStructure("AVID", "c#2"), PrimitiveEventStructure("AMZN", "am#2")
                    ),
        TrueCondition(),
        timedelta(minutes=5)
    )
    pattern4 = Pattern(
        AndOperator(PrimitiveEventStructure("AAPL", "a#3"),
                    PrimitiveEventStructure("AVID", "c1"), PrimitiveEventStructure("AVID", "c2"),
                    PrimitiveEventStructure("AMZN", "am#3")
                    ),
        TrueCondition(),
        timedelta(minutes=5)
    )

    selectivityMatrix = [[1.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0],
                         ]
    arrivalRates = [0.1, 0.1, 0.1, 0.1]

    pattern1.set_statistics({StatisticsTypes.SELECTIVITY_MATRIX: selectivityMatrix,
                             StatisticsTypes.ARRIVAL_RATES: arrivalRates})

    pattern2.set_statistics({StatisticsTypes.SELECTIVITY_MATRIX: selectivityMatrix,
                             StatisticsTypes.ARRIVAL_RATES: arrivalRates})

    pattern3.set_statistics({StatisticsTypes.SELECTIVITY_MATRIX: selectivityMatrix,
                             StatisticsTypes.ARRIVAL_RATES: arrivalRates})

    pattern4.set_statistics({StatisticsTypes.SELECTIVITY_MATRIX: selectivityMatrix,
                             StatisticsTypes.ARRIVAL_RATES: arrivalRates})

    runMultiTest("leftDeepSimple", [pattern1, pattern2, pattern3, pattern4], createTestFile,
                 greedy_left_deep_tree_local_search_eval_mechanism_params,
                 eventStream=nasdaqEventStreamTiny)

def bushySimple(createTestFile=False):
    pattern1 = Pattern(
        AndOperator(PrimitiveEventStructure("AAPL", "a1"), PrimitiveEventStructure("AAPL", "a2"),
                    PrimitiveEventStructure("AVID", "c#0"), PrimitiveEventStructure("AMZN", "am#0")
                    ),
        TrueCondition(),
        timedelta(minutes=5)
    )
    pattern2 = Pattern(
        AndOperator(PrimitiveEventStructure("AAPL", "a1"), PrimitiveEventStructure("AAPL", "a2"),
                    PrimitiveEventStructure("AVID", "c1"), PrimitiveEventStructure("AVID", "c2"),
                    ),
        TrueCondition(),
        timedelta(minutes=100)
    )
    pattern3 = Pattern(
        AndOperator(PrimitiveEventStructure("AAPL", "a1"), PrimitiveEventStructure("AAPL", "a2"),
                    PrimitiveEventStructure("AVID", "c#2"), PrimitiveEventStructure("AMZN", "am#2")
                    ),
        TrueCondition(),
        timedelta(minutes=5)
    )
    pattern4 = Pattern(
        AndOperator(PrimitiveEventStructure("AAPL", "a#3"),
                    PrimitiveEventStructure("AVID", "c1"), PrimitiveEventStructure("AVID", "c2"),
                    PrimitiveEventStructure("AMZN", "am#3")
                    ),
        TrueCondition(),
        timedelta(minutes=5)
    )

    selectivityMatrix = [[1.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0],
                         ]
    arrivalRates = [0.1, 0.1, 0.1, 0.1]

    pattern1.set_statistics({StatisticsTypes.SELECTIVITY_MATRIX: selectivityMatrix,
                             StatisticsTypes.ARRIVAL_RATES: arrivalRates})

    pattern2.set_statistics({StatisticsTypes.SELECTIVITY_MATRIX: selectivityMatrix,
                             StatisticsTypes.ARRIVAL_RATES: arrivalRates})

    pattern3.set_statistics({StatisticsTypes.SELECTIVITY_MATRIX: selectivityMatrix,
                             StatisticsTypes.ARRIVAL_RATES: arrivalRates})

    pattern4.set_statistics({StatisticsTypes.SELECTIVITY_MATRIX: selectivityMatrix,
                             StatisticsTypes.ARRIVAL_RATES: arrivalRates})

    runMultiTest("bushySimple", [pattern1, pattern2, pattern3, pattern4], createTestFile,
                 bushy_tree_local_search_eval_mechanism_params,
                 eventStream=nasdaqEventStreamTiny)


def nestedSeq(createTestFile=False):
    pattern1 = Pattern(
        AndOperator(AndOperator(PrimitiveEventStructure("AMZN", "am#0"), PrimitiveEventStructure("AVID", "c#0")),
                    PrimitiveEventStructure("AAPL", "a1"),PrimitiveEventStructure("AAPL", "a2")),
        TrueCondition(),
        timedelta(minutes=5)
    )
    pattern2 = Pattern(
        AndOperator(AndOperator(PrimitiveEventStructure("AMZN", "am#0"), PrimitiveEventStructure("AVID", "c#0")),
                    PrimitiveEventStructure("AAPL", "a1"), PrimitiveEventStructure("AAPL", "a2"),
                    PrimitiveEventStructure("AAPL", "a3")),
        TrueCondition(),
        timedelta(minutes=5)
    )
    pattern3 = Pattern(
        AndOperator(AndOperator(PrimitiveEventStructure("AMZN", "am#0"), PrimitiveEventStructure("AVID", "c#0")),
                    PrimitiveEventStructure("AAPL", "a1"), PrimitiveEventStructure("AAPL", "a2"),
                    PrimitiveEventStructure("AVID", "av")),
        TrueCondition(),
        timedelta(minutes=5)
    )

    selectivityMatrix = [[1.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0],
                         ]
    arrivalRates = [0.1, 0.1, 0.1, 0.1]

    pattern1.set_statistics({StatisticsTypes.SELECTIVITY_MATRIX: selectivityMatrix,
                             StatisticsTypes.ARRIVAL_RATES: arrivalRates})

    mutual_selectivityMatrix = [[1.0, 1.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0, 1.0]
                         ]
    mutual_arrivalRates = [0.1, 0.1, 0.1, 0.1, 0.1]

    pattern2.set_statistics({StatisticsTypes.SELECTIVITY_MATRIX: mutual_selectivityMatrix,
                             StatisticsTypes.ARRIVAL_RATES: arrivalRates})

    pattern3.set_statistics({StatisticsTypes.SELECTIVITY_MATRIX: mutual_arrivalRates,
                             StatisticsTypes.ARRIVAL_RATES: arrivalRates})

    runMultiTest("nestedSeq", [pattern1, pattern2, pattern3], createTestFile,
                 bushy_tree_local_search_eval_mechanism_params,
                 eventStream=nasdaqEventStreamTiny)




"""
Cost Tests- Those are structual tests, we verifay that in the end of the local search we got the pattern to tree plan map 
we expect to. 
"""
def shouldShare(createTestFile=False):
    """
    In this test, the initial plan is the best one and thus the returned pattern to tree plan map from local search should be the initial one.
    """
    pattern1 = Pattern(
        AndOperator(PrimitiveEventStructure("AAPL", "a"), PrimitiveEventStructure("AAPL", "b"),
                    PrimitiveEventStructure("AAPL", "c")
                   ),
        AndCondition(),
        timedelta(minutes=1)
    )
    pattern2 = Pattern(
        AndOperator(PrimitiveEventStructure("AAPL", "g"), PrimitiveEventStructure("AAPL", "b"),
                    PrimitiveEventStructure("AAPL", "a")
                    ),
        AndCondition(
        ),
        timedelta(minutes=1)
    )
    selectivityMatrix = [[1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0],
                         ]
    arrivalRates = [0.1, 0.1, 0.1]

    pattern1.set_statistics({StatisticsTypes.SELECTIVITY_MATRIX: selectivityMatrix,
                             StatisticsTypes.ARRIVAL_RATES: arrivalRates})

    pattern2.set_statistics({StatisticsTypes.SELECTIVITY_MATRIX: selectivityMatrix,
                             StatisticsTypes.ARRIVAL_RATES: arrivalRates})

    runMultiTest("shouldShare", [pattern1, pattern2], createTestFile,
                 trivial_left_deep_tree_local_search_eval_mechanism_params,
                 eventStream=nasdaqEventStreamTiny)

def shouldNotShare(createTestFile=False):
    """
    In this test, the initial plan is the best one and thus the returned pattern to tree plan map from local search should be the initial one.
    For explnations see...
    """
    pattern1 = Pattern(
        AndOperator(PrimitiveEventStructure("AAPL", "a"), PrimitiveEventStructure("AAPL", "e"),
                    PrimitiveEventStructure("AAPL", "g")
                   ),
        AndCondition(),
        timedelta(minutes=1)
    )
    pattern2 = Pattern(
        AndOperator(PrimitiveEventStructure("AAPL", "g"), PrimitiveEventStructure("AAPL", "e"),
                    PrimitiveEventStructure("AAPL", "b")
                    ),
        AndCondition(),
        timedelta(minutes=1)
    )
    selectivityMatrix1 = [[1.0, 0.0001, 0.0001],
                         [0.0001, 1.0, 1.0],
                         [0.0001, 1.0, 1.0],
                         ]
    arrivalRates1 = [1.0, 1.0, 1.0]

    pattern1.set_statistics({StatisticsTypes.SELECTIVITY_MATRIX: selectivityMatrix1,
                             StatisticsTypes.ARRIVAL_RATES: arrivalRates1})

    selectivityMatrix2 = [[1.0, 1.0, 0.0001],
                          [1.0, 1.0, 0.0001],
                          [0.0001, 0.0001, 1.0],
                          ]
    arrivalRates2 = [1.0, 1.0, 1.0]

    pattern2.set_statistics({StatisticsTypes.SELECTIVITY_MATRIX: selectivityMatrix2,
                             StatisticsTypes.ARRIVAL_RATES: arrivalRates2})


    runMultiTest("shouldShare", [pattern1, pattern2], createTestFile,
                 trivial_left_deep_tree_local_search_eval_mechanism_params,
                 eventStream=nasdaqEventStreamTiny)





















""" MPG(maximal common subpatterns graph) only tests """
#todo comment: in mcs we dont care about the conditions

def two_and_operator(createTestFile=False):
    pattern0 = Pattern(
        AndOperator(PrimitiveEventStructure("AAPL", "aa"), PrimitiveEventStructure("AMZN", "am"),
                    PrimitiveEventStructure("AVID", "av"),PrimitiveEventStructure("CBRL", "cb")),
        AndCondition(
            GreaterThanCondition(Variable("aa", lambda x: x["Opening Price"]),
                                 Variable("am", lambda x: x["Opening Price"])),
            GreaterThanEqCondition(Variable("av", lambda x: x["Peak Price"]), 136)),
        timedelta(minutes=2)
    )
    pattern1 = Pattern(
        AndOperator(PrimitiveEventStructure("AAPL", "aa"), PrimitiveEventStructure("AMZN", "am"),
                    PrimitiveEventStructure("AVID", "av"), PrimitiveEventStructure("CBRL", "gh")),
        AndCondition(
            GreaterThanCondition(Variable("aa", lambda x: x["Opening Price"]),
                                 Variable("am", lambda x: x["Opening Price"])),
            GreaterThanEqCondition(Variable("av", lambda x: x["Peak Price"]), 135)),
        timedelta(minutes=2)
    )

    mpg = MPG(patterns=[pattern0, pattern1])
    #mcs = should be AND(am, aa)
    return True

#todo comment: can use runStructuralTest for testing mpg only(i do not think i can use it directly)-think about bushy when immplementing

def one_pattern_inside_other(createTestFile=False):
    pattern0 = Pattern(
        AndOperator(SeqOperator(PrimitiveEventStructure("AAPL", "a"), PrimitiveEventStructure("AMZN", "b")),
                    PrimitiveEventStructure("BIDU", "d")),
        AndCondition(
            GreaterThanCondition(Variable("a", lambda x: x["Opening Price"]),
                                 Variable("b", lambda x: x["Opening Price"])),
            GreaterThanEqCondition(Variable("d", lambda x: x["Peak Price"]), 135)),
        timedelta(minutes=5)
    )
    pattern1 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "a"), PrimitiveEventStructure("AMZN", "b")),
        AndCondition(
            GreaterThanCondition(Variable("a", lambda x: x["Opening Price"]),
                                 Variable("b", lambda x: x["Opening Price"]))),
        timedelta(minutes=2)
    )


# SEQ(Z,OR(a,b),or(c,d)) AND(OR(a,b),Y,or(c,d)) ====> OR(a,b)
def nested_OR_MPG(createTestFile=False):
    pattern0 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "z"), OrOperator(PrimitiveEventStructure("AAPL", "a"),
                    PrimitiveEventStructure("AAPL", "b")), OrOperator(PrimitiveEventStructure("AAPL", "c"),
                    PrimitiveEventStructure("AAPL", "d"))),
        AndCondition(GreaterThanEqCondition(Variable("z", lambda x: x["Peak Price"]), 135),
                    GreaterThanCondition(Variable("a", lambda x: x["Opening Price"]),Variable("b", lambda x: x["Opening Price"])),
                    GreaterThanCondition(Variable("c", lambda x: x["Opening Price"]), Variable("d", lambda x: x["Opening Price"]))
                    ),
        timedelta(minutes=5)
    )
    pattern1 = Pattern(
        SeqOperator(OrOperator(PrimitiveEventStructure("AAPL", "a"),
                    PrimitiveEventStructure("AAPL", "b")),PrimitiveEventStructure("AAPL", "y"),
                    OrOperator(PrimitiveEventStructure("AAPL", "c"), PrimitiveEventStructure("AAPL", "d"))),
        AndCondition(GreaterThanEqCondition(Variable("y", lambda x: x["Peak Price"]), 135),
                     GreaterThanCondition(Variable("a", lambda x: x["Opening Price"]),
                                          Variable("b", lambda x: x["Opening Price"])),
                     SmallerThanCondition(Variable("c", lambda x: x["Opening Price"]),
                                          Variable("d", lambda x: x["Opening Price"]))
                     ),
        timedelta(minutes=5)
    )
    mpg = MPG(patterns=[pattern0, pattern1])
    # mcs = should be OR(a,b)
    return True



def seqABC_seqACB_MPG(createTestFile=False):
    pattern0 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "a"), PrimitiveEventStructure("AAPL", "b"), PrimitiveEventStructure("AAPL", "c")),
        AndCondition(GreaterThanEqCondition(Variable("a", lambda x: x["Peak Price"]), 135)),
        timedelta(minutes=5)
    )
    pattern1 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "a"), PrimitiveEventStructure("AAPL", "c"), PrimitiveEventStructure("AAPL", "b")),
        AndCondition(GreaterThanEqCondition(Variable("a", lambda x: x["Peak Price"]), 135)),
        timedelta(minutes=2)
    )

    mpg = MPG(patterns=[pattern0, pattern1])
    #2 mcs = seq(A,B), seq(A,C)
    return True

def andABC_seqACB(createTestFile=False):
    pattern0 = Pattern(
        AndOperator(PrimitiveEventStructure("AAPL", "a"), PrimitiveEventStructure("AAPL", "b"), PrimitiveEventStructure("AAPL", "c")),
        AndCondition(GreaterThanEqCondition(Variable("a", lambda x: x["Peak Price"]), 135)),
        timedelta(minutes=5)
    )
    pattern1 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "a"), PrimitiveEventStructure("AAPL", "c"), PrimitiveEventStructure("AAPL", "b")),
        AndCondition(GreaterThanEqCondition(Variable("a", lambda x: x["Peak Price"]), 135)),
        timedelta(minutes=2)
    )

    mpg = MPG(patterns=[pattern0, pattern1])
    #mcs = None
    return True

# we want to return the return mcs with no top level operator
def one_pattern_inside_other(createTestFile=False):
    pattern0 = Pattern(
        AndOperator(SeqOperator(PrimitiveEventStructure("AAPL", "a"), PrimitiveEventStructure("AMZN", "b")),
                    PrimitiveEventStructure("BIDU", "d")),
        AndCondition(
            GreaterThanCondition(Variable("a", lambda x: x["Opening Price"]),
                                 Variable("b", lambda x: x["Opening Price"])),
            GreaterThanEqCondition(Variable("d", lambda x: x["Peak Price"]), 135)),
        timedelta(minutes=5)
    )
    pattern1 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "a"), PrimitiveEventStructure("AMZN", "b")),
        AndCondition(
            GreaterThanCondition(Variable("a", lambda x: x["Opening Price"]),
                                 Variable("b", lambda x: x["Opening Price"]))),
        timedelta(minutes=2)
    )

    mpg = MPG(patterns=[pattern0, pattern1])
    # mcs = should be seq(a,b)
    return True

    #todo: sharing of leaf, binary node, nested node cost calculation

"""
cost function tests
"""
def leaf_sharing(createTestFile=False):
    pattern0 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "a"), PrimitiveEventStructure("AMZN", "b")),
        AndCondition(),
        timedelta(minutes=5)
    )
    pattern1 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "a"), PrimitiveEventStructure("AMZN", "c")),
        AndCondition(),
        timedelta(minutes=2)
    )

    mpg = MPG(patterns=[pattern0, pattern1])
    # mcs = should be seq(a,b)
    return True

def nestedNodeSharing(createTestFile=False):
    pattern0 = Pattern(
        SeqOperator(PrimitiveEventStructure("CBRL", "z"),
                    SeqOperator(PrimitiveEventStructure("AAPL", "a"), PrimitiveEventStructure("AAPL", "b")),
                    PrimitiveEventStructure("CBRL", "c")),
        AndCondition(),
        timedelta(minutes=5)
    )
    pattern1 = Pattern(
        SeqOperator(PrimitiveEventStructure("CBRL", "d"), SeqOperator(PrimitiveEventStructure("AAPL", "a"), PrimitiveEventStructure("AAPL", "b")),
                    PrimitiveEventStructure("CBRL", "c")),
        AndCondition(),
        timedelta(minutes=5)
    )
    selectivityMatrix = [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    arrivalRates = [0.2, 0.2, 0.2, 0.2]
    pattern0.set_statistics({StatisticsTypes.SELECTIVITY_MATRIX: selectivityMatrix,
                             StatisticsTypes.ARRIVAL_RATES: arrivalRates})

    selectivityMatrix = [[1.0, 1.0, 1.0,1.0],[1.0, 1.0, 1.0,1.0],
                         [1.0, 1.0, 1.0,1.0],[1.0, 1.0, 1.0,1.0]]
    arrivalRates = [0.2, 0.2, 0.2,0.2]
    pattern1.set_statistics({StatisticsTypes.SELECTIVITY_MATRIX: selectivityMatrix,
                             StatisticsTypes.ARRIVAL_RATES: arrivalRates})

    runMultiTest("other", [pattern0, pattern1], createTestFile, local_search_eval_mechanism_params,
                 eventStream=nasdaqEventStreamTiny)


def one_pattern_inside_other(createTestFile=False):
    pattern0 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "d"), PrimitiveEventStructure("AMZN", "b"),
                    PrimitiveEventStructure("AAPL", "a")),
        AndCondition(
            GreaterThanCondition(Variable("a", lambda x: x["Opening Price"]),
                                 Variable("b", lambda x: x["Opening Price"]))),
        timedelta(minutes=2)
    )
    pattern1 = Pattern(
        AndOperator(SeqOperator(PrimitiveEventStructure("AAPL", "a"), PrimitiveEventStructure("AMZN", "b")),
                    PrimitiveEventStructure("BIDU", "d")),
        AndCondition(
            GreaterThanCondition(Variable("a", lambda x: x["Opening Price"]),
                                 Variable("b", lambda x: x["Opening Price"])),
            GreaterThanEqCondition(Variable("d", lambda x: x["Peak Price"]), 135)),
        timedelta(minutes=5)
    )

    runMultiTest("other", [pattern0, pattern1], createTestFile, sub_tree_sharing_eval_mechanism_params,
                 eventStream=nasdaqEventStreamTiny)










"""
full tests
"""

# SEQ(Z,seq(a,b),And(c,d)) SEQ(seq(a,b),Y,And(c,d)) ====> seq(a,b)
def nested_And(createTestFile=False):
    pattern0 = Pattern(
        SeqOperator(PrimitiveEventStructure("CBRL", "z"), SeqOperator(PrimitiveEventStructure("MSFT", "a"),
                    PrimitiveEventStructure("ORLY", "b")),
                    AndOperator(PrimitiveEventStructure("CBRL", "c"), PrimitiveEventStructure("CBRL", "d"))),
        AndCondition(GreaterThanEqCondition(Variable("z", lambda x: x["Peak Price"]), 135),
                     GreaterThanCondition(Variable("a", lambda x: x["Opening Price"]),
                                          Variable("b", lambda x: x["Opening Price"])),
                     GreaterThanCondition(Variable("c", lambda x: x["Opening Price"]),
                                          Variable("d", lambda x: x["Opening Price"]))
                     ),
        timedelta(minutes=5)
    )
    pattern1 = Pattern(
        SeqOperator(SeqOperator(PrimitiveEventStructure("MSFT", "a"),
                    PrimitiveEventStructure("ORLY", "b")),PrimitiveEventStructure("CBRL", "y"),
                    AndOperator(PrimitiveEventStructure("CBRL", "c"), PrimitiveEventStructure("CBRL", "d"))),
        AndCondition(GreaterThanEqCondition(Variable("y", lambda x: x["Peak Price"]), 135),
                     GreaterThanCondition(Variable("a", lambda x: x["Opening Price"]),
                                          Variable("b", lambda x: x["Opening Price"])),
                     SmallerThanCondition(Variable("c", lambda x: x["Opening Price"]),
                                          Variable("d", lambda x: x["Opening Price"]))
                     ),
        timedelta(minutes=5)
    )

    runMultiTest("other", [pattern0,pattern1], createTestFile, sub_tree_sharing_eval_mechanism_params, eventStream=nasdaqEventStreamTiny)



def seq_resarch_nested(createTestFile=False):
    pattern0 = Pattern(
        SeqOperator(PrimitiveEventStructure("DRIV", "x"), AndOperator(PrimitiveEventStructure("ORLY", "a"),
                    PrimitiveEventStructure("ORLY", "b")),
                    AndOperator(PrimitiveEventStructure("CBRL", "c"), PrimitiveEventStructure("CBRL", "d")), PrimitiveEventStructure("MSFT", "y")),
        AndCondition(),
        timedelta(minutes=5)
    )

    pattern0_with_mcs = Pattern(
        SeqOperator(AndOperator(PrimitiveEventStructure("ORLY", "a"),
                    PrimitiveEventStructure("ORLY", "b")),
                    AndOperator(PrimitiveEventStructure("CBRL", "c"), PrimitiveEventStructure("CBRL", "d")),SeqOperator(PrimitiveEventStructure("DRIV", "x"),PrimitiveEventStructure("MSFT", "y"))),
        AndCondition(),
        timedelta(minutes=5)
    )


    runMultiTest("nested_And", [pattern0, pattern0_with_mcs], createTestFile, sub_tree_sharing_eval_mechanism_params, eventStream= nasdaqEventStreamMedium)

def seq_resarch(createTestFile=False):
    pattern0 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "a"), PrimitiveEventStructure("AMZN", "b"), PrimitiveEventStructure("AVID", "c")),
        AndCondition(),
        timedelta(minutes=5)
    )
    pattern1 = Pattern(
        SeqOperator(SeqOperator(PrimitiveEventStructure("AAPL", "a"), PrimitiveEventStructure("AVID", "c")),
                    PrimitiveEventStructure("AMZN", "b")),
        AndCondition(),
        timedelta(minutes=5)
    )

    runMultiTest("seq_resarch", [pattern1], createTestFile, sub_tree_sharing_eval_mechanism_params,eventStream= nasdaqEventStreamTiny)

#todo: seq(a,b) seq(a,b,c,d) => share all first pattern as nested even thouh it will not reutrn as "contained pattern" in the mpg.


def new_cost_machnizem(createTestFile=False):
    # SEQ(Z,And(a,b),And(c,d)) y,AND(And(a,b),And(c,d)) ====> And(a,b)
    pattern0 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "z"), AndOperator(PrimitiveEventStructure("AAPL", "a"),
                    PrimitiveEventStructure("AAPL", "b")),
                    AndOperator(PrimitiveEventStructure("AAPL", "c"), PrimitiveEventStructure("AAPL", "d"))),
        AndCondition(GreaterThanEqCondition(Variable("z", lambda x: x["Peak Price"]), 135),
                     GreaterThanCondition(Variable("a", lambda x: x["Opening Price"]),
                                          Variable("b", lambda x: x["Opening Price"])),
                     GreaterThanCondition(Variable("c", lambda x: x["Opening Price"]),
                                          Variable("d", lambda x: x["Opening Price"]))
                     ),
        timedelta(minutes=5)
    )
    selectivityMatrix = [[0.8, 0.6, 1.0, 1.0,1.0], [0.9457796098355941, 1.0, 0.15989723367389616, 1.0,1.0],
                         [1.0, 0.15989723367389616, 1.0, 0.9992557393942864,1.0], [1.0, 1.0, 0.9992557393942864, 1.0,1.0],
                         [0.8, 0.6, 1.0, 1.0,1.0]]
    arrivalRates = [0.8, 0.2, 0.2, 0.2,0.2]
    pattern0.set_statistics({StatisticsTypes.SELECTIVITY_MATRIX: selectivityMatrix,
                             StatisticsTypes.ARRIVAL_RATES: arrivalRates})

    pattern1 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "y"),AndOperator(PrimitiveEventStructure("AAPL", "a"),
                    PrimitiveEventStructure("AAPL", "b")),
                    AndOperator(PrimitiveEventStructure("AAPL", "c"), PrimitiveEventStructure("AAPL", "d"))),
        AndCondition(GreaterThanEqCondition(Variable("y", lambda x: x["Peak Price"]), 135),
                     GreaterThanCondition(Variable("a", lambda x: x["Opening Price"]),
                                          Variable("b", lambda x: x["Opening Price"])),
                     SmallerThanCondition(Variable("c", lambda x: x["Opening Price"]),
                                          Variable("d", lambda x: x["Opening Price"]))
                     ),
        timedelta(minutes=5)
    )

    selectivityMatrix = [[0.9, 0.6, 1.0, 1.0, 1.0], [0.9457796098355941, 1.0, 0.15989723367389616, 1.0, 1.0],
                         [1.0, 0.15989723367389616, 1.0, 0.9992557393942864, 1.0],
                         [1.0, 1.0, 0.9992557393942864, 1.0, 1.0],[0.9, 0.6, 1.0, 1.0, 1.0]]
    arrivalRates = [0.9, 0.3, 0.3, 0.3,0.3]
    pattern1.set_statistics({StatisticsTypes.SELECTIVITY_MATRIX: selectivityMatrix,
                             StatisticsTypes.ARRIVAL_RATES: arrivalRates})

    runMultiTest("nested_And", [pattern0, pattern1], createTestFile, sub_tree_sharing_eval_mechanism_params)


def seqABC_seqACB(createTestFile=False):
    pattern0 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "aa"), PrimitiveEventStructure("Amzn", "am"), PrimitiveEventStructure("AVID", "av")),
        AndCondition(),
        timedelta(minutes=5)
    )
    pattern1 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "aa"), PrimitiveEventStructure("AVID", "av"), PrimitiveEventStructure("Amzn", "am")),
        AndCondition(),
        timedelta(minutes=2)
    )

    runTest("seqABC_seqACB", [pattern0, pattern1], createTestFile, local_search_eval_mechanism_params, events=nasdaqEventStreamTiny)

def dpBPatternSearchTestTomCheck(createTestFile=False):
    pattern0 = Pattern(
        SeqOperator(PrimitiveEventStructure("MSFT", "a"), PrimitiveEventStructure("DRIV", "b"),
                    PrimitiveEventStructure("ORLY", "c"), PrimitiveEventStructure("CBRL", "d")),
        AndCondition(
            SmallerThanCondition(Variable("a", lambda x: x["Peak Price"]),
                                 Variable("b", lambda x: x["Peak Price"])),
            BinaryCondition(Variable("b", lambda x: x["Peak Price"]),
                            Variable("c", lambda x: x["Peak Price"]),
                            relation_op=lambda x, y: x < y),
            SmallerThanCondition(Variable("c", lambda x: x["Peak Price"]),
                                 Variable("d", lambda x: x["Peak Price"]))
        ),
        timedelta(minutes=3)
    )
    selectivityMatrix = [[1.0, 0.6, 1.0, 1.0], [0.9457796098355941, 1.0, 0.15989723367389616, 1.0],
                         [1.0, 0.15989723367389616, 1.0, 0.9992557393942864], [1.0, 1.0, 0.9992557393942864, 1.0]]
    arrivalRates = [0.016597077244258872, 0.01454418928322895, 0.013917884481558803, 0.012421711899791231]
    pattern0.set_statistics({StatisticsTypes.SELECTIVITY_MATRIX: selectivityMatrix,
                            StatisticsTypes.ARRIVAL_RATES: arrivalRates})

    pattern1 = Pattern(
        SeqOperator(PrimitiveEventStructure("MSFT", "a"), PrimitiveEventStructure("DRIV", "b"),
                    PrimitiveEventStructure("ORLY", "c"), PrimitiveEventStructure("CBRL", "d")),
        AndCondition(
            SmallerThanCondition(Variable("a", lambda x: x["Peak Price"]),
                                 Variable("b", lambda x: x["Peak Price"])),
            BinaryCondition(Variable("b", lambda x: x["Peak Price"]),
                            Variable("c", lambda x: x["Peak Price"]),
                            relation_op=lambda x, y: x < y),
            SmallerThanCondition(Variable("c", lambda x: x["Peak Price"]),
                                 Variable("d", lambda x: x["Peak Price"]))
        ),
        timedelta(minutes=3)
    )
    selectivityMatrix1 = [[1.0, 0.7, 1.0, 1.0], [0.9457796098355941, 1.0, 0.15989723367389616, 1.0],
                         [1.0, 0.15989723367389616, 1.0, 0.9992557393942864], [1.0, 1.0, 0.9992557393942864, 1.0]]
    arrivalRates1 = [0.5, 0.5, 0.5, 0.5]
    pattern1.set_statistics({StatisticsTypes.SELECTIVITY_MATRIX: selectivityMatrix1,
                             StatisticsTypes.ARRIVAL_RATES: arrivalRates1})



    eval_params = TreeBasedEvaluationMechanismParameters(
        optimizer_params=StatisticsDeviationAwareOptimizerParameters(
            tree_plan_params=TreePlanBuilderParameters(TreePlanBuilderTypes.DYNAMIC_PROGRAMMING_BUSHY_TREE),
            statistics_collector_params=StatisticsCollectorParameters(statistics_types=[StatisticsTypes.ARRIVAL_RATES, StatisticsTypes.SELECTIVITY_MATRIX])),
        storage_params=DEFAULT_TESTING_EVALUATION_MECHANISM_SETTINGS.storage_params)

    runMultiTest('dpBPatternSearchTestTomCheck', [pattern0, pattern1], createTestFile, eval_mechanism_params=eval_params, events=nasdaqEventStream)

#todo: support negative and other operators




    #todo: types of test
    #mpg tests: for beauty least important
    #cost tests:
    #corrrectness test: no nested, with nested, unary opertaor



def samePatternSharingRoot_local_search_correctness(createTestFile=False):
    hierarchyPattern = Pattern(
        AndOperator(PrimitiveEventStructure("AMZN", "a"), PrimitiveEventStructure("AAPL", "b"),
                    PrimitiveEventStructure("GOOG", "c")),
        AndCondition(
            SmallerThanCondition(Variable("a", lambda x: x["Peak Price"]),
                                 Variable("b", lambda x: x["Peak Price"])),
            SmallerThanCondition(Variable("b", lambda x: x["Peak Price"]),
                                 Variable("c", lambda x: x["Peak Price"]))
        ),
        timedelta(minutes=1)

    )

    hierarchyPattern2 = Pattern(
        AndOperator(PrimitiveEventStructure("AMZN", "a"), PrimitiveEventStructure("AAPL", "b"),
                    PrimitiveEventStructure("GOOG", "c")),
        AndCondition(
            SmallerThanCondition(Variable("a", lambda x: x["Peak Price"]),
                                 Variable("b", lambda x: x["Peak Price"])),
            SmallerThanCondition(Variable("b", lambda x: x["Peak Price"]),
                                 Variable("c", lambda x: x["Peak Price"]))
        ),
        timedelta(minutes=0.5)
    )

    hierarchyPattern3 = Pattern(
        AndOperator(PrimitiveEventStructure("AMZN", "a"), PrimitiveEventStructure("AAPL", "b"),
                    PrimitiveEventStructure("GOOG", "c")),
        AndCondition(
            SmallerThanCondition(Variable("a", lambda x: x["Peak Price"]),
                                 Variable("b", lambda x: x["Peak Price"])),
            SmallerThanCondition(Variable("b", lambda x: x["Peak Price"]),
                                 Variable("c", lambda x: x["Peak Price"]))
        ),
        timedelta(minutes=0.1)
    )

    selectivityMatrix = [[0.11, 0.12, 0.13],
                         [0.21, 0.22, 0.23],
                         [0.31, 0.32, 0.33],
                         ]
    arrivalRates = [0.1, 0.2, 0.3]

    hierarchyPattern.set_statistics({StatisticsTypes.SELECTIVITY_MATRIX: selectivityMatrix,
                                     StatisticsTypes.ARRIVAL_RATES: arrivalRates})

    hierarchyPattern2.set_statistics({StatisticsTypes.SELECTIVITY_MATRIX: selectivityMatrix,
                                      StatisticsTypes.ARRIVAL_RATES: arrivalRates})

    hierarchyPattern3.set_statistics({StatisticsTypes.SELECTIVITY_MATRIX: selectivityMatrix,
                                      StatisticsTypes.ARRIVAL_RATES: arrivalRates})

    runMultiTest('hierarchyMultiPattern', [hierarchyPattern, hierarchyPattern2, hierarchyPattern3], createTestFile,
                 local_search_eval_mechanism_params)
