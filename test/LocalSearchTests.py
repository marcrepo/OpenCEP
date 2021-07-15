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

def leftNestedSeq(createTestFile=False):
    pattern1 = Pattern(
        AndOperator(SeqOperator(PrimitiveEventStructure("AMZN", "am"), PrimitiveEventStructure("AVID", "c")),
                    PrimitiveEventStructure("AAPL", "a1"),PrimitiveEventStructure("AAPL", "a2")),
        TrueCondition(),
        timedelta(minutes=9)
    )
    pattern2 = Pattern(
        AndOperator(SeqOperator(PrimitiveEventStructure("AMZN", "am"), PrimitiveEventStructure("AVID", "c")),
                    PrimitiveEventStructure("AAPL", "a1"), PrimitiveEventStructure("AAPL", "a2"),
                    PrimitiveEventStructure("AAPL", "a#3")),
        TrueCondition(),
        timedelta(minutes=9)
    )
    pattern3 = Pattern(
        AndOperator(SeqOperator(PrimitiveEventStructure("AMZN", "am"), PrimitiveEventStructure("AVID", "c")),
                    PrimitiveEventStructure("AAPL", "a1"), PrimitiveEventStructure("AAPL", "a2"),
                    PrimitiveEventStructure("AVID", "av#")),
        TrueCondition(),
        timedelta(minutes=9)
    )

    selectivityMatrix = [[1.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0]]
    arrivalRates = [0.1, 0.1, 0.1, 0.1]

    pattern1.set_statistics({StatisticsTypes.SELECTIVITY_MATRIX: selectivityMatrix,
                             StatisticsTypes.ARRIVAL_RATES: arrivalRates})

    mutual_selectivityMatrix = [[1.0, 1.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0, 1.0],
                         ]
    mutual_arrivalRates = [0.1, 0.1, 0.1, 0.1, 0.1]

    pattern2.set_statistics({StatisticsTypes.SELECTIVITY_MATRIX: mutual_selectivityMatrix,
                             StatisticsTypes.ARRIVAL_RATES: mutual_arrivalRates})

    pattern3.set_statistics({StatisticsTypes.SELECTIVITY_MATRIX: mutual_selectivityMatrix,
                             StatisticsTypes.ARRIVAL_RATES: mutual_arrivalRates})

    runMultiTest("leftNestedSeq", [pattern1, pattern2, pattern3], createTestFile,
                 greedy_left_deep_tree_local_search_eval_mechanism_params,
                 eventStream=nasdaqEventStreamTiny)

def bushyNestedSeq(createTestFile=False):
    pattern1 = Pattern(
        AndOperator(SeqOperator(PrimitiveEventStructure("AMZN", "am"), PrimitiveEventStructure("AVID", "c")),
                    PrimitiveEventStructure("AAPL", "a1"),PrimitiveEventStructure("AAPL", "a2")),
        TrueCondition(),
        timedelta(minutes=9)
    )
    pattern2 = Pattern(
        AndOperator(SeqOperator(PrimitiveEventStructure("AMZN", "am"), PrimitiveEventStructure("AVID", "c")),
                    PrimitiveEventStructure("AAPL", "a1"), PrimitiveEventStructure("AAPL", "a2"),
                    PrimitiveEventStructure("AAPL", "a#3")),
        TrueCondition(),
        timedelta(minutes=9)
    )
    pattern3 = Pattern(
        AndOperator(SeqOperator(PrimitiveEventStructure("AMZN", "am"), PrimitiveEventStructure("AVID", "c")),
                    PrimitiveEventStructure("AAPL", "a1"), PrimitiveEventStructure("AAPL", "a2"),
                    PrimitiveEventStructure("AVID", "av#")),
        TrueCondition(),
        timedelta(minutes=9)
    )

    selectivityMatrix = [[1.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0]]
    arrivalRates = [0.1, 0.1, 0.1, 0.1]

    pattern1.set_statistics({StatisticsTypes.SELECTIVITY_MATRIX: selectivityMatrix,
                             StatisticsTypes.ARRIVAL_RATES: arrivalRates})

    mutual_selectivityMatrix = [[1.0, 1.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0, 1.0],
                         ]
    mutual_arrivalRates = [0.1, 0.1, 0.1, 0.1, 0.1]

    pattern2.set_statistics({StatisticsTypes.SELECTIVITY_MATRIX: mutual_selectivityMatrix,
                             StatisticsTypes.ARRIVAL_RATES: mutual_arrivalRates})

    pattern3.set_statistics({StatisticsTypes.SELECTIVITY_MATRIX: mutual_selectivityMatrix,
                             StatisticsTypes.ARRIVAL_RATES: mutual_arrivalRates})

    runMultiTest("bushyNestedSeq", [pattern1, pattern2, pattern3], createTestFile,
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

"""
MPG - MultiPatternGraph TESTS - check the correctness of the data structure for storing mutual part in patterns
"""

def mpgNoNestedSharing(createTestFile=False):
    pattern0 = Pattern(
        AndOperator(SeqOperator(PrimitiveEventStructure("AAPL", "aa"), PrimitiveEventStructure("AMZN", "am"))),
        AndCondition(),
        timedelta(minutes=2)
    )
    pattern1 = Pattern(
        AndOperator(SeqOperator(PrimitiveEventStructure("AAPL", "aa"), PrimitiveEventStructure("AMZN", "am"))),
        AndCondition(),
        timedelta(minutes=2)
    )

    mpg = MPG(patterns=[pattern0, pattern1])
    #we check there is no mcs that is shared by checking the crossponding dict is empty
    assert(mpg.mcs_to_patterns == {})

def mpgNoOneArgumentSharing(createTestFile=False):
    pattern0 = Pattern(
        AndOperator(PrimitiveEventStructure("AAPL", "a"), PrimitiveEventStructure("AMZN", "am")),
        AndCondition(),
        timedelta(minutes=2)
    )
    pattern1 = Pattern(
        AndOperator(PrimitiveEventStructure("AAPL", "a"), PrimitiveEventStructure("AVID", "av")),
        AndCondition(),
        timedelta(minutes=2)
    )

    mpg = MPG(patterns=[pattern0, pattern1])
    #we check there is no mcs that is shared by checking the crossponding dict is empty
    assert(mpg.mcs_to_patterns == {})














