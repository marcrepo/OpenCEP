from datetime import timedelta

from adaptive.optimizer.OptimizerFactory import MultiPatternStatisticsDeviationAwareOptimizerParameters
from base.Pattern import Pattern
from base.PatternStructure import AndOperator, SeqOperator, PrimitiveEventStructure, NegationOperator
from condition.BaseRelationCondition import GreaterThanCondition, SmallerThanCondition, GreaterThanEqCondition, \
    SmallerThanEqCondition
from condition.CompositeCondition import AndCondition
from condition.Condition import Variable
from plan.multi.MultiPatternTreePlanMergeApproaches import MultiPatternTreePlanMergeApproaches
from plan.multi.TreePlanMergerFactory import TreePlanMergerParameters
from test.testUtils import *
from tree.evaluation.TreeEvaluationMechanismUpdateTypes import TreeEvaluationMechanismUpdateTypes

currentPath = pathlib.Path(os.path.dirname(__file__))
absolutePath = str(currentPath.parent)
sys.path.append(absolutePath)

leaf_sharing_eval_mechanism_params = TreeBasedEvaluationMechanismParameters(
    optimizer_params=MultiPatternStatisticsDeviationAwareOptimizerParameters(tree_merger_params=
    TreePlanMergerParameters(
        MultiPatternTreePlanMergeApproaches.TREE_PLAN_TRIVIAL_SHARING_LEAVES),
        tree_plan_params=
        TreePlanBuilderParameters(
            builder_type=TreePlanBuilderTypes.TRIVIAL_LEFT_DEEP_TREE,
            cost_model_type=TreeCostModels.INTERMEDIATE_RESULTS_TREE_COST_MODEL,
            tree_plan_merger_type=MultiPatternTreePlanMergeApproaches.TREE_PLAN_TRIVIAL_SHARING_LEAVES),
        statistics_updates_wait_time=timedelta(minutes=10)),
    storage_params=TreeStorageParameters(sort_storage=False, clean_up_interval=10,
                                         prioritize_sorting_by_timestamp=True),
    tree_update_type=TreeEvaluationMechanismUpdateTypes.MULTI_PATTERN_TREE_EVALUATION)

subtree_sharing_eval_mechanism_params = TreeBasedEvaluationMechanismParameters(
    optimizer_params=MultiPatternStatisticsDeviationAwareOptimizerParameters(tree_merger_params=
    TreePlanMergerParameters(
        MultiPatternTreePlanMergeApproaches.TREE_PLAN_SUBTREES_UNION),
        tree_plan_params=TreePlanBuilderParameters(
            builder_type=TreePlanBuilderTypes.TRIVIAL_LEFT_DEEP_TREE,
            cost_model_type=TreeCostModels.INTERMEDIATE_RESULTS_TREE_COST_MODEL,
            tree_plan_merger_type=MultiPatternTreePlanMergeApproaches.TREE_PLAN_SUBTREES_UNION),
        statistics_updates_wait_time=timedelta(
            minutes=10)),
    storage_params=TreeStorageParameters(sort_storage=False, clean_up_interval=10,
                                         prioritize_sorting_by_timestamp=True),
    tree_update_type=TreeEvaluationMechanismUpdateTypes.MULTI_PATTERN_TREE_EVALUATION)

"""
Simple multi-pattern test with 2 patterns
"""
def adaptiveLeafIsRoot(createTestFile=False):
    pattern1 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "a")),
        GreaterThanCondition(Variable("a", lambda x: x["Peak Price"]), 135),
        timedelta(minutes=5)
    )
    pattern2 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "a"), NegationOperator(PrimitiveEventStructure("AMZN", "b")),
                    PrimitiveEventStructure("GOOG", "c")),
        AndCondition(
            GreaterThanCondition(Variable("a", lambda x: x["Opening Price"]),
                                 Variable("b", lambda x: x["Opening Price"])),
            SmallerThanCondition(Variable("b", lambda x: x["Opening Price"]),
                                 Variable("c", lambda x: x["Opening Price"]))),
        timedelta(minutes=5)
    )

    patterns = [pattern1, pattern2]
    multi_pattern = MultiPattern(patterns)
    runMultiTest("FirstMultiPattern_adaptive", multi_pattern, createTestFile, leaf_sharing_eval_mechanism_params)


"""
multi-pattern test 2 completely distinct patterns
"""


def adaptiveDistinctPatterns(createTestFile=False):
    pattern1 = Pattern(
        SeqOperator(PrimitiveEventStructure("GOOG", "a"), PrimitiveEventStructure("GOOG", "b"),
                    PrimitiveEventStructure("GOOG", "c")),
        AndCondition(
            SmallerThanCondition(Variable("a", lambda x: x["Peak Price"]),
                                 Variable("b", lambda x: x["Peak Price"])),
            SmallerThanCondition(Variable("b", lambda x: x["Peak Price"]),
                                 Variable("c", lambda x: x["Peak Price"]))
        ),
        timedelta(minutes=3)
    )
    pattern2 = Pattern(
        SeqOperator(PrimitiveEventStructure("AMZN", "x1"), PrimitiveEventStructure("AMZN", "x2"),
                    PrimitiveEventStructure("AMZN", "x3")),
        AndCondition(
            SmallerThanEqCondition(Variable("x1", lambda x: x["Lowest Price"]), 75),
            GreaterThanEqCondition(Variable("x2", lambda x: x["Peak Price"]), 78),
            SmallerThanEqCondition(Variable("x3", lambda x: x["Lowest Price"]),
                                   Variable("x1", lambda x: x["Lowest Price"]))
        ),
        timedelta(days=1)
    )

    patterns = [pattern1, pattern2]
    multi_pattern = MultiPattern(patterns)
    runMultiTest("BigMultiPattern_adaptive", multi_pattern, createTestFile, leaf_sharing_eval_mechanism_params)


"""
multi-pattern test with 3 patterns and leaf sharing
"""


def adaptiveThreePatternsTest(createTestFile=False):
    pattern1 = Pattern(
        AndOperator(PrimitiveEventStructure("AAPL", "a"), PrimitiveEventStructure("AMZN", "b"),
                    PrimitiveEventStructure("GOOG", "c")),
        AndCondition(
            SmallerThanCondition(Variable("a", lambda x: x["Peak Price"]),
                                 Variable("b", lambda x: x["Peak Price"])),
            SmallerThanCondition(Variable("b", lambda x: x["Peak Price"]),
                                 Variable("c", lambda x: x["Peak Price"]))
        ),
        timedelta(minutes=1)
    )
    pattern2 = Pattern(
        SeqOperator(PrimitiveEventStructure("MSFT", "a"), PrimitiveEventStructure("DRIV", "b"),
                    PrimitiveEventStructure("MSFT", "c"), PrimitiveEventStructure("DRIV", "d"),
                    PrimitiveEventStructure("MSFT", "e")),
        AndCondition(
            SmallerThanCondition(Variable("a", lambda x: x["Peak Price"]),
                                 Variable("b", lambda x: x["Peak Price"])),
            SmallerThanCondition(Variable("b", lambda x: x["Peak Price"]),
                                 Variable("c", lambda x: x["Peak Price"])),
            SmallerThanCondition(Variable("c", lambda x: x["Peak Price"]),
                                 Variable("d", lambda x: x["Peak Price"])),
            SmallerThanCondition(Variable("d", lambda x: x["Peak Price"]),
                                 Variable("e", lambda x: x["Peak Price"]))
        ),
        timedelta(minutes=10)
    )
    pattern3 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "a"), PrimitiveEventStructure("AMZN", "b"),
                    PrimitiveEventStructure("GOOG", "c")),
        AndCondition(
            GreaterThanCondition(Variable("a", lambda x: x["Opening Price"]),
                                 Variable("c", lambda x: x["Opening Price"])),
            GreaterThanCondition(Variable("a", lambda x: x["Opening Price"]),
                                 Variable("b", lambda x: x["Opening Price"]))),
        timedelta(minutes=5)
    )

    patterns = [pattern1, pattern2, pattern3]
    multi_pattern = MultiPattern(patterns)
    runMultiTest("ThreePatternTest_adaptive", multi_pattern, createTestFile, leaf_sharing_eval_mechanism_params)


"""
multi-pattern test checking case where output node is not a root
"""


def adaptiveSamePatternDifferentTimeStamps(createTestFile=False):
    # similar to leafIsRoot, but the time windows are different
    pattern1 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "a")),
        GreaterThanEqCondition(Variable("a", lambda x: x["Peak Price"]), 135),
        timedelta(minutes=5)
    )
    pattern2 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "a"), PrimitiveEventStructure("AMZN", "b"),
                    PrimitiveEventStructure("GOOG", "c")),
        AndCondition(
            GreaterThanEqCondition(Variable("a", lambda x: x["Peak Price"]), 135),
            SmallerThanCondition(Variable("b", lambda x: x["Peak Price"]),
                                 Variable("c", lambda x: x["Peak Price"]))
        ),
        timedelta(minutes=3)
    )

    patterns = [pattern1, pattern2]
    multi_pattern = MultiPattern(patterns)
    runMultiTest("RootAndInner", multi_pattern, createTestFile, leaf_sharing_eval_mechanism_params)


"""
multi-pattern test 2 identical patterns with different time stamp
"""


def adaptiveRootAndInner(createTestFile=False):
    pattern1 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "a"), PrimitiveEventStructure("AMZN", "b"),
                    PrimitiveEventStructure("GOOG", "c")),
        AndCondition(
            GreaterThanEqCondition(Variable("a", lambda x: x["Peak Price"]), 135),
            SmallerThanCondition(Variable("b", lambda x: x["Peak Price"]),
                                 Variable("c", lambda x: x["Peak Price"]))
        ),
        timedelta(minutes=5)
    )
    pattern2 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "a"), PrimitiveEventStructure("AMZN", "b"),
                    PrimitiveEventStructure("GOOG", "c")),
        AndCondition(
            GreaterThanEqCondition(Variable("a", lambda x: x["Peak Price"]), 135),
            SmallerThanCondition(Variable("b", lambda x: x["Peak Price"]),
                                 Variable("c", lambda x: x["Peak Price"]))
        ),
        timedelta(minutes=2)
    )

    patterns = [pattern1, pattern2]
    multi_pattern = MultiPattern(patterns)
    runMultiTest("DifferentTimeStamp_adaptive", multi_pattern, createTestFile, leaf_sharing_eval_mechanism_params)


"""
multi-pattern test sharing equivalent subtrees
"""


def adaptiveOnePatternIncludesOther(createTestFile=False):
    pattern1 = Pattern(
        SeqOperator(PrimitiveEventStructure("GOOG", "a"), PrimitiveEventStructure("GOOG", "b"),
                    PrimitiveEventStructure("AAPL", "c")),
        AndCondition(
            SmallerThanCondition(Variable("a", lambda x: x["Peak Price"]),
                                 Variable("b", lambda x: x["Peak Price"])),
            GreaterThanCondition(Variable("b", lambda x: x["Peak Price"]),
                                 Variable("c", lambda x: x["Peak Price"]))
        ),
        timedelta(minutes=3)
    )

    pattern2 = Pattern(
        SeqOperator(PrimitiveEventStructure("GOOG", "a"), PrimitiveEventStructure("GOOG", "b")),
        SmallerThanCondition(Variable("a", lambda x: x["Peak Price"]),
                             Variable("b", lambda x: x["Peak Price"]))
        ,
        timedelta(minutes=3)
    )

    patterns = [pattern1, pattern2]
    multi_pattern = MultiPattern(patterns)
    runMultiTest("onePatternIncludesOther_adaptive", multi_pattern, createTestFile, leaf_sharing_eval_mechanism_params)


"""
multi-pattern test multiple patterns share the same output node
"""


def adaptiveSamePatternSharingRoot(createTestFile=False):
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

    patterns = [hierarchyPattern, hierarchyPattern2, hierarchyPattern3]
    multi_pattern = MultiPattern(patterns)
    runMultiTest('hierarchyMultiPattern_adaptive', multi_pattern , createTestFile,
                 leaf_sharing_eval_mechanism_params)


"""
multi-pattern test several patterns sharing the same subtree
"""


def adaptiveSeveralPatternShareSubtree(createTestFile=False):
    pattern = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "a"),
                    PrimitiveEventStructure("AMZN", "b"),
                    PrimitiveEventStructure("GOOG", "c"),
                    NegationOperator(PrimitiveEventStructure("TYP1", "x")),
                    NegationOperator(PrimitiveEventStructure("TYP2", "y")),
                    NegationOperator(PrimitiveEventStructure("TYP3", "z"))),
        AndCondition(
            GreaterThanCondition(Variable("a", lambda x: x["Opening Price"]),
                                 Variable("b", lambda x: x["Opening Price"])),
            SmallerThanCondition(Variable("b", lambda x: x["Opening Price"]),
                                 Variable("c", lambda x: x["Opening Price"]))),
        timedelta(minutes=5)
    )

    pattern2 = Pattern(SeqOperator(PrimitiveEventStructure("AAPL", "a"), PrimitiveEventStructure("AMZN", "b"),
                                   PrimitiveEventStructure("TYP1", "x")),
                       GreaterThanCondition(Variable("a", lambda x: x["Opening Price"]),
                                            Variable("b", lambda x: x["Opening Price"])),
                       timedelta(minutes=5)
                       )

    pattern3 = Pattern(SeqOperator(PrimitiveEventStructure("AAPL", "a"), PrimitiveEventStructure("AMZN", "b")),
                       GreaterThanCondition(Variable("a", lambda x: x["Opening Price"]),
                                            Variable("b", lambda x: x["Opening Price"])),
                       timedelta(minutes=5)
                       )

    patterns = [pattern, pattern2, pattern3]
    multi_pattern = MultiPattern(patterns)
    runMultiTest("threeSharingSubtrees_adaptive", multi_pattern, createTestFile,
                 leaf_sharing_eval_mechanism_params)


def adaptiveNotInTheBeginningShare(createTestFile=False):
    getattr_func = lambda x: x["Opening Price"]

    pattern1 = Pattern(
        SeqOperator(NegationOperator(PrimitiveEventStructure("TYP1", "x")),
                    NegationOperator(PrimitiveEventStructure("TYP2", "y")),
                    NegationOperator(PrimitiveEventStructure("TYP3", "z")), PrimitiveEventStructure("AAPL", "a"),
                    PrimitiveEventStructure("AMZN", "b"), PrimitiveEventStructure("GOOG", "c")),
        AndCondition(
            GreaterThanCondition(Variable("a", getattr_func),
                                 Variable("b", getattr_func)),
            SmallerThanCondition(Variable("b", getattr_func),
                                 Variable("c", getattr_func))),
        timedelta(minutes=5)
    )

    pattern2 = Pattern(
        SeqOperator(NegationOperator(PrimitiveEventStructure("TYP1", "x")),
                    NegationOperator(PrimitiveEventStructure("TYP2", "y")),
                    PrimitiveEventStructure("AAPL", "a"),
                    PrimitiveEventStructure("AMZN", "b")),
        GreaterThanCondition(Variable("a", getattr_func),
                             Variable("b", getattr_func)),
        timedelta(minutes=5)
    )

    pattern3 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "a"),
                    PrimitiveEventStructure("AMZN", "b"),
                    PrimitiveEventStructure("GOOG", "c")),
        AndCondition(
            GreaterThanCondition(Variable("a", getattr_func),
                                 Variable("b", getattr_func)),
            GreaterThanCondition(Variable("c", getattr_func),
                                 Variable("b", getattr_func))
        ),
        timedelta(minutes=5)
    )

    patterns = [pattern1, pattern2, pattern3]
    multi_pattern = MultiPattern(patterns)
    runMultiTest("MultipleNotBeginningShare_adaptive", multi_pattern, createTestFile,
                 leaf_sharing_eval_mechanism_params)


"""
multi-pattern test sharing internal node between patterns
"""


def adaptiveMultipleParentsForInternalNode(createTestFile=False):
    pattern1 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "a"),
                    PrimitiveEventStructure("AMZN", "b"),
                    PrimitiveEventStructure("GOOG", "c")),
        AndCondition(
            GreaterThanCondition(Variable("a", lambda x: x["Opening Price"]),
                                 Variable("b", lambda x: x["Opening Price"])),
            GreaterThanCondition(Variable("c", lambda x: x["Peak Price"]), 500)
        ),
        timedelta(minutes=5)
    )

    pattern2 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "a"),
                    PrimitiveEventStructure("AMZN", "b"),
                    PrimitiveEventStructure("GOOG", "c")),
        AndCondition(
            GreaterThanCondition(Variable("a", lambda x: x["Opening Price"]),
                                 Variable("b", lambda x: x["Opening Price"])),
            GreaterThanCondition(Variable("c", lambda x: x["Peak Price"]), 530)
        ),
        timedelta(minutes=3)
    )

    pattern3 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "a"),
                    PrimitiveEventStructure("AMZN", "b"), PrimitiveEventStructure("FB", "e")),
        AndCondition(
            GreaterThanCondition(Variable("a", lambda x: x["Opening Price"]),
                                 Variable("b", lambda x: x["Opening Price"])),
            GreaterThanCondition(Variable("e", lambda x: x["Peak Price"]), 520)
        ),
        timedelta(minutes=5)
    )

    pattern4 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "a"),
                    PrimitiveEventStructure("AMZN", "b"), PrimitiveEventStructure("LI", "c")),
        AndCondition(
            GreaterThanCondition(Variable("a", lambda x: x["Opening Price"]),
                                 Variable("b", lambda x: x["Opening Price"])),
            GreaterThanCondition(Variable("c", lambda x: x["Peak Price"]), 100)
        ),
        timedelta(minutes=2)
    )

    patterns = [pattern1, pattern2, pattern3, pattern4]
    multi_pattern = MultiPattern(patterns)
    runMultiTest("multipleParentsForInternalNode_adaptive", multi_pattern, createTestFile,
                 leaf_sharing_eval_mechanism_params)


"""
Same tests as above are repeated below, but with subtree sharing
"""


def adaptiveLeafIsRootFullSharing(createTestFile=False):
    pattern1 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "a")),
        GreaterThanCondition(Variable("a", lambda x: x["Peak Price"]), 135),
        timedelta(minutes=5)
    )
    pattern2 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "a"), NegationOperator(PrimitiveEventStructure("AMZN", "b")),
                    PrimitiveEventStructure("GOOG", "c")),
        AndCondition(
            GreaterThanCondition(Variable("a", lambda x: x["Opening Price"]),
                                 Variable("b", lambda x: x["Opening Price"])),
            SmallerThanCondition(Variable("b", lambda x: x["Opening Price"]),
                                 Variable("c", lambda x: x["Opening Price"]))),
        timedelta(minutes=5)
    )

    patterns = [pattern1, pattern2]
    multi_pattern = MultiPattern(patterns)
    runMultiTest("FirstMultiPatternFullSharing_adaptive", multi_pattern, createTestFile,
                 subtree_sharing_eval_mechanism_params,
                 expected_file_name="FirstMultiPattern")


"""
multi-pattern test 2 completely distinct patterns
"""


def adaptiveDistinctPatternsFullSharing(createTestFile=False):
    pattern1 = Pattern(
        SeqOperator(PrimitiveEventStructure("GOOG", "a"), PrimitiveEventStructure("GOOG", "b"),
                    PrimitiveEventStructure("GOOG", "c")),
        AndCondition(
            SmallerThanCondition(Variable("a", lambda x: x["Peak Price"]),
                                 Variable("b", lambda x: x["Peak Price"])),
            SmallerThanCondition(Variable("b", lambda x: x["Peak Price"]),
                                 Variable("c", lambda x: x["Peak Price"]))
        ),
        timedelta(minutes=3)
    )
    pattern2 = Pattern(
        SeqOperator(PrimitiveEventStructure("AMZN", "x1"), PrimitiveEventStructure("AMZN", "x2"),
                    PrimitiveEventStructure("AMZN", "x3")),
        AndCondition(
            SmallerThanEqCondition(Variable("x1", lambda x: x["Lowest Price"]), 75),
            GreaterThanEqCondition(Variable("x2", lambda x: x["Peak Price"]), 78),
            SmallerThanEqCondition(Variable("x3", lambda x: x["Lowest Price"]),
                                   Variable("x1", lambda x: x["Lowest Price"]))
        ),
        timedelta(days=1)
    )

    patterns = [pattern1, pattern2]
    multi_pattern = MultiPattern(patterns)
    runMultiTest("BigMultiPatternFullSharing_adaptive", multi_pattern, createTestFile,
                 subtree_sharing_eval_mechanism_params,
                 expected_file_name="BigMultiPattern")


"""
multi-pattern test with 3 patterns and leaf sharing
"""


def adaptiveThreePatternsTestFullSharing(createTestFile=False):
    pattern1 = Pattern(
        AndOperator(PrimitiveEventStructure("AAPL", "a"), PrimitiveEventStructure("AMZN", "b"),
                    PrimitiveEventStructure("GOOG", "c")),
        AndCondition(
            SmallerThanCondition(Variable("a", lambda x: x["Peak Price"]),
                                 Variable("b", lambda x: x["Peak Price"])),
            SmallerThanCondition(Variable("b", lambda x: x["Peak Price"]),
                                 Variable("c", lambda x: x["Peak Price"]))
        ),
        timedelta(minutes=1)
    )
    pattern2 = Pattern(
        SeqOperator(PrimitiveEventStructure("MSFT", "a"), PrimitiveEventStructure("DRIV", "b"),
                    PrimitiveEventStructure("MSFT", "c"), PrimitiveEventStructure("DRIV", "d"),
                    PrimitiveEventStructure("MSFT", "e")),
        AndCondition(
            SmallerThanCondition(Variable("a", lambda x: x["Peak Price"]),
                                 Variable("b", lambda x: x["Peak Price"])),
            SmallerThanCondition(Variable("b", lambda x: x["Peak Price"]),
                                 Variable("c", lambda x: x["Peak Price"])),
            SmallerThanCondition(Variable("c", lambda x: x["Peak Price"]),
                                 Variable("d", lambda x: x["Peak Price"])),
            SmallerThanCondition(Variable("d", lambda x: x["Peak Price"]),
                                 Variable("e", lambda x: x["Peak Price"]))
        ),
        timedelta(minutes=10)
    )
    pattern3 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "a"), PrimitiveEventStructure("AMZN", "b"),
                    PrimitiveEventStructure("GOOG", "c")),
        AndCondition(
            GreaterThanCondition(Variable("a", lambda x: x["Opening Price"]),
                                 Variable("c", lambda x: x["Opening Price"])),
            GreaterThanCondition(Variable("a", lambda x: x["Opening Price"]),
                                 Variable("b", lambda x: x["Opening Price"]))),
        timedelta(minutes=5)
    )

    patterns = [pattern1, pattern2, pattern3]
    multi_pattern = MultiPattern(patterns)
    runMultiTest("ThreePatternTestFullSharing_adaptive", multi_pattern, createTestFile,
                 subtree_sharing_eval_mechanism_params,
                 expected_file_name="ThreePatternTest")


"""
multi-pattern test checking case where output node is not a root
"""


def adaptiveRootAndInnerFullSharing(createTestFile=False):
    # similar to leafIsRoot, but the time windows are different
    pattern1 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "a")),
        GreaterThanEqCondition(Variable("a", lambda x: x["Peak Price"]), 135),
        timedelta(minutes=5)
    )
    pattern2 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "a"), PrimitiveEventStructure("AMZN", "b"),
                    PrimitiveEventStructure("GOOG", "c")),
        AndCondition(
            GreaterThanEqCondition(Variable("a", lambda x: x["Peak Price"]), 135),
            SmallerThanCondition(Variable("b", lambda x: x["Peak Price"]),
                                 Variable("c", lambda x: x["Peak Price"]))
        ),
        timedelta(minutes=3)
    )

    patterns = [pattern1, pattern2]
    multi_pattern = MultiPattern(patterns)
    runMultiTest("RootAndInnerFullSharing_adaptive", multi_pattern, createTestFile, subtree_sharing_eval_mechanism_params,
                 expected_file_name="RootAndInner")


"""
multi-pattern test 2 identical patterns with different time stamp
"""


def adaptiveSamePatternDifferentTimeStampsFullSharing(createTestFile=False):
    pattern1 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "a"), PrimitiveEventStructure("AMZN", "b"),
                    PrimitiveEventStructure("GOOG", "c")),
        AndCondition(
            GreaterThanEqCondition(Variable("a", lambda x: x["Peak Price"]), 135),
            SmallerThanCondition(Variable("b", lambda x: x["Peak Price"]),
                                 Variable("c", lambda x: x["Peak Price"]))
        ),
        timedelta(minutes=5)
    )
    pattern2 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "a"), PrimitiveEventStructure("AMZN", "b"),
                    PrimitiveEventStructure("GOOG", "c")),
        AndCondition(
            GreaterThanEqCondition(Variable("a", lambda x: x["Peak Price"]), 135),
            SmallerThanCondition(Variable("b", lambda x: x["Peak Price"]),
                                 Variable("c", lambda x: x["Peak Price"]))
        ),
        timedelta(minutes=2)
    )

    patterns = [pattern1, pattern2]
    multi_pattern = MultiPattern(patterns)
    runMultiTest("DifferentTimeStampFullSharing_adaptive", multi_pattern, createTestFile,
                 subtree_sharing_eval_mechanism_params,
                 expected_file_name="DifferentTimeStamp")


"""
multi-pattern test sharing equivalent subtrees
"""


def adaptiveOnePatternIncludesOtherFullSharing(createTestFile=False):
    pattern1 = Pattern(
        SeqOperator(PrimitiveEventStructure("GOOG", "a"), PrimitiveEventStructure("GOOG", "b"),
                    PrimitiveEventStructure("AAPL", "c")),
        AndCondition(
            SmallerThanCondition(Variable("a", lambda x: x["Peak Price"]),
                                 Variable("b", lambda x: x["Peak Price"])),
            GreaterThanCondition(Variable("b", lambda x: x["Peak Price"]),
                                 Variable("c", lambda x: x["Peak Price"]))
        ),
        timedelta(minutes=3)
    )

    pattern2 = Pattern(
        SeqOperator(PrimitiveEventStructure("GOOG", "a"), PrimitiveEventStructure("GOOG", "b")),
        SmallerThanCondition(Variable("a", lambda x: x["Peak Price"]),
                             Variable("b", lambda x: x["Peak Price"]))
        ,
        timedelta(minutes=3)
    )

    patterns = [pattern1, pattern2]
    multi_pattern = MultiPattern(patterns)
    runMultiTest("onePatternIncludesOtherFullSharing_adaptive", multi_pattern, createTestFile,
                 subtree_sharing_eval_mechanism_params,
                 expected_file_name="onePatternIncludesOther")


"""
multi-pattern test multiple patterns share the same output node
"""


def adaptiveSamePatternSharingRootFullSharing(createTestFile=False):
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

    patterns = [hierarchyPattern, hierarchyPattern2, hierarchyPattern3]
    multi_pattern = MultiPattern(patterns)
    runMultiTest('hierarchyMultiPatternFullSharing_adaptive', multi_pattern,
                 createTestFile, subtree_sharing_eval_mechanism_params,
                 expected_file_name='hierarchyMultiPattern')


"""
multi-pattern test several patterns sharing the same subtree
"""


def adaptiveSeveralPatternShareSubtreeFullSharing(createTestFile=False):
    pattern = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "a"),
                    PrimitiveEventStructure("AMZN", "b"),
                    PrimitiveEventStructure("GOOG", "c"),
                    NegationOperator(PrimitiveEventStructure("TYP1", "x")),
                    NegationOperator(PrimitiveEventStructure("TYP2", "y")),
                    NegationOperator(PrimitiveEventStructure("TYP3", "z"))),
        AndCondition(
            GreaterThanCondition(Variable("a", lambda x: x["Opening Price"]),
                                 Variable("b", lambda x: x["Opening Price"])),
            SmallerThanCondition(Variable("b", lambda x: x["Opening Price"]),
                                 Variable("c", lambda x: x["Opening Price"]))),
        timedelta(minutes=5)
    )

    pattern2 = Pattern(SeqOperator(PrimitiveEventStructure("AAPL", "a"), PrimitiveEventStructure("AMZN", "b"),
                                   PrimitiveEventStructure("TYP1", "x")),
                       GreaterThanCondition(Variable("a", lambda x: x["Opening Price"]),
                                            Variable("b", lambda x: x["Opening Price"])),
                       timedelta(minutes=5)
                       )

    pattern3 = Pattern(SeqOperator(PrimitiveEventStructure("AAPL", "a"), PrimitiveEventStructure("AMZN", "b")),
                       GreaterThanCondition(Variable("a", lambda x: x["Opening Price"]),
                                            Variable("b", lambda x: x["Opening Price"])),
                       timedelta(minutes=5)
                       )

    patterns = [pattern, pattern2, pattern3]
    multi_pattern = MultiPattern(patterns)
    runMultiTest("threeSharingSubtreesFullSharing_adaptive", multi_pattern, createTestFile,
                 subtree_sharing_eval_mechanism_params,
                 expected_file_name="threeSharingSubtrees")


def adaptiveNotInTheBeginningShareFullSharing(createTestFile=False):
    getattr_func = lambda x: x["Opening Price"]

    pattern1 = Pattern(
        SeqOperator(NegationOperator(PrimitiveEventStructure("TYP1", "x")),
                    NegationOperator(PrimitiveEventStructure("TYP2", "y")),
                    NegationOperator(PrimitiveEventStructure("TYP3", "z")), PrimitiveEventStructure("AAPL", "a"),
                    PrimitiveEventStructure("AMZN", "b"), PrimitiveEventStructure("GOOG", "c")),
        AndCondition(
            GreaterThanCondition(Variable("a", getattr_func),
                                 Variable("b", getattr_func)),
            SmallerThanCondition(Variable("b", getattr_func),
                                 Variable("c", getattr_func))),
        timedelta(minutes=5)
    )

    pattern2 = Pattern(
        SeqOperator(NegationOperator(PrimitiveEventStructure("TYP1", "x")),
                    NegationOperator(PrimitiveEventStructure("TYP2", "y")),
                    PrimitiveEventStructure("AAPL", "a"),
                    PrimitiveEventStructure("AMZN", "b")),
        GreaterThanCondition(Variable("a", getattr_func),
                             Variable("b", getattr_func)),
        timedelta(minutes=5)
    )

    pattern3 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "a"),
                    PrimitiveEventStructure("AMZN", "b"),
                    PrimitiveEventStructure("GOOG", "c")),
        AndCondition(
            GreaterThanCondition(Variable("a", getattr_func),
                                 Variable("b", getattr_func)),
            GreaterThanCondition(Variable("c", getattr_func),
                                 Variable("b", getattr_func))
        ),
        timedelta(minutes=5)
    )

    patterns = [pattern1, pattern2, pattern3]
    multi_pattern = MultiPattern(patterns)
    runMultiTest("MultipleNotBeginningShareFullSharing_adaptive", multi_pattern, createTestFile,
                 subtree_sharing_eval_mechanism_params,
                 expected_file_name="MultipleNotBeginningShare")


"""
multi-pattern test sharing internal node between patterns
"""


def adaptiveMultipleParentsForInternalNodeFullSharing(createTestFile=False):
    pattern1 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "a"),
                    PrimitiveEventStructure("AMZN", "b"),
                    PrimitiveEventStructure("GOOG", "c")),
        AndCondition(
            GreaterThanCondition(Variable("a", lambda x: x["Opening Price"]),
                                 Variable("b", lambda x: x["Opening Price"])),
            GreaterThanCondition(Variable("c", lambda x: x["Peak Price"]), 500)
        ),
        timedelta(minutes=5)
    )

    pattern2 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "a"),
                    PrimitiveEventStructure("AMZN", "b"), PrimitiveEventStructure("GOOG", "c")),
        AndCondition(
            GreaterThanCondition(Variable("a", lambda x: x["Opening Price"]),
                                 Variable("b", lambda x: x["Opening Price"])),
            GreaterThanCondition(Variable("c", lambda x: x["Peak Price"]), 530)
        ),
        timedelta(minutes=3)
    )

    pattern3 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "a"),
                    PrimitiveEventStructure("AMZN", "b"), PrimitiveEventStructure("FB", "e")),
        AndCondition(
            GreaterThanCondition(Variable("a", lambda x: x["Opening Price"]),
                                 Variable("b", lambda x: x["Opening Price"])),
            GreaterThanCondition(Variable("e", lambda x: x["Peak Price"]), 520)
        ),
        timedelta(minutes=5)
    )

    pattern4 = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "a"),
                    PrimitiveEventStructure("AMZN", "b"), PrimitiveEventStructure("LI", "c")),
        AndCondition(
            GreaterThanCondition(Variable("a", lambda x: x["Opening Price"]),
                                 Variable("b", lambda x: x["Opening Price"])),
            GreaterThanCondition(Variable("c", lambda x: x["Peak Price"]), 100)
        ),
        timedelta(minutes=2)
    )

    patterns = [pattern1, pattern2, pattern3, pattern4]
    multi_pattern = MultiPattern(patterns)
    runMultiTest("multipleParentsForInternalNodeFullSharing_adaptive", multi_pattern, createTestFile,
                 subtree_sharing_eval_mechanism_params,
                 expected_file_name="multipleParentsForInternalNode")
