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

currentPath = pathlib.Path(os.path.dirname(__file__))
absolutePath = str(currentPath.parent)
sys.path.append(absolutePath)

local_search_eval_mechanism_params = TreeBasedEvaluationMechanismParameters(
    optimizer_params=OptimizerParameters(opt_type=OptimizerTypes.TRIVIAL_OPTIMIZER,
                                         tree_plan_params=
                                         TreePlanBuilderParameters(builder_type=TreePlanBuilderTypes.TRIVIAL_LEFT_DEEP_TREE,
                              cost_model_type=TreeCostModels.INTERMEDIATE_RESULTS_TREE_COST_MODEL,
                              tree_plan_merger_type=MultiPatternTreePlanMergeApproaches.TREE_PLAN_LOCAL_SEARCH,
                              tree_plan_merger_params=['TabuSearch', timedelta(seconds=30)])),
    storage_params=TreeStorageParameters(sort_storage=False, clean_up_interval=10, prioritize_sorting_by_timestamp=True))

sub_tree_sharing_eval_mechanism_params = TreeBasedEvaluationMechanismParameters(
    optimizer_params=OptimizerParameters(opt_type=OptimizerTypes.TRIVIAL_OPTIMIZER,
                                         tree_plan_params=
                                         TreePlanBuilderParameters(builder_type=TreePlanBuilderTypes.TRIVIAL_LEFT_DEEP_TREE,
                              cost_model_type=TreeCostModels.INTERMEDIATE_RESULTS_TREE_COST_MODEL,
                              tree_plan_merger_type=MultiPatternTreePlanMergeApproaches.TREE_PLAN_SUBTREES_UNION,
                              tree_plan_merger_params=['TabuSearch', timedelta(seconds=30)])),
    storage_params=TreeStorageParameters(sort_storage=False, clean_up_interval=10, prioritize_sorting_by_timestamp=True))


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

#todo comment: can use runStructuralTest for testing mpg only(i do not think i can use it directly)

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


#todo: what is should return in order to ilyas response
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


"""
full tests
"""

# SEQ(Z,And(a,b),And(c,d)) AND(And(a,b),Y,And(c,d)) ====> And(a,b)
def nested_And(createTestFile=False):
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
    pattern1 = Pattern(
        SeqOperator(AndOperator(PrimitiveEventStructure("AAPL", "a"),
                    PrimitiveEventStructure("AAPL", "b")),PrimitiveEventStructure("AAPL", "y"),
                    AndOperator(PrimitiveEventStructure("AAPL", "c"), PrimitiveEventStructure("AAPL", "d"))),
        AndCondition(GreaterThanEqCondition(Variable("y", lambda x: x["Peak Price"]), 135),
                     GreaterThanCondition(Variable("a", lambda x: x["Opening Price"]),
                                          Variable("b", lambda x: x["Opening Price"])),
                     SmallerThanCondition(Variable("c", lambda x: x["Opening Price"]),
                                          Variable("d", lambda x: x["Opening Price"]))
                     ),
        timedelta(minutes=5)
    )

    runMultiTest("nested_And", [pattern0, pattern1], createTestFile, local_search_eval_mechanism_params)

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
