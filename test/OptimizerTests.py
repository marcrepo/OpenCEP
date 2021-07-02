from datetime import timedelta
from typing import Dict

from adaptive.statistics.StatisticsCollectorFactory import StatisticsCollectorFactory
from adaptive.statistics.StatisticsTypes import StatisticsTypes
from base.Pattern import Pattern
from base.PatternStructure import SeqOperator, PrimitiveEventStructure
from condition.Condition import SimpleCondition, Variable
from adaptive.optimizer.OptimizerFactory import OptimizerFactory, OptimizerParameters
from test.EvalTestsDefaults import DEFAULT_TESTING_GREEDY_INVARIANT_OPTIMIZER_SETTINGS, \
    DEFAULT_TESTING_ZSTREAM_INVARIANT_OPTIMIZER_SETTINGS, \
    DEFAULT_TESTING_STATISTICS_COLLECTOR_SELECTIVITY_AND_ARRIVAL_RATES_STATISTICS
from unittest.mock import MagicMock


def get_pattern_test():
    pattern = Pattern(
        SeqOperator(PrimitiveEventStructure("AAPL", "a"), PrimitiveEventStructure("AMZN", "b"),
                    PrimitiveEventStructure("LOCM", "c")),
        SimpleCondition(Variable("a", lambda x: x["Opening Price"]),
                        Variable("b", lambda x: x["Opening Price"]),
                        Variable("c", lambda x: x["Opening Price"]),
                        relation_op=lambda x, y, z: x > y > z),
        timedelta(minutes=5)
    )
    return pattern


def greedyInvariantOptimizerTreeChangeFailTest_1():
    """
    Basic test, check if greedy invariant aware optimizer say that
    need to generate new tree in the case that statistics doesnt change
    Expected:
    The optimizer will say that there is no need to re-optimize
    """
    optimizer_parameters = DEFAULT_TESTING_GREEDY_INVARIANT_OPTIMIZER_SETTINGS

    old_arrival_rates = [1, 5, 8, 2]
    old_selectivity_matrix = [[1.0, 0.9457796098355941, 1.0, 1.0], [0.9457796098355941, 1.0, 0.15989723367389616, 1.0],
                              [1.0, 0.15989723367389616, 1.0, 0.9992557393942864], [1.0, 1.0, 0.9992557393942864, 1.0]]
    old_statistics = {StatisticsTypes.ARRIVAL_RATES: old_arrival_rates,
                      StatisticsTypes.SELECTIVITY_MATRIX: old_selectivity_matrix}

    new_arrival_rates = [1, 5, 8, 2]
    new_selectivity_matrix = [[1.0, 0.9457796098355941, 1.0, 1.0], [0.9457796098355941, 1.0, 0.15989723367389616, 1.0],
                              [1.0, 0.15989723367389616, 1.0, 0.9992557393942864], [1.0, 1.0, 0.9992557393942864, 1.0]]
    new_statistics = {StatisticsTypes.ARRIVAL_RATES: new_arrival_rates,
                      StatisticsTypes.SELECTIVITY_MATRIX: new_selectivity_matrix}

    is_changed = optimizer_test(old_statistics, new_statistics, optimizer_parameters)
    result = "Succeeded" if not is_changed else "Failed"
    print(f"Test greedyInvariantOptimizerTreeChangeFailTest_1 result: {result}")


def greedyInvariantOptimizerTreeChangeFailTest_2():
    """
    In this test:
    We change the statistics so that the conditions in invariants should not change yet.
    Expected:
    The optimizer will say that there is no need to re-optimize
    """
    optimizer_parameters = DEFAULT_TESTING_GREEDY_INVARIANT_OPTIMIZER_SETTINGS

    old_arrival_rates = [1, 5, 8, 2]
    old_selectivity_matrix = [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0],
                              [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    old_statistics = {StatisticsTypes.ARRIVAL_RATES: old_arrival_rates,
                      StatisticsTypes.SELECTIVITY_MATRIX: old_selectivity_matrix}

    new_arrival_rates = [1, 7, 8, 2]
    new_selectivity_matrix = [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0],
                              [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    new_statistics = {StatisticsTypes.ARRIVAL_RATES: new_arrival_rates,
                      StatisticsTypes.SELECTIVITY_MATRIX: new_selectivity_matrix}

    is_changed = optimizer_test(old_statistics, new_statistics, optimizer_parameters)
    result = "Succeeded" if not is_changed else "Failed"
    print(f"Test greedyInvariantOptimizerTreeChangeFailTest_2 result: {result}")


def greedyInvariantOptimizerTreeChangeTest_1():
    """
    In this test:
    We change the statistics so that the conditions in invariants should be change.
    Expected:
    The optimizer will say that need to re-optimize
    """
    optimizer_parameters = DEFAULT_TESTING_GREEDY_INVARIANT_OPTIMIZER_SETTINGS

    old_arrival_rates = [1, 5, 8, 2]
    old_selectivity_matrix = [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0],
                              [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    old_statistics = {StatisticsTypes.ARRIVAL_RATES: old_arrival_rates,
                      StatisticsTypes.SELECTIVITY_MATRIX: old_selectivity_matrix}

    new_arrival_rates = [2, 5, 8, 1]
    new_selectivity_matrix = [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0],
                              [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    new_statistics = {StatisticsTypes.ARRIVAL_RATES: new_arrival_rates,
                      StatisticsTypes.SELECTIVITY_MATRIX: new_selectivity_matrix}

    is_changed = optimizer_test(old_statistics, new_statistics, optimizer_parameters)
    result = "Failed" if not is_changed else "Succeeded"
    print(f"Test greedyInvariantOptimizerTreeChangeTest_1 result: {result}")


def zstreamInvariantOptimizerTreeChangeFailTest_1():
    """
    Basic test, check if zstream invariant aware optimizer say that
    need to generate new tree in the case that statistics doesnt change
    Expected:
    The optimizer will say that there is no need to re-optimize
    """
    optimizer_parameters = DEFAULT_TESTING_ZSTREAM_INVARIANT_OPTIMIZER_SETTINGS

    old_arrival_rates = [5, 1, 8, 2]
    old_selectivity_matrix = [[1.0, 0.9457796098355941, 1.0, 1.0], [0.9457796098355941, 1.0, 0.15989723367389616, 1.0],
                              [1.0, 0.15989723367389616, 1.0, 0.9992557393942864], [1.0, 1.0, 0.9992557393942864, 1.0]]
    old_statistics = {StatisticsTypes.ARRIVAL_RATES: old_arrival_rates,
                      StatisticsTypes.SELECTIVITY_MATRIX: old_selectivity_matrix}
    new_arrival_rates = [1, 5, 8, 2]
    new_selectivity_matrix = [[1.0, 0.9457796098355941, 1.0, 1.0], [0.9457796098355941, 1.0, 0.15989723367389616, 1.0],
                              [1.0, 0.15989723367389616, 1.0, 0.9992557393942864], [1.0, 1.0, 0.9992557393942864, 1.0]]
    new_statistics = {StatisticsTypes.ARRIVAL_RATES: new_arrival_rates,
                      StatisticsTypes.SELECTIVITY_MATRIX: new_selectivity_matrix}

    is_changed = optimizer_test(old_statistics, new_statistics, optimizer_parameters)
    result = "Succeeded" if not is_changed else "Failed"
    print(f"Test zstreamInvariantOptimizerTreeChangeFailTest_1 result: {result}")


def zstreamInvariantOptimizerTreeChangeTest_1():
    """
    Basic test, check if zstream invariant aware optimizer say that
    need to generate new tree in the case that statistics doesnt change
    Expected:
    The optimizer will say that need to re-optimize
    """
    optimizer_parameters = DEFAULT_TESTING_ZSTREAM_INVARIANT_OPTIMIZER_SETTINGS

    old_arrival_rates = [8, 5, 1, 2, 3]
    old_selectivity_matrix = [[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0],
                              [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]]
    old_statistics = {StatisticsTypes.ARRIVAL_RATES: old_arrival_rates,
                      StatisticsTypes.SELECTIVITY_MATRIX: old_selectivity_matrix}
    new_arrival_rates = [1, 5, 3, 2, 1]
    new_selectivity_matrix = [[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0],
                              [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]]

    new_statistics = {StatisticsTypes.ARRIVAL_RATES: new_arrival_rates,
                      StatisticsTypes.SELECTIVITY_MATRIX: new_selectivity_matrix}

    is_changed = optimizer_test(old_statistics, new_statistics, optimizer_parameters)
    result = "Succeeded" if is_changed else "Failed"
    print(f"Test zstreamInvariantOptimizerTreeChangeTest_1 result: {result}")


def zstreamInvariantOptimizerTreeChangeTest_2():
    """
    Basic test, check if zstream invariant aware optimizer say that
    need to generate new tree in the case that statistics doesnt change
    Expected:
    The optimizer will say that need to re-optimize
    """
    optimizer_parameters = DEFAULT_TESTING_ZSTREAM_INVARIANT_OPTIMIZER_SETTINGS

    old_arrival_rates = [8, 5, 1, 2, 3]
    old_selectivity_matrix = [[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0],
                              [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]]
    old_statistics = {StatisticsTypes.ARRIVAL_RATES: old_arrival_rates,
                      StatisticsTypes.SELECTIVITY_MATRIX: old_selectivity_matrix}

    new_arrival_rates = [8, 5, 9, 2, 3]
    new_selectivity_matrix = [[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0],
                              [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]]
    new_statistics = {StatisticsTypes.ARRIVAL_RATES: new_arrival_rates,
                      StatisticsTypes.SELECTIVITY_MATRIX: new_selectivity_matrix}

    is_changed = optimizer_test(old_statistics, new_statistics, optimizer_parameters)
    result = "Succeeded" if is_changed else "Failed"
    print(f"Test zstreamInvariantOptimizerTreeChangeTest_2 result: {result}")


def optimizer_test(old_statistics: Dict, new_statistics: Dict, optimizer_parameters: OptimizerParameters):
    """
    Given old_statistics and new_statistics, the Optimizer decide if need to re-optimize.
    If so, return True, else return False
    """
    pattern = [get_pattern_test()]
    statistics_collector_parameters = DEFAULT_TESTING_STATISTICS_COLLECTOR_SELECTIVITY_AND_ARRIVAL_RATES_STATISTICS
    statistics_collector = StatisticsCollectorFactory.build_statistics_collector(statistics_collector_parameters,
                                                                                 pattern)
    optimizer = OptimizerFactory.build_optimizer(optimizer_parameters, statistics_collector, pattern)

    statistics_collector.get_pattern_statistics = MagicMock(return_value=old_statistics)
    optimizer.try_optimize()

    statistics_collector.get_pattern_statistics = MagicMock(return_value=new_statistics)
    new_tree_plan = optimizer.try_optimize()

    if new_tree_plan:
        return True

    return False
