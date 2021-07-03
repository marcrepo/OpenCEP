import copy
from datetime import timedelta
from typing import List
from base.Pattern import Pattern
from adaptive.statistics.StatisticsTypes import StatisticsTypes
from adaptive.statistics.Statistics import SelectivityStatistics, ArrivalRatesStatistics


class StatisticsFactory:
    """
    Creates a statistics given its specification.
    """

    @staticmethod
    def create_statistics(patterns: List[Pattern], stat_type: StatisticsTypes, statistics_time_window: timedelta):

        predefined_statistics = StatisticsFactory.__extract_statistics(patterns, stat_type)

        if stat_type == StatisticsTypes.ARRIVAL_RATES:
            return ArrivalRatesStatistics(statistics_time_window, patterns, predefined_statistics)
        if stat_type == StatisticsTypes.SELECTIVITY_MATRIX:
            return SelectivityStatistics(patterns, predefined_statistics)
        raise Exception("Unknown statistics type: %s" % (StatisticsTypes.stat_type,))

    @staticmethod
    def __extract_statistics(patterns, stat_type):
        """
        Extracts statistics from each pattern and returns one mapping
        """
        stat_object_to_statistics_map = {}
        for pattern in patterns:
            if pattern.statistics and stat_type in pattern.statistics:
                pattern_statistics = copy.deepcopy(pattern.statistics)
                for stat_object, statistics in pattern_statistics[stat_type].items():
                    if stat_object not in stat_object_to_statistics_map:
                        stat_object_to_statistics_map[stat_object] = statistics

        return stat_object_to_statistics_map

