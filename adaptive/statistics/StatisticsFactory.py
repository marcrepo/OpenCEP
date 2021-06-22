from datetime import timedelta
from typing import List

from base.Pattern import Pattern
from adaptive.statistics.StatisticsTypes import StatisticsTypes
from adaptive.statistics.Statistics import SelectivityStatistics, ArrivalRatesStatistics


class StatisticsFactory:
    """
    Creates a statistics collector given its specification.
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
        items_to_item_map = {}
        for pattern in patterns:
            if pattern.statistics and stat_type in pattern.statistics:
                pattern.prior_statistics_exist = True

                for item, statistics in pattern.statistics[stat_type].items():
                    if item not in items_to_item_map:
                        items_to_item_map[item] = statistics

        return items_to_item_map

