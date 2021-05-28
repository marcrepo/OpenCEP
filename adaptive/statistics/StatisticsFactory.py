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
        StatisticsFactory.__update_patterns(patterns, stat_type, predefined_statistics)

        if stat_type == StatisticsTypes.ARRIVAL_RATES:
            return ArrivalRatesStatistics(statistics_time_window, patterns, predefined_statistics)
        if stat_type == StatisticsTypes.SELECTIVITY_MATRIX:
            return SelectivityStatistics(patterns, predefined_statistics)
        raise Exception("Unknown statistics type: %s" % (StatisticsTypes.stat_type,))

    @staticmethod
    def __extract_statistics(patterns, stat_type):
        items_to_item_map = {}
        # for pattern in patterns:
        #     if pattern.statistics and stat_type in pattern.statistics:
        #
        #         for item, statistics in pattern.statistics[stat_type].items():
        #             if item not in items_to_item_map:
        #                 items_to_item_map[item] = statistics

        return items_to_item_map

    @staticmethod
    def __update_patterns(patterns, stat_type, items_to_item_map):
        """
        after extract from every pattern his pre defines statistics, update all patterns if they get
        predefined statistics from other pattern. For example, 2 pattern with 'a' type and just one of them contain
        statistics for this arrival rates type, so we wand updata the other pattern to know.
        """
        for pattern in patterns:
            for item in items_to_item_map:
                if item in pattern.statistics[stat_type].keys():
                    pattern.prior_statistics_exist = True
                    continue
