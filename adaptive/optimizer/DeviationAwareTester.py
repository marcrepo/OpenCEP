from abc import ABC, abstractmethod
from typing import List, Pattern


class DeviationAwareTester(ABC):
    """
    An abstract class for deviation-aware testing functions.
    """
    def __init__(self, deviation_threshold: float):
        self._deviation_threshold = deviation_threshold

    @abstractmethod
    def is_deviated_by_t(self, new_statistics, prev_statistics):
        """
        Checks if there was a deviation in one of the statistics by the given factor.
        """
        raise NotImplementedError()


class ArrivalRatesDeviationAwareTester(DeviationAwareTester):
    """
    Checks for deviations in the arrival rates statistics by the given factor.
    In single pattern, when ew find change we can stop the tester.
    """
    def is_deviated_by_t(self, new_statistics: List[int], prev_statistics: List[int]):
        for i in range(len(new_statistics)):
            if prev_statistics[i] * (1 + self._deviation_threshold) < new_statistics[i] or \
                    prev_statistics[i] * (1 - self._deviation_threshold) > new_statistics[i]:
                return True
        return False


class SelectivityDeviationAwareOptimizerTester(DeviationAwareTester):
    """
    Checks for deviations in the selectivity matrix statistics by the given factor.
    In single pattern, when ew find change we can stop the tester.
    """
    def is_deviated_by_t(self, new_statistics: List[List[float]], prev_statistics: List[List[float]]):
        for i in range(len(new_statistics)):
            for j in range(i+1):
                if prev_statistics[i][j] * (1 + self._deviation_threshold) < new_statistics[i][j] or \
                        prev_statistics[i][j] * (1 - self._deviation_threshold) > new_statistics[i][j]:
                    return True
        return False


# class SinglePatternArrivalRatesDeviationAwareTester(ArrivalRatesDeviationAwareTester):
#     """
#     Checks for deviations in the arrival rates statistics by the given factor.
#     In single pattern, when ew find change we can stop the tester.
#     """
#     def is_deviated_by_t(self, new_statistics: List[int], prev_statistics: List[int]):
#         for i in range(len(new_statistics)):
#             if prev_statistics[i] * (1 + self._deviation_threshold) < new_statistics[i] or \
#                     prev_statistics[i] * (1 - self._deviation_threshold) > new_statistics[i]:
#                 return True
#         return False


# class SinglePatternSelectivityDeviationAwareOptimizerTester(SelectivityDeviationAwareTester):
#     """
#     Checks for deviations in the selectivity matrix statistics by the given factor.
#     In single pattern, when ew find change we can stop the tester.
#     """
#     def is_deviated_by_t(self, new_statistics: List[List[float]], prev_statistics: List[List[float]]):
#         for i in range(len(new_statistics)):
#             for j in range(i+1):
#                 if prev_statistics[i][j] * (1 + self._deviation_threshold) < new_statistics[i][j] or \
#                         prev_statistics[i][j] * (1 - self._deviation_threshold) > new_statistics[i][j]:
#                     return True
#         return False


# class MultiPatternPatternArrivalRatesDeviationAwareTester(ArrivalRatesDeviationAwareTester):
#     """
#     Checks for deviations in the arrival rates statistics by the given factor.
#     Unlike single pattern, here it is not enough just to see if there was
#     a change but We need to know all the changes that were so that later we
#     can find what parts of the forest we want to change. Hence we  cant stop
#     where we find change.
#     """
#     def is_deviated_by_t(self, new_statistics, prev_statistics: List[int]):
#         pattern_was_changed = set()
#         indices_was_changed = []
#         statistics = new_statistics.statistics
#         indices_to_pattern_map = new_statistics.indices_to_pattern_map
#         for i in range(len(statistics)):
#             if prev_statistics[i] * (1 + self._deviation_threshold) < statistics[i] or \
#                     prev_statistics[i] * (1 - self._deviation_threshold) > statistics[i]:
#                 indices_was_changed.append(i)
#
#         for index in indices_was_changed:
#             pattern_was_changed.add(indices_to_pattern_map[index])
#
#         return pattern_was_changed
#
#
# class MultiPatternSelectivityDeviationAwareOptimizerTester(SelectivityDeviationAwareTester):
#     """
#     Checks for deviations in the selectivity statistics by the given factor.
#     Unlike single pattern, here it is not enough just to see if there was
#     a change but We need to know all the changes that were so that later we
#     can find what parts of the forest we want to change. Hence we  cant stop
#     where we find change.
#     """
#     def __init__(self, patterns: List[Pattern], deviation_threshold: float):
#         super().__init__(deviation_threshold)
#         self.__patterns = patterns
#
#     def is_deviated_by_t(self, new_statistics: StatisticsWrapper, prev_statistics: List[List[float]]):
#         pattern_was_changed = []
#         for pattern_id, statistics in new_statistics.get_statistics():
#             if super().is_deviated_by_t(statistics, prev_statistics):
#                 pattern_was_changed.append(pattern_id)
#         return pattern_was_changed
