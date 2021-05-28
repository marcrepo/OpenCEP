import copy
from abc import ABC, abstractmethod
from datetime import timedelta, datetime
from typing import List
from base.Event import Event
from base.Pattern import Pattern


class StatisticEventData:
    """
    A container class that contains the event type along with event timestamp
    """
    def __init__(self, timestamp: datetime, event_type: str):
        self.timestamp = timestamp
        self.event_type = event_type


class Statistics(ABC):
    """
    An abstract class for the statistics.
    """

    def update(self, data):
        """
        Given the newly arrived event, update the statistics.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_all_statistics(self):
        """
        Return the current statistics.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_specific_statistics(self, pattern):
        """
        Given pattern, return his statistics
        """
        raise NotImplementedError()


class ArrivalRatesStatistics(Statistics):
    """
    Represents the arrival rates statistics.
    """

    def __init__(self, arrival_rates_time_window: timedelta, patterns: List[Pattern],
                 predefined_statistics=None):
        self.__arrival_rates_time_window = arrival_rates_time_window
        self.__events_arrival_time = []

        self.__event_type_to_patterns_map = {}
        self.__event_type_to_arrival_rates = {}
        self.__pattern_to_event_types_map = {}
        self.__pattern_to_arrival_rates = {}

        # Initialize
        self.__init_maps(patterns, predefined_statistics)

    def __create_initialize_statistics(self, pattern: Pattern, args_len: int):
        self.__pattern_to_arrival_rates[pattern] = [0.0] * args_len

    def update(self, event: Event):
        """
        Increases the arrival rate of the current event type by 1 and decreases the arrival rates of the expired events.
        """
        event_type = event.type
        event_timestamp = event.timestamp

        if event_type in self.__event_type_to_arrival_rates:
            self.__events_arrival_time.append(StatisticEventData(event_timestamp, event_type))

            self.__event_type_to_arrival_rates[event_type] += 1

        self.__remove_expired_events(event_timestamp)

    def __remove_expired_events(self, last_timestamp: datetime):
        """
        Lowers the arrival rates of the events that left the time window.
        """
        is_removed_elements = False
        for i, event_time in enumerate(self.__events_arrival_time):
            if last_timestamp - event_time.timestamp > self.__arrival_rates_time_window:
                self.__event_type_to_arrival_rates[event_time.event_type] -= 1

            else:
                is_removed_elements = True
                self.__events_arrival_time = self.__events_arrival_time[i:]
                break

        if not is_removed_elements:
            self.__events_arrival_time = []

    def get_specific_statistics(self, pattern):
        """
        Given pattern, return his corresponding selectivity statistics for tree builder
        """
        arrival_rates = self.__pattern_to_arrival_rates[pattern]
        for i, event_type in enumerate(self.__pattern_to_event_types_map[pattern]):
            arrival_rates[i] = self.__event_type_to_arrival_rates[event_type]

        return copy.deepcopy(arrival_rates)

    def get_all_statistics(self):
        """
        Calculates the selectivity of all the conditions.
        If there is no condition between any pair of events then the selectivity is always 1.0
        """
        return copy.deepcopy(self.__event_type_to_arrival_rates)

    def __init_maps(self, patterns: List[Pattern], predefined_arrival_rates):
        for pattern in patterns:
            primitive_events = pattern.get_primitive_events()
            # For reconstruction
            self.__pattern_to_event_types_map[pattern] = [primitive_event.type for primitive_event in primitive_events]

            self.__create_initialize_statistics(pattern, len(primitive_events))

            #  This for know after which pattern was changed(saves us run time-eliminate from recursion)
            for primitive_event in primitive_events:
                event_type = primitive_event.type
                if event_type not in self.__event_type_to_patterns_map:
                    self.__event_type_to_patterns_map[event_type] = set()
                self.__event_type_to_patterns_map[event_type].add(pattern)

                if event_type not in self.__event_type_to_arrival_rates:

                    if event_type in predefined_arrival_rates:
                        arrival_rates = predefined_arrival_rates[event_type]
                    else:
                        arrival_rates = 0.0

                    self.__event_type_to_arrival_rates[event_type] = arrival_rates


class SelectivityStatistics(Statistics):

    def __init__(self, patterns: List[Pattern], predefined_statistics=None):
        self.__atomic_condition_to_total_map = {}
        self.__atomic_condition_to_success_map = {}
        self.__indices_to_atomic_condition_map = {}
        self.__atomic_condition_to_patterns_map = {}
        self.__atomic_condition_to_selectivity_map = {}
        self.__pattern_to_selectivity_matrix_map = {}
        self.__pattern_to_another_dict = {}

        # Initialize
        self.__init_maps(patterns, predefined_statistics)

    def __create_initialize_statistics(self, pattern: Pattern, args_len: int):
        """
        Given pattern, initialize his corresponding selectivity statistics with ones.
        Note that not all entrances in the matrix will update by run time (because just
        2 events and conditions between them can be changed).
        Hence entrances that will not change remain 1.
        """
        self.__pattern_to_selectivity_matrix_map[pattern] = [[1.0 for _ in range(args_len)] for _ in range(args_len)]

    def update(self, data):
        """
        Updates the selectivity of an atomic condition.
        """
        (atomic_condition, is_condition_success) = data

        if atomic_condition:
            atomic_condition_id = str(atomic_condition)
            if atomic_condition_id in self.__atomic_condition_to_total_map:
                self.__atomic_condition_to_total_map[atomic_condition_id] += 1
                if is_condition_success:
                    self.__atomic_condition_to_success_map[atomic_condition_id] += 1

    def get_specific_statistics(self, pattern):
        """
        Given pattern, return his corresponding selectivity statistics for tree builder
        """
        selectivity_matrix = self.__pattern_to_selectivity_matrix_map[pattern]

        #  Thank to initialization, currently, we can update just entrances in the matrix
        #  that could be change in run time
        indices_to_atomic_condition_map = self.__pattern_to_another_dict[pattern]
        for (i, j), atomic_conditions_id in indices_to_atomic_condition_map.items():
            selectivity = self.__atomic_condition_to_selectivity_map[atomic_conditions_id]
            selectivity_matrix[i][j] = selectivity_matrix[j][i] = selectivity

        return copy.deepcopy(selectivity_matrix)

    def get_all_statistics(self):
        """
        Calculates the selectivity of all the conditions.
        If there is no condition between any pair of events then the selectivity is always 1.0
        """
        for atomic_condition_id in self.__atomic_condition_to_total_map:
            selectivity = self.__compute_selectivity(atomic_condition_id)
            self.__atomic_condition_to_selectivity_map[atomic_condition_id] = selectivity

        return copy.deepcopy(self.__atomic_condition_to_selectivity_map)

    def __compute_selectivity(self, atomic_condition_id):
        selectivity = 1.0
        numerator = self.__atomic_condition_to_success_map[atomic_condition_id]
        denominator = self.__atomic_condition_to_total_map[atomic_condition_id]
        if denominator != 0.0:
            selectivity *= (numerator / denominator)

        return selectivity

    def __init_maps(self, patterns: List[Pattern], predefined_selectivity):
        """
        Initiates the success counters and total evaluation counters for each pair of event types.
        """
        for pattern in patterns:
            primitive_events = pattern.get_primitive_events()
            self.__create_initialize_statistics(pattern, len(primitive_events))

            indices_to_atomic_condition_map = {}

            for i in range(len(primitive_events)):
                for j in range(i + 1):
                    conditions = pattern.condition.get_condition_of({primitive_events[i].name, primitive_events[j].name})
                    atomic_conditions = conditions.extract_atomic_conditions()
                    for atomic_condition in atomic_conditions:
                        if atomic_condition:
                            atomic_condition_id = str(atomic_condition)

                            #  This for know after which pattern was changed(saves us run time-eliminate from recursion)
                            if atomic_condition_id in self.__atomic_condition_to_patterns_map:
                                self.__atomic_condition_to_patterns_map[atomic_condition_id].append(pattern)
                            else:
                                self.__atomic_condition_to_patterns_map[atomic_condition_id] = [pattern]

                            if atomic_condition_id not in self.__atomic_condition_to_selectivity_map:
                                if atomic_condition_id in predefined_selectivity:
                                    success, total = predefined_selectivity[atomic_condition_id]
                                else:
                                    success = total = 0.0
                                self.__atomic_condition_to_success_map[atomic_condition_id] = success
                                self.__atomic_condition_to_total_map[atomic_condition_id] = total
                                self.__atomic_condition_to_selectivity_map[atomic_condition_id] = \
                                    self.__compute_selectivity(atomic_condition_id)

                            indices_to_atomic_condition_map[(i, j)] = atomic_condition_id

            self.__pattern_to_another_dict[pattern] = indices_to_atomic_condition_map
