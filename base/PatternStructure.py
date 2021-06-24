"""
The pattern structure consists of an arbitrarily nested expression over the set of the supported operators.
As of now, OpenCEP supports three n-ary operators (SEQ, AND, OR) and two unary operators (NOT, KL). More operators
could be added in the future.
"""
from abc import ABC
from functools import reduce

KC_MIN_SIZE = 1
KC_MAX_SIZE = None


class PatternStructure(ABC):
    """
    The top class in the pattern structure hierarchy.
    """

    def get_top_operator(self):
        """
        Returns the operator type at the top of this pattern structure.
        """
        return type(self)

    def duplicate(self):
        """
        Returns a deep copy of this pattern structure.
        """
        raise NotImplementedError()

    def contains_event(self, event_name: str):
        """
        Returns True if this structure contains an event specified by the given name and False otherwise.
        """
        raise NotImplementedError()

    def get_all_event_names(self):
        """
        Returns all event names participating in this structure.
        """
        raise NotImplementedError()


class PrimitiveEventStructure(PatternStructure):
    """
    Represents a simple primitive event, defined by a type and a name.
    """
    def __init__(self, event_type: str, name: str):
        self.type = event_type
        self.name = name

    def duplicate(self):
        return PrimitiveEventStructure(self.type, self.name)

    def __eq__(self, other):
        return type(self) == type(other) and self.name == other.name and self.type == other.type


    def contains_event(self, event_name: str):
        return self.name == event_name

    def get_all_event_names(self):
        return [self.name]

    def __hash__(self):
        return hash((self.type, self.name))

    def hash_objects(self):
        return [self.type, self.name]

    def __repr__(self):
        return "%s %s" % (self.type, self.name)


class UnaryStructure(PatternStructure, ABC):
    """
    Represents a pattern structure with an unary operator at the top level.
    """
    def __init__(self, arg):
        self.arg = arg

    def __eq__(self, other):
        return type(self) == type(other) and self.arg == other.arg

    def contains_event(self, event_name: str):
        return self.arg.contains_event(event_name)

    def get_all_event_names(self):
        return self.arg.get_all_event_names()

    def calc_structure_intersections(self, other):
        """
        returns the structure intersections of the operator args
        """
        raise NotImplementedError()


class CompositeStructure(PatternStructure, ABC):
    """
    Represents a pattern structure with a multinary operator at the top level.
    """
    def __init__(self, *args):
        self.args = list(args)

    def duplicate(self):
        new_structure = self.duplicate_top_operator()
        new_structure.args = [arg.duplicate() for arg in self.args]
        return new_structure

    def duplicate_top_operator(self):
        raise NotImplementedError()

    def __eq__(self, other):
        if type(self) != type(other) or len(self.args) != len(other.args):
            return False
        for i in range(len(self.args)):
            if self.args[i] != other.args[i]:
                return False
        return True

    def __hash__(self):
        hash_objects = []
        for arg in self.args:
            hash_objects.extend(arg.hash_objects())
        return hash(frozenset(hash_objects))

    def contains_event(self, event_name: str):
        for arg in self.args:
            if arg.contains_event(event_name):
                return True
        return False

    def get_all_event_names(self):
        return reduce(lambda x, y: x+y, [arg.get_all_event_names() for arg in self.args])

    def calc_structure_intersections(self, other):
        """
        returns the structure intersections of the operator args
        """
        raise NotImplementedError()


class AndOperator(CompositeStructure):
    def duplicate_top_operator(self):
        return AndOperator()

    def __repr__(self):
        return "AND(%s)" % (self.args,)

    def calc_structure_intersections(self, other):
        self_events = set(self.args)
        other_events = set(other.args)
        events_intersection = self_events.intersection(other_events)
        return [events_intersection]


class OrOperator(CompositeStructure):
    def duplicate_top_operator(self):
        return OrOperator()

    def __repr__(self):
        return "OR(%s)" % (self.args,)


class SeqOperator(CompositeStructure):
    def duplicate_top_operator(self):
        return SeqOperator()

    def __repr__(self):
        return "SEQ(%s)" % (self.args,)

    def calc_structure_intersections(self, other):
        """
        Here could be return 2 structure intersections like in that case:
        seq(a,b,c)
        seq(a,c,b)  =====> seq(a,b) ,seq(a,c)
        """
        self_events = set(self.args)
        other_events = set(other.args)
        self_mutual_event_list = [arg for arg in self.args if arg in other_events]
        other_mutual_event_list = [arg for arg in other.args if arg in self_events]
        if self_mutual_event_list == other_mutual_event_list:
            return [set(self_mutual_event_list)]
        return [set(self.calc_structure_intersections_according_to_first(self_mutual_event_list, other_mutual_event_list)),
                set(self.calc_structure_intersections_according_to_first(other_mutual_event_list, self_mutual_event_list))]

    def calc_structure_intersections_according_to_first(self, first_event_intersection, second_event_intersection):
        """
        calc the event intersection according to first events list.
        """
        second_idx = 0
        second_len = len(second_event_intersection)
        result = []
        for event in first_event_intersection:
            while second_idx < second_len and second_event_intersection[second_idx] != event:
                second_idx += 1
            if second_idx == second_len:
                break
            result.append(event)
        return result





class KleeneClosureOperator(UnaryStructure):
    def __init__(self, arg: PatternStructure, min_size=KC_MIN_SIZE, max_size=KC_MAX_SIZE):
        super().__init__(arg)
        if min_size <= 0:
            raise Exception("Invalid Argument: KleeneClosure node min_size <= 0!")
        # enforce min_size <= max_size
        if max_size is not None and max_size < min_size:
            raise Exception("Invalid Argument: KleeneClosure node max_size < min_size!")
        self.min_size = min_size
        self.max_size = max_size

    def duplicate(self):
        return KleeneClosureOperator(self.arg.duplicate(), self.min_size, self.max_size)

    def __repr__(self):
        return "(%s)+" % (self.arg,)


class NegationOperator(UnaryStructure):
    def duplicate(self):
        return NegationOperator(self.arg.duplicate())

    def __repr__(self):
        return "NOT(%s)" % (self.arg,)
