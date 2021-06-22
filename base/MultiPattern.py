from typing import List

from base.Pattern import Pattern


class MultiPattern:
    """
    Container of patterns
    """

    def __init__(self, patterns: List[Pattern]):
        self.__patterns = patterns
        self.__set_pattern_ids()

    def __set_pattern_ids(self):
        i = 1  # pattern IDs starts from 1
        for pattern in self.__patterns:
            pattern.id = i
            i += 1

    def get_patterns(self):
        return self.__patterns
