from adaptive.optimizer.Optimizer import Optimizer
from base.Pattern import Pattern


class TrivialOptimizer(Optimizer):
    """
    Represents the trivial optimizer that always initiates plan reconstruction ignoring the statistics.
    """

    def try_optimize(self):
        new_statistics = self._statistics_collector.get_specific_statistics(self._patterns[0])
        if self._should_optimize(new_statistics, self._patterns[0]):
            self.count += 1
            new_tree_plan = self._build_new_plan(new_statistics, self._patterns[0])
            return new_tree_plan
        return None

    def _should_optimize(self, new_statistics: dict, pattern):
        return True

    def _build_new_plan(self, new_statistics: dict, pattern: Pattern):
        tree_plan = self._tree_plan_builder.build_tree_plan(pattern, new_statistics)
        return tree_plan
