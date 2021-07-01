from adaptive.optimizer.Optimizer import Optimizer
from base.Pattern import Pattern


class TrivialOptimizer(Optimizer):
    """
    Represents the trivial optimizer that always initiates plan reconstruction ignoring the statistics.
    """

    def _should_optimize(self, new_statistics: dict, pattern):
        return True

    def _build_new_plan(self, new_statistics: dict, pattern: Pattern):
        tree_plan = self._tree_plan_builder.build_tree_plan(pattern, new_statistics)
        return tree_plan
