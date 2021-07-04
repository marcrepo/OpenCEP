from typing import Dict, Set
from base.Pattern import Pattern
from plan.TreePlan import TreePlan, TreePlanNode, TreePlanLeafNode, TreePlanNestedNode, TreePlanUnaryNode, \
    TreePlanBinaryNode
from plan.multi.TreePlanMerger import TreePlanMerger


class RecursiveTraversalTreePlanMerger(TreePlanMerger):
    """
    An abstract class for tree plan mergers functioning by recursively traversing the provided tree plans.
    """

    def merge_tree_plans(self, pattern_to_tree_plan_map: Dict[Pattern, TreePlan], known_unique_tree_plan_nodes):
        merged_pattern_to_tree_plan_map = {}
        for pattern, tree_plan in pattern_to_tree_plan_map.items():
            merged_pattern_to_tree_plan_map[pattern] = TreePlan(self.traverse_tree_plan(tree_plan.root,
                                                                                        known_unique_tree_plan_nodes))
        return merged_pattern_to_tree_plan_map

    def traverse_tree_plan(self, current: TreePlanNode, known_unique_tree_plan_nodes: Set[TreePlanNode],
                           is_merged: list = None):
        """
        Recursively traverses a tree plan and attempts to merge it with previously traversed subtrees. 
        """
        equivalent_node = self.__find_node_to_share(current, known_unique_tree_plan_nodes)
        if equivalent_node is not None:
            if is_merged is not None:
                is_merged[0] = True
            return equivalent_node
        known_unique_tree_plan_nodes.add(current)
        if isinstance(current, TreePlanLeafNode):
            return current
        if isinstance(current, TreePlanNestedNode):
            current.sub_tree_plan = self.traverse_tree_plan(current.sub_tree_plan, known_unique_tree_plan_nodes, is_merged)
            return current
        if isinstance(current, TreePlanUnaryNode):
            current.child = self.traverse_tree_plan(current.child, known_unique_tree_plan_nodes, is_merged)
            return current
        if isinstance(current, TreePlanBinaryNode):
            current.left_child = self.traverse_tree_plan(current.left_child, known_unique_tree_plan_nodes, is_merged)
            current.right_child = self.traverse_tree_plan(current.right_child, known_unique_tree_plan_nodes, is_merged)
            return current
        raise Exception("Unexpected node type: %s" % (type(current),))

    def __find_node_to_share(self, node: TreePlanNode, known_unique_tree_plan_nodes: Set[TreePlanNode]):
        """
        Attempts to find a node in the given collection of potentially shared nodes with which the given unshared
        node can be replaced.
        """
        for candidate_node in known_unique_tree_plan_nodes:
            if self._are_suitable_for_share(candidate_node, node):
                return candidate_node
        return None

    def _are_suitable_for_share(self, first_node: TreePlanNode, second_node: TreePlanNode):
        """
        Returns True if the two given nodes can be represented by a single, shared node in the global plan, and False
        otherwise.
        """
        raise NotImplementedError()
