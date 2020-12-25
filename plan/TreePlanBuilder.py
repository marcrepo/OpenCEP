from abc import ABC
from datetime import timedelta
from typing import Dict, List

from base.Pattern import Pattern
from base.PatternStructure import AndOperator, SeqOperator, PatternStructure, UnaryStructure, PrimitiveEventStructure, \
    CompositeStructure
from misc.ConsumptionPolicy import ConsumptionPolicy
from plan.TreeCostModel import TreeCostModelFactory
from plan.TreeCostModels import TreeCostModels
from plan.TreePlan import TreePlan, TreePlanNode, OperatorTypes, TreePlanBinaryNode, TreePlanLeafNode, \
    TreePlanUnaryNode, TreePlanInternalNode
from plan.multi.MultiPatternUnifiedTreePlanApproaches import MultiPatternTreePlanUnionApproaches
from tree.Tree import Tree
from tree.nodes.LeafNode import LeafNode
from tree.nodes.Node import Node


class TreePlanBuilder(ABC):
    """
    The base class for the builders of tree-based plans.
    """

    def __init__(self, cost_model_type: TreeCostModels):
        self.__cost_model = TreeCostModelFactory.create_cost_model(cost_model_type)

    def build_tree_plan(self, pattern: Pattern):
        """
        Creates a tree-based evaluation plan for the given pattern.
        """
        return TreePlan(self._create_tree_topology(pattern))

    def _create_tree_topology(self, pattern: Pattern):
        """
        An abstract method for creating the actual tree topology.
        """
        raise NotImplementedError()

    def _union_tree_plans(self, pattern_to_tree_plan_map: Dict[Pattern, TreePlan] or TreePlan,
                          tree_plan_union_approach: MultiPatternTreePlanUnionApproaches):
        if isinstance(pattern_to_tree_plan_map, TreePlan) or len(pattern_to_tree_plan_map) == 1:
            return pattern_to_tree_plan_map
        if tree_plan_union_approach == MultiPatternTreePlanUnionApproaches.TREE_PLAN_TRIVIAL_SHARING_LEAVES:
            return self.__share_leaves(pattern_to_tree_plan_map)
        if tree_plan_union_approach == MultiPatternTreePlanUnionApproaches.TREE_PLAN_SUBTREES_UNION:
            return self.__construct_subtrees_union_tree_plan(pattern_to_tree_plan_map)
        else:
            raise Exception("Unsupported union algorithm, yet")

    def _get_plan_cost(self, pattern: Pattern, plan: TreePlanNode):
        """
        Returns the cost of a given plan for the given plan according to a predefined cost model.
        """
        return self.__cost_model.get_plan_cost(pattern, plan)

    @staticmethod
    def _instantiate_binary_node(pattern: Pattern, left_subtree: TreePlanNode, right_subtree: TreePlanNode):
        """
        A helper method for the subclasses to instantiate tree plan nodes depending on the operator.
        """
        pattern_structure = pattern.positive_structure
        if isinstance(pattern_structure, AndOperator):
            operator_type = OperatorTypes.AND
        elif isinstance(pattern_structure, SeqOperator):
            operator_type = OperatorTypes.SEQ
        else:
            raise Exception("Unsupported binary operator")
        return TreePlanBinaryNode(operator_type, left_subtree, right_subtree)

    def __try_to_share_and_merge_nodes(self, root, node):
        """
            This method is trying to share the node (and its subtree) and the tree of root.
            If the root and node are not equivalent, trying to share the children of node and root.
        """
        if self.__find_and_merge_node_into_subtree(root, node) == node:
            return node
        if isinstance(node, TreePlanBinaryNode):
            partial_root = self.__try_to_share_and_merge_nodes(root, node.left_child)
            if partial_root != root:
                return node.left_child

            partial_root = self.__try_to_share_and_merge_nodes(root, node.right_child)
            if partial_root != root:
                return node.right_child

        if isinstance(node, TreePlanUnaryNode):
            partial_root = self.__try_to_share_and_merge_nodes(root, node.child)
            if partial_root != root:
                return node.child

        return root

    def __find_and_merge_node_into_subtree(self, root: TreePlanNode, node: TreePlanNode):
        """
               This method is trying to find node in the subtree of root (or an equivalent node).
               If such a node is found, it merges the equivalent nodes.
        """
        if TreePlanBuilder.is_equivalent(root, node, self.leaves_dict):
            return node

        elif isinstance(root, TreePlanBinaryNode):
            root.left_child = self.__find_and_merge_node_into_subtree(root.left_child, node)
            root.right_child = self.__find_and_merge_node_into_subtree(root.right_child, node)

        elif isinstance(root, TreePlanUnaryNode):
            root.child = self.__find_and_merge_node_into_subtree(root.child, node)

        return root

    def __merge_nodes(self, node: TreePlanNode, other: TreePlanNode):
        """
        Merge two nodes, and update all the required information
        """
        # merges other into node
        node = other

    def __construct_subtrees_union_tree_plan(self, pattern_to_tree_plan_map: Dict[Pattern, TreePlan] or TreePlan):
        """
        This method gets patterns, builds a single-pattern tree to each one of them,
        and merges equivalent subtrees from different trees.
        We are sharing the maximal subtree that exists in the tree.
        We are assuming that each pattern appears only once in patterns (which is a legitimate assumption).
        """
        # trees = self.__construct_trees_for_patterns(pattern_to_tree_plan_map, storage_params)
        leaves_dict = {}
        # first_pattern = list(pattern_to_tree_plan_map.keys())[0]
        for i, pattern in enumerate(pattern_to_tree_plan_map):
            tree_plan = pattern_to_tree_plan_map[pattern]
            leaves_dict[pattern] = TreePlanBuilder.tree_get_leaves(pattern.positive_structure, tree_plan.root,
                                                                   Tree.__get_operator_arg_list(pattern.positive_structure),
                                                                   pattern.window, pattern.consumption_policy)

        self.leaves_dict = leaves_dict

        output_nodes = []
        pattern_to_output_node_dict = {}
        for i, pattern in enumerate(pattern_to_tree_plan_map):
            tree_plan = pattern_to_tree_plan_map[pattern]
            current_root = tree_plan.root
            pattern_id = i + 1
            output_nodes.append(current_root)
            pattern_to_output_node_dict[pattern_id] = current_root
            for root in output_nodes:
                if root == current_root:
                    break
                if self.__try_to_share_and_merge_nodes(root, current_root):
                    break
        return output_nodes

    def __share_leaves(self, pattern_to_tree_plan_map: Dict[Pattern, TreePlan] or TreePlan):
        """ this function is the core implementation for the trivial_sharing_trees algorithm """
        leaves_dict = {}
        first_pattern = list(pattern_to_tree_plan_map.keys())[0]
        for i, pattern in enumerate(pattern_to_tree_plan_map):
            tree_plan = pattern_to_tree_plan_map[pattern]
            leaves_dict[pattern] = TreePlanBuilder.tree_get_leaves(pattern.positive_structure, tree_plan.root,
                                                                   Tree.__get_operator_arg_list(pattern.positive_structure),
                                                                   pattern.window, pattern.consumption_policy)
        shared_leaves_dict = {}
        for leaf in leaves_dict[first_pattern]:
            shared_leaves_dict[leaf] = [first_pattern,
                                        TreePlanLeafNode(leaf.event_index, leaf.event_type, leaf.event_name)]
        for pattern in list(pattern_to_tree_plan_map)[1:]:
            curr_leaves = leaves_dict[pattern]
            for curr_leaf in curr_leaves.keys():
                leaf_tree_plan_node = TreePlanLeafNode(curr_leaf.event_index, curr_leaf.event_type,
                                                       curr_leaf.event_name)
                for leaf in shared_leaves_dict.keys():
                    leaf_pattern, _ = shared_leaves_dict[leaf]
                    curr_leaf_pattern = pattern
                    curr_tree_node = leaves_dict[curr_leaf_pattern][curr_leaf]
                    leaf_tree_node = leaves_dict[leaf_pattern][leaf]
                    if curr_tree_node.get_event_name() == leaf_tree_node.get_event_name() \
                            and curr_tree_node.is_equivalent(leaf_tree_node):
                        _, leaf_tree_plan_node = shared_leaves_dict[leaf]
                        break
                shared_leaves_dict[curr_leaf] = [pattern, leaf_tree_plan_node]
        return TreePlanBuilder._tree_plans_update_leaves(pattern_to_tree_plan_map,
                                                         shared_leaves_dict)  # the unified tree

    @staticmethod
    def is_equivalent(plan_node1: TreePlanNode, plan_node2: TreePlanNode, leaves_dict: Dict[TreePlanNode, Node]):

        if type(plan_node1) != type(plan_node2):
            return False

        nodes_type = type(plan_node1)

        if nodes_type == TreePlanInternalNode:
            if plan_node1.operator != plan_node2.operator:
                return False
            if nodes_type == TreePlanUnaryNode:
                return TreePlanBuilder.is_equivalent(plan_node1.child, plan_node2.child, leaves_dict)

            if nodes_type == TreePlanBinaryNode:
                return TreePlanBuilder.is_equivalent(plan_node1.left_child, plan_node2.left_child, leaves_dict) \
                       and TreePlanBuilder.is_equivalent(plan_node1.right_child, plan_node2.right_child, leaves_dict)

        if nodes_type == TreePlanLeafNode:
            leaf_node1 = leaves_dict.get(plan_node1)
            leaf_node2 = leaves_dict.get(plan_node2)
            if not leaf_node1 and not leaf_node2:
                return leaf_node1.get_event_name() == leaf_node2.get_event_name() and leaf_node1.is_equivalent(leaf_node2)

        return False

    @staticmethod
    def tree_get_leaves(root_operator: PatternStructure, tree_plan: TreePlanNode,
                        args: List[PatternStructure], sliding_window: timedelta,
                        consumption_policy: ConsumptionPolicy):
        if tree_plan is None:
            return {}
        leaves_nodes = {}
        if type(tree_plan) == TreePlanLeafNode:
            # a special case where the top operator of the entire pattern is an unary operator
            leaves_nodes[tree_plan] = LeafNode(sliding_window, tree_plan.event_index, root_operator.args[tree_plan.event_index], None)
            return leaves_nodes
        leaves_nodes = TreePlanBuilder.tree_get_leaves(root_operator, tree_plan.left_child, args, sliding_window, consumption_policy)
        leaves_nodes.update(TreePlanBuilder.tree_get_leaves(root_operator, tree_plan.right_child, args, sliding_window, consumption_policy))
        return leaves_nodes

    @staticmethod
    def _tree_plans_update_leaves(pattern_to_tree_plan_map: Dict[Pattern, TreePlan], shared_leaves_dict):
        """at this function we share same leaves in different patterns to share the same root ,
        that says one instance of each unique leaf is created ,
        (if a leaf exists in two trees , the leaf will have two parents)
        """
        for pattern, tree_plan in pattern_to_tree_plan_map.items():
            updated_tree_plan_root = TreePlanBuilder._single_tree_plan_update_leaves(tree_plan.root, shared_leaves_dict)
            pattern_to_tree_plan_map[pattern].root = updated_tree_plan_root
        return pattern_to_tree_plan_map

    @staticmethod
    def _single_tree_plan_update_leaves(tree_plan_root_node: TreePlanNode, shared_leaves_dict):

        if tree_plan_root_node is None:
            return tree_plan_root_node

        if type(tree_plan_root_node) == TreePlanLeafNode:
            pattern, updated_tree_plan_leaf = shared_leaves_dict[tree_plan_root_node]
            tree_plan_root_node = updated_tree_plan_leaf
            return tree_plan_root_node

        assert issubclass(type(tree_plan_root_node), TreePlanInternalNode)

        if type(tree_plan_root_node) == TreePlanUnaryNode:
            updated_child = TreePlanBuilder._single_tree_plan_update_leaves(tree_plan_root_node.child,
                                                                            shared_leaves_dict)
            tree_plan_root_node.child = updated_child
            return tree_plan_root_node

        if type(tree_plan_root_node) == TreePlanBinaryNode:
            updated_left_child = TreePlanBuilder._single_tree_plan_update_leaves(tree_plan_root_node.left_child,
                                                                                 shared_leaves_dict)
            updated_right_child = TreePlanBuilder._single_tree_plan_update_leaves(tree_plan_root_node.right_child,
                                                                                  shared_leaves_dict)
            tree_plan_root_node.left_child = updated_left_child
            tree_plan_root_node.right_child = updated_right_child
            return tree_plan_root_node

        else:
            raise Exception("Unsupported Node type")
