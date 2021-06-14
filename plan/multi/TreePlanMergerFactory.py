from misc import DefaultConfig
from plan.multi.MultiPatternTreePlanMergeApproaches import MultiPatternTreePlanMergeApproaches
from plan.multi.ShareLeavesTreePlanMerger import ShareLeavesTreePlanMerger
from plan.multi.SubTreeSharingTreePlanMerger import SubTreeSharingTreePlanMerger


class TreePlanMergerParameters:

    def __init__(self, tree_plan_merge_approach: MultiPatternTreePlanMergeApproaches =
                 DefaultConfig.DEFAULT_TREE_PLAN_MERGER):
        self.tree_plan_merge_approach = tree_plan_merge_approach


class TreePlanMergerFactory:

    @staticmethod
    def build_tree_plan_merger(tree_plan_merger_params: TreePlanMergerParameters):
        if tree_plan_merger_params is None:
            tree_plan_merger_params = TreePlanMergerFactory.__create_default_tree_plan_merger_parameters()
        return TreePlanMergerFactory.create_tree_plan_merger(tree_plan_merger_params)

    @staticmethod
    def create_tree_plan_merger(tree_plan_merger_params: TreePlanMergerParameters):

        tree_plan_merge_approach = tree_plan_merger_params.tree_plan_merge_approach

        if tree_plan_merge_approach == MultiPatternTreePlanMergeApproaches.TREE_PLAN_TRIVIAL_SHARING_LEAVES:
            return ShareLeavesTreePlanMerger()
        if tree_plan_merge_approach == MultiPatternTreePlanMergeApproaches.TREE_PLAN_SUBTREES_UNION:
            return SubTreeSharingTreePlanMerger()
        if tree_plan_merge_approach == MultiPatternTreePlanMergeApproaches.TREE_PLAN_LOCAL_SEARCH:
            # TODO: not yet implemented
            pass
        raise Exception("Unsupported multi-pattern merge algorithm %s" % (tree_plan_merge_approach,))

    @staticmethod
    def __create_default_tree_plan_merger_parameters():
        """
        Uses the default configuration to create tree_plan_merger parameters.
        """
        return TreePlanMergerParameters()
