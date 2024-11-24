from src.tree import TreeNode
from src.policy import Policy
from typing import Iterable

class TreeBandit:
    """
    Integrates a tree structure and a selection policy for bandit tasks.
    """
    def __init__(self, root: TreeNode, policy: Policy):
        self.root = root
        self.policy = policy
    
    def select(self, n: Iterable[int], node: TreeNode = None) -> set[TreeNode]:
        """
        Traverse the tree from the root to a leaf node using the policy.
        """
        if node is None:
            node = self.root
        
        if node.is_leaf:
            return {node}
        
        selected = self.policy.select(node.children, n[0])
        return set.union(*[self.select(n[1:], child) for child in selected])


    def update(self, leaf_node, reward):
        """
        Update the policy with a reward from a selected leaf node.
        """
        self.policy.update(leaf_node, reward)