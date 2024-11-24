from src.tree import TreeNode
from src.policy import Policy

class TreeBandit:
    """
    Integrates a tree structure and a selection policy for bandit tasks.
    """
    def __init__(self, root: TreeNode, policy: Policy):
        self.root = root
        self.policy = policy

    def initialize_tree(self, node: TreeNode):
        """
        Initialize the policy for all nodes in the tree.
        """
        self.policy.init(node)
        if not node.is_leaf:
            for child in node.children:
                self.initialize_tree(child)

    def select(self):
        """
        Traverse the tree from the root to a leaf node using the policy.
        """
        current_node = self.root
        while not current_node.is_leaf:
            current_node = self.policy.select(current_node.children)
        return current_node

    def update(self, leaf_node, reward):
        """
        Update the policy with a reward from a selected leaf node.
        """
        self.policy.update(leaf_node, reward)