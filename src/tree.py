from typing import Iterable

class TreeNode:
    """
    Represents a node in the tree.
    """
    def __init__(self, value):
        self.value = value
        self.children = []

    def add_child(self, child):
        """
        Add a child node to this node.
        """
        self.children.append(child)

    @property
    def is_leaf(self):
        """
        Check if the node is a leaf (has no children).
        """
        return self.children == []

    def __repr__(self):
        return f"TreeNode({self.value})"
    
    def __str__(self, level=0):
        ret = "\t" * level + str(self.value) + "\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret


class TreeBandit:
    """
    Integrates a tree structure and a selection policy for bandit tasks.
    """
    def __init__(self, root: TreeNode, policy):
        self.root = root
        self.policy = policy
    
    def select(self, n: Iterable[int], node: TreeNode = None, **kwargs) -> set[TreeNode]:
        """
        Traverse the tree from the root to a leaf node using the policy.
        """
        self.policy.reset() # Reset the cache for the current call
        return self.__recursive_select(n, node, **kwargs)

    def __recursive_select(self, n, node, **kwargs):
        if node is None:
            node = self.root
        
        if node.is_leaf:
            return {node}
        
        selected = self.policy.select(node.children, n[0], **kwargs)
        return set.union(*[self.__recursive_select(n[1:], child, **kwargs) for child in selected])