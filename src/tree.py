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
        return len(self.children) == 0

    def __repr__(self):
        return f"TreeNode({self.value})"