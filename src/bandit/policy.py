from abc import ABC, abstractmethod
from src.tree import TreeNode

class Policy(ABC):
    """
    Abstract base class for a selection policy.
    """
    @abstractmethod
    def init(self, **kwargs):
        """
        Initialize the policy.
        """
        pass

    @abstractmethod
    def select(self, nodes: list[TreeNode], n: int, **kwargs) -> set[TreeNode]:
        """
        Select a node from a list of nodes.
        """
        pass

    @abstractmethod
    def update(self, **kwargs):
        """
        Update the stats for a node after a reward is received.
        """
        pass