from abc import ABC, abstractmethod

class Policy(ABC):
    """
    Abstract base class for a selection policy.
    """
    @abstractmethod
    def init(self, *args, **kwargs):
        """
        Initialize the policy.
        """
        pass

    @abstractmethod
    def select(self, nodes, *args, **kwargs):
        """
        Select a node from a list of nodes.
        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        """
        Update the stats for a node after a reward is received.
        """
        pass