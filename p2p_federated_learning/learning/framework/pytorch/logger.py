"""Logger for P2PFL for FedAvg."""

from typing import Any

from lightning.pytorch.loggers.logger import Logger
from p2p_federated_learning.management.logger import logger as P2PLogger


class FederatedLogger(Logger):
    """
    Pytorch  Logger for Federated Learning. Handles local training loggin.

    Args:
        node_name: Name of the node.

    """

    def __init__(self, addr: str) -> None:
        """Initialize the logger."""
        super().__init__()
        self.__addr = addr

    @property
    def name(self) -> None:
        """Name of the logger."""
        pass

    @property
    def version(self) -> None:
        """Version of the logger."""
        pass

    def log_hyperparams(self, *args: Any, **kwargs: Any) -> None:
        """Log hyperparameters."""
        pass

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics (in a pytorch format)."""
        for k, v in metrics.items():
            P2PLogger.log_metric(self.__addr, k, v, step=step)

    def save(self) -> None:
        """Save the logger."""
        pass

    def finalize(self, status: str) -> None:
        """Finalize the logger."""
        pass
