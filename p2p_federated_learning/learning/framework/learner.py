"""NodeLearning Interface - Template Pattern."""

from abc import ABC, abstractmethod

import numpy as np

from p2p_federated_learning.learning.aggregators.aggregator import Aggregator
from p2p_federated_learning.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2p_federated_learning.framework.p2p_framework import P2PFLModel
from p2p_federated_learning.utils.node_component import NodeComponent, allow_no_addr_check


class Learner(ABC, NodeComponent):
    """
    Template to implement learning processes, including metric monitoring during training.

    Args:
        model: The model of the learner.
        data: The data of the learner.
        self_addr: The address of the learner.

    """

    def __init__(self, model: P2PFLModel | None = None, data: P2PFLDataset | None = None, aggregator: Aggregator | None = None) -> None:
        """Initialize the learner."""
        NodeComponent.__init__(self)
        self.epochs: int = 1  
        self.__model: P2PFLModel | None = None
        if model:
            self.set_model(model)
        self.__data: P2PFLDataset | None = None
        if data:
            self.set_data(data)

    @allow_no_addr_check
    def set_model(self, model: P2PFLModel | list[np.ndarray] | bytes) -> None:
        """
        Set the model of the learner.

        Args:
            model: The model of the learner.

        """
        if isinstance(model, P2PFLModel):
            self.__model = model
        elif isinstance(model, list | bytes):
            self.get_model().set_parameters(model)

    @allow_no_addr_check
    def get_model(self) -> P2PFLModel:
        """
        Get the model of the learner.

        Returns:
            The model of the learner.

        """
        if self.__model is None:
            raise ValueError("Model not initialized, please ensure to set the model before accessing it. Use .set_model() method.")
        return self.__model

    @allow_no_addr_check
    def set_data(self, data: P2PFLDataset) -> None:
        """
        Set the data of the learner. It is used to fit the model.

        Args:
            data: The data of the learner.

        """
        self.__data = data

    @allow_no_addr_check
    def get_data(self) -> P2PFLDataset:
        """
        Get the data of the learner.

        Returns:
            The data of the learner.

        """
        if self.__data is None:
            raise ValueError("Data not initialized, please ensure to set the data before accessing it. Use .set_data() method.")
        return self.__data

    @allow_no_addr_check
    def set_epochs(self, epochs: int) -> None:
        """
        Set the number of epochs of the model.

        Args:
            epochs: The number of epochs of the model.

        """
        self.epochs = epochs

    @abstractmethod
    def fit(self) -> P2PFLModel:
        """Fit the model."""
        pass

    @abstractmethod
    def interrupt_fit(self) -> None:
        """Interrupt the fit process."""
        pass

    @abstractmethod
    def evaluate(self) -> dict[str, float]:
        """
        Evaluate the model with actual parameters.

        Returns:
            The evaluation results.

        """
        pass

    @abstractmethod
    def get_framework(self) -> str:
        """
        Retrieve the learner name.

        Returns:
            The name of the learner class.

        """
        pass
