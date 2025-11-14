import threading

import numpy as np

from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.utils.node_component import NodeComponent

class SimpleAggregator(NodeComponent):
    """
    A simple aggregator that only sopports FedAvg algorithm

    Args:
        node_addr: Address of the node。
    """

    def __init__(self) -> None:
        """Initialization"""
        self.__train_set: list[str] = []
        self.__models: list[P2PFLModel] = []

        NodeComponent.__init__(self)

        self.__agg_lock = threading.Lock()
        self._finish_aggregation_event = threading.Event()
        self._finish_aggregation_event.set()

    def aggregate(self, models: list[P2PFLModel]) -> P2PFLModel:
        """
        Use FedAvg Algorithm to aggregate the model
        
        Args:
            models: the Dictionary with the models to aggregate.

        Returns:
            The model after aggregated。

        Raises:
            NoModelsToAggregateError: If there are no models to aggregate。

        """
        if len(models) == 0:
            raise NoModelsToAggregateError(f"({self.addr}) lacks of usable models when trying to aggregate")

        total_samples = sum([m.get_num_samples() for m in models])

        first_model_weights = models[0].get_parameters()
        accum = [np.zeros_like(layer) for layer in first_model_weights]

        for m in models:
            for i, layer in enumerate(m.get_parameters()):
                accum[i] = np.add(accum[i], layer * m.get_num_samples())

        accum = [np.divide(layer, total_samples) for layer in accum]

        contributors: list[str] = []
        for m in models:
            contributors = contributors + m.get_contributors()

        return models[0].build_copy(params=accum, num_samples=total_samples, contributors=contributors)

    def set_nodes_to_aggregate(self, nodes_to_aggregate: list[str]) -> None:
        """
        List with the name of nodes to aggregate. By setting new nodes, the actual aggregation will be lost.

        Args:
            nodes_to_aggregate: List of nodes to aggregate. Empty for no aggregation.

        Raises:
            Exception: If the aggregation is running.

        """
        if not self._finish_aggregation_event.is_set():
            raise Exception("It is not allowed to set nodes to aggregate when the aggregation is running.")

        self.__train_set = nodes_to_aggregate
        self._finish_aggregation_event.clear()

    def clear(self) -> None:
        """Clear the aggregation which includes removing trainset and releasing the locks"""
        with self.__agg_lock:
            self.__train_set = []
            self.__models = []
            self._finish_aggregation_event.set()

    def get_aggregated_models(self) -> list[str]:
        """
        Get the list of moedls that have been aggregated.

        Returns:
            The list of the node that has joined in the aggregation

        """
        models_added = []
        for n in self.__models:
            models_added += n.get_contributors()
        return models_added
    
    def get_aggregated_models(self) -> list[str]:
        """
        Get the list of aggregated models。

        Returns:
            Name of nodes that colaborated to get the model

        """
        models_added = []
        for n in self.__models:
            models_added += n.get_contributors()
        return models_added

    def add_model(self, model: P2PFLModel) -> list[str]:
        """
        Add a model. The first model to be added starts the `run` method (timeout).

        Args:
            model: Model to add.

        Returns:
            List of contributors.

        """

        if model.get_contributors() == []:
            logger.debug(self.addr, "Received a model without a list of contributors. ")
            return []

        self.__agg_lock.acquire()

        try:
            if len(self.__train_set) > len(self.get_aggregated_models()):
                if all(n in self.__train_set for n in model.get_contributors()):
                    any_model_added = any(n in self.get_aggregated_models() for n in model.get_contributors())
                    if not any_model_added:
                        self.__models.append(model)
                        models_added = str(len(self.get_aggregated_models()))
                        logger.info(
                            self.addr,
                            f"Model added ({models_added}/{str(len(self.__train_set))}) from {str(model.get_contributors())}",
                        )
                        if len(self.get_aggregated_models()) >= len(self.__train_set):
                            self._finish_aggregation_event.set()

                        return self.get_aggregated_models()
                    else:
                        logger.debug(
                            self.addr,
                            f"Can't add a model from a node ({model.get_contributors()}) because it's already aggregated.",
                        )
                else:
                    logger.debug(
                        self.addr,
                        f"Can't add a model from a node ({model.get_contributors()}) because it's not in the training set",
                    )
            else:
                logger.debug(self.addr, "Received a model when is not needed. Saving a iteration to affor bandwith.")

        finally:
            self.__agg_lock.release()

        return []

    def wait_and_get_aggregation(self, timeout: int = Settings.training.AGGREGATION_TIMEOUT) -> P2PFLModel:
        """
        Wait for aggregation to finish.

        Args:
            timeout: Timeout in seconds.

        Returns:
            Aggregated model.

        """
        event_set = self._finish_aggregation_event.wait(timeout=timeout)
        missing_models = self.get_missing_models()
        
        if not event_set:
            logger.info(self.addr, f"Time out. The missing model: {missing_models}")
        else:
            if len(missing_models) > 0:
                logger.info(
                    self.addr,
                    f"We have set the aggregation even. But missing model: {missing_models}",
                )
            else:
                logger.info(self.addr, "Aggragating models now.")

        return self.aggregate(self.__models)

    def get_missing_models(self) -> set:
        """
        Obtain missing models for the aggregation.

        Returns:
            A set of missing models.

        """
        agg_models = []
        for m in self.__models:
            agg_models += m.get_contributors()
        missing_models = set(self.__train_set) - set(agg_models)
        return missing_models
    
class NoModelsToAggregateError(Exception):
    """exception raised when there are no models to aggregate"""
    pass