from p2p_federated_learning.learning.aggregators.aggregator import SimpleAggregator


class FedAvg(SimpleAggregator):
    """
    Federated Averaging (FedAvg) [McMahan et al., 2016].

    Paper: https://arxiv.org/abs/1602.05629.
    
    Implemented based on SimpleAggregator
    
    
    Using examples:
        >>> aggregator = FedAvg()
        >>> aggregator.set_addr("127.0.0.1:8000")
        >>> aggregator.set_nodes_to_aggregate(["node1", "node2", "node3"])
        >>> aggregator.add_model(model1)
        >>> aggregator.add_model(model2)
        >>> aggregated_model = aggregator.wait_and_get_aggregation()
    """

    def __init__(self) -> None:
        """
        Initialize the FedAvg aggregator
        """
        super().__init__()