"""Data partitioning strategies for P2PFL Datasets."""
import random
from abc import abstractmethod

from dataset import Dataset

from p2p_federated_learning.settings import Settings


class DataPartitionStrategy:
    """
    Abstract class for defining data partitioning strategies in federated learning.

    This class provides a common interface for generating partitions of a dataset, which can be used
    to simulate different data distributions across clients.
    """

    @staticmethod
    @abstractmethod
    def generate_partitions(
        train_data: Dataset,
        test_data: Dataset,
        num_partitions: int,
        **kwargs,
    ) -> tuple[list[list[int]], list[list[int]]]:
        """
        Generate partitions of the dataset based on the specific strategy.

        Args:
            train_data: The training Dataset object to partition.
            test_data: The test Dataset object to partition.
            num_partitions: The number of partitions to create.
            **kwargs: Additional keyword arguments that may be required by specific strategies.

        Returns:
            A tuple containing two lists of lists:
                - The first list contains lists of indices for the training data partitions.
                - The second list contains lists of indices for the test data partitions.

        """
        pass


class RandomIIDPartitionStrategy(DataPartitionStrategy):
    """
    Partition the dataset randomly, resulting in an IID distribution of data across clients.

    Using example:
        strategy = RandomIIDPartitionStrategy()
        train_partitions, test_partitions = strategy.generate_partitions(
            train_data=train_dataset,
            test_data=test_dataset,
            num_partitions=5  # 5 nodes
        )
    """

    @staticmethod
    def generate_partitions(
        train_data: Dataset,
        test_data: Dataset,
        num_partitions: int,
        **kwargs,
    ) -> tuple[list[list[int]], list[list[int]]]:
        """
        Generate partitions of the dataset using random sampling.

        Args:
            train_data: The training Dataset object to partition.
            test_data: The test Dataset object to partition.
            num_partitions: The number of partitions to create.
            **kwargs: Additional keyword arguments that may be required by specific strategies.

        Returns:
            A tuple containing two lists of lists:
                - The first list contains lists of indices for the training data partitions.
                - The second list contains lists of indices for the test data partitions.

        """
        return (
            RandomIIDPartitionStrategy._partition_data(train_data, num_partitions),
            RandomIIDPartitionStrategy._partition_data(test_data, num_partitions),
        )

    @staticmethod
    def _partition_data(data: Dataset, num_partitions: int) -> list[list[int]]:
        """
       Randomly partition the data. 

        Args:
            data: The dataset to be partitioned.
            num_partitions: The number of partitions. 

        Returns:
            The index list of each partition.
        """
        indices = list(range(len(data)))
        random.Random(Settings.general.SEED).shuffle(indices)
        samples_per_partition = len(data) // num_partitions
        remainder = len(data) % num_partitions  
        partitions = []
        start_idx = 0
        for i in range(num_partitions):
            partition_size = samples_per_partition + (1 if i < remainder else 0)
            end_idx = start_idx + partition_size
            partitions.append(indices[start_idx:end_idx])
            start_idx = end_idx
        return partitions

