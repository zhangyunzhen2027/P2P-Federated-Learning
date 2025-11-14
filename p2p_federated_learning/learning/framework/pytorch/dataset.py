"""PyTorch dataset integration for FedAvg."""

from datasets import Dataset  # type: ignore
from torch.utils.data import DataLoader

from p2p_federated_learning.learning.dataset.p2pfl_dataset import DataExportStrategy
from p2p_federated_learning.settings import Settings


class PyTorchExportStrategy(DataExportStrategy):
    """Export strategy for PyTorch tensors."""

    @staticmethod
    def export(
        data: Dataset,
        batch_size: int | None = None,
        num_workers: int = 0,
        **kwargs,
    ) -> DataLoader:
        """
        Export the data using the PyTorch strategy.

        Args:
            data: The data to export. Transforms should already be applied to the dataset via set_transform.
            batch_size: The batch size to use for the exported data.
            num_workers: The number of workers to use for the exported
            kwargs: Additional keyword arguments.

        Returns:
            The exported data.

        """
        if not batch_size:
            batch_size = Settings.training.DEFAULT_BATCH_SIZE
        if hasattr(data, "format") and data.format["type"] is not None:
            return DataLoader(data, batch_size=batch_size, num_workers=num_workers)
        else:
            return DataLoader(data.with_format(type="torch", output_all_columns=True), batch_size=batch_size, num_workers=num_workers)
