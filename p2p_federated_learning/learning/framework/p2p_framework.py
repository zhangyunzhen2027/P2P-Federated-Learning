"""P2PFL model abstraction."""

import copy
from typing import Any

import numpy as np

from p2p_federated_learning.learning.compression.manager import CompressionManager
from p2p_federated_learning.learning.framework.exceptions import DecodingParamsError


class P2PFLModel:
    """
    P2PFL model abstraction.

    This class encapsulates PyTorch Lightning models.

    The key concept is the extraction of the model weights in a common format.

    Args:
        model: The model to encapsulate (PyTorch Lightning module).

    """

    def __init__(
        self,
        model: Any,
        params: list[np.ndarray] | bytes | None = None,
        num_samples: int | None = None,
        contributors: list[str] | None = None,
        additional_info: dict[str, Any] | None = None,
        compression: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the model."""
        self.model = model
        self.contributors: list[str] = []
        if contributors is not None:
            self.contributors = contributors
        self.num_samples = 0
        if num_samples is not None:
            self.num_samples = num_samples
        self.additional_info: dict[str, Any] = {}
        if additional_info is not None:
            self.additional_info = additional_info
        if params is not None:
            self.set_parameters(params)
        if compression is not None:
            self.compression = compression
        else:
            self.compression = {}

    def get_model(self) -> Any:
        """Get the model."""
        return self.model

    def encode_parameters(self, params: list[np.ndarray] | None = None) -> bytes:
        """
        Encode the parameters of the model.

        Args:
            params: The parameters of the model.

        """
        if params is None:
            params = self.get_parameters()

        return CompressionManager.apply(params, self.additional_info, self.compression)

    def decode_parameters(self, data: bytes) -> tuple[list[np.ndarray], dict[str, Any]]:
        """
        Decode the parameters of the model.

        Args:
            data: The parameters of the model.

        """
        try:
            return CompressionManager.reverse(data)
        except Exception as e:
            raise DecodingParamsError("Error decoding parameters") from e

    def get_parameters(self) -> list[np.ndarray]:
        """
        Get the parameters of the model.

        Returns:
            The parameters of the model

        """
        raise NotImplementedError

    def set_parameters(self, params: list[np.ndarray] | bytes) -> None:
        """
        Set the parameters of the model.

        Args:
            params: The parameters of the model.

        Raises:
            ModelNotMatchingError: If parameters don't match the model.

        """
        raise NotImplementedError

    def set_contribution(self, contributors: list[str], num_samples: int) -> None:
        """
        Set the contribution of the model.

        Args:
            contributors: The contributors of the model.
            num_samples: The number of samples used to train this model.

        """
        self.contributors = contributors
        self.num_samples = num_samples

    def get_contributors(self) -> list[str]:
        """Get the contributors of the model."""
        if self.contributors == []:
            raise ValueError("Contributors are empty")
        return self.contributors

    def get_num_samples(self) -> int:
        """Get the number of samples used to train this model."""
        if self.num_samples == 0:
            raise ValueError("Number of samples required")
        return self.num_samples

    def build_copy(self, **kwargs) -> "P2PFLModel":
        """
        Build a copy of the model.

        Args:
            **kwargs: Parameters of the model initialization.

        Returns:
            A copy of the model.

        """
        return self.__class__(copy.deepcopy(self.model), **kwargs)

    def get_framework(self) -> str:
        """
        Retrieve the model framework name.

        Returns:
            The name of the model framework.

        """
        raise NotImplementedError
