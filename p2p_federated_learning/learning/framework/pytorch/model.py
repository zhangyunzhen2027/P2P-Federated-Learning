"""PyTorch model wrapper for P2PFL."""

from typing import Any

import lightning as L
import numpy as np
import torch

from p2p_federated_learning.learning.framework import Framework
from p2p_federated_learning.learning.framework.exceptions import ModelNotMatchingError
from p2p_federated_learning.learning.framework.p2p_framework import P2PFLModel

class LightningModel(P2PFLModel):
    """
    P2PFL model abstraction for PyTorch.

    Args:
        model: The model to encapsulate.
        params: The parameters of the model.
        num_samples: The number of samples.
        contributors: The contributors of the model.
        additional_info: Additional information.
        compression: Optional dictionary for compression settings.

    """

    def __init__(
        self,
        model: L.LightningModule,
        params: list[np.ndarray] | bytes | None = None,
        num_samples: int | None = None,
        contributors: list[str] | None = None,
        additional_info: dict[str, Any] | None = None,
        compression: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the model."""
        super().__init__(model, params, num_samples, contributors, additional_info, compression)

    def get_parameters(self) -> list[np.ndarray]:
        """
        Get the parameters of the model.

        Returns:
            The parameters of the model

        """
        return [param.cpu().numpy() for _, param in self.model.state_dict().items()]

    def set_parameters(self, params: list[np.ndarray] | bytes) -> None:
        """
        Set the parameters of the model.

        Args:
            params: The parameters of the model.

        Raises:
            ModelNotMatchingError: If parameters don't match the model.

        """
        if isinstance(params, bytes):
            params, additional_info = self.decode_parameters(params)
            self.additional_info.update(additional_info)

        state_dict = self.model.state_dict()
        for (layer_name, param), new_param in zip(state_dict.items(), params, strict=False):
            if param.shape != new_param.shape:
                raise ModelNotMatchingError("Not matching models")
            state_dict[layer_name] = torch.tensor(new_param)

        try:
            self.model.load_state_dict(state_dict)
        except Exception as e:
            raise ModelNotMatchingError("Not matching models") from e

    def get_framework(self) -> str:
        """
        Retrieve the model framework name.

        Returns:
            The name of the model framework.

        """
        return Framework.PYTORCH.value
