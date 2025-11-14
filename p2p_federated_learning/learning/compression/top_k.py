"""Top-K Sparsification compression strategy."""

import numpy as np

from p2p_federated_learning.learning.compression.base_strategy  import TensorCompressor

class TopKSparsification2(TensorCompressor):
    """
    Top-K sparsification.
    
    Keeps only the top k% largest values in model parameters, sets others to zero.
    This reduces the number of non-zero parameters, decreasing transmission size.
    """

    def apply_strategy(self, params: list[np.ndarray], k: float = 0.1) -> tuple[list[np.ndarray], dict]:
        """
        Compress parameters by keeping only the top k% largest values.
        
        Args:
            params: The parameters to compress.
            k: Percentage of parameters to keep (between 0 and 1).
        
        Returns:
            Tuple of (sparse_params, metadata) where sparse_params contains only
            the top k% values, and metadata stores indices and shapes for reconstruction.
        """
        new_params = []
        sparse_metadata = {}

        for pos, param in enumerate(params):
            k_elements = max(1, int(k * param.size))
            flattened = param.flatten()
            indices = np.argpartition(flattened, -k_elements)[-k_elements:]
            values = flattened[indices]
            sparse_metadata[pos] = {"indices": indices.astype(np.uint32), "shape": param.shape}
            new_params.append(values)

        return new_params, {"topk_sparse_metadata": sparse_metadata}

    def reverse_strategy(self, params: list[np.ndarray], additional_info: dict) -> list[np.ndarray]:
        """
        Reconstruct full parameters from sparse representation.
        
        Args:
            params: The sparse parameters (only top k% values).
            additional_info: Contains indices and shapes for reconstruction.
        
        Returns:
            Reconstructed parameters with zeros where values were removed.
        """
        reconstructed_params = []
        sparse_metadata = additional_info["topk_sparse_metadata"]

        for pos, values in enumerate(params):
            if pos in sparse_metadata:
                meta = sparse_metadata[pos]
                indices = meta["indices"]
                shape = meta["shape"]
                full_array = np.zeros(np.prod(shape), dtype=values.dtype)
                full_array[indices] = values
                full_array = full_array.reshape(shape)
                reconstructed_params.append(full_array)
            else:
                reconstructed_params.append(values)

        return reconstructed_params

