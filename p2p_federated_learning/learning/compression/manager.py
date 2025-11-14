import pickle
from typing import Any

import numpy as np

from p2p_federated_learning.learning.compression.base_strategy import ByteCompressor
from p2p_federated_learning.learning.compression.dp import DifferentialPrivacyCompressor
from p2p_federated_learning.learning.compression.quantization import PTQuantization
from p2p_federated_learning.learning.compression.top_k import TopKSparsification
from p2p_federated_learning.learning.compression.zlib import ZlibCompressor

COMPRESSION_STRATEGIES_REGISTRY = {
    "ptq": PTQuantization,              # Quantization
    "dp": DifferentialPrivacyCompressor,  # Differential Privacy
    "topk": TopKSparsification,         # Top-K Sparsification
    "zlib": ZlibCompressor,             # zlib byte compression
}


class CompressionManager2:
    """
    Compression manager for Quantization, DP, Top-K, and zlib.
    
    Supports:
    - Tensor compression: ptq (quantization), dp (differential privacy), topk (sparsification)
    - Byte compression: zlib (applied after serialization)
    """

    @staticmethod
    def get_registry() -> dict[str, Any]:
        """Return the registry of compression strategies."""
        return COMPRESSION_STRATEGIES_REGISTRY

    @staticmethod
    def apply(params: list[np.ndarray], additional_info: dict, techniques: dict[str, dict[str, Any]]) -> bytes:
        """
        Apply compression techniques in sequence to the data.
        
        Execution order:
        1. Apply tensor compressors (ptq, dp, topk) in the order specified
        2. Serialize the data (pickle)
        3. Apply byte compressor (zlib) if specified
        
        Args:
            params: The model parameters to compress (list of numpy arrays).
            additional_info: Additional information to include in the compressed data.
            techniques: Dictionary mapping technique names to their parameters.
                       Example: {"ptq": {"dtype": "int8"}, "dp": {"epsilon": 1.0}, "zlib": {"level": 6}}
        
        Returns:
            Compressed data as bytes.
        
        Raises:
            ValueError: If unknown compression technique is specified.
        """
        registry = CompressionManager2.get_registry()
        applied_techniques = []
        byte_compressor: ByteCompressor | None = None
        encoder_key: str | None = None

        for name, fn_params in techniques.items():
            if name not in registry:
                raise ValueError(
                    f"Unknown compression technique: {name}. "
                    f"Supported techniques: {list(registry.keys())}"
                )
            
            instance = registry[name]()
            
            if isinstance(instance, ByteCompressor):
                if byte_compressor is not None:
                    raise ValueError("Only one byte compressor (zlib) can be applied at a time")
                byte_compressor = instance
                encoder_key = name
            else:
                params, compression_settings = instance.apply_strategy(params, **fn_params)
                applied_techniques.append([name, compression_settings])

        data = {
            "params": params,
            "additional_info": additional_info | {"applied_techniques": applied_techniques},
        }
        data_bytes = pickle.dumps(data)

        if byte_compressor is not None:
            data_bytes = byte_compressor.apply_strategy(data_bytes)

        return pickle.dumps(
            {
                "byte_compressor": encoder_key,
                "bytes": data_bytes,
            }
        )

    @staticmethod
    def reverse(data: bytes) -> tuple[list[np.ndarray], dict]:
        """
        Reverse compression techniques in reverse order.
        
        Execution order (opposite of apply):
        1. Deserialize outer wrapper
        2. Reverse byte compressor (zlib) if applied
        3. Deserialize inner data
        4. Reverse tensor compressors (topk, dp, ptq) in reverse order
        
        Args:
            data: The compressed data bytes to decompress.
        
        Returns:
            Tuple of (decompressed_params, additional_info).
        
        Raises:
            ValueError: If data format is invalid or decompression fails.
        """
        registry = CompressionManager2.get_registry()
        raw_data = pickle.loads(data)

        encoder_key = raw_data.get("byte_compressor", None)
        if encoder_key is not None:
            byte_compressor = registry[encoder_key]()
            data_bytes = byte_compressor.reverse_strategy(raw_data["bytes"])
        else:
            data_bytes = raw_data["bytes"]

        data = pickle.loads(data_bytes)
        if not isinstance(data, dict):
            raise ValueError("Invalid data format: expected dictionary after deserialization")

        params = data["params"]
        if "additional_info" not in data:
            raise ValueError(
                "No additional info found in data. "
                "Cannot reverse compression without technique metadata."
            )
        
        applied_techniques = data["additional_info"].pop("applied_techniques")

        for compressor_name, compressor_info in reversed(applied_techniques):
            if compressor_name not in registry:
                raise ValueError(
                    f"Unknown compression technique in data: {compressor_name}. "
                    f"Supported techniques: {list(registry.keys())}"
                )
            instance = registry[compressor_name]()
            params = instance.reverse_strategy(params, compressor_info)

        return params, data["additional_info"]

