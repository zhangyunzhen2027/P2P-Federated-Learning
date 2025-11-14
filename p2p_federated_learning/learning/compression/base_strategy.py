"""Compression strategy interface for Quantization, DP, Top-K, and zlib."""

from abc import ABC, abstractmethod

import numpy as np

class CompressionStrategy(ABC):
    """Base abstract class for all compression strategies."""

    @abstractmethod
    def apply_strategy(self, *args, **kwargs):
        """Apply compression strategy."""
        pass

    @abstractmethod
    def reverse_strategy(self, *args, **kwargs):
        """Reverse the compression strategy."""
        pass


class TensorCompressor(CompressionStrategy):
    """
    Base class for tensor compression strategies.
    
    Used by:
    - Quantization (PTQuantization)
    - Differential Privacy (DifferentialPrivacyCompressor)
    - Top-K Sparsification (TopKSparsification)
    """

    @abstractmethod
    def apply_strategy(self, params: list[np.ndarray], **kwargs) -> tuple[list[np.ndarray], dict]:
        """
        Apply compression to model parameters.
        
        Args:
            params: List of numpy arrays representing model parameters.
            **kwargs: Strategy-specific parameters.
            
        Returns:
            Tuple of (compressed_params, compression_info_dict)
        """
        pass

    @abstractmethod
    def reverse_strategy(self, params: list[np.ndarray], additional_info: dict) -> list[np.ndarray]:
        """
        Reverse the compression.
        
        Args:
            params: Compressed parameters.
            additional_info: Dictionary containing compression metadata.
            
        Returns:
            Decompressed parameters.
        """
        pass


class ByteCompressor(CompressionStrategy):
    """
    Base class for byte-level compression strategies.
    
    Used by:
    - zlib compression (ZlibCompressor)
    """

    @abstractmethod
    def apply_strategy(self, data: bytes, **kwargs) -> bytes:
        """
        Apply compression to byte data.
        
        Args:
            data: Input bytes to compress.
            **kwargs: Strategy-specific parameters.
            
        Returns:
            Compressed bytes.
        """
        pass

    @abstractmethod
    def reverse_strategy(self, data: bytes) -> bytes:
        """
        Reverse the compression.
        
        Args:
            data: Compressed bytes.
            
        Returns:
            Decompressed bytes.
        """
        pass

