"""zlib byte compression strategy."""

import zlib

from p2p_federated_learning.learning.compression.base_strategy import ByteCompressor

class ZlibCompressor(ByteCompressor):
    """
    Simplified lossless compression using zlib.
    
    Compresses byte data after serialization. This is a generic compression
    that works on any byte stream, not specific to model parameters.
    """

    def apply_strategy(self, data: bytes, level: int = 6) -> bytes:
        """
        Compress byte data using zlib.
        
        Args:
            data: Input bytes to compress.
            level: Compression level (0-9). Higher = better compression but slower.
        
        Returns:
            Compressed bytes.
        """
        return zlib.compress(data, level=level)

    def reverse_strategy(self, data: bytes) -> bytes:
        """
        Decompress zlib-compressed byte data.
        
        Args:
            data: Compressed bytes.
        
        Returns:
            Decompressed bytes.
        """
        return zlib.decompress(data)

