"""P2PFL compressions - Quantization, DP, Top-K, and zlib"""

from .dp import DifferentialPrivacyCompressor
from .quantization import PTQuantization
from .top_k import TopKSparsification
from .zlib import ZlibCompressor

COMPRESSION_STRATEGIES_REGISTRY = {
    "ptq": PTQuantization,              # Quantization
    "dp": DifferentialPrivacyCompressor,  # Differential Privacy
    "topk": TopKSparsification,         # Top-K Sparsification
    "zlib": ZlibCompressor,             # zlib byte compression
}

__all__ = [
    "PTQuantization",
    "DifferentialPrivacyCompressor",
    "TopKSparsification",
    "ZlibCompressor",
    "COMPRESSION_STRATEGIES_REGISTRY",
]

