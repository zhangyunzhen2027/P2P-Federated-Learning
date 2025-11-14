"""Post-Training Differential Privacy compression strategy."""

try:
    import opendp.prelude as dp
except ImportError as err:
    raise ImportError("Please install with `pip install p2pfl[dp]`") from err

import numpy as np

from p2p_federated_learning.learning.compression.base_strategy  import TensorCompressor

dp.enable_features("contrib")


class DifferentialPrivacyCompressor2(TensorCompressor):
    """
    Post-Training Local Differential Privacy compressor.
    
    Applies DP to model parameters after training (not during training).
    This protects privacy by adding noise to model updates before transmission.
    """

    def _get_noise_mechanism(
        self, noise_type: str, clip_norm: float, epsilon: float, delta: float, vec_len: int
    ) -> tuple[dp.Measurement, float]:
        """Create an OpenDP noise mechanism (Gaussian or Laplace)."""
        if noise_type == "laplace":
            scale = clip_norm / epsilon
            space = dp.vector_domain(dp.atom_domain(T=float, nan=False), vec_len), dp.l1_distance(T=float)
            mech = space >> dp.m.then_laplace(scale=scale)
        elif noise_type == "gaussian":
            scale = clip_norm * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
            space = dp.vector_domain(dp.atom_domain(T=float, nan=False), vec_len), dp.l2_distance(T=float)
            mech = space >> dp.m.then_gaussian(scale=scale)
        else:
            raise ValueError("The parameter 'noise_type' must be 'gaussian' or 'laplace'")
        return mech, scale

    def apply_strategy(
        self,
        params: list[np.ndarray],
        clip_norm: float = 1.0,
        epsilon: float = 3.0,
        delta: float = 1e-5,
        noise_type: str = "gaussian",
        stability_constant: float = 1e-6,
    ) -> tuple[list[np.ndarray], dict]:
        """
        Apply differential privacy to model parameters.
        
        Args:
            params: Model update (delta) after local training.
            clip_norm: Maximum L2 norm for clipping (C).
            epsilon: Privacy budget (ε). Lower = stronger privacy. Typical: 0.1-10.0.
            delta: Privacy budget (δ). Probability of privacy breach. Typical: 1e-5.
            noise_type: "gaussian" or "laplace".
            stability_constant: Small constant to avoid division by zero.
        
        Returns:
            Tuple of (dp_params, dp_info) with privacy metadata.
        """
        if not params:
            raise ValueError("DifferentialPrivacyCompressor: list 'params' must not be empty")

        flat_update = np.concatenate([p.flatten() for p in params])
        total_norm = np.linalg.norm(flat_update)

        if total_norm > clip_norm:
            clip_factor = clip_norm / (total_norm + stability_constant)
            clipped_flat_update = flat_update * clip_factor
        else:
            clipped_flat_update = flat_update.copy()

        mech, scale = self._get_noise_mechanism(noise_type, clip_norm, epsilon, delta, clipped_flat_update.size)
        noisy_flat_update = mech(clipped_flat_update.tolist())

        dp_params = []
        current_pos = 0
        for p in params:
            shape = p.shape
            size = p.size
            dtype = p.dtype
            dp_params.append(np.array(noisy_flat_update[current_pos : current_pos + size], dtype=dtype).reshape(shape))
            current_pos += size

        dp_info = {
            "dp_applied": True,
            "clip_norm": clip_norm,
            "epsilon": epsilon,
            "delta": delta if noise_type == "gaussian" else None,
            "noise_type": noise_type,
            "noise_scale": scale,
            "original_norm": float(total_norm),
            "was_clipped": bool(total_norm > clip_norm),
        }

        return dp_params, dp_info

    def reverse_strategy(self, params: list[np.ndarray], additional_info: dict) -> list[np.ndarray]:
        """
        Reverse the differential privacy transformation.
        
        Note: DP is inherently irreversible (that's the point!).
        This just returns the parameters as-is.
        
        Args:
            params: The DP-protected parameters.
            additional_info: Contains DP metadata.
        
        Returns:
            The same parameters (DP cannot be reversed).
        """
        return params

