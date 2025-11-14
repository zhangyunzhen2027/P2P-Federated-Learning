"""Post-Training Quantization (PTQ) compression strategy."""

from typing import Literal

import numpy as np


from p2p_federated_learning.learning.compression.base_strategy  import TensorCompressor


class PTQuantization(TensorCompressor):
    """
    Post-Training Quantization (PTQ).
    
    Reduces parameter precision (e.g., float32 → int8) to decrease model size.
    Supports symmetric/asymmetric quantization and per-tensor/per-channel granularity.
    """

    _SUPPORTED_FLOAT_TYPES = {np.float16, np.float32, np.float64}
    _SUPPORTED_INT_TYPES = {np.int8, np.uint8, np.int16, np.uint16, np.int32}

    def apply_strategy(
        self,
        params: list[np.ndarray],
        dtype: str = "float16",
        scheme: Literal["symmetric", "asymmetric"] = "symmetric",
        granularity: Literal["per_tensor", "per_channel"] = "per_tensor",
        channel_axis: int = 0,
    ) -> tuple[list[np.ndarray], dict]:
        """
        Quantize model parameters to lower precision.
        
        Args:
            params: Parameters to quantize.
            dtype: Target precision (e.g., "float16", "int8").
            scheme: "symmetric" (centered at 0) or "asymmetric" (full range).
            granularity: "per_tensor" (one scale) or "per_channel" (per-channel scales).
            channel_axis: Axis for per-channel quantization.
        
        Returns:
            Tuple of (quantized_params, quantization_info).
        """
        if not params:
            raise ValueError("Empty parameter list provided for quantization")

        try:
            target_dtype = np.dtype(dtype)
        except TypeError as e:
            raise ValueError(f"Invalid dtype: {dtype}") from e

        if np.issubdtype(target_dtype, np.floating):
            if target_dtype.type not in self._SUPPORTED_FLOAT_TYPES:
                raise ValueError(f"Unsupported float dtype: {dtype}")
        elif np.issubdtype(target_dtype, np.integer):
            if target_dtype.type not in self._SUPPORTED_INT_TYPES:
                raise ValueError(f"Unsupported integer dtype: {dtype}")
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        if scheme not in ["symmetric", "asymmetric"]:
            raise ValueError(f"Unsupported scheme: {scheme}")
        if granularity not in ["per_tensor", "per_channel"]:
            raise ValueError(f"Unsupported granularity: {granularity}")

        original_dtype = params[0].dtype

        if np.issubdtype(target_dtype, np.floating):
            return [param.astype(target_dtype) for param in params], {
                "ptq_original_dtype": original_dtype,
                "ptq_type": "float",
            }

        quantized_params = []
        scales: list[float | np.ndarray] = []
        zero_points: list[int | np.ndarray] = []

        for param in params:
            if granularity == "per_tensor":
                q_param, scale, zero_point = self._quantize_tensor(param, target_dtype, scheme)
                scales.append(scale)
                zero_points.append(zero_point)
            else: 
                q_param, channel_scales, channel_zero_points = self._quantize_per_channel(
                    param, target_dtype, scheme, channel_axis
                )
                scales.append(channel_scales)
                zero_points.append(channel_zero_points)
            quantized_params.append(q_param)

        return quantized_params, {
            "ptq_original_dtype": original_dtype,
            "ptq_type": "int",
            "ptq_scheme": scheme,
            "ptq_granularity": granularity,
            "ptq_scales": scales,
            "ptq_zero_points": zero_points,
            "ptq_channel_axis": channel_axis if granularity == "per_channel" else None,
        }

    def reverse_strategy(self, params: list[np.ndarray], additional_info: dict) -> list[np.ndarray]:
        """
        Dequantize parameters back to original precision.
        
        Args:
            params: Quantized parameters.
            additional_info: Quantization metadata.
        
        Returns:
            Dequantized parameters.
        """
        if not params:
            raise ValueError("Empty parameter list provided for dequantization")

        if "ptq_original_dtype" not in additional_info:
            raise ValueError("Missing 'ptq_original_dtype' in additional_info")
        original_dtype = additional_info["ptq_original_dtype"]

        if additional_info.get("ptq_type") == "float":
            return [param.astype(original_dtype) for param in params]
        
        required_keys = ["ptq_scheme", "ptq_granularity", "ptq_scales", "ptq_zero_points"]
        for key in required_keys:
            if key not in additional_info:
                raise ValueError(f"Missing required key '{key}' in additional_info")

        scheme = additional_info["ptq_scheme"]
        granularity = additional_info["ptq_granularity"]
        scales = additional_info["ptq_scales"]
        zero_points = additional_info["ptq_zero_points"]
        channel_axis = additional_info.get("ptq_channel_axis")

        dequantized_params = []
        for i, param in enumerate(params):
            if granularity == "per_tensor":
                dq_param = self._dequantize_tensor(param, scales[i], zero_points[i], original_dtype)
            else:  
                if channel_axis is None:
                    raise ValueError("Missing 'ptq_channel_axis' for per_channel dequantization")
                dq_param = self._dequantize_per_channel(param, scales[i], zero_points[i], channel_axis, original_dtype)
            dequantized_params.append(dq_param)

        return dequantized_params

    def _quantize_tensor(self, tensor: np.ndarray, target_dtype: np.dtype, scheme: str) -> tuple[np.ndarray, float, int]:
        """Quantize a tensor with a single scale factor."""
        qmin, qmax = self._get_quantization_range(target_dtype)

        if tensor.size == 0:
            return np.array([], dtype=target_dtype), 1.0, 0

        if scheme == "symmetric":
            abs_max = max(abs(tensor.min()), abs(tensor.max()))
            scale = 1.0 if abs_max == 0 else abs_max / max(abs(qmin), abs(qmax))
            zero_point = 0
            raw_quantized = np.round(tensor / scale)
            quantized = np.clip(raw_quantized, qmin, qmax).astype(target_dtype)
        else: 
            tmin, tmax = tensor.min(), tensor.max()
            if tmin == tmax:
                if tmin == 0:
                    return np.zeros_like(tensor, dtype=target_dtype), 1.0, 0
                mid_q = (qmin + qmax) // 2
                return np.full_like(tensor, mid_q, dtype=target_dtype), tmin / mid_q, 0
            scale = (tmax - tmin) / (qmax - qmin)
            zero_point = qmin - round(tmin / scale)
            zero_point = max(qmin, min(qmax, zero_point))
            raw_quantized = np.round(tensor / scale + zero_point)
            quantized = np.clip(raw_quantized, qmin, qmax).astype(target_dtype)

        return quantized, float(scale), int(zero_point)

    def _quantize_per_channel(
        self, tensor: np.ndarray, target_dtype: np.dtype, scheme: str, channel_axis: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Quantize a tensor with per-channel scale factors."""
        if tensor.ndim == 0:
            raise ValueError("Cannot perform per-channel quantization on a scalar tensor")
        if not 0 <= channel_axis < tensor.ndim:
            raise ValueError(f"Invalid channel_axis {channel_axis} for tensor with {tensor.ndim} dimensions")

        if tensor.ndim <= 1:
            q_tensor, scale, zero_point = self._quantize_tensor(tensor, target_dtype, scheme)
            return q_tensor, np.array([scale], dtype=np.float32), np.array([zero_point], dtype=np.int32)

        qmin, qmax = self._get_quantization_range(target_dtype)
        num_channels = tensor.shape[channel_axis]

        if num_channels == 0 or tensor.size == 0:
            return (np.array([], dtype=target_dtype), np.array([], dtype=np.float32), np.array([], dtype=np.int32))

        scales = np.zeros(num_channels, dtype=np.float32)
        zero_points = np.zeros(num_channels, dtype=np.int32)
        transposed_tensor = np.moveaxis(tensor, channel_axis, 0)
        quantized = np.zeros_like(tensor, dtype=target_dtype)
        transposed_quantized = np.moveaxis(quantized, channel_axis, 0)

        for c in range(num_channels):
            channel_tensor = transposed_tensor[c]

            if scheme == "symmetric":
                if channel_tensor.size == 0 or (channel_tensor.min() == channel_tensor.max() == 0):
                    scales[c] = 1.0
                    zero_points[c] = 0
                    continue
                abs_max = max(abs(channel_tensor.min()), abs(channel_tensor.max()))
                scale = 1.0 if abs_max == 0 else abs_max / max(abs(qmin), abs(qmax))
                zero_point = 0
                raw_quantized = np.round(channel_tensor / scale)
                channel_quantized = np.clip(raw_quantized, qmin, qmax).astype(target_dtype)
            else:  
                tmin, tmax = channel_tensor.min(), channel_tensor.max()
                if tmin == tmax:
                    if tmin == 0:
                        transposed_quantized[c] = np.zeros_like(channel_tensor, dtype=target_dtype)
                        scales[c] = 1.0
                        zero_points[c] = 0
                        continue
                    mid_q = (qmin + qmax) // 2
                    transposed_quantized[c] = np.full_like(channel_tensor, mid_q, dtype=target_dtype)
                    scales[c] = float(tmin / mid_q)
                    zero_points[c] = 0
                    continue
                scale = (tmax - tmin) / (qmax - qmin)
                if scale == 0:
                    scale = 1.0
                zero_point = qmin - round(tmin / scale)
                zero_point = max(qmin, min(qmax, zero_point))
                channel_quantized = np.round(channel_tensor / scale + zero_point).astype(target_dtype)
                channel_quantized = np.clip(channel_quantized, qmin, qmax)

            transposed_quantized[c] = channel_quantized
            scales[c] = float(scale)
            zero_points[c] = int(zero_point)

        return quantized, scales, zero_points

    def _dequantize_tensor(self, tensor: np.ndarray, scale: float, zero_point: int, original_dtype: np.dtype) -> np.ndarray:
        """Dequantize a tensor with a single scale factor."""
        if not isinstance(scale, (int, float)) or scale <= 0:
            raise ValueError(f"Invalid scale factor: {scale}")
        if not isinstance(zero_point, (int, np.integer)):
            raise ValueError(f"Invalid zero point: {zero_point}")

        if tensor.size == 0:
            return np.array([], dtype=original_dtype)

        if zero_point == 0:
            dequantized = tensor.astype(np.float32) * scale
        else:
            dequantized = (tensor.astype(np.float32) - zero_point) * scale

        return dequantized.astype(original_dtype)

    def _dequantize_per_channel(
        self, tensor: np.ndarray, scales: np.ndarray, zero_points: np.ndarray, channel_axis: int, original_dtype: np.dtype
    ) -> np.ndarray:
        """Dequantize a tensor with per-channel scale factors."""
        if not isinstance(scales, np.ndarray) or scales.ndim != 1:
            raise ValueError("Scales must be a 1D numpy array")
        if not isinstance(zero_points, np.ndarray) or zero_points.ndim != 1:
            raise ValueError("Zero points must be a 1D numpy array")
        if len(scales) != len(zero_points):
            raise ValueError(f"Number of scales ({len(scales)}) must match number of zero points ({len(zero_points)})")

        if tensor.size == 0 or scales.size == 0:
            return np.array([], dtype=original_dtype)

        if scales.size == 1:
            return self._dequantize_tensor(tensor, scales[0], zero_points[0], original_dtype)

        if tensor.ndim == 0:
            raise ValueError("Cannot perform per-channel dequantization on a scalar tensor")
        if not 0 <= channel_axis < tensor.ndim:
            raise ValueError(f"Invalid channel_axis {channel_axis} for tensor with {tensor.ndim} dimensions")

        num_channels = tensor.shape[channel_axis]
        if num_channels != len(scales):
            raise ValueError(f"Number of channels ({num_channels}) must match number of scales ({len(scales)})")

        transposed_tensor = np.moveaxis(tensor, channel_axis, 0)
        dequantized = np.zeros_like(tensor, dtype=np.float32)
        transposed_dequantized = np.moveaxis(dequantized, channel_axis, 0)

        for c in range(len(scales)):
            channel_tensor = transposed_tensor[c]
            scale = float(scales[c])
            zero_point = zero_points[c]

            if scale <= 0 or not np.isfinite(scale):
                raise ValueError(f"Invalid scale factor at index {c}: {scale}")
            if not isinstance(zero_point, (int, np.integer)):
                raise ValueError(f"Invalid zero point at index {c}: {zero_point}")

            if zero_point == 0:
                channel_dequantized = channel_tensor.astype(np.float32) * scale
            else:
                channel_dequantized = (channel_tensor.astype(np.float32) - zero_point) * scale

            transposed_dequantized[c] = channel_dequantized

        return dequantized.astype(original_dtype)

    def _get_quantization_range(self, target_dtype: np.dtype) -> tuple[int, int]:
        """Get quantization range for a given dtype."""
        if target_dtype == np.int8:
            return -128, 127
        elif target_dtype == np.uint8:
            return 0, 255
        elif target_dtype == np.int16:
            return -32768, 32767
        elif target_dtype == np.uint16:
            return 0, 65535
        elif target_dtype == np.int32:
            return -2147483648, 2147483647
        else:
            raise ValueError(f"Unsupported integer dtype for quantization: {target_dtype}")

