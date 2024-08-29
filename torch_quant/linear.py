import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizedLinear(nn.Module):
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_quant: torch.Tensor,
        weight_scale: torch.Tensor) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_quant = nn.Parameter(weight_quant)
        self.weight_scale = nn.Parameter(weight_scale)
    
    def forward(self, inputs):
        float_weight = self._dequantize_tensor(self.weight_quant, self.weight_scale)
        outputs = F.linear(inputs, float_weight)
        return outputs
    
    @staticmethod
    def _get_amax(inputs: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.abs(inputs))
    
    @staticmethod
    def _quantize_tensor(inputs: torch.Tensor, narrow_range: bool=True):
        """Shared function body between TensorQuantFunction and FakeTensorQuantFunction"""
        # Fine scale, per channel scale will be handled by broadcasting, which could be tricky. Pop a warning.
        amax = QuantizedLinear._get_amax(inputs)
        if isinstance(amax, torch.Tensor) and inputs.dim() != amax.dim():
            logging.debug("amax %s has different shape than inputs %s. Make sure broadcast works as expected!", amax.size(),
                        inputs.size())

        # Computation must be in FP32 to prevent potential over flow.
        input_dtype = inputs.dtype
        if inputs.dtype == torch.half:
            inputs = inputs.float()
        if amax.dtype == torch.half:
            amax = amax.float()

        min_amax = amax.min()
        if min_amax < 0:
            raise ValueError("Negative values in amax")

        max_bound = torch.tensor((2.0**(8 - 1)) - 1.0, device=amax.device)
        if narrow_range:
            min_bound = -max_bound
        else:
            min_bound = -max_bound - 1
        scale = max_bound / amax

        epsilon = 1. / (1 << 24)
        if min_amax <= epsilon:  # Treat amax smaller than minimum representable of fp16 0
            zero_amax_mask = (amax <= epsilon)
            scale[zero_amax_mask] = 0  # Value quantized with amax=0 should all be 0

        quant = torch.clamp((inputs * scale).round_(), min_bound, max_bound)

        if min_amax <= epsilon:
            scale[zero_amax_mask] = 1.  # Return 1 makes more sense for values quantized to 0 with amax=0

        if input_dtype == torch.half:
            quant = quant.half()

        return quant, scale
    
    @staticmethod
    def _dequantize_tensor(quant: torch.Tensor, scale: torch.Tensor):
        return (quant / scale).float()
    
    @classmethod
    def from_linear(cls, linear: nn.Linear):
        assert linear.bias is None
        float_weight = linear.weight
        weight_quant, weight_scale = cls._quantize_tensor(float_weight.data)
        return cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            weight_quant=weight_quant,
            weight_scale=weight_scale
        )


if __name__ == '__main__':
    inputs = torch.randn(4, 256)
    float_linear = nn.Linear(in_features=256, out_features=8, bias=False)
    quantized_linear = QuantizedLinear.from_linear(float_linear)
    float_output = float_linear(inputs)
    quantized_output = quantized_linear(inputs)
    print(float_output)
    print(quantized_output)