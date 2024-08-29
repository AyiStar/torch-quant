import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantLinear(nn.Module):
    
    def __init__(self, in_features: int, out_features: int, quantized_weight: torch.Tensor, quantized_bias: Optional[torch.Tensor]=None) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(quantized_weight)
        if quantized_bias is not None:
            self.bias = nn.Parameter(quantized_bias)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, inputs):
        float_weight = self._dequantize(self.weight)
        float_bias = None
        if self.bias:
            float_bias = self._dequantize(self.bias)
        outputs = F.linear(inputs, float_weight, float_bias)
        return outputs
    
    @staticmethod
    def _get_amax(inputs: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.abs(inputs))
    
    @staticmethod
    def _quantize_tensor(inputs: torch.Tensor, narrow_range: bool=True):
        """Shared function body between TensorQuantFunction and FakeTensorQuantFunction"""
        # Fine scale, per channel scale will be handled by broadcasting, which could be tricky. Pop a warning.
        if isinstance(amax, torch.Tensor) and inputs.dim() != amax.dim():
            logging.debug("amax %s has different shape than inputs %s. Make sure broadcast works as expected!", amax.size(),
                        inputs.size())
        
        amax = QuantLinear._get_amax(inputs)

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

        outputs = torch.clamp((inputs * scale).round_(), min_bound, max_bound)

        if min_amax <= epsilon:
            scale[zero_amax_mask] = 1.  # Return 1 makes more sense for values quantized to 0 with amax=0

        if input_dtype == torch.half:
            outputs = outputs.half()

        return outputs, scale
    
    def forward(self, inputs):
        pass
    
    @classmethod
    def from_linear(cls, linear: nn.Linear):
        float_weight = linear.weight
        float_bias = linear.bias
        quantized_weight = cls._quantize_tensor(float_weight.data)
        quantized_bias = None
        if float_bias is not None:
            quantized_bias = cls._quantize_tensor(float_bias.data)
        return cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            quantized_weight=quantized_weight,
            quantized_bias=quantized_bias,
        )
