import os

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

model_path = os.environ.get("LLAMA_MODEL_PATH")
if model_path is None:
    raise EnvironmentError(
        "The environment variable 'LLAMA_MODEL_PATH' is not set or is invalid."
    )
model = LlamaForCausalLM.from_pretrained(model_path, torchscript=True, torch_dtype=torch.float32)
model.config.use_cache = False

print(model)