import torch
from .arch.modeling_mmamba import MeteorMambaForCausalLM

def load_mmamba(link):

    # huggingface model configuration
    huggingface_config = {}
    huggingface_config.update(dict(
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ))

    # Meteor Mamba Model (no fp32)
    mmamba = MeteorMambaForCausalLM.from_pretrained(link, **huggingface_config)

    return mmamba