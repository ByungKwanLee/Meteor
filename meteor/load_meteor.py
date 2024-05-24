import torch
import warnings
from config import *
from transformers import BitsAndBytesConfig
from .arch.modeling_meteor import MeteorForCausalLM
from .arch.tokenization_internlm2 import InternLM2Tokenizer
warnings.filterwarnings(action='ignore')


def load_meteor(link, bits):
    
    # huggingface model configuration
    huggingface_config = {}

    # Bit quantization
    if bits in [4, 8]:
        huggingface_config.update(dict(
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=bits == 4,
                load_in_8bit=bits == 8,
                llm_int8_skip_modules=["vit", "vision_proj", "output", "ffn"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
        ))
    else:
        huggingface_config.update(dict(
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        ))

    # loading backbone model
    meteor = MeteorForCausalLM.from_pretrained(link, **huggingface_config)

    # loading meteor tokenizer
    # adding <image> and <tor> special token
    tok_meteor = InternLM2Tokenizer.from_pretrained(link, padding_side='left')
    tok_meteor.add_tokens("<image>", special_tokens=True)
    tok_meteor.add_tokens("<tor>", special_tokens=True)
    return meteor, tok_meteor