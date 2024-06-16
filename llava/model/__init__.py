# try:
from .language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig
from .language_model.llava_mpt import LlavaMptForCausalLM, LlavaMptConfig
from .language_model.llava_mistral import LlavaMistralForCausalLM, LlavaMistralConfig

from .language_model.deepstack_llama import DeepstackLlamaForCausalLM, DeepstackConfig
from .language_model.deepstack_phi import DeepstackPhiForCausalLM, DeepstackPhiConfig


MODEL_REGISTRY = {
    'llama': DeepstackLlamaForCausalLM,
    'phi-3': DeepstackPhiForCausalLM,
    'phi3': DeepstackPhiForCausalLM,
}

LLAVA_MODEL_REGISTRY = {
    'llama': LlavaLlamaForCausalLM,
    'mpt': LlavaMptForCausalLM,
    'mistral': LlavaMistralForCausalLM,
}

def ModelSelect(model_name_or_path):
    model = None

    registry = MODEL_REGISTRY if not 'llava' in model_name_or_path.lower() else LLAVA_MODEL_REGISTRY
    for name in registry.keys():
        if name.lower() in model_name_or_path.lower():
            model = registry[name]
    if model is None:
        model = registry['llama']
    return model