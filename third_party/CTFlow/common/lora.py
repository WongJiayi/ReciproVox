
import os
from functools import partial

import torch
import safetensors

import diffusers
from diffusers.loaders import PeftAdapterMixin
from diffusers.utils.peft_utils import get_peft_kwargs, recurse_remove_peft_layers

from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
from peft.tuners.lora.layer import LoraLayer
from peft.tuners.tuners_utils import BaseTunerLayer

def enforce_lora_support(model):
    """
    Adds support for LoRA to a model, by creating a new class 
    that inherits the PeftAdapterMixin class and the original model class.

    Args:
        model (object): The model to add LoRA support to.
    
    Returns:
        modelPA: The model with LoRA support.
    """

    # Check if model is a subclass of PeftAdapterMixin
    if issubclass(model.__class__, PeftAdapterMixin):
        print(f"Model {model.__class__.__name__} already has LoRA support.")
        return model

    # print(f"Adding LoRA support to {model.__class__.__name__}... ", end="")
    
    current_klass = model.__class__

    # define the class name for the new class
    new_klass_name = getattr(current_klass, "__class_name__", current_klass.__name__) + "PA"

    # define the init method for the new class, which calls the init method of the original model
    # PeftAdapterMixin does not do anything in the init method
    def __init__(self, *args, **kwargs):
        super(new_class, self).__init__(*args, **kwargs)

    # dynamically create the new class
    new_klass = type(new_klass_name, (current_klass, PeftAdapterMixin), {'__init__': __init__})

    # change the class of the model to the new class
    # this works because PeftAdapterMixin only uses existing attributes of the model
    model.__class__ = new_klass

    # print("Done.")
    print(f"Model {current_klass.__name__} was wrapped as {model.__class__.__name__} with LoRA support.")

    return model

def revert_class_to_original(model):
    # Test if the model is a custom subclass of PeftAdapterMixin
    if not model.__class__.__name__.endswith("PA"):
        return model
    
    # get original class
    assert len(model.__class__.__bases__) == 2, "Expected model to have two base classes: PeftAdapterMixin and the original class."
    bases = list(model.__class__.__bases__)
    bases.remove(diffusers.loaders.peft.PeftAdapterMixin)
    original_class = bases[0]

    # change the class of the model to the original class
    model.__class__ = original_class
    return model

def write_lora_layers(state_dict, save_directory, weight_name="pytorch_lora_weights.safetensors"):
    if os.path.isfile(save_directory):
        print(f"Provided path ({save_directory}) should be a directory, not a file")
        return

    os.makedirs(save_directory, exist_ok=True)
    save_path = os.path.join(save_directory, weight_name)
    safetensors.torch.save_file(state_dict, save_path, metadata={"format": "pt"})
    print(f"Model weights saved in {save_path}")

def save_lora_weights(lora_layers, save_directory, weight_name="pytorch_lora_weights.safetensors"):

    state_dict = {}

    # def pack_weights(layers, prefix):
    #     layers_weights = layers.state_dict() if isinstance(layers, torch.nn.Module) else layers
    #     layers_state_dict = {f"{prefix}.{module_name}": param for module_name, param in layers_weights.items()}
    #     return layers_state_dict
    
    # state_dict.update(pack_weights(lora_layers, "transformer"))
    state_dict.update(lora_layers)

    write_lora_layers(
        state_dict=state_dict,
        save_directory=save_directory,
        weight_name=weight_name,
    )

def merge_lora_weights(denoiser, lora_state_dict):
    # Constants
    TRANSFORMER="transformer"
    adapter_name = "default"
    network_alphas = None

    # 1. Load the LoRA weights
    state_dict = lora_state_dict
    keys = list(state_dict.keys())
    # print(state_dict.keys())

    # 2. Remove the "transformer" prefix
    # TODO: Check why this was necessary in the first place
    # if denoiser.__class
    # transformer_keys = [k for k in keys if k.startswith(TRANSFORMER)]
    # state_dict = { k.replace(f"{TRANSFORMER}.", ""): v for k, v in state_dict.items() if k in transformer_keys }
    # print(state_dict.keys())

    # 3. Re-Instantiate the LoRA config
    rank = {}
    for key, val in state_dict.items():
        if "lora_B" in key:
            rank[key] = val.shape[1]

    lora_config_kwargs = get_peft_kwargs(rank, network_alphas, state_dict)
    lora_config = LoraConfig(**lora_config_kwargs)
    # print(lora_config_kwargs)

    # 4. Inject the LoRA adapter in the model
    # TODO: Check if the adapter is already in the model
    inject_adapter_in_model(lora_config, denoiser, adapter_name=adapter_name) # Inject the empty adapter in the model
    incompatible_keys = set_peft_model_state_dict(denoiser, state_dict, adapter_name) # Inject the LoRA weights in the adapter
    # print(incompatible_keys.unexpected_keys)

    # 5. Merge the LoRA adapter with the model
    def _fuse_lora_apply(module, model):
        merge_kwargs = {"safe_merge": True} # check if values are NaN and avoid merging them if they are
        if isinstance(module, BaseTunerLayer):
            module.merge(**merge_kwargs)
    denoiser.apply(partial(_fuse_lora_apply, model=denoiser))

    # 6. Remove the LoRA adapter
    recurse_remove_peft_layers(denoiser)
    if hasattr(denoiser, "peft_config"):
        del denoiser.peft_config

    # 7. Set the class back to the original class
    denoiser = revert_class_to_original(denoiser)
    return denoiser
