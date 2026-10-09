# =========================================================
# 0. Imports
# =========================================================
import torch
import torch.nn as nn

from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoModel,
    CONFIG_MAPPING,
    MODEL_MAPPING,
)


# =========================================================
# 1. Clean State Dict Helper
# =========================================================
def _clean_state_dict(sd: dict):
    new_sd = {}
    for k, v in sd.items():
        k = k.replace("module.", "")
        k = k.replace("_orig_mod.", "")
        new_sd[k] = v
    return new_sd


# =========================================================
# 2. SigVLPConfig (must be defined BEFORE registration)
# =========================================================
class SigVLPConfig(PretrainedConfig):
    model_type = "sigvlp"

    def __init__(
        self,
        siglip_model_path="google/siglip2-large-patch16-256",
        in_channels=1,
        patch_size=16,
        text_max_len=256,
        pos_embed_type="3d",
        rope=None,
        ckpt_path=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.siglip_model_path = siglip_model_path
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.text_max_len = text_max_len
        self.pos_embed_type = pos_embed_type
        self.ckpt_path = ckpt_path

        # Default RoPE config
        default_rope = {
            "text": {"enable": True, "base": 1000.0},
            "vision": {"enable": True, "mode": "videorope", "theta": 10000.0, "t_scale": 1.0},
        }

        if rope is None:
            rope = default_rope
        else:
            # fill missing fields
            for k, v in default_rope.items():
                rope.setdefault(k, v)
                for kk, vv in v.items():
                    rope[k].setdefault(kk, vv)

        self.rope = rope


# =========================================================
# 3. SigVLPModel
# =========================================================
from .models.rope_generic import (
    monkey_patch_siglip_rope,
    configure_text_pos_mode,
    enable_text_rope_in_attention,
)
from .models.rope_kernels import prepare_vision_rope
from .utils_v2 import inject_3d_embeddings


class SigVLPModel(PreTrainedModel):
    config_class = SigVLPConfig

    def __init__(self, config: SigVLPConfig):
        super().__init__(config)

        # ---- Load SigLIP2 backbone ----
        base_model = AutoModel.from_pretrained(
            config.siglip_model_path,
            local_files_only=True
        )

        # ---- Apply RoPE monkey patch ----
        monkey_patch_siglip_rope()

        # ---- Inject 3D Embeddings ----
        inject_3d_embeddings(
            base_model,
            in_channels=int(config.in_channels),
            embed_dim=base_model.config.vision_config.hidden_size,
            patch_size=int(config.patch_size),
            pos_embed_type=config.pos_embed_type,
        )

        # ---- Configure text RoPE ----
        rope_text_cfg = config.rope["text"]
        if rope_text_cfg["enable"]:
            configure_text_pos_mode(
                base_model,
                mode="rope",
                rope_base=float(rope_text_cfg["base"]),
                max_seq_len=int(config.text_max_len),
            )
            enable_text_rope_in_attention(
                rope_base=float(rope_text_cfg["base"])
            )
        else:
            configure_text_pos_mode(
                base_model,
                mode="abs",
                max_seq_len=int(config.text_max_len),
            )

        # ---- Load finetuned checkpoint if provided ----
        if config.ckpt_path:
            print(f"[SigVLP] Loading checkpoint: {config.ckpt_path}")
            ckpt = torch.load(config.ckpt_path, map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            ckpt = _clean_state_dict(ckpt)
            missing, unexpected = base_model.load_state_dict(ckpt, strict=False)
            print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

        # ---- Expose models ----
        self.base_model = base_model
        self.vision_model = base_model.vision_model
        self.text_model = getattr(base_model, "text_model", None)

        self.config.vision_config = base_model.config.vision_config

        self.post_init()

    # Forward = vision forward
    def forward(
        self,
        pixel_values=None,
        input_ids=None,
        attention_mask=None,
        return_dict=True,
        **kwargs
    ):
        cfg = self.config
        vis_rope = cfg.rope["vision"]

        # ===== Vision branch =====
        if pixel_values is not None and vis_rope["enable"]:
            prepare_vision_rope(
                self.vision_model.encoder,
                pixel_values,
                patch_size=cfg.patch_size,
                head_dim=self.config.vision_config.hidden_size
                        // self.config.vision_config.num_attention_heads,
                theta=float(vis_rope["theta"]),
                mode=str(vis_rope["mode"]),
                t_scale=float(vis_rope["t_scale"]),
                device=pixel_values.device,
            )

        vision_out = None
        if pixel_values is not None:
            vision_out = self.vision_model(pixel_values=pixel_values)

        # ===== No text: return vision only =====
        if input_ids is None:
            text_out = None
        else:
            # ===== Text branch =====
            text_out = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        # ===== Projection =====
        image_embeds = vision_out
        text_embeds  = text_out

        #image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        #text_embeds  = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        #logit_scale = self.logit_scale.exp()
        #logits_per_image = logit_scale * image_embeds @ text_embeds.T
        #logits_per_text  = logits_per_image.T

        return {
            #"logits_per_image": logits_per_image,
            #"logits_per_text": logits_per_text,
            "image_embeds": image_embeds,
            "text_embeds": text_embeds,
        }



# =========================================================
# 4. Register model with HuggingFace
# =========================================================
CONFIG_MAPPING.register("sigvlp", SigVLPConfig)
MODEL_MAPPING.register(SigVLPConfig, SigVLPModel)
