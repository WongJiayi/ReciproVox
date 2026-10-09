import torch
import torch.nn as nn
import torch.nn.functional as F

from positional_encodings.torch_encodings import PositionalEncoding3D, Summer


class SiglipVisionEmbeddings3D(nn.Module):
    """
    3D patch embedding for SigLIP vision tower.

    - Input:  x of shape (B, C, D, H, W)
    - Output: embeddings of shape (B, N, embed_dim) where N = (D/pD)*(H/pH)*(W/pW)

    pos_embed_type:
        "none"    -> no absolute PE (recommended when using RoPE)
        "learned" -> learned 3D PE (interpolated across volumes of different sizes)
        "3d"      -> sinusoidal 3D PE from `positional_encodings` (added in conv space)

    Notes:
      * We accept extra kwargs in forward so it can be dropped into HF SigLIP2 code path
        which calls embeddings(pixel_values, spatial_shapes=...).
      * If you plan to use RoPE, set pos_embed_type="none".
    """
    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 768,
        patch_size=16,                  # int or (pD, pH, pW)
        pos_embed_type: str = "learned" # "none" | "learned" | "3d"
    ):
        super().__init__()

        # Normalize patch_size into (pD, pH, pW)
        if isinstance(patch_size, int):
            pD = pH = pW = patch_size
        else:
            assert len(patch_size) == 3, "patch_size must be int or a 3-tuple (pD, pH, pW)"
            pD, pH, pW = patch_size

        self.embed_dim = embed_dim
        self.patch_size = (pD, pH, pW)
        self.pos_embed_type = pos_embed_type

        # 3D patchify conv
        self.patch_embed = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(pD, pH, pW),
            stride=(pD, pH, pW)
        )

        # Positional embeddings
        if pos_embed_type == "none":
            self.position_embed = None
            self._learned_grid_shape = None
        elif pos_embed_type == "3d":
            # Add sinusoidal 3D PE in conv space (B, C, D', H', W') then flatten
            self.position_embed = Summer(PositionalEncoding3D(embed_dim))
            self._learned_grid_shape = None
        elif pos_embed_type == "learned":
            # We'll keep a learnable table that represents a 3D grid (D0, H0, W0)
            # The actual (D', H', W') is unknown at init; on first forward we will
            # initialize this table to (D', H', W'). If later shapes differ, we
            # interpolate the table to the new shape.
            self.position_embed = nn.Parameter(torch.zeros(1, embed_dim, 1, 1, 1))  # lazily expanded
            nn.init.trunc_normal_(self.position_embed, std=0.02)
            self._learned_grid_shape = None  # (D0, H0, W0) recorded at first forward
        else:
            raise ValueError(f"Invalid pos_embed_type '{pos_embed_type}'. Use 'none' | 'learned' | '3d'.")

    @torch.no_grad()
    def _interpolate_learned_pe(self, target_shape, device, dtype):
        """
        Interpolate the learned positional embedding to `target_shape` = (D', H', W').
        We store self.position_embed as (1, C, D0, H0, W0) internally.
        """
        Dp, Hp, Wp = target_shape

        if self._learned_grid_shape is None:
            # First time: expand parameter to current shape
            self.position_embed.data = self.position_embed.data.expand(1, self.embed_dim, Dp, Hp, Wp).clone()
            self._learned_grid_shape = (Dp, Hp, Wp)
            return

        D0, H0, W0 = self._learned_grid_shape
        if (D0, H0, W0) == (Dp, Hp, Wp):
            return  # shapes match, no-op

        # Interpolate from (D0, H0, W0) to (Dp, Hp, Wp)
        pe = self.position_embed.data  # (1, C, D0, H0, W0)
        pe = F.interpolate(pe, size=(Dp, Hp, Wp), mode="trilinear", align_corners=False)
        self.position_embed.data = pe
        self._learned_grid_shape = (Dp, Hp, Wp)

    def forward(self, x: torch.Tensor, **_ignored_kwargs) -> torch.Tensor:
        """
        x: (B, C, D, H, W)
        returns: (B, N, C) with N = D' * H' * W'
        """
        # Patchify
        target_dtype = self.patch_embed.weight.dtype
        x = x.to(dtype=target_dtype)
        patches = self.patch_embed(x)  # (B, C', D', H', W')

        # Add positional encoding (if any)
        if self.pos_embed_type == "none":
            pass  # do nothing

        elif self.pos_embed_type == "3d":
            # PositionalEncoding3D expects (B, C, D, H, W) and sums in place
            patches = self.position_embed(patches)

        elif self.pos_embed_type == "learned":
            # Make sure the learnable table is at the current grid size
            B, C, Dp, Hp, Wp = patches.shape
            self._interpolate_learned_pe(target_shape=(Dp, Hp, Wp),
                                         device=patches.device, dtype=patches.dtype)
            # Add to conv-space feature map
            # position_embed is (1, C, Dp, Hp, Wp)
            patches = patches + self.position_embed.to(patches.dtype)

        # Flatten to (B, N, C)
        patches = patches.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        return patches


class TextEmbeddingsNoAbsPos(nn.Module):
    """
    复用原 token embedding，不再相加 position embedding。
    尽量兼容 SigLIP v1 / v2：只要原模块有 .token_embedding 就能接上。
    """
    def __init__(self, embeddings: nn.Module):
        super().__init__()
        if hasattr(embeddings, "token_embedding"):
            self.token_embedding = embeddings.token_embedding
        else:
            # 部分实现可能叫 word_embeddings / embeddings 等
            # 你可以按需扩展检查
            raise AttributeError("Cannot find token_embedding on text embeddings module.")
        # 可保留 position_ids 以免上游代码索引
        if hasattr(embeddings, "position_ids"):
            self.register_buffer("position_ids", embeddings.position_ids, persistent=False)

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)
        return inputs_embeds