"""Atlas codebook: tokens <-> voxels.

* tokens_to_volume: look up each of the 4,096 ids and tile the 16^3 blocks into a 256^3 mosaic.
* AtlasTokenizer:   encode a CT into ids. Each 16^3 block is embedded by SigVLP *block by block*
                    (not the whole volume at once), mean-pooled, L2-normalized, and matched to the
                    codebook features by cosine similarity (exact, no FAISS). This reproduces the
                    stored CT-RATE training tokens exactly.

Volumes are uint8 (HU = v/255*2000 - 1000), shape (D, 256, 256), in the training
("mp4") orientation: axcodes ('S','P','R'), frame 0 = most inferior.
"""
import os, numpy as np
from .config import path, setup_third_party
from .sequence import CODEBOOK_SIZE, GRID

BS = 16


def load_atlas(p=None):
    return np.memmap(p or path("atlas_pixels"), dtype=np.uint8, mode="r").reshape(CODEBOOK_SIZE, BS, BS, BS)


def tokens_to_volume(ids, atlas):
    b = np.asarray(atlas[np.asarray(ids).reshape(-1)], np.uint8).reshape(GRID, GRID, GRID, BS, BS, BS)
    return b.transpose(0, 3, 1, 4, 2, 5).reshape(GRID * BS, GRID * BS, GRID * BS)


def center_crop_depth(v, n=256):
    """Depth centre crop used everywhere in training. Returns None if D < n."""
    D = v.shape[0]
    if D < n: return None
    mid = D // 2; s = max(0, mid - n // 2); e = min(D, s + n); s = max(0, e - n)
    return v[s:s + n]


class AtlasTokenizer:
    def __init__(self, device="cuda"):
        import torch, torch.nn.functional as F
        setup_third_party()
        from SigVLP.modeling_sigvlp import SigVLPConfig, SigVLPModel
        cfg_dir = os.path.join(os.path.dirname(__file__), "..", "third_party", "SigVLP")
        c = SigVLPConfig.from_pretrained(cfg_dir)
        from .config import CFG
        if CFG.get("siglip_base"): c.siglip_model_path = CFG["siglip_base"]   # SigLIP-2 backbone (HF id or local dir)
        m = SigVLPModel(c)
        st = torch.load(path("sigvlp_ckpt"), map_location="cpu")
        m.load_state_dict({k.replace("module.", "").replace("_orig_mod.", ""): v
                           for k, v in st.get("model", st).items()}, strict=False)
        self.vm = m.vision_model.to(device).eval(); del m, st
        raw = np.fromfile(path("atlas_feats"), dtype=np.float32).reshape(CODEBOOK_SIZE, 1025)
        assert (raw[:, 0].view(np.int32) == 1024).all(), "unexpected .fvecs layout"
        self.cb = F.normalize(torch.from_numpy(raw[:, 1:].copy()).to(device), dim=1)
        self.device, self.F, self.torch = device, F, torch

    def __call__(self, vol_u8, batch=512):
        """(D,256,256) uint8 -> (4096,) int32, or None if D < 256."""
        torch, F = self.torch, self.F
        v = center_crop_depth(vol_u8)
        if v is None: return None
        P = torch.from_numpy(np.ascontiguousarray(v)).reshape(16, 16, 16, 16, 16, 16) \
                 .permute(0, 2, 4, 1, 3, 5).reshape(-1, 16, 16, 16)
        ids = []
        with torch.no_grad():
            for b in range(0, P.shape[0], batch):
                x = ((P[b:b + batch].float() / 255. - 0.5) * 2.).unsqueeze(1).to(self.device)
                o = self.vm(x)
                f = (o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0]).mean(1).float()
                ids.append((F.normalize(f, dim=1) @ self.cb.T).argmax(1).cpu())
        return torch.cat(ids).numpy().astype(np.int32)
