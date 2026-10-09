"""Mosaic -> CT refinement (two stages, both in the latent space of a CT fine-tuned FLUX VAE).

  uint8 mosaic --VAE.encode--> z_mosaic
     Stage 1  Deblocker (3D U-Net), 16-frame windows, stride 8, Hann blending -> z_mean = E[z_GT | z_mosaic]
     Stage 2  Rectifier (CTFlow STDiT + ControlNet), SDEdit from t* = 0.8 in N steps,
              block-autoregressive over depth (each block sees the previous block's output)
  --VAE.decode--> refined CT

Flow convention (CTFlow): x_t = (1-t) z + t eps, t=0 is data, v = eps - z, sample by x -= v dt.
"""
import numpy as np, torch
from .config import path, setup_third_party

LAT_MEAN, LAT_STD = -0.08395224064588547, 1.187322974205017
BLK = 16
sc = lambda z: (z - LAT_MEAN) / LAT_STD
unsc = lambda z: z * LAT_STD + LAT_MEAN


def to_latent_world(m):
    """training ("mp4") orientation -> VAE latent orientation."""
    return np.ascontiguousarray(np.asarray(m).transpose(0, 2, 1)[:, ::-1, :])


def to_mp4_world(m):
    return np.ascontiguousarray(np.asarray(m)[:, ::-1, :].transpose(0, 2, 1))


class Refiner:
    def __init__(self, rectifier_ckpt=None, device="cuda", t_star=0.8, N=8, stride=16, zmean_stride=8):
        setup_third_party()
        import diffusers
        from CTFlow.common.models import DiffuserSTDiT
        from .models.deblocker_unet3d import DeblockerUNet3D
        from .models.rectifier_flow import Rectifier
        self.vae = diffusers.AutoencoderKL.from_pretrained(path("vae_pretrained"), torch_dtype=torch.float32) \
                            .to(device, dtype=torch.bfloat16).eval()
        self.vae.load_state_dict(torch.load(path("vae_ft_ckpt"), map_location=device, weights_only=False))
        self.D = DeblockerUNet3D().to(device).eval()
        self.D.load_state_dict(torch.load(path("deblocker_ckpt"), map_location="cpu")["model"])
        sd = torch.load(rectifier_ckpt or path("rectifier_ckpt"), map_location="cpu")
        self.R = Rectifier(DiffuserSTDiT.from_pretrained(path("stdit_backbone")), mode="controlnet").to(device).eval()
        self.R.load_state_dict(sd["model"], strict=False)
        for m in (self.vae, self.D, self.R):
            for p in m.parameters(): p.requires_grad = False
        self.dev, self.t_star, self.N, self.stride, self.zmean_stride = device, t_star, N, stride, zmean_stride
        self._w = torch.hann_window(BLK + 2, periodic=False)[1:-1].view(1, BLK, 1, 1)

    def _starts(self, T, stride):
        st = list(range(0, T - BLK + 1, stride))
        if st[-1] != T - BLK: st.append(T - BLK)
        return st

    @torch.no_grad()
    def encode(self, vol):
        """(T,H,W) uint8, latent orientation -> (16,T,h,w) posterior mean (no scale factor)."""
        v = torch.as_tensor(np.ascontiguousarray(vol)).float() / 255. * 2. - 1.
        v = v.unsqueeze(1).repeat(1, 3, 1, 1)
        z = [self.vae.encoder(v[k:k + 32].to(self.dev, dtype=torch.bfloat16))[:, :16].float().cpu()
             for k in range(0, len(v), 32)]
        return torch.cat(z, 0).permute(1, 0, 2, 3)

    @torch.no_grad()
    def deblock(self, zm):
        T = zm.shape[1]
        out = torch.zeros(zm.shape[0], T, *zm.shape[2:]); ws = torch.zeros(1, T, 1, 1)
        for d in self._starts(T, self.zmean_stride):
            out[:, d:d + BLK] += self.D(zm[:, d:d + BLK][None].to(self.dev)).float()[0].cpu() * self._w
            ws[:, d:d + BLK] += self._w
        return out / ws.clamp(min=1e-6)

    @torch.no_grad()
    def rectify(self, zmu, zmos, seed=1234):
        g = torch.Generator(device=self.dev).manual_seed(seed)
        T = zmu.shape[1]; out = torch.zeros_like(zmu); ws = torch.zeros(1, T, 1, 1)
        prev, h = None, self.t_star / self.N
        full = lambda v: torch.full((1,), v, device=self.dev)
        for d in self._starts(T, self.stride):
            mu = sc(zmu[:, d:d + BLK][None].to(self.dev)); mo = sc(zmos[:, d:d + BLK][None].to(self.dev))
            x = (1 - self.t_star) * mu + self.t_star * torch.randn(mu.shape, generator=g, device=self.dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for k in range(self.N):
                    x = x - h * self.R(x, full(self.t_star - k * h), mo, gap=full(h), z_prev=prev, z_mean=mu).float()
            prev = x
            out[:, d:d + BLK] += unsc(x)[0].cpu() * self._w; ws[:, d:d + BLK] += self._w
        return out / ws.clamp(min=1e-6)

    @torch.no_grad()
    def decode(self, z):
        return torch.cat([self.vae.decoder(z[:, k:k + 32].permute(1, 0, 2, 3).to(self.dev, dtype=torch.bfloat16))
                          .mean(1).float().cpu() for k in range(0, z.shape[1], 32)], 0).clamp(-1, 1)

    def __call__(self, mosaic_u8, seed=1234, stage=2):
        """(T,256,256) uint8 mosaic in training orientation -> refined uint8 volume, same orientation.
        stage=1 stops after the Deblocker (no Rectifier)."""
        v = to_latent_world(mosaic_u8)
        zmos = self.encode(v); z = self.deblock(zmos)
        if stage >= 2: z = self.rectify(z, zmos, seed)
        o = to_mp4_world(self.decode(z).numpy())
        return np.clip(np.rint((o + 1) * 127.5), 0, 255).astype(np.uint8)
