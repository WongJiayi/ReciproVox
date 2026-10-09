---
license: mit
---
# SigVLP (SigLIP-based Vision–Language Pretraining for 3D CT)

This repository provides the pretrained **SigVLP** model, a 3D vision–language encoder built on top of **Google SigLIP-2 Large** and extended with:

- **3D Patch Embedding**
- **3D RoPE (Rotary Position Embedding)** on Vision Attention
- **Text RoPE** for SigLIP’s text encoder
- **Cross-modality alignment for CT + Radiology text**
- **Trained on CT-RATE** (3D CT volumes with radiology reports)