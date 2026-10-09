"""Volume I/O. All volumes are uint8, HU = v/255*2000 - 1000, shape (D,256,256), training orientation:
axcodes ('S','P','R') -> frame 0 is most inferior, rows run anterior->posterior, columns patient-left->right."""
import os, re, glob, numpy as np


def u8_to_hu(v): return np.asarray(v, np.float32) / 255. * 2000. - 1000.


def hu_to_u8(h): return np.clip(np.rint((np.clip(h, -1000, 1000) + 1000) / 2000 * 255), 0, 255).astype(np.uint8)


def save_frames(vol, out_dir):
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    for i, f in enumerate(vol): Image.fromarray(f).save(f"{out_dir}/frame_{i:03d}.png", compress_level=1)


def load_frames(d):
    from PIL import Image
    fs = sorted(glob.glob(f"{d}/*.png"), key=lambda s: int(re.findall(r"\d+", os.path.basename(s))[-1]))
    return np.stack([np.asarray(Image.open(f).convert("L")) for f in fs])


def nifti_to_train_format(img, z_mm=1.5, size=256):
    """nibabel image -> (D,size,size) uint8 in training orientation.

    Calibrated against the training videos (pixel MAE 0.7%): reorient to ('S','P','R'), clip HU to
    [-1000, 1000], resample z to `z_mm` (linear), resize every frame bilinearly to size x size.
    In-plane the whole field of view is resized, so in-plane spacing = FOV / size.
    """
    import torch, torch.nn.functional as F
    from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform
    img = img.as_reoriented(ornt_transform(io_orientation(img.affine), axcodes2ornt(("S", "P", "R"))))
    v = np.clip(np.asanyarray(img.dataobj).astype(np.float32), -1000, 1000)
    sz = float(img.header.get_zooms()[0])
    t = torch.from_numpy(v)[None, None]
    nz = max(1, round(v.shape[0] * sz / z_mm))
    t = F.interpolate(t, size=(nz, v.shape[1], v.shape[2]), mode="trilinear", align_corners=False)[0, 0]
    t = F.interpolate(t[:, None], size=(size, size), mode="bilinear", align_corners=False)[:, 0]
    return hu_to_u8(t.numpy())


def load_volume(p, z_mm=1.5):
    """.nii/.nii.gz -> converted; directory of PNG frames or .npy -> as is."""
    if os.path.isdir(p): return load_frames(p)
    if p.endswith(".npy"): return np.load(p)
    import nibabel as nib
    return nifti_to_train_format(nib.load(p), z_mm=z_mm)
