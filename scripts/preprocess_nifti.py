"""NIfTI -> training format (PNG frames), e.g. to prepare an external test set.

  python scripts/preprocess_nifti.py --inputs a.nii.gz b.nii.gz --out data/frames --z_mm 1.5

Orientation ('S','P','R'), HU clipped to [-1000, 1000], z resampled to --z_mm, frames resized to 256.
The calibration against the training videos used z = 1.5 mm; the BIMCV-R experiments used 1.0 mm.
"""
import _bootstrap  # noqa
import os, argparse
from reciprovox.io import load_volume, save_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--z_mm", type=float, default=1.5)
    a = ap.parse_args()
    for p in a.inputs:
        name = os.path.basename(p).replace(".nii.gz", "").replace(".nii", "")
        v = load_volume(p, z_mm=a.z_mm); save_frames(v, f"{a.out}/{name}")
        print(name, v.shape)


if __name__ == "__main__":
    main()
