"""CT -> report.

  python scripts/generate_report.py --inputs scan1.nii.gz scan2.nii.gz --out reports.jsonl
  python scripts/generate_report.py --input_dir frames_root/ --out reports.jsonl   # one sub-dir of PNG frames per case

Inputs may be NIfTI (converted to the training format first), PNG-frame directories, .npy volumes
in training format, or precomputed token files (--tokens_dir). Volumes need >= 256 frames after
z-resampling (the centre 256 are used), as in training. Decoding is greedy.
Output jsonl: {"id", "volume_name", "generated"} — the format read by evaluation/report_metrics.py.
"""
import _bootstrap  # noqa
import os, glob, json, argparse, numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", default=[])
    ap.add_argument("--input_dir", default="")
    ap.add_argument("--tokens_dir", default="", help="skip tokenization: <name>.npy (4096,) ids")
    ap.add_argument("--out", required=True)
    ap.add_argument("--z_mm", type=float, default=1.5, help="z spacing for NIfTI conversion")
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--max_new", type=int, default=384)
    ap.add_argument("--lora", default=None)
    a = ap.parse_args()

    items = []
    if a.tokens_dir:
        items = [(os.path.basename(f)[:-4], f) for f in sorted(glob.glob(f"{a.tokens_dir}/*.npy"))]
    else:
        srcs = list(a.inputs) + ([d for d in sorted(glob.glob(f"{a.input_dir}/*")) if os.path.isdir(d)] if a.input_dir else [])
        from reciprovox.atlas import AtlasTokenizer
        from reciprovox.io import load_volume
        T = AtlasTokenizer(); tmp = os.path.splitext(a.out)[0] + "_tokens"; os.makedirs(tmp, exist_ok=True)
        for s in srcs:
            name = os.path.basename(s.rstrip("/")).replace(".nii.gz", "").replace(".nii", "").replace(".npy", "")
            f = f"{tmp}/{name}.npy"
            if not os.path.exists(f):
                t = T(load_volume(s, z_mm=a.z_mm))
                if t is None: print(f"  skip {name}: fewer than 256 frames"); continue
                np.save(f, t)
            items.append((name, f))
        del T
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["volume_name"] for l in open(a.out)}
    items = [x for x in items if x[0] not in done]
    print(f"[generate_report] {len(items)} volumes", flush=True)

    from reciprovox.lm import ReciproVoxLM
    lm = ReciproVoxLM(lora=a.lora)
    with open(a.out, "a") as fo:
        for k in range(0, len(items), a.bs):
            chunk = items[k:k + a.bs]
            texts = lm.generate_reports([np.load(f) for _, f in chunk], max_new=a.max_new)
            for i, ((n, _), t) in enumerate(zip(chunk, texts)):
                fo.write(json.dumps({"id": k + i, "volume_name": n, "generated": t}) + "\n")
            fo.flush(); print(f"  {k + len(chunk)}/{len(items)}", flush=True)


if __name__ == "__main__":
    main()
