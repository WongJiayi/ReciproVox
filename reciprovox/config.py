"""Central path configuration.

Lookup order: $RECIPROVOX_CONFIG, configs/paths.local.yaml, configs/paths.yaml.
Paths starting with checkpoints/ or ./ are resolved against the repository root.
"""
import os, sys, types, importlib.machinery

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THIRD_PARTY = os.path.join(ROOT, "third_party")


def _load():
    import yaml
    for p in (os.environ.get("RECIPROVOX_CONFIG", ""),
              os.path.join(ROOT, "configs", "paths.local.yaml"),
              os.path.join(ROOT, "configs", "paths.yaml")):
        if p and os.path.exists(p):
            cfg = yaml.safe_load(open(p)) or {}
            for k, v in cfg.items():
                # repo-relative paths start with checkpoints/ or ./ ; anything else (e.g. a
                # Hugging Face id like google/siglip2-large-patch16-256) is kept as is
                if isinstance(v, str) and v.startswith(("checkpoints/", "./", "../")):
                    cfg[k] = os.path.normpath(os.path.join(ROOT, v))
            cfg["_file"] = p
            return cfg
    raise FileNotFoundError("no configs/paths.yaml found")


CFG = _load()


def path(key):
    v = CFG.get(key, "")
    if not v:
        raise KeyError(f"'{key}' is not set in {CFG['_file']}")
    return v


def setup_third_party():
    """Put vendored CTFlow / SigVLP on sys.path and stub optional imports they never use at inference."""
    if THIRD_PARTY not in sys.path:
        sys.path.insert(0, THIRD_PARTY)
    for name, attrs in (("faiss", {}), ("muon", {"MuonWithAuxAdam": object})):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__spec__ = importlib.machinery.ModuleSpec(name, None)
            for a, v in attrs.items(): setattr(m, a, v)
            sys.modules[name] = m
