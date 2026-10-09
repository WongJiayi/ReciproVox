import os
import torch
import random
from torch.utils.data import Dataset

class LatentBlockDataset(Dataset):
    def __init__(self, root_dir, embedding_dir, block_size=16):
        self.root_dir = root_dir
        self.embedding_dir = embedding_dir
        self.block_size = block_size

        self.file_paths = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".pt")
        ])

        # concat embedding path
        self.embedding_paths = [
            os.path.join(embedding_dir, os.path.basename(f))
            for f in self.file_paths
        ]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        latent_path = self.file_paths[idx]
        embed_path = self.embedding_paths[idx]

        latent = torch.load(latent_path, map_location="cpu")  # [C, T, H, W]
        embedding = torch.load(embed_path, map_location="cpu")  # [N, D]，如 [128, 796]
        embedding = embedding[0].unsqueeze(0)
        #print(embedding.norm())
        embedding = embedding / (embedding.norm(p=2) + 1e-6) * 1


        C, T, H, W = latent.shape
        max_start = T - 2 * self.block_size
        t = random.randint(0, max_start)

        block_curr = latent[:, t:t + self.block_size]          # [C, T, H, W]
        block_next = latent[:, t + self.block_size:t + 2 * self.block_size]

        return {
            "image": block_curr,
            "video": block_next,
            "embedding": embedding  # shape: [N, D]
        }


def instantiate_dataset(configs, split=None):
    datasets = []
    for cfg in configs:
        if not cfg.get("active", False):
            continue
        name = cfg.name
        params = dict(cfg.params)

        if name == "LatentBlock":
            dataset = LatentBlockDataset(**params)
        else:
            raise ValueError(f"Unknown dataset name: {name}")
        datasets.append(dataset)

    if len(datasets) == 1:
        return datasets[0]
    from torch.utils.data import ConcatDataset
    return ConcatDataset(datasets)

if __name__ == "__main__":
    pass