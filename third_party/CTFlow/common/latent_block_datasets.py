# echosyn/data/latent_block.py

import os
import torch
from torch.utils.data import Dataset

class LatentBlockDataset(Dataset):
    def __init__(self, root_dir, block_size=16):
        self.root_dir = root_dir
        self.block_size = block_size
        self.file_paths = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".pt")
        ])
        self.index_map = self._build_index_map()

    def _build_index_map(self):
        """
        构造 (file_idx, start_t) 索引列表，每个表示一对 block（当前 + 下一）
        """
        index_map = []
        for file_idx, path in enumerate(self.file_paths):
            latent = torch.load(path, map_location="cpu")
            T = latent.shape[0]
            for t in range(0, T - 2 * self.block_size + 1):
                index_map.append((file_idx, t))
        return index_map

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, t = self.index_map[idx]
        path = self.file_paths[file_idx]
        latent = torch.load(path, map_location="cpu")  # [T, C, H, W]

        block_curr = latent[t:t+self.block_size]       # [B, C, H, W]
        block_next = latent[t+self.block_size:t+2*self.block_size]

        # 转为 [C, T, H, W]
        cond_image = block_curr.permute(1, 0, 2, 3)
        video = block_next.permute(1, 0, 2, 3)

        return {
            "image": cond_image,  # 用于 condition 拼接
            "video": video        # 作为目标输出
        }

# 直接使用 config.datasets[0] 初始化
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