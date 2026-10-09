import ctypes
import os
import sys
from glob import glob

import decord
import numpy as np
import pandas as pd
import torch
# import multiprocessing as mp
import torch.multiprocessing as mp
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler
from torchvision import transforms as T
from torchvision.transforms.functional import center_crop

from CTFlow.common import loadvideo

decord.bridge.set_bridge("torch")

# VIEWS = ["A4C", "PSAX", "PLAX"]
from CTFlow.common.datasets import VIEWS


class MPCacheArray:
    def __init__(
        self, shape: list, load_func: callable, e_ctype=ctypes.c_float, verbose=False
    ) -> None:
        # ctypes: ctypes.c_uint8, ctypes.c_int, ctypes.c_float
        self.shape = shape
        shared_array_base = mp.Array(e_ctype, int(np.prod(shape)))
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array = shared_array.reshape(*shape)
        self.shared_array = torch.from_numpy(shared_array)

        idx_is_cached_base = mp.Array(ctypes.c_bool, shape[0])
        idx_is_cached = np.ctypeslib.as_array(idx_is_cached_base.get_obj())
        self.idx_is_cached = torch.from_numpy(idx_is_cached)

        self.load_func = load_func
        self.verbose = verbose

    def load(self, idx, *args, **kwargs):
        if self.idx_is_cached[idx]:
            if self.verbose:
                print(f"Cache hit for idx {idx}")
            return self.shared_array[idx]
        else:
            if self.verbose:
                print(f"Cache miss for idx {idx}")
            data = self.load_func(*args, **kwargs)  # This returns a numpy array
            data_tensor = torch.from_numpy(data).to(
                dtype=self.shared_array.dtype
            )  # Convert numpy array to torch.Tensor with the correct dtype
            self.shared_array[idx] = data_tensor
            self.idx_is_cached[idx] = True
            return data_tensor

    def cache_status(self):
        print(
            f"Cache status ({(self.idx_is_cached == True).sum()}/{self.idx_is_cached.shape[0]})"
        )


class MPCacheVideo:
    def __init__(
        self, shape: list, load_func: callable, e_ctype=ctypes.c_float, verbose=False
    ) -> None:
        # ctypes: ctypes.c_uint8, ctypes.c_int, ctypes.c_float
        if verbose:
            print(f"Creating data array of shape {shape}")
        self.shape = shape
        shared_array_base = mp.Array(e_ctype, int(np.prod(shape)))
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array = shared_array.reshape(*shape)
        self.shared_array = torch.from_numpy(shared_array)
        size_in_bytes_pt = (
            self.shared_array.element_size() * self.shared_array.nelement()
        )

        if verbose:
            print(f"Creating index array of shape {shape[0]}")
        idx_is_cached_base = mp.Array(ctypes.c_bool, shape[0])
        idx_is_cached = np.ctypeslib.as_array(idx_is_cached_base.get_obj())
        self.idx_is_cached = torch.from_numpy(idx_is_cached)

        self.load_func = load_func
        self.verbose = verbose

    def load(self, start_idx, end_idx, *args, **kwargs):
        if sum(self.idx_is_cached[start_idx:end_idx]) == (end_idx - start_idx):
            if self.verbose:
                print(f"Cache hit for idx {start_idx}:{end_idx}")
            return self.shared_array[start_idx:end_idx]
        else:
            if self.verbose:
                print(f"Cache miss for idx {start_idx}:{end_idx}")
            data_tensor = self.load_func(*args, **kwargs).to(
                dtype=self.shared_array.dtype
            )
            # data_tensor = torch.from_numpy(data) # Convert numpy array to torch.Tensor with the correct dtype
            self.shared_array[start_idx:end_idx] = data_tensor
            self.idx_is_cached[start_idx:end_idx] = torch.ones(
                end_idx - start_idx, dtype=torch.bool
            )
            return data_tensor

    def cache_status(self):
        print(
            f"Cache status ({(self.idx_is_cached == True).sum()}/{self.idx_is_cached.shape[0]})"
        )


class ContrastiveEchoNet(Dataset):
    def __init__(self, root, split=["TRAIN", "VAL", "TEST"]):

        self.split = split

        self.metadata = pd.read_csv(os.path.join(root, "FileList.csv"))
        self.metadata = self.metadata[self.metadata.Split.isin(self.split)]

        self.start_stop_indices = {}
        start = 0
        for row in self.metadata.itertuples():
            start, stop = start, start + int(row.NumberOfFrames)
            self.start_stop_indices[row.FileName] = (start, stop)
            start = stop
        print(f"Ready to load {start:_} frames. This will take some time.")
        total_frames = start

        def load_video_np(path):
            video = loadvideo(path)
            video = video.permute(0, 3, 1, 2)
            return video

        self.video_folder = os.path.join(root, "Videos")

        N, C, H, W = total_frames, 3, 112, 112

        self.video_cache = MPCacheVideo(
            shape=[N, C, H, W],
            load_func=load_video_np,
            e_ctype=ctypes.c_uint8,
            verbose=False,
        )

    def __len__(self):
        return len(self.metadata)

    def __get_one_video__(self, idx):
        row = self.metadata.iloc[idx]
        start, stop = self.start_stop_indices[row.FileName]
        video = self.video_cache.load(
            start, stop, os.path.join(self.video_folder, row.FileName + ".avi")
        )
        return video

    def __getitem__(self, idx):
        video1 = self.__get_one_video__(idx)
        video2 = self.__get_one_video__(np.random.randint(len(self.metadata)))

        print(video1.shape, video2.shape)

        selected_indices1 = np.random.choice(video1.shape[0], 2, replace=False)
        selected_indices2 = np.random.choice(video2.shape[0], 2, replace=False)
        frames = torch.cat(
            [video1[selected_indices1], video2[selected_indices2]], dim=0
        )

        # Apply transformations
        frames = frames / 128.0 - 1.0

        framesA1, framesA2, framesB1, framesB2 = frames.split(1, dim=0)

        return framesA1[0], framesA2[0], framesB1[0], framesB2[0]


class InfiniteTensor:
    def __init__(self, base_tensor):
        self.base_tensor = base_tensor.unsqueeze(0)  # Shape (1, H, W)

    def __getitem__(self, idx):
        # Always return the same base tensor, treating the first dimension as "infinite"
        return self.base_tensor[0]  # Return the single base tensor


def Identity(x):
    return x


def load_pt(path):
    video = torch.load(path)
    return video


def load_avi(path):
    video = loadvideo(path)
    video = video.permute(0, 3, 1, 2)
    return video


def load_seg(path):
    downscale_factor = 8
    seg = torch.load(path)  # B H*8 W*8
    seg = torch.nn.functional.max_pool2d(seg.long(), kernel_size=downscale_factor)
    return seg


class CachedEchoNet(Dataset):
    def __init__(
        self, config, split=["TRAIN", "VAL", "TEST"], datafolder="Videos", ext=".avi"
    ) -> None:
        super().__init__()

        if type(split) == str:
            split = [split]
        assert [
            s in ["TRAIN", "VAL", "TEST"] for s in split
        ], "Splits must be a list of TRAIN, VAL, TEST"

        self.target_fps = config.target_fps
        self.resolution = config.target_resolution
        self.outputs = config.outputs
        if type(self.outputs) == str:
            self.outputs = [self.outputs]
        assert [
            o in ["video", "image", "lvef", "view"] for o in self.outputs
        ], "Outputs must be a list of video, image, lvef, view"

        self.duration_frames = config.target_nframes
        self.duration_seconds = (
            self.duration_frames / self.target_fps
            if type(self.target_fps) == int
            else None
        )

        self.output_key = config.get("output_key", None)

        self.view_label = config.get("view_label", "A4C")
        assert self.view_label in VIEWS, f"View label must be one of {VIEWS}"
        self.view_label_index = VIEWS.index(self.view_label)

        # LOAD DATA
        assert hasattr(config, "root"), "No root folder specified in config"
        assert os.path.exists(
            os.path.join(config.root, datafolder)
        ), f"Data folder {os.path.join(config.root, datafolder)} does not exist"
        assert os.path.exists(
            os.path.join(config.root, "FileList.csv")
        ), f"FileList.csv does not exist in {config.root}"

        self.metadata = pd.read_csv(os.path.join(config.root, "FileList.csv"))
        self.metadata = self.metadata[
            self.metadata["Split"].isin(split)
        ]  # filter by split
        self.len_before_filter = len(self.metadata)
        # add duration column
        self.metadata["Duration"] = (
            self.metadata["NumberOfFrames"] / self.metadata["FPS"]
        )  # won't work for pediatrics
        # filter by duration

        if self.duration_seconds is not None:
            self.metadata = self.metadata[
                self.metadata["Duration"] >= self.duration_seconds
            ]
       
        # check if videos are reachable
        self.metadata["VideoPath"] = self.metadata["FileName"].apply(
            lambda x: (
                os.path.join(config.root, datafolder, x)
                if x.endswith(ext)
                else os.path.join(config.root, datafolder, x.split(".")[0] + ext)
            )
        )

        self.metadata["VideoExists"] = self.metadata["VideoPath"].apply(
            lambda x: os.path.exists(x)
        )
        self.metadata = self.metadata[self.metadata["VideoExists"]]
        self.metadata.reset_index(inplace=True, drop=True)
        if len(self.metadata) == 0:
            raise ValueError(
                f"No data found in folder {os.path.join(config.root, datafolder)}"
            )

        self.transform = Identity
        if hasattr(config, "transforms"):
            transforms = []
            for transform in config.transforms:
                tklass = getattr(T, transform.name)
                tobj = tklass(**transform.params)
                transforms.append(tobj)
            self.transform = T.Compose(transforms)

        # cached videos
        self.start_stop_indices = {}
        start = 0
        for row in self.metadata.itertuples():
            start, stop = start, start + int(row.NumberOfFrames)
            self.start_stop_indices[row.FileName] = (start, stop)
            start = stop
        total_frames = start
        self.total_frames = total_frames

        first_data_point_path = self.metadata.iloc[0].VideoPath

        if ext == ".avi":
            load_data = load_avi
            data_ctype = ctypes.c_uint8
            N = total_frames
            _, C, H, W = load_data(first_data_point_path).shape
            self.need_normalize = True
        elif ext == ".pt":
            load_data = load_pt
            data_ctype = ctypes.c_float
            N = total_frames
            _, C, H, W = load_data(first_data_point_path).shape
            self.need_normalize = False

        data_ctype_in_bytes = ctypes.sizeof(data_ctype)
        bytes_in_gb = 1024**3
        print(
            f"Allocating {(N*C*H*W*data_ctype_in_bytes)/bytes_in_gb:.2f} GB in RAM. This could take some time."
        )
        self.video_cache = MPCacheVideo(
            shape=[N, C, H, W], load_func=load_data, e_ctype=data_ctype, verbose=False
        )

    def __len__(self):
        # the dataloader takes long time restart at every epoch
        # to avoid this, we return a multiple of the length of the dataset
        # which keeps the dataloader running for a longer time, before reseting.
        # this has no effect on the training process.
        return int(len(self.metadata))

    def __get_one_video__(self, idx):
        row = self.metadata.iloc[idx]
        start, stop = self.start_stop_indices[row.FileName]
        video = self.video_cache.load(start, stop, row.VideoPath)
        return video

    def __getitem__(self, idx, return_row=False):
        idx = idx % len(self.metadata)
        row = self.metadata.iloc[idx]
        output = {
            "filename": row["FileName"],
            "still": False,
        }

        if "image" in self.outputs or "video" in self.outputs:
            video = self.__get_one_video__(idx)
            if self.need_normalize:
                video = video / 128.0 - 1.0
            og_fps = row.FPS
            og_frame_count = len(video)

        if "image" in self.outputs:
            # Sample a random frame
            self.image_idx = np.random.randint(0, len(video))
            frame = video[self.image_idx]
            # frame = frame.float() / 128.0 - 1
            # frame = frame.permute(2, 0, 1) # H x W x C -> C x H x W
            output["image"] = self.transform(frame)

        if "video" in self.outputs:
            # Generate indices to resample
            # Generate a random starting point to cover all frames
            if self.target_fps == "original":
                target_fps = og_fps
            elif self.target_fps == "random":
                target_fps = np.random.randint(16, 120)
            elif self.target_fps == "half":
                target_fps = int(og_fps // 2)
            elif self.target_fps == "exponential":
                rnd, offset = np.random.randint(0, 100), 11
                target_fps = int(np.exp(rnd / offset) + offset)  # min: 12, max: ~8000
            else:
                target_fps = self.target_fps
            new_frame_count = np.floor(target_fps / og_fps * og_frame_count).astype(int)
            resample_indices_a = (
                np.linspace(0, og_frame_count - 1, new_frame_count, endpoint=False)
                .round()
                .astype(int)
            )
            start_idx = (
                np.random.choice(np.arange(0, resample_indices_a[1]))
                if len(resample_indices_a) > 1 and resample_indices_a[1] > 1
                else 0
            )
            resample_indices_a = resample_indices_a + start_idx

            # Sample a random chunk to cover the requested duration
            start_idx = (
                np.random.choice(
                    np.arange(0, len(resample_indices_a) - self.duration_frames)
                )
                if len(resample_indices_a) > self.duration_frames
                else 0
            )
            end_idx = start_idx + self.duration_frames
            resample_indices_b = resample_indices_a[start_idx:end_idx]
            resample_indices_b = resample_indices_b[
                resample_indices_b < len(video)
            ]  # remove indices that are out of bounds
            # video = reader.get_batch(resample_indices_b) # T x H x W x C, uint8
            video = video[resample_indices_b]
            self.resample_indices_b = resample_indices_b

            # Check if padding is needed
            p_index = len(video)
            if len(video) < self.duration_frames:
                padding_element = torch.zeros_like(video[0])
                padding = torch.stack(
                    [padding_element] * (self.duration_frames - len(video))
                )
                video = torch.cat((video, padding), dim=0)
                assert (
                    len(video) == self.duration_frames
                ), f"Video length is {len(video)} but should be {self.duration_frames}"

            # video = video.float() / 128.0 - 1 # normalize to [-1, 1]
            # video = video.permute(3, 0, 1, 2) # T x H x W x C -> C x T x H x W
            video = video.permute(1, 0, 2, 3)  # T x C x H x W -> C x T x H x W
            output["video"] = self.transform(video)
            output["fps"] = target_fps
            output["padding"] = p_index
            if self.target_fps == "exponential":
                resample_indices_b[1:] = (
                    resample_indices_b[1:] - resample_indices_b[:-1] >= 1
                ).cumsum(0)
                resample_indices_b[0] = 0
                output["indices"] = np.concatenate(
                    (
                        resample_indices_b,
                        np.repeat(
                            resample_indices_b[-1],
                            self.duration_frames - len(resample_indices_b),
                        ),
                    )
                )

        if "lvef" in self.outputs:
            lvef = row["EF"] / 100.0  # normalize to [0, 1]
            output["lvef"] = torch.tensor(lvef, dtype=torch.float32)

        if "view" in self.outputs:
            view = torch.tensor(self.view_label_index, dtype=torch.int64)
            output["view"] = view

        if "split" in self.outputs:
            output["split"] = row["Split"]

        if self.output_key is not None:
            output = output[self.output_key]

        if return_row:
            return output, row

        return output


class Dynamic(CachedEchoNet):
    def __init__(self, config, split=["TRAIN", "VAL", "TEST"]) -> None:
        super().__init__(config, split, datafolder="Videos", ext=".avi")


class Pediatric(CachedEchoNet):
    def __init__(self, config, split=["TRAIN", "VAL", "TEST"]) -> None:
        super().__init__(config, split, datafolder="Videos", ext=".avi")


class Latent(CachedEchoNet):
    def __init__(self, config, split=["TRAIN", "VAL", "TEST"]) -> None:
        super().__init__(config, split, datafolder="Latents", ext=".pt")


class LatentSeg(CachedEchoNet):
    def __init__(self, config, split=["TRAIN", "VAL", "TEST"]) -> None:
        super().__init__(config, split, datafolder="Latents", ext=".pt")

        self.segmentation_root = config.segmentation_root

        # create cache for the segmentation masks

        load_data = load_seg
        data_ctype = ctypes.c_uint8
        N, H, W = (
            self.total_frames,
            config.target_resolution,
            config.target_resolution,
        )

        data_ctype_in_bytes = ctypes.sizeof(data_ctype)
        bytes_in_gb = 1024**3

        if self.segmentation_root == "no_seg":
            infinite_tensor = InfiniteTensor(torch.zeros((H, W), dtype=torch.uint8))
            self.__get_one_segmentation__ = lambda idx: infinite_tensor
        else:
            print(
                f"Allocating {(N*H*W*data_ctype_in_bytes)/bytes_in_gb:.2f} GB in RAM (segmentations). This could take some time."
            )
            self.segmentation_cache = MPCacheVideo(
                shape=[N, H, W], load_func=load_data, e_ctype=data_ctype, verbose=False
            )

    def __get_one_segmentation__(self, idx):
        row = self.metadata.iloc[idx]
        start, stop = self.start_stop_indices[row.FileName]
        seg = self.segmentation_cache.load(
            start, stop, os.path.join(self.segmentation_root, row.FileName + ".pt")
        )
        return seg

    def __getitem__(self, idx, return_row=False):
        idx = idx % len(self.metadata)
        if return_row:
            output, row = super().__getitem__(idx, return_row)
        else:
            output = super().__getitem__(idx, return_row)

        seg = self.__get_one_segmentation__(idx)  # T H W

        if "image" in self.outputs:
            output["image_segmentation"] = seg[self.image_idx]  # H W

        if "video" in self.outputs:
            output["video_segmentation"] = seg[self.resample_indices_b]  # T H W

        if return_row:
            return output, row
        return output


class ContrastivePair(Dataset):
    def __init__(
        self, root, folder="Latents", extension="pt", split=["TRAIN", "VAL", "TEST"]
    ):
        self.root = root
        self.folder = folder
        self.extension = extension

        assert self.folder in ["Latents", "Videos"]
        assert self.extension in ["pt", "avi"]

        self.video_path = os.path.join(self.root, self.folder)
        self.all_videos = glob(os.path.join(self.video_path, f"*.{self.extension}"))

        split = [split] if isinstance(split, str) else split

        self.metadata = pd.read_csv(os.path.join(self.root, "FileList.csv"))
        self.metadata = self.metadata[self.metadata["Split"].isin(split)]
        self.metadata.reset_index(inplace=True, drop=True)

        # Mapping from FileName to VideoPath
        self.metadata["VideoPath"] = self.metadata["FileName"].apply(
            lambda x: (
                os.path.join(self.video_path, x)
                if x.endswith(f".{self.extension}")
                else os.path.join(self.video_path, f"{x}.{self.extension}")
            )
        )
        self.metadata = self.metadata[self.metadata["VideoPath"].apply(os.path.exists)]
        self.metadata.reset_index(inplace=True, drop=True)

        # Map from index to FileName
        self.index_to_filename = self.metadata["FileName"].tolist()

        # Build start_stop_indices for caching
        self.start_stop_indices = {}
        start = 0
        for row in self.metadata.itertuples():
            video_length = self._get_video_length(row.VideoPath)
            start, stop = start, start + video_length
            self.start_stop_indices[row.FileName] = (start, stop)
            start = stop
        total_frames = start

        if self.extension == "pt":
            load_data = load_pt  # Assume load_pt is defined
            data_ctype = ctypes.c_float
            first_item = load_pt(self.all_videos[0])
            _, C, H, W = first_item.size()
        elif self.extension == "avi":
            load_data = load_avi  # Assume load_avi is defined
            data_ctype = ctypes.c_uint8
            first_item = load_avi(self.all_videos[0])
            _, H, W, C = first_item.shape

        N = total_frames
        data_ctype_in_bytes = ctypes.sizeof(data_ctype)
        bytes_in_gb = 1024**3
        print(
            f"Allocating {(N * C * H * W * data_ctype_in_bytes) / bytes_in_gb:.2f} GB in RAM. This could take some time."
        )
        self.video_cache = MPCacheVideo(
            shape=[N, C, H, W], load_func=load_data, e_ctype=data_ctype, verbose=False
        )

    def __len__(self):
        return len(self.index_to_filename)

    def __getitem__(self, idx):
        videoA_filename = self.index_to_filename[idx]
        videoA = self._get_one_video(videoA_filename)
        frameA_idx = torch.randint(0, videoA.size(0), (1,)).item()
        frameA = videoA[frameA_idx, :, :, :]  # C x H x W

        if torch.rand(1) < 0.5:
            frameB_idx = torch.randint(0, videoA.size(0), (1,)).item()
            frameB = videoA[frameB_idx, :, :, :]
            label = torch.tensor(1.0)
        else:
            idx_B = torch.randint(0, len(self.index_to_filename), (1,)).item()
            videoB_filename = self.index_to_filename[idx_B]
            videoB = self._get_one_video(videoB_filename)
            frameB_idx = torch.randint(0, videoB.size(0), (1,)).item()
            frameB = videoB[frameB_idx, :, :, :]
            label = torch.tensor(0.0)

        return frameA, frameB, label

    def _get_one_video(self, filename):
        start, stop = self.start_stop_indices[filename]
        row = self.metadata[self.metadata["FileName"] == filename].iloc[0]
        video = self.video_cache.load(start, stop, row.VideoPath)
        return video  # T x C x H x W

    def _get_video_length(self, path):
        if self.extension == "pt":
            video = torch.load(path)
            return video.size(0)
        else:
            video = load_avi(path)
            return video.size(0)


def instantiate_cached_dataset(configs, split=["TRAIN", "VAL", "TEST"]):
    # Check if number of frames and resolution are the same for all datasets
    target_nframes = None
    target_resolution = None
    reference_name = None
    for dataset_config in configs:
        if dataset_config.get("active", True):
            if reference_name is None:
                reference_name = dataset_config.name
            if target_nframes is None:
                target_nframes = dataset_config.params.target_nframes
            else:
                newd = dataset_config.params.target_nframes
                assert (
                    newd == target_nframes
                ), f"All datasets must ouput the same number of frames, got {reference_name}: {target_nframes} frames and {dataset_config.name}: {newd} frames."
            if target_resolution is None:
                target_resolution = dataset_config.params.target_resolution
            else:
                assert (
                    dataset_config.params.target_resolution == target_resolution
                ), f"All datasets must have the same target_resolution, got {reference_name}: {target_resolution} and {dataset_config.name}: {dataset_config.params.target_resolution}."

    datasets = []
    for dataset_config in configs:
        if dataset_config.get("active", True):
            datasets.append(
                globals()[dataset_config.name](dataset_config.params, split=split)
            )

    if len(datasets) == 1:
        return datasets[0]
    else:
        return ConcatDataset(datasets)
