import ctypes
import hashlib
import os
import time
from multiprocessing import shared_memory
from typing import Any, Callable, Dict, List, Optional

import cv2  # Ensure OpenCV is installed
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms as T

from CTFlow.common.cached_datasets import (
    Identity,
    InfiniteTensor,
    load_avi,
    load_pt,
    load_seg,
)
from CTFlow.common.datasets import VIEWS

logger = get_logger(__name__, log_level="INFO")


def get_load_seg_function(factor):
    def load_seg(path):
        seg = torch.load(path)  # B H*8 W*8
        seg = torch.nn.functional.max_pool2d(seg.long(), kernel_size=factor)
        return seg

    return load_seg


def generate_shm_name(dataset_name: str, root: str, suffix: str) -> str:
    """
    Generates a unique shared memory name based on dataset name and root path.

    Args:
        dataset_name (str): Name of the dataset.
        root (str): Root directory of the dataset.
        suffix (str): Suffix to differentiate between data and cache.

    Returns:
        str: Unique shared memory name.
    """

    gpu_str = os.environ.get("CUDA_VISIBLE_DEVICES") or ""
    gpu_str = "_" + gpu_str.replace(",", "") if gpu_str else ""

    unique_str = f"{dataset_name}_{os.path.abspath(root)}_{suffix}_{gpu_str}"
    unique_hash = hashlib.md5(unique_str.encode()).hexdigest()
    return f"shm_{unique_hash}"


class MPCacheVideoSM:
    def __init__(
        self,
        shape: List[int],
        load_func: Callable[[str], np.ndarray],
        dataset_name: str,
        root: str,
        e_ctype=ctypes.c_float,
        verbose: bool = False,
        is_main_process: bool = True,
    ):
        """
        Initializes the MPCacheVideoSM instance with named shared memory.

        Args:
            shape (List[int]): Shape of the video data (N, C, H, W).
            load_func (Callable[[str], np.ndarray]): Function to load video data.
            dataset_name (str): Name of the dataset.
            root (str): Root directory of the dataset.
            e_ctype (ctypes type, optional): CType of the elements (e.g., ctypes.c_float).
            verbose (bool, optional): If True, logs detailed status messages.
            is_main_process (bool, optional): Indicates if the current process is the main process.
        """
        self.shape = shape
        self.load_func = load_func
        self.dataset_name = dataset_name
        self.root = root
        self.verbose = verbose
        self.e_ctype = e_ctype
        self.dtype = np.dtype(e_ctype)
        self.size = int(np.prod(shape)) * ctypes.sizeof(e_ctype)
        self.total_memory_gb = self.size / (1024**3)

        logger.info(
            f"Initializing shared memory for dataset '{dataset_name}' "
            f"with size {self.total_memory_gb:.2f} GB in RAM."
        )

        # Generate deterministic shared memory names
        self.shm_name = generate_shm_name(dataset_name, root, "video")
        self.cache_shm_name = generate_shm_name(dataset_name, root, "video_cache")

        # Initialize shared memory segments
        self.initialize_shared_memory(is_main_process)

    def initialize_shared_memory(self, is_main_process: bool):
        """
        Initializes shared memory segments based on ownership.
        The main process creates the shared memory and initializes it.
        Other processes connect to the existing shared memory.

        Args:
            is_main_process (bool): Indicates if the current process is the main process.
        """
        self.is_owner = is_main_process

        if self.is_owner:
            logger.info(
                f"Main Process: Allocating {self.total_memory_gb:.2f} GB of RAM for shared memory '{self.shm_name}'.",
            )

            # Create shared memory for video data
            try:
                self.shm = shared_memory.SharedMemory(
                    name=self.shm_name, create=True, size=self.size
                )
            except FileExistsError as e:
                # logger.error(
                #     f"Shared memory '{self.shm_name}' already exists. Possible race condition.\n{e}"
                # )
                # raise RuntimeError(f"Shared memory '{self.shm_name}' already exists.")
                # if memory already exists, delete it and recreate
                logger.warning(
                    f"Main Process: Shared memory '{self.shm_name}' already exists. Deleting and recreating."
                )
                self.shm = shared_memory.SharedMemory(name=self.shm_name)
                # self.shm = shared_memory.SharedMemory(name=self.shm_name)

            self.shared_array = np.ndarray(
                self.shape, dtype=self.dtype, buffer=self.shm.buf
            )
            self.shared_tensor = torch.from_numpy(self.shared_array)

            # Create shared memory for cache status
            cache_size = self.shape[0]  # Number of frames
            cache_shm_size = cache_size * ctypes.sizeof(ctypes.c_bool)
            try:
                self.cache_shm = shared_memory.SharedMemory(
                    name=self.cache_shm_name, create=True, size=cache_shm_size
                )
            except FileExistsError:
                # logger.error(
                #     f"Cache shared memory '{self.cache_shm_name}' already exists. Possible race condition."
                # )
                # raise RuntimeError(
                #     f"Cache shared memory '{self.cache_shm_name}' already exists."
                # )
                # if memory already exists, delete it and recreate
                logger.warning(
                    f"Main Process: Cache shared memory '{self.cache_shm_name}' already exists. Deleting and recreating."
                )
                self.cache_shm = shared_memory.SharedMemory(name=self.cache_shm_name)

            self.idx_is_cached = np.ndarray(
                (cache_size,), dtype=bool, buffer=self.cache_shm.buf
            )
            self.idx_is_cached[:] = False  # Initialize cache status

            logger.info(
                f"Main Process: Shared memory '{self.shm_name}' initialized.",
            )
        else:
            logger.debug(
                f"Process {os.getpid()}: Waiting to connect to shared memory '{self.shm_name}'.",
                main_process_only=False,
            )

            # Non-owner processes wait until the main process initializes the shared memory
            while True:
                try:
                    self.shm = shared_memory.SharedMemory(name=self.shm_name)
                    break
                except FileNotFoundError:
                    time.sleep(0.1)  # Wait and retry

            self.shared_array = np.ndarray(
                self.shape, dtype=self.dtype, buffer=self.shm.buf
            )
            self.shared_tensor = torch.from_numpy(self.shared_array)

            # Connect to cache shared memory
            while True:
                try:
                    self.cache_shm = shared_memory.SharedMemory(
                        name=self.cache_shm_name
                    )
                    break
                except FileNotFoundError:
                    time.sleep(0.1)  # Wait and retry

            self.idx_is_cached = np.ndarray(
                (self.shape[0],), dtype=bool, buffer=self.cache_shm.buf
            )

            logger.debug(
                f"Process {os.getpid()}: Connected to shared memory '{self.shm_name}'.",
                main_process_only=False,
            )

    def load(self, start_idx: int, end_idx: int, video_path: str) -> torch.Tensor:
        """
        Loads video data into shared memory if not already cached.

        Args:
            start_idx (int): Start index of the video frames.
            end_idx (int): End index of the video frames.
            video_path (str): Path to the video file.

        Returns:
            torch.Tensor: Loaded video frames of shape (T, C, H, W).
        """
        if np.all(self.idx_is_cached[start_idx:end_idx]):
            if self.verbose:
                logger.info(f"Cache hit for frames {start_idx}:{end_idx}")
            return self.shared_tensor[start_idx:end_idx]
        else:
            if self.verbose:
                logger.info(
                    f"Cache miss for frames {start_idx}:{end_idx}. Loading from {video_path}"
                )

            # Load video data
            data = self.load_func(
                video_path
            )  # Expected to return a NumPy array of shape (T, C, H, W)

            # Validate the loaded data shape
            expected_shape = (end_idx - start_idx, *self.shape[1:])
            if data.shape != expected_shape:
                logger.error(
                    f"Loaded data shape {data.shape} does not match expected shape {expected_shape} for frames {start_idx}:{end_idx}."
                )
                raise ValueError(
                    f"Loaded data shape {data.shape} does not match expected shape {expected_shape} for frames {start_idx}:{end_idx}."
                )

            # Store data in shared memory
            self.shared_array[start_idx:end_idx] = data

            # Update cache status
            self.idx_is_cached[start_idx:end_idx] = True

            return self.shared_tensor[start_idx:end_idx]

    def close(self):
        """
        Closes the shared memory segments.
        """
        self.shm.close()
        self.cache_shm.close()
        logger.info(f"Shared memory '{self.shm_name}' closed.")

    def unlink(self):
        """
        Unlinks (deletes) the shared memory segments. Should be called by the owner.
        """
        if self.is_owner:
            self.shm.unlink()
            self.cache_shm.unlink()
            logger.info(
                f"Main Process: Unlinked shared memory '{self.shm_name}' and cache '{self.cache_shm_name}'."
            )

    def cache_status(self):
        """
        Prints the current cache status.
        """
        cached = np.sum(self.idx_is_cached)
        total = self.idx_is_cached.size
        logger.info(f"Cache status: {cached}/{total} frames cached.")


class CachedEchoNetSM(Dataset):
    def __init__(
        self,
        config: Dict[str, Any],
        split: List[str] = ["TRAIN", "VAL", "TEST"],
        datafolder: str = "Videos",
        ext: str = ".avi",
        is_main_process: bool = True,
    ) -> None:
        """
        Initializes the CachedEchoNetSM dataset with shared memory.

        Args:
            config (Dict[str, Any]): Configuration object.
            split (List[str], optional): Dataset splits to include.
            datafolder (str, optional): Folder containing video data.
            ext (str, optional): File extension of video data.
            is_main_process (bool, optional): Indicates if the current process is the main process.
        """
        super().__init__()
        self.is_main_process = is_main_process
        self.verbose = config.get("verbose", False)
        self.config = config
        if isinstance(split, str):
            split = [split]
        assert all(
            s in ["TRAIN", "VAL", "TEST"] for s in split
        ), "Splits must be a list of TRAIN, VAL, TEST"

        self.target_fps = config.get("target_fps", "original")
        self.target_nframes = config.get("target_nframes", 16)
        self.duration_frames = self.target_nframes  # As per user code
        self.outputs = config.get("outputs", ["image", "video"])
        if isinstance(self.outputs, str):
            self.outputs = [self.outputs]
        assert all(
            o in ["video", "image", "lvef", "view"] for o in self.outputs
        ), "Outputs must be a list of video, image, lvef, view"

        self.output_key = config.get("output_key", None)

        self.view_label = config.get("view_label", "A4C")
        assert self.view_label in VIEWS, f"View label must be one of {VIEWS}"
        self.view_label_index = VIEWS.index(self.view_label)

        # LOAD DATA
        self.root = config.get("root", "")
        assert self.root, "No root folder specified in config"
        assert os.path.exists(
            os.path.join(self.root, datafolder)
        ), f"Data folder {os.path.join(self.root, datafolder)} does not exist"
        metadata_path = os.path.join(self.root, "FileList.csv")
        assert os.path.exists(
            metadata_path
        ), f"FileList.csv does not exist in {self.root}"

        self.metadata = pd.read_csv(metadata_path)
        self.metadata = self.metadata[
            self.metadata["Split"].isin(split)
        ]  # filter by split
        self.len_before_filter = len(self.metadata)
        # add duration column
        self.metadata["Duration"] = (
            self.metadata["NumberOfFrames"] / self.metadata["FPS"]
        )  # won't work for pediatrics
        # filter by duration
        if self.target_fps is not None and isinstance(self.target_fps, (int, float)):
            self.duration_seconds = self.duration_frames / self.target_fps
            self.metadata = self.metadata[
                self.metadata["Duration"] >= self.duration_seconds
            ]

        # check if videos are reachable
        self.metadata["VideoPath"] = self.metadata["FileName"].apply(
            lambda x: (
                os.path.join(self.root, datafolder, x)
                if x.endswith(ext)
                else os.path.join(self.root, datafolder, os.path.splitext(x)[0] + ext)
            )
        )
        self.metadata["VideoExists"] = self.metadata["VideoPath"].apply(
            lambda x: os.path.exists(x)
        )
        self.metadata = self.metadata[self.metadata["VideoExists"]]
        self.metadata.reset_index(inplace=True, drop=True)
        if len(self.metadata) == 0:
            raise ValueError(
                f"No data found in folder {os.path.join(self.root, datafolder)} after filtering."
            )

        # Define transforms
        self.transform = Identity

        # Build start_stop_indices
        self.start_stop_indices = {}
        start = 0
        for row in self.metadata.itertuples():
            stop = start + int(row.NumberOfFrames)
            self.start_stop_indices[row.FileName] = (start, stop)
            start = stop
        total_frames = start
        self.total_frames = total_frames

        if self.is_main_process and config.get("verbose", False):
            print(f"Total frames across all videos: {self.total_frames}")

        # Determine data loading function and dtype based on file extension
        first_data_point_path = self.metadata.iloc[0].VideoPath

        if ext == ".avi":
            load_data = load_avi
            data_ctype = ctypes.c_uint8
            data_sample = load_data(first_data_point_path)
            C, H, W = data_sample.shape[0], data_sample.shape[1], data_sample.shape[2]
            self.need_normalize = True
        elif ext == ".pt":
            load_data = load_pt
            data_ctype = ctypes.c_float
            data_sample = load_data(first_data_point_path)
            C, H, W = data_sample.shape[1], data_sample.shape[2], data_sample.shape[3]
            self.need_normalize = False
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        # Define the shape for shared memory: (total_frames, C, H, W)
        self.shape = [self.total_frames, C, H, W]

        # Initialize shared memory cache using MPCacheVideoSM
        self.unique_name = self.root.split("/")[-1]
        self.video_cache = MPCacheVideoSM(
            shape=self.shape,
            load_func=load_data,
            dataset_name=self.unique_name + "_" + "_".join(split),
            root=self.root,
            e_ctype=data_ctype,
            verbose=config.get("verbose", False),
            is_main_process=is_main_process,
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def __get_one_video__(self, idx: int) -> torch.Tensor:
        """
        Retrieves one video from the cache.

        Args:
            idx (int): Index of the video.

        Returns:
            torch.Tensor: Video frames as a tensor of shape (T, C, H, W).
        """
        row = self.metadata.iloc[idx]
        start, stop = self.start_stop_indices[row.FileName]
        video = self.video_cache.load(start, stop, row.VideoPath)  # Shape: (T, C, H, W)
        return video.clone()  # clone to avoid modifying the shared memory

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
            output["padding"] = torch.tensor(p_index)
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


class LatentSM(CachedEchoNetSM):
    def __init__(
        self,
        config: Dict[str, Any],
        split: List[str] = ["TRAIN", "VAL", "TEST"],
        is_main_process: bool = True,
    ) -> None:
        """
        Initializes the LatentSM dataset with shared memory.

        Args:
            config (Dict[str, Any]): Configuration object.
            split (List[str], optional): Dataset splits to include.
            is_main_process (bool, optional): Indicates if the current process is the main process.
        """
        super().__init__(
            config=config,
            split=split,
            datafolder="Latents",
            ext=".pt",
            is_main_process=is_main_process,
        )


class LatentSegSM(CachedEchoNetSM):
    def __init__(
        self,
        config: Dict[str, Any],
        split: List[str] = ["TRAIN", "VAL", "TEST"],
        is_main_process: bool = True,
    ) -> None:
        """
        Initializes the LatentSegSM dataset with shared memory, including segmentation masks.

        Args:
            config (Dict[str, Any]): Configuration object.
            split (List[str], optional): Dataset splits to include.
            is_main_process (bool, optional): Indicates if the current process is the main process.
        """
        super().__init__(
            config=config,
            split=split,
            datafolder="Latents",
            ext=".pt",
            is_main_process=is_main_process,
        )

        self.segmentation_root = config.get("segmentation_root", "no_seg")

        # Create cache for the segmentation masks
        vae_factor = 112 // self.shape[2]
        load_data = get_load_seg_function(vae_factor)
        data_ctype = ctypes.c_uint8
        N, H, W = (
            self.total_frames,
            self.shape[2],
            self.shape[3],
        )

        data_ctype_in_bytes = ctypes.sizeof(data_ctype)
        bytes_in_gb = 1024**3

        if self.segmentation_root == "no_seg":
            infinite_tensor = InfiniteTensor(torch.zeros((H, W), dtype=torch.uint8))
            self.__get_one_segmentation__ = lambda idx: infinite_tensor
        else:
            if self.is_main_process and self.verbose:
                print(
                    f"Allocating {(N*H*W*data_ctype_in_bytes)/bytes_in_gb:.2f} GB in RAM (segmentations). This could take some time."
                )
            self.segmentation_cache = MPCacheVideoSM(
                shape=[N, H, W],
                load_func=load_data,
                dataset_name=self.unique_name + "_seg_" + "_".join(split),
                root=self.segmentation_root,
                e_ctype=data_ctype,
                verbose=self.verbose,
                is_main_process=is_main_process,
            )

    def __get_one_segmentation__(self, idx):
        row = self.metadata.iloc[idx]
        start, stop = self.start_stop_indices[row.FileName]
        seg = self.segmentation_cache.load(
            start, stop, os.path.join(self.segmentation_root, row.FileName + ".pt")
        )
        return seg.clone()  # clone to avoid modifying the shared memory

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


def instantiate_cached_dataset_sm(
    configs: List[Dict[str, Any]],
    split: List[str] = ["TRAIN", "VAL", "TEST"],
    is_main_process: bool = True,
) -> Dataset:
    """
    Instantiates cached datasets with shared memory management.

    Args:
        configs (List[Dict[str, Any]]): List of dataset configurations.
        split (List[str], optional): Dataset splits to include.
        is_main_process (bool, optional): Indicates if the current process is the main process.

    Returns:
        torch.utils.data.Dataset: Combined dataset.
    """
    datasets = []
    for dataset_config in configs:
        if not dataset_config.get("active", True):
            continue

        dataset_name = dataset_config["name"]  # e.g., "CachedEchoNet"
        params = dataset_config["params"]

        # Dynamically determine the shared memory dataset class
        shared_dataset_class = f"{dataset_name}SM"
        if shared_dataset_class not in globals():
            raise ValueError(
                f"Shared memory dataset class '{shared_dataset_class}' not found."
            )

        # Instantiate the shared memory dataset
        dataset_class = globals()[shared_dataset_class]
        dataset = dataset_class(
            config=params,
            split=split,
            is_main_process=is_main_process,
        )
        datasets.append(dataset)

    if len(datasets) == 1:
        return datasets[0]
    else:
        return ConcatDataset(datasets)


def custom_collate_fn(batch):
    """
    Custom collate function to handle dictionary-based batches.

    Args:
        batch (List[Dict[str, Any]]): List of dataset items.

    Returns:
        Dict[str, Any]: Batched dataset items.
    """
    batched = {}
    # Assume all items have the same keys
    keys = batch[0].keys()
    for key in keys:
        # Extract all values for this key
        items = [item[key] for item in batch]
        if isinstance(items[0], torch.Tensor):
            batched[key] = torch.stack(items, dim=0)
        elif isinstance(items[0], (int, float, str)):
            batched[key] = items  # Handle other data types as needed
        elif isinstance(items[0], list):
            batched[key] = items  # Handle lists if necessary
        else:
            # Add more conditions as needed
            batched[key] = items
    return batched
