import collections
import os
import sys
from glob import glob

import cv2
import decord
import numpy as np
import pandas as pd
import skimage.draw
import torch
import torchvision
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler
from torchvision import transforms as T
from torchvision.transforms.functional import center_crop

from CTFlow.common import loadvideo

decord.bridge.set_bridge("torch")

VIEWS = ["A4C", "PSAX", "PLAX"]


# support video and image + additional info (lvef, view, etc)
class Dynamic(Dataset):
    def __init__(
        self, config, split=["TRAIN", "VAL", "TEST"], datafolder="Videos", ext=".avi"
    ) -> None:
        super().__init__()
        # the config here is only the config for this dataset, ie config.dataset.dynamic

        if type(split) == str:
            split = [split]
        assert [
            s in ["TRAIN", "VAL", "TEST"] for s in split
        ], "Splits must be a list of TRAIN, VAL, TEST"

        # assert type(config.target_fps) == int or config.target_fps in ["original", "random", "exponential"], "target_fps must be an integer, 'original', 'random' or 'exponential'"
        self.target_fps = config.target_fps
        # self.duration_seconds = config.target_duration
        self.resolution = config.target_resolution
        self.outputs = config.outputs
        if type(self.outputs) == str:
            self.outputs = [self.outputs]
        assert [
            o in ["video", "image", "lvef"] for o in self.outputs
        ], "Outputs must be a list of video, image, lvef"

        # self.duration_frames = int(self.target_fps * self.duration_seconds)
        self.duration_frames = config.target_nframes
        self.duration_seconds = (
            self.duration_frames / self.target_fps
            if type(self.target_fps) == int
            else None
        )

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

        self.transform = lambda x: x
        if hasattr(config, "transforms"):
            transforms = []
            for transform in config.transforms:
                tklass = getattr(T, transform.name)
                tobj = tklass(**transform.params)
                transforms.append(tobj)
            self.transform = T.Compose(transforms)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx, return_row=False):
        row = self.metadata.iloc[idx]
        output = {
            "filename": row["FileName"],
            "still": False,
        }

        if "image" in self.outputs or "video" in self.outputs:
            reader = decord.VideoReader(
                row["VideoPath"],
                ctx=decord.cpu(),
                width=self.resolution,
                height=self.resolution,
            )
            og_fps = reader.get_avg_fps()
            og_frame_count = len(reader)

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
                resample_indices_b < len(reader)
            ]  # remove indices that are out of bounds
            video = reader.get_batch(resample_indices_b)  # T x H x W x C, uint8

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

            video = video.float() / 128.0 - 1  # normalize to [-1, 1]
            video = video.permute(3, 0, 1, 2)  # T x H x W x C -> C x T x H x W
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

        if "image" in self.outputs:
            image = reader.get_batch(np.random.randint(0, og_frame_count, 1))[
                0
            ]  # H x W x C, uint8
            image = image.float() / 128.0 - 1
            image = image.permute(2, 0, 1)  # H x W x C -> C x H x W
            output["image"] = self.transform(image)

        if return_row:
            return output, row

        return output


class Pediatric(Dynamic):
    def __init__(self, config, split=["TRAIN", "VAL", "TEST"]) -> None:
        super().__init__(config, split)

        # View
        self.view = config.get("views", "ALL")  # A4C, PSAX, ALL
        if self.view == "ALL":
            pass
        else:
            self.metadata = self.metadata[self.metadata["View"] == self.view]
            self.metadata.reset_index(inplace=True, drop=True)
            if len(self.metadata) == 0:
                raise ValueError(f"No videos found for view {self.view}")

    def __getitem__(self, idx):
        output, row = super().__getitem__(idx, return_row=True)
        if "view" in self.outputs:
            output["view"] = row["View"]

        return output


class Latent(Dynamic):
    def __init__(self, config, split=["TRAIN", "VAL", "TEST"]) -> None:
        self.config = config

        super().__init__(config, split, datafolder="Latents", ext=".pt")

        self.view = config.get("views", "ALL")  # A4C, PSAX, ALL
        self.view_label = config.get("view_label", "A4C").upper()
        assert self.view_label in VIEWS, f"View label must be one of {VIEWS}"
        self.view_label_index = VIEWS.index(self.view_label)
        if self.view == "ALL":
            pass
        else:
            self.metadata = self.metadata[self.metadata["View"] == self.view]
            self.metadata.reset_index(inplace=True, drop=True)
            if len(self.metadata) == 0:
                raise ValueError(f"No videos found for view {self.view}")

    def __getitem__(self, idx, return_row=False):
        row = self.metadata.iloc[idx]
        output = {
            "filename": row["FileName"],
        }

        if "image" in self.outputs or "video" in self.outputs:
            latent_file = row["VideoPath"]
            latent_video_tensor = torch.load(latent_file)  # T x C x H x W
            og_fps = row["FPS"]
            og_frame_count = len(latent_video_tensor)

        if "video" in self.outputs:
            if self.target_fps == "original":
                target_fps = og_fps
            elif self.target_fps == "random":
                target_fps = np.random.randint(8, 50)
            else:
                target_fps = self.target_fps

            new_frame_count = np.floor(target_fps / og_fps * og_frame_count).astype(int)
            resample_indices = (
                np.linspace(0, og_frame_count, new_frame_count, endpoint=False)
                .round()
                .astype(int)
            )
            start_idx = (
                np.random.choice(np.arange(0, resample_indices[1]))
                if len(resample_indices) > 1 and resample_indices[1] > 1
                else 0
            )
            resample_indices = resample_indices + start_idx

            # Sample a random chunk to cover the requested duration
            start_idx = (
                np.random.choice(
                    np.arange(0, len(resample_indices) - self.duration_frames)
                )
                if len(resample_indices) > self.duration_frames
                else 0
            )
            end_idx = start_idx + self.duration_frames
            resample_indices = resample_indices[start_idx:end_idx]
            resample_indices = resample_indices[
                resample_indices < og_frame_count
            ]  # remove indices that are out of bounds

            latent_video_sample = latent_video_tensor[resample_indices]

            # Check if padding is needed
            p_index = len(latent_video_sample)
            if len(latent_video_sample) < self.duration_frames:
                padding_element = torch.zeros_like(latent_video_sample[0])
                padding = torch.stack(
                    [padding_element]
                    * (self.duration_frames - len(latent_video_sample))
                )
                latent_video_sample = torch.cat((latent_video_sample, padding), dim=0)
                assert (
                    len(latent_video_sample) == self.duration_frames
                ), f"Video length is {len(latent_video_sample)} but should be {self.duration_frames}"

            latent_video_sample = latent_video_sample.permute(
                1, 0, 2, 3
            )  # T x C x H x W -> C x T x H x W
            output["video"] = self.transform(latent_video_sample)
            output["fps"] = target_fps
            output["padding"] = p_index

        if "lvef" in self.outputs:
            lvef = row["EF"] / 100.0
            output["lvef"] = torch.tensor(lvef, dtype=torch.float32)

        if "image" in self.outputs:
            latent_image_tensor = latent_video_tensor[
                np.random.randint(0, og_frame_count, 1)
            ][
                0
            ]  # C x H x W
            output["image"] = self.transform(latent_image_tensor)

        if "view" in self.outputs:
            output["view"] = self.view_label_index

        if return_row:
            return output, row

        return output


class RandomVideo(Dataset):
    def __init__(self, config, split=["TRAIN", "VAL", "TEST"]) -> None:
        super().__init__()

        self.config = config
        self.root = config.root

        self.target_nframes = config.target_nframes
        self.target_resolution = config.target_resolution

        self.outputs = config.outputs
        assert len(self.outputs) > 0, "Outputs must not be empty"
        assert all(
            [o in ["video", "image"] for o in self.outputs]
        ), "Outputs can only be video or image (or both) for RandomVideo"

        assert os.path.exists(self.root), f"Root folder {self.root} does not exist"
        assert os.path.isdir(self.root), f"Root folder {self.root} is not a directory"
        self.all_frames = os.listdir(self.root)

        assert len(self.all_frames) > 0, f"No frames found in {self.root}"

        self.still_image_p = config.get(
            "still_image_p", 0
        )  # probability of returning a still image instead of a video

    def __len__(self):
        return len(self.all_frames)

    def __getitem__(self, idx):

        output = {
            "filename": "Fake",
        }

        if "image" in self.outputs:
            path = os.path.join(self.root, self.all_frames[idx])
            image = Image.open(path)  # H x W x C, uint8
            image = np.array(image)  # H x W x C, uint8
            image = torch.from_numpy(image)
            image = image.permute(2, 0, 1).float()  # C x H x W
            image = image / 128.0 - 1  # [-1, 1]
            output["image"] = image

        if "video" in self.outputs:
            if (
                self.still_image_p
                > torch.rand(
                    1,
                ).item()
            ):
                random_indices = np.random.randint(0, len(self.all_frames), 1)
                path = os.path.join(self.root, self.all_frames[random_indices[0]])
                image = Image.open(path)  # H x W x C, uint8
                image = np.array(image)  # H x W x C, uint8
                image = torch.from_numpy(image)
                image = image.permute(2, 0, 1).float()  # C x H x W
                image = image / 128.0 - 1
                image = image[:, None, :, :].repeat(1, self.target_nframes, 1, 1)
                output["video"] = image
                output["still"] = True
            else:
                random_indices = np.random.randint(
                    0, len(self.all_frames), self.target_nframes
                )
                paths = [
                    os.path.join(self.root, self.all_frames[ridx])
                    for ridx in random_indices
                ]
                images = [Image.open(path) for path in paths]
                images = [np.array(image) for image in images]
                images = np.stack(images, axis=0)  # T x H x W x C
                images = torch.from_numpy(images)  # T x H x W x C
                images = images.permute(3, 0, 1, 2).float()  # C x T x H x W
                images = images / 128.0 - 1  # [-1, 1]
                output["video"] = images
                output["still"] = False
            output["fps"] = 0
            output["padding"] = 0

        return output


class FrameFolder(Dataset):
    """config:
    - name: FrameFolder
      active: true
      params:
        video_folder: path/to/video_folders
        meta_path: path/to/FileList.csv
        outputs: ['video', 'image', 'lvef']
    """

    def __init__(self, config, split=["TRAIN", "VAL", "TEST"]) -> None:
        super().__init__()

        self.config = config
        self.video_folder = config.video_folder
        self.meta_path = config.meta_path

        self.target_nframes = config.target_nframes
        self.target_resolution = config.target_resolution

        self.metadata = pd.read_csv(self.meta_path)
        self.metadata = self.metadata[
            self.metadata["Split"].isin(split)
        ]  # filter by split

        # check if videos are reachable
        self.metadata["VideoPath"] = self.metadata["FileName"].apply(
            lambda x: os.path.join(config.video_folder, x.split(".")[0])
        )
        self.metadata["VideoExists"] = self.metadata["VideoPath"].apply(
            lambda x: (os.path.isdir(x) and len(os.listdir(x)) > 0)
        )
        self.metadata = self.metadata[self.metadata["VideoExists"]]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        row = self.metadata.iloc[idx]

        output = {
            "filename": row["FileName"],
        }

        if "image" in self.outputs:
            fpath = os.path.join(row["VideoPath"])
            rand_item = np.random.choice(os.listdir(fpath))
            image = Image.open(os.path.join(fpath, rand_item))  # H x W x C, uint8
            image = np.array(image)  # H x W x C, uint8
            image = torch.from_numpy(image)
            image = image.permute(2, 0, 1).float()  # C x H x W
            image = image / 128.0 - 1  # [-1, 1]
            output["image"] = image

        if "video" in self.outputs:
            fpath = os.path.join(row["VideoPath"])
            fc = self.target_nframes
            all_frames_names = sorted(os.listdir(fpath))
            if len(all_frames_names) > fc:
                start_idx = np.random.randint(0, len(all_frames_names) - fc)
                end_idx = start_idx + fc
            else:
                start_idx = 0
                end_idx = -1
            all_frames_names = all_frames_names[start_idx:end_idx]
            all_frames_path = [os.path.join(fpath, f) for f in all_frames_names]
            all_frames = [Image.open(f) for f in all_frames_path]
            all_frames = [np.array(f) for f in all_frames]
            all_frames = np.stack(all_frames, axis=0)  # T x H x W x C
            all_frames = torch.from_numpy(all_frames)  # T x H x W x C
            all_frames = all_frames.permute(3, 0, 1, 2).float()  # C x T x H x W
            all_frames = all_frames / 128.0 - 1  # [-1, 1]

            if len(all_frames) < fc:
                padding_element = torch.zeros_like(all_frames[0])
                padding = torch.stack([padding_element] * (fc - len(all_frames)))
                all_frames = torch.cat((all_frames, padding), dim=0)
                assert (
                    len(all_frames) == fc
                ), f"Video length is {len(all_frames)} but should be {fc}"

            output["video"] = all_frames

        if "lvef" in self.outputs:
            lvef = row["EF"] / 100.0
            output["lvef"] = torch.tensor(lvef, dtype=torch.float32)

        return output


def instantiate_dataset(configs, split=["TRAIN", "VAL", "TEST"]):
    # config = config.copy()
    # assert config.get("datasets", False), "No 'datasets' key found in config"

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


class RFBalancer(Dataset):  # Real - Fake Balancer
    """
    Balances the dataset by sampling from each dataset with equal probability.

    """

    def __init__(self, real_dataset=None, fake_dataset=None, transform=None) -> None:
        super().__init__()

        # self.datasets = [fake_dataset, real_dataset]
        self.datasets = []
        if fake_dataset is not None:
            self.datasets.append(fake_dataset)
        if real_dataset is not None:
            self.datasets.append(real_dataset)

        if len(self.datasets) == 0:
            raise ValueError("At least one dataset must be provided")

        if len(self.datasets) > 1:
            self.ds_idx = (
                np.random.rand(
                    1,
                )
                < 0.5
            )[
                0
            ]  # pick the first dataset to start with
        else:
            self.ds_idx = 0

        self.ds_current = [0] * len(self.datasets)

        self.transforms = transform

    def __len__(self):
        return np.sum([len(ds) for ds in self.datasets])

    def _get_index_for_ds(self, idx):
        ds_idx = 0
        while True:
            if idx < len(self.datasets[ds_idx]):
                break
            else:
                idx -= len(self.datasets[ds_idx])
                ds_idx = (ds_idx + 1) % len(self.datasets)
        return ds_idx, idx

    def __getitem__(self, idx):
        ds_idx, idx = self._get_index_for_ds(idx)
        output = self.datasets[ds_idx][idx]  # get item from dataset
        output["real"] = float(ds_idx)  # add real/fake label

        if self.transforms is not None and "video" in output:
            output["video"] = self.transforms(output["video"])
        if self.transforms is not None and "image" in output:
            output["image"] = self.transforms(output["image"])

        return output


class SimaseUSVideoDataset(Dataset):
    def __init__(
        self,
        phase="training",
        transform=None,
        latents_csv="./",
        training_latents_base_path="./",
        in_memory=True,
        generator_seed=None,
    ):
        self.phase = phase
        self.training_latents_base_path = training_latents_base_path

        self.in_memory = in_memory
        self.videos = []

        PHASE_TO_SPLIT = {"training": "TRAIN", "validation": "VAL", "testing": "TEST"}
        self.df = pd.read_csv(latents_csv)
        self.df = self.df[self.df["Split"] == PHASE_TO_SPLIT[self.phase]].reset_index(
            drop=True
        )
        self.transform = transform

        if generator_seed is None:
            self.generator = np.random.default_rng()
            # unseeded
        else:
            self.generator_seed = generator_seed
            print(f"Set {self.phase} dataset seed to {self.generator_seed}")

        if self.in_memory:
            self.load_videos()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        vid_a = self.get_vid(index)
        if self.transform is not None:
            vid_a = self.transform(vid_a)
        return vid_a

    def reset_generator(self):
        self.generator = np.random.default_rng(self.generator_seed)

    def get_vid(self, index, from_disk=False):
        if self.in_memory and not from_disk:
            return self.videos[index]
        else:
            path = self.df.iloc[index]["FileName"].split(".")[0] + ".pt"
            path = os.path.join(self.training_latents_base_path, path)
            return torch.load(path)

    def load_videos(self):
        self.videos = []
        print("Preloading videos")
        for i in range(len(self)):
            self.videos.append(self.get_vid(i, from_disk=True))


class SiameseUSDataset(Dataset):
    def __init__(
        self,
        phase="training",
        transform=None,
        latents_csv="./",
        training_latents_base_path="./",
        in_memory=True,
        generator_seed=None,
    ):
        self.phase = phase
        self.training_latents_base_path = training_latents_base_path

        self.in_memory = in_memory
        self.videos = []

        PHASE_TO_SPLIT = {"training": "TRAIN", "validation": "VAL", "testing": "TEST"}
        self.df = pd.read_csv(latents_csv)
        self.df = self.df[self.df["Split"] == PHASE_TO_SPLIT[self.phase]].reset_index(
            drop=True
        )

        self.transform = transform

        if generator_seed is None:
            self.generator = np.random.default_rng()
            # unseeded
        else:
            self.generator_seed = generator_seed
            print(f"Set {self.phase} dataset seed to {self.generator_seed}")

        if self.in_memory:
            self.load_videos()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        vid_a = torch.clone(self.get_vid(index))
        if self.generator.uniform() < 0.5:
            vid_b = torch.clone(
                self.get_vid(
                    (index + self.generator.integers(low=1, high=len(self))) % len(self)
                )
            )  # random different vid
            y = 0.0
        else:
            vid_b = torch.clone(vid_a)
            y = 1.0

        if self.transform is not None:
            vid_a = self.transform(vid_a)
            vid_b = self.transform(vid_b)

        frame_a = self.generator.integers(len(vid_a))
        frame_b = (frame_a + self.generator.integers(low=1, high=len(vid_b))) % len(
            vid_b
        )
        # print(f"Dataloader: framea {frame_a} - frame_b {frame_b} - y: {y}")
        return vid_a[frame_a], vid_b[frame_b], y

    def reset_generator(self):
        self.generator = np.random.default_rng(self.generator_seed)

    def get_vid(self, index, from_disk=False):
        if self.in_memory and not from_disk:
            return self.videos[index]
        else:
            path = self.df.iloc[index]["FileName"].split(".")[0] + ".pt"
            path = os.path.join(self.training_latents_base_path, path)
            return torch.load(path)

    def load_videos(self):
        self.videos = []
        print("Preloading videos")
        for i in range(len(self)):
            self.videos.append(self.get_vid(i, from_disk=True))


class ImageSet(Dataset):
    def __init__(self, root, ext=".jpg"):
        self.root = root
        self.all_images = glob(os.path.join(root, "*.jpg"))

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        image = Image.open(self.all_images[idx])
        image = np.array(image) / 128.0 - 1  # [0, 255] -> [-1, 1]
        image = image.transpose(2, 0, 1)  # H x W x C -> C x H x W
        return image


class TensorSet(Dataset):
    def __init__(self, root):
        self.root = root
        self.all_tensors = glob(os.path.join(root, "*.pt"))

    def __len__(self):
        return len(self.all_tensors)

    def __getitem__(self, idx):
        tensor = torch.load(self.all_tensors[idx], map_location="cpu")
        if tensor.ndim == 4:
            # get random frame
            tensor = tensor[np.random.randint(0, tensor.shape[0])]
        return tensor


class OGEchoNet(torchvision.datasets.VisionDataset):
    """EchoNet-Dynamic Dataset.

    Args:
        root (string): Root directory of dataset (defaults to `echonet.config.DATA_DIR`)
        split (string): One of {``train'', ``val'', ``test'', ``all'', or ``external_test''}
        target_type (string or list, optional): Type of target to use,
            ``Filename'', ``EF'', ``EDV'', ``ESV'', ``LargeIndex'',
            ``SmallIndex'', ``LargeFrame'', ``SmallFrame'', ``LargeTrace'',
            or ``SmallTrace''
            Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``Filename'' (string): filename of video
                ``EF'' (float): ejection fraction
                ``EDV'' (float): end-diastolic volume
                ``ESV'' (float): end-systolic volume
                ``LargeIndex'' (int): index of large (diastolic) frame in video
                ``SmallIndex'' (int): index of small (systolic) frame in video
                ``LargeFrame'' (np.array shape=(3, height, width)): normalized large (diastolic) frame
                ``SmallFrame'' (np.array shape=(3, height, width)): normalized small (systolic) frame
                ``LargeTrace'' (np.array shape=(height, width)): left ventricle large (diastolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
                ``SmallTrace'' (np.array shape=(height, width)): left ventricle small (systolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
            Defaults to ``EF''.
        mean (int, float, or np.array shape=(3,), optional): means for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not shifted).
        std (int, float, or np.array shape=(3,), optional): standard deviation for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not scaled).
        length (int or None, optional): Number of frames to clip from video. If ``None'', longest possible clip is returned.
            Defaults to 16.
        period (int, optional): Sampling period for taking a clip from the video (i.e. every ``period''-th frame is taken)
            Defaults to 2.
        max_length (int or None, optional): Maximum number of frames to clip from video (main use is for shortening excessively
            long videos when ``length'' is set to None). If ``None'', shortening is not applied to any video.
            Defaults to 250.
        clips (int, optional): Number of clips to sample. Main use is for test-time augmentation with random clips.
            Defaults to 1.
        pad (int or None, optional): Number of pixels to pad all frames on each side (used as augmentation).
            and a window of the original size is taken. If ``None'', no padding occurs.
            Defaults to ``None''.
        noise (float or None, optional): Fraction of pixels to black out as simulated noise. If ``None'', no simulated noise is added.
            Defaults to ``None''.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        external_test_location (string): Path to videos to use for external testing.
    """

    def __init__(
        self,
        root,
        split="train",
        target_type="EF",
        mean=0.0,
        std=1.0,
        length=16,
        period=2,
        max_length=250,
        clips=1,
        pad=None,
        noise=None,
        target_transform=None,
        external_test_location=None,
    ):
        import cv2

        super().__init__(root, target_transform=target_transform)

        self.split = split.upper()
        if not isinstance(target_type, list):
            target_type = list(target_type)
        self.target_type = target_type
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.pad = pad
        self.noise = noise
        self.target_transform = target_transform
        self.external_test_location = external_test_location

        self.fnames, self.outcome = [], []

        if self.split == "EXTERNAL_TEST":
            self.fnames = sorted(os.listdir(self.external_test_location))
        else:
            # Load video-level labels
            with open(os.path.join(self.root, "FileList.csv")) as f:
                data = pd.read_csv(f)
            data["Split"].map(lambda x: x.upper())

            if self.split != "ALL":
                data = data[data["Split"] == self.split]

            self.header = data.columns.tolist()
            self.fnames = data["FileName"].tolist()
            self.fnames = [
                fn + ".avi" for fn in self.fnames if os.path.splitext(fn)[1] == ""
            ]  # Assume avi if no suffix
            self.outcome = data.values.tolist()

            # Check that files are present
            missing = set(self.fnames) - set(
                os.listdir(os.path.join(self.root, "Videos"))
            )
            if len(missing) != 0:
                print(
                    "{} videos could not be found in {}:".format(
                        len(missing), os.path.join(self.root, "Videos")
                    )
                )
                for f in sorted(missing):
                    print("\t", f)
                raise FileNotFoundError(
                    os.path.join(self.root, "Videos", sorted(missing)[0])
                )

            # Load traces
            self.frames = collections.defaultdict(list)
            self.trace = collections.defaultdict(_defaultdict_of_lists)

            with open(os.path.join(self.root, "VolumeTracings.csv")) as f:
                header = f.readline().strip().split(",")
                # assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]
                if header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]:
                    self.tracing_mode = "dynamic"
                elif header == ["FileName", "X", "Y", "Frame"]:
                    self.tracing_mode = "pediatric"
                else:
                    raise ValueError(
                        "Unrecognized header in VolumeTracings.csv", header
                    )

                if self.tracing_mode == "dynamic":
                    for line in f:
                        filename, x1, y1, x2, y2, frame = line.strip().split(",")
                        x1 = float(x1)
                        y1 = float(y1)
                        x2 = float(x2)
                        y2 = float(y2)
                        frame = int(frame)
                        if frame not in self.trace[filename]:
                            self.frames[filename].append(frame)
                        self.trace[filename][frame].append((x1, y1, x2, y2))
                elif self.tracing_mode == "pediatric":
                    for i, line in enumerate(f):
                        filename, x, y, frame = line.strip().split(",")
                        frame = int(frame) if frame.isdigit() else frame

                        if isinstance(frame, str):
                            self.trace[filename][frame] = [(0, 0, 0, 0), (0, 0, 0, 0)]
                            continue

                        if frame not in self.trace[filename]:
                            self.frames[filename].append(frame)

                        x = float(x)
                        y = float(y)
                        if i % 2 == 0:
                            first_point = (x, y)
                        else:
                            second_point = (x, y)
                            self.trace[filename][frame].append(
                                (
                                    first_point[0],
                                    first_point[1],
                                    second_point[0],
                                    second_point[1],
                                )
                            )

            for filename in self.frames:
                for frame in self.frames[filename]:
                    self.trace[filename][frame] = np.array(self.trace[filename][frame])

            # add None to trace for videos without traces
            for f in self.fnames:
                if f not in self.frames:
                    self.frames[f] = [0, 0]
                    self.trace[f] = None
            keep = [True for f in self.fnames]

            # A small number of videos are missing traces; remove these videos
            # keep = [len(self.frames[f]) >= 2 for f in self.fnames]
            self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
            self.outcome = [f for (f, k) in zip(self.outcome, keep) if k]

    def __getitem__(self, index):
        # Find filename of video
        if self.split == "EXTERNAL_TEST":
            video = os.path.join(self.external_test_location, self.fnames[index])
        elif self.split == "CLINICAL_TEST":
            video = os.path.join(
                self.root, "ProcessedStrainStudyA4c", self.fnames[index]
            )
        else:
            video = os.path.join(self.root, "Videos", self.fnames[index])

        # Load video into np.array
        video = self.loadvideo(video).astype(np.float32)

        # Add simulated noise (black out random pixels)
        # 0 represents black at this point (video has not been normalized yet)
        if self.noise is not None:
            n = video.shape[1] * video.shape[2] * video.shape[3]
            ind = np.random.choice(n, round(self.noise * n), replace=False)
            f = ind % video.shape[1]
            ind //= video.shape[1]
            i = ind % video.shape[2]
            ind //= video.shape[2]
            j = ind
            video[:, f, i, j] = 0

        # Apply normalization
        if isinstance(self.mean, (float, int)):
            video -= self.mean
        else:
            video -= self.mean.reshape(3, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            video /= self.std.reshape(3, 1, 1, 1)

        # Set number of frames
        c, f, h, w = video.shape
        if self.length is None:
            # Take as many frames as possible
            length = f // self.period
        else:
            # Take specified number of frames
            length = self.length

        if self.max_length is not None:
            # Shorten videos to max_length
            length = min(length, self.max_length)

        if f < length * self.period:
            # Pad video with frames filled with zeros if too short
            # 0 represents the mean color (dark grey), since this is after normalization
            video = np.concatenate(
                (video, np.zeros((c, length * self.period - f, h, w), video.dtype)),
                axis=1,
            )
            c, f, h, w = video.shape  # pylint: disable=E0633

        if self.clips == "all":
            # Take all possible clips of desired length
            start = np.arange(f - (length - 1) * self.period)
        else:
            # Take random clips from video
            start = np.random.choice(f - (length - 1) * self.period, self.clips)

        # Gather targets
        target = []
        for t in self.target_type:
            key = self.fnames[index]
            if t == "Filename":
                target.append(self.fnames[index])
            elif t == "LargeIndex":
                # Traces are sorted by cross-sectional area
                # Largest (diastolic) frame is first
                target.append(int(self.frames[key][0]))
            elif t == "SmallIndex":
                # Smallest (systolic) frame is last
                target.append(int(self.frames[key][-1]))
            elif t == "LargeFrame":
                tmp_idx = min(video.shape[1] - 1, self.frames[key][0])
                target.append(video[:, tmp_idx, :, :])
            elif t == "SmallFrame":
                tmp_idx = min(video.shape[1] - 1, self.frames[key][-1])
                target.append(video[:, tmp_idx, :, :])
            elif t in ["LargeTrace", "SmallTrace"]:
                if self.trace[key] is None:
                    mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
                else:
                    if t == "LargeTrace":
                        trace = self.trace[key][self.frames[key][0]]
                    else:
                        trace = self.trace[key][self.frames[key][-1]]

                    if self.tracing_mode == "pediatric":
                        traceN2 = trace.reshape(-1, 2)
                        x = traceN2[:, 0]
                        y = traceN2[:, 1]
                    else:
                        x1, y1, x2, y2 = (
                            trace[:, 0],
                            trace[:, 1],
                            trace[:, 2],
                            trace[:, 3],
                        )
                        x = np.concatenate((x1[1:], np.flip(x2[1:])))
                        y = np.concatenate((y1[1:], np.flip(y2[1:])))

                    r, c = skimage.draw.polygon(
                        np.rint(y).astype(int),
                        np.rint(x).astype(int),
                        (video.shape[2], video.shape[3]),
                    )
                    mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
                    mask[r, c] = 1
                target.append(mask)
            elif t in ["RawLargeTrace", "RawSmallTrace"]:
                if t == "RawLargeTrace":
                    trace = self.trace[key][self.frames[key][0]]
                else:
                    trace = self.trace[key][self.frames[key][-1]]
                if trace is not None and len(trace) != 21:
                    trace = np.concatenate([trace[:11, :], trace[-10:, :]])
                target.append(trace)
            else:
                if self.split == "CLINICAL_TEST" or self.split == "EXTERNAL_TEST":
                    target.append(np.float32(0))
                else:
                    if isinstance(self.outcome[index][self.header.index(t)], str):
                        target.append(self.outcome[index][self.header.index(t)])
                    else:
                        target.append(
                            np.float32(self.outcome[index][self.header.index(t)])
                        )

        if target != []:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)

        # Select clips from video
        video = tuple(
            video[:, s + self.period * np.arange(length), :, :] for s in start
        )
        if self.clips == 1:
            video = video[0]
        else:
            video = np.stack(video)

        if self.pad is not None:
            # Add padding of zeros (mean color of videos)
            # Crop of original size is taken out
            # (Used as augmentation)
            c, l, h, w = video.shape
            temp = np.zeros(
                (c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype
            )
            temp[:, :, self.pad : -self.pad, self.pad : -self.pad] = (
                video  # pylint: disable=E1130
            )
            i, j = np.random.randint(0, 2 * self.pad, 2)
            video = temp[:, :, i : (i + h), j : (j + w)]

        return video, target

    def __len__(self):
        return len(self.fnames)

    def extra_repr(self) -> str:
        """Additional information to add at end of __repr__."""
        lines = ["Target type: {target_type}", "Split: {split}"]
        return "\n".join(lines).format(**self.__dict__)

    def loadvideo(self, filename: str) -> np.ndarray:
        """Loads a video from a file.

        Args:
            filename (str): filename of video

        Returns:
            A np.ndarray with dimensions (channels=3, frames, height, width). The
            values will be uint8's ranging from 0 to 255.

        Raises:
            FileNotFoundError: Could not find `filename`
            ValueError: An error occurred while reading the video
        """

        if not os.path.exists(filename):
            raise FileNotFoundError(filename)
        capture = cv2.VideoCapture(filename)

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        v = np.zeros((frame_count, frame_height, frame_width, 3), np.uint8)

        for count in range(frame_count):
            ret, frame = capture.read()
            if not ret:
                raise ValueError(
                    "Failed to load frame #{} of {}.".format(count, filename)
                )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            v[count, :, :] = frame

        v = v.transpose((3, 0, 1, 2))

        return v


def _defaultdict_of_lists():
    """Returns a defaultdict of lists.

    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)


class BaseVideoDataset(Dataset):
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

        split = [split.upper()] if isinstance(split, str) else split

        self.metadata = pd.read_csv(os.path.join(self.root, "FileList.csv"))
        self.metadata = self.metadata[self.metadata["Split"].isin(split)]
        self.metadata.reset_index(inplace=True, drop=True)

        self.index_to_filename = self.metadata["FileName"].tolist()

    def __len__(self):
        return len(self.index_to_filename)

    def _get_path(self, idx):
        fname = self.index_to_filename[idx]
        path = os.path.join(self.video_path, f"{fname}.{self.extension}")
        return path

    def _load_data(self, path):
        if self.extension == "pt":
            return torch.load(path)  # T x C x H x W
        else:
            # TODO: this has not been tested
            print("Loading video from avi")
            video = loadvideo(path)
            print("Video shape should be T C H W, is this right ?:", video.shape)
            print("Exiting before until this is confirmed")
            exit()
            return loadvideo(path)  # ?? T x C x H x W ??


class ContrastivePair(BaseVideoDataset):
    def __init__(
        self, root, folder="Latents", extension="pt", split=["TRAIN", "VAL", "TEST"]
    ):
        super().__init__(root, folder, extension, split)

    def __getitem__(self, idx):
        videoA_path = self._get_path(idx)
        videoA = self._load_data(videoA_path)
        frameA_idx = torch.randint(0, videoA.size(0), (1,)).item()
        frameA = videoA[frameA_idx, :, :, :]  # C x H x W

        if torch.rand(1) < 0.5:
            # Same video
            frameB_idx = torch.randint(0, videoA.size(0), (1,)).item()
            frameB = videoA[frameB_idx, :, :, :]  # C x H x W
            label = torch.tensor(1.0)  # 1 = same video
        else:
            # Different video
            videoB_path = self._get_path(torch.randint(0, len(self), (1,)).item())
            videoB = self._load_data(videoB_path)
            frameB_idx = torch.randint(0, videoB.size(0), (1,)).item()
            frameB = videoB[frameB_idx, :, :, :]
            label = torch.tensor(0.0)  # 0 = different video

        return frameA, frameB, label


class FirstFrame(BaseVideoDataset):
    def __init__(
        self, root, folder="Latents", extension="pt", split=["TRAIN", "VAL", "TEST"]
    ):
        super().__init__(root, folder, extension, split)

    def __getitem__(self, idx):
        video_path = self._get_path(idx)
        video = self._load_data(video_path)
        frame_idx = 0
        frame = video[frame_idx, :, :, :]  # C x H x W

        return frame


class AllFramesSegmented(BaseVideoDataset):
    def __init__(
        self,
        root,
        folder="Latents",
        extension="pt",
        split=["TRAIN", "VAL", "TEST"],
        segmentation_path=None,
        vae_scale=8,
    ):

        super().__init__(root, folder, extension, split)
        assert (
            segmentation_path is not None
        ), "Segmentation path must be provided for AllFramesSegmented Dataset"

        self.segmentation_path = segmentation_path
        self.all_segmentations = glob(os.path.join(self.segmentation_path, "*.pt"))
        self.vae_scale = vae_scale

    def __getitem__(self, idx):
        video_path = self._get_path(idx)
        video = self._load_data(video_path)

        segmentation_path = self._get_segmentation_path(idx)
        segmentation_v = torch.load(segmentation_path)  # T x H x W

        frame_idx = torch.randint(0, video.size(0), (1,)).item()

        frame = video[frame_idx, :, :, :]  # C x H x W
        segmentation = segmentation_v[frame_idx, :, :]  # H x W
        segmentation = self._downsize_segmentation(segmentation)

        assert segmentation.size(0) == frame.size(1)

        return frame, segmentation

    def _get_segmentation_path(self, idx):
        fname = self.index_to_filename[idx]
        path = os.path.join(self.segmentation_path, f"{fname}.pt")
        return path

    def _downsize_segmentation(self, segmentation):
        return F.max_pool2d(segmentation, self.vae_scale)


if __name__ == "__main__":
    pass
