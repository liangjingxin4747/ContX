import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, ImageTFPaths, NumpyPaths, ConcatDatasetWithIndex


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example


class CustomTrain(CustomBase):
    # def __init__(self, size, training_images_list_file):
    def __init__(self, size, training_images_list_file, training_images_dis_file, augment=None):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        with open(training_images_dis_file, "r") as f:
            dis_paths = f.read().splitlines()
        # self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.data = ImagePaths(paths=paths, dis_paths=dis_paths, size=size, random_crop=True, augment=augment)


class CustomTest(CustomBase):
    # def __init__(self, size, test_images_list_file):
    def __init__(self, size, test_images_list_file, test_images_dis_file):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        with open(test_images_dis_file, "r") as f:
            dis_paths = f.read().splitlines()
        # self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.data = ImagePaths(paths=paths, dis_paths=dis_paths, size=size, random_crop=False)


class CustomTFTrain(CustomBase):
    def __init__(self, training_images_list_file, training_images_dis_file, size, crop_size=None, coord=False, augment=None):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        with open(training_images_dis_file, "r") as f:
            dis_paths = f.read().splitlines()
        # self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.data = ImageTFPaths(paths=paths, dis_paths=dis_paths, crop_size=size, augment=augment)
        self.coord = coord
        if crop_size is not None:
            self.cropper = albumentations.RandomCrop(height=crop_size, width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image", "dis_image": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex = self.data[i]
        if hasattr(self, "cropper"):
            if not self.coord:
                out = self.cropper(image=ex["image"])
                ex["image"] = out["image"]
            else:  # !
                h, w, _ = ex["image"].shape  # (h,w,87)
                coord = np.arange(h * w).reshape(h, w, 1) / (h * w)
                out = self.cropper(image=ex["image"], coord=coord, dis_image=ex["dis_image"])
                ex["image"] = out["image"]
                ex["dis_image"] = out["dis_image"]
                ex["coord"] = out["coord"]
        # ex["class"] = y
        return ex


class CustomTFTest(CustomBase):
    def __init__(self, test_images_list_file, test_images_dis_file, size, crop_size=None, coord=False):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        with open(test_images_dis_file, "r") as f:
            dis_paths = f.read().splitlines()
        # self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.data = ImageTFPaths(paths=paths, dis_paths=dis_paths, crop_size=size)
        self.coord = coord
        if crop_size is not None:
            self.cropper = albumentations.RandomCrop(height=crop_size, width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image", "dis_image": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex = self.data[i]
        if hasattr(self, "cropper"):
            if not self.coord:
                out = self.cropper(image=ex["image"])
                ex["image"] = out["image"]
            else:  # !
                h, w, _ = ex["image"].shape  # (h,w,87)
                coord = np.arange(h * w).reshape(h, w, 1) / (h * w)
                out = self.cropper(image=ex["image"], coord=coord, dis_image=ex["dis_image"])
                ex["image"] = out["image"]
                ex["dis_image"] = out["dis_image"]
                ex["coord"] = out["coord"]
        return ex

