import bisect
import numpy as np
import albumentations
import torch
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
# from torchvision.transforms.functional import InterpolationMode
import cv2



class ConcatDatasetWithIndex(ConcatDataset):
    """Modified from original pytorch code to return dataset idx"""
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx


class ImagePaths(Dataset):
    #def __init__(self, paths, size=None, random_crop=False, labels=None):
    def __init__(self, paths, dis_paths, size=None, random_crop=False, labels=None, augment=None):
        self.size = size
        self.random_crop = random_crop

        self.augment = augment
        if self.augment != None:
            paths = [p for p in paths for i in range(self.augment)]
            dis_paths = [p for p in dis_paths for i in range(self.augment)]
        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self.labels["dis_path_"] = dis_paths
        self._length = len(paths)

        if self.augment != None:
            if self.size is not None and self.size > 0:
                crop_size = int(self.size * 1.2)
                self.preprocessor = albumentations.Compose([
                    albumentations.SmallestMaxSize(crop_size),
                    albumentations.RandomCrop(height=self.size, width=self.size),
                    albumentations.HorizontalFlip(p=0.5)
                ]
                )
            else:
                self.preprocessor = lambda **kwargs: kwargs
        else:
            if self.size is not None and self.size > 0:
                self.rescaler = albumentations.SmallestMaxSize(max_size=self.size, interpolation=cv2.INTER_NEAREST)
                if not self.random_crop:
                    self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
                else:
                    self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
                self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
            else:
                self.preprocessor = lambda **kwargs: kwargs

        self.cat_dict = dict()   # gray value --- class
        with open("./concat_cat.txt", "r") as f:   # dyh data(87)
        # with open("/home/liangxin01/code/datasets/coco-stuff/my_category.txt", "r") as f:   # ljx data(85)
            cat_f = f.read().splitlines()
            for i, j in enumerate(cat_f):
                self.cat_dict[int(j)] = i

        # self.cats_object_path = "/home/liangxin01/code/datasets/coco-stuff/cats_train_object.txt"
        # self.cats_object = dict()   # gray value --- class
        # with open(self.cats_object_path, "r") as f:
        #     cat_f = f.read().splitlines()
        #     for i, j in enumerate(cat_f):
        #         self.cats_object[int(j)] = i
        #
        # self.cats_context_path = "/home/liangxin01/code/datasets/coco-stuff/cats_train_context.txt"
        # self.cats_context = dict()   # gray value --- class
        # with open(self.cats_context_path, "r") as f:
        #     cat_f = f.read().splitlines()
        #     for i, j in enumerate(cat_f):
        #         self.cats_context[int(j)] = i

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path, type=None):
        '''
            L(h*w) --> array(h*w*c)
        '''
        # print(image_path)
        image = Image.open(image_path)
        '''
        if not image.mode == "RGB":
            image = image.convert("RGB")
        '''
        image = np.array(image).astype(np.uint8)
        b = np.unique(image)
        image = self.preprocessor(image=image)["image"]
        q, w = image.shape
        # c = [[0, -1], [0, 1], [-1, 0], [0, 1], [-1, -1], [1, -1], [-1, 1], [1, 1]]  # 上,下,左,右,左上，右上，左下，右下
        # for i in range(q):
        #     for j in range(w):
        #         if image[i][j] not in b:
        #             temp = 0
        #             for k in c:
        #                 e1 = k[0] + i
        #                 e2 = k[1] + j
        #                 if e1 < q and e2 < w and image[e1][e2] in b:
        #                     image[i][j] = image[e1][e2]
        #                     temp = 1
        #                     break
        #             if temp == 0:
        #                 image[i][j] = 255

        # print("-----------")
        # print(b)
        # cat_dic = {0: 0, 34: 1, 119: 2, 158: 3, 255: 4}
        if type == "object":
            self.cat_dict = self.cats_object
        elif type == "context":
            self.cat_dict = self.cats_context
        new_image = np.zeros((self.size, self.size, len(self.cat_dict)))
        for i in range(q):
            for j in range(w):
                dic_poz = self.cat_dict[image[i][j]]   # class
                new_image[i][j][dic_poz] = 1
        # image = self.rescaler(image=image)["image"]
        # image = (image/127.5 - 1.0).astype(np.float32)
        return new_image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        example["dis_image"] = self.preprocess_image(self.labels["dis_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example



class ImageTFPaths(Dataset):
    #def __init__(self, paths, size=None, random_crop=False, labels=None):
    def __init__(self, paths, dis_paths, crop_size, random_crop=False, labels=None, augment=None):
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.size = crop_size
        self.augment = augment

        if self.augment is not None:
            paths = [path for path in paths for i in range(self.augment)]
            dis_paths = [path for path in dis_paths for i in range(self.augment)]

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self.labels["dis_path_"] = dis_paths
        self._length = len(paths)

        if self.augment != None:
            if self.size is not None and self.size > 0:
                crop_size = int(self.size * 1.2)
                self.preprocessor = albumentations.Compose([
                    albumentations.SmallestMaxSize(crop_size),
                    albumentations.RandomCrop(height=self.size, width=self.size),
                    albumentations.HorizontalFlip(p=0.5)
                ]
                )
            else:
                self.preprocessor = lambda **kwargs: kwargs
        else:
            if self.size is not None and self.size > 0:
                self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
                if not self.random_crop:
                    self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
                else:
                    self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
                self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
            else:
                self.preprocessor = lambda **kwargs: kwargs

        self.cat_dict = dict()   # gray value --- class
        with open("./concat_cat.txt", "r") as f:   # dyh data(87)
        # with open("/home/liangxin01/code/datasets/coco-stuff/cats_my_train2017.txt", "r") as f:   # ljx data(85)
            cat_f = f.read().splitlines()
            for i, j in enumerate(cat_f):
                self.cat_dict[int(j)] = i

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path, type=None):
        '''
            rgb(h*w*3) --> array(h*w*c)
        '''
        # print(image_path)
        image = Image.open(image_path)
        '''
        if not image.mode == "RGB":
            image = image.convert("RGB")
        '''
        image = np.array(image).astype(np.uint8)
        b = np.unique(image)
        image = self.preprocessor(image=image)["image"]
        h, w = image.shape
        c = [[0, -1], [0, 1], [-1, 0], [0, 1], [-1, -1], [1, -1], [-1, 1], [1, 1]]  # 上,下,左,右,左上，右上，左下，右下
        # for i in range(h):
        #     for j in range(w):
        #         if image[i][j] not in b:
        #             temp = 0
        #             for k in c:
        #                 e1 = k[0] + i
        #                 e2 = k[1] + j
        #                 if e1 < h and e2 < w and image[e1][e2] in b:
        #                     image[i][j] = image[e1][e2]
        #                     temp = 1
        #                     break
        #             if temp == 0:
        #                 image[i][j] = 255

        new_image = np.zeros((h, w, len(self.cat_dict)))
        for i in range(h):
            for j in range(w):
                dic_poz = self.cat_dict[image[i][j]]   # class
                new_image[i][j][dic_poz] = 1
        # image = self.rescaler(image=image)["image"]
        # image = (image/127.5 - 1.0).astype(np.float32)
        return new_image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        example["dis_image"] = self.preprocess_image(self.labels["dis_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example

'''
class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example
'''

class NumpyPaths(ImagePaths):
    def preprocess_image(self, image_path):
        image = np.load(image_path).squeeze(0)  # 3 x 1024 x 1024
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(image, mode="RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image


def preprocess_image(image_path, size):
    rescaler = albumentations.SmallestMaxSize(max_size=size)   # 缩放
    cropper = albumentations.CenterCrop(height=size, width=size)   # 中心裁剪
    preprocessor = albumentations.Compose([rescaler, cropper])

    save_dir = '../../img/'
    image = Image.open(image_path)
    print("原图：", type(image), image.size)
    # image.save(save_dir + 'RGB_img.png')
    image = np.array(image).astype(np.uint8)
    print("np.array：", type(image), image.shape)
    # cv2.imwrite(save_dir + 'ndarray_img.png', image)
    b = np.unique(image)
    print("b = ", b)
    image = preprocessor(image=image)["image"]
    print("preprocessor：", type(image), image.shape)
    # cv2.imwrite(save_dir + 'preprocess_img.png', image)
    q, w = image.shape
    c = [[0, -1], [0, 1], [-1, 0], [0, 1], [-1, -1], [1, -1], [-1, 1], [1, 1]]  # 上,下,左,右,左上，右上，左下，右下
    count = 0
    for i in range(q):   # 128
        for j in range(w):   # 128
            if image[i][j] not in b:   # 邻值填充
                count = count + 1
                temp = 0
                for k in c:
                    e1 = k[0] + i
                    e2 = k[1] + j
                    if e1 < q and e2 < w and image[e1][e2] in b:
                        image[i][j] = image[e1][e2]
                        temp = 1
                        break
                if temp == 0:   # missing
                    image[i][j] = 255
    print("interpolate：", type(image), image.shape)
    print("count：", count)
    # cv2.imwrite(save_dir + 'interpolate_img.png', image)

    # category_dict {rgb_value : category, ...}
    cat_dic = dict()
    with open("../../concat_cat.txt", "r") as f:
    # with open("/home/liangxin01/code/datasets/coco-stuff/cats_my_train2017.txt", "r") as f:
        cat_f = f.read().splitlines()
        for i, j in enumerate(cat_f):
            cat_dic[int(j)] = i
    new_image = np.zeros((size, size, len(cat_dic)))   # 128*128*67
    for i in range(q):
        for j in range(w):
            dic_poz = cat_dic[image[i][j]]   # 找出cat_dic中rgb值对应的类别值
            new_image[i][j][dic_poz] = 1     # one-hot
    # image = self.rescaler(image=image)["image"]
    # image = (image/127.5 - 1.0).astype(np.float32)
    print("new_image：", type(new_image), new_image.shape)


if __name__ == '__main__':
    image_path = '/home/liangxin01/code/dyh/ljx/remove_image1207/train_con/000000067115.png'
    preprocess_image(image_path, size=128)