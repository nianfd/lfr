import random

from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
import torch.multiprocessing
import json

# import imgaug.augmenters as iaa
# from perlin import rand_perlin_2d_np

torch.multiprocessing.set_sharing_strategy('file_system')


def get_data_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms


def get_strong_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomResizedCrop((isize, isize), scale=(0.6, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    return data_transforms


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.cls_idx = 0

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type, "*.png")) + \
                           glob.glob(os.path.join(self.gt_path, defect_type, "*.bmp"))
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if label == 0:
            gt = torch.zeros([1, img.size(-2), img.size(-1)])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


class RealIADDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, transform, gt_transform, phase):
        self.img_path = os.path.join(root, 'realiad_1024', category)
        self.transform = transform
        self.gt_transform = gt_transform
        self.phase = phase

        json_path = os.path.join(root, 'realiad_jsons', 'realiad_jsons', category + '.json')
        with open(json_path) as file:
            class_json = file.read()
        class_json = json.loads(class_json)

        self.img_paths, self.gt_paths, self.labels, self.types = [], [], [], []

        data_set = class_json[phase]
        for sample in data_set:
            self.img_paths.append(os.path.join(root, 'realiad_1024', category, sample['image_path']))
            label = sample['anomaly_class'] != 'OK'
            if label:
                self.gt_paths.append(os.path.join(root, 'realiad_1024', category, sample['mask_path']))
            else:
                self.gt_paths.append(None)
            self.labels.append(label)
            self.types.append(sample['anomaly_class'])

        self.img_paths = np.array(self.img_paths)
        self.gt_paths = np.array(self.gt_paths)
        self.labels = np.array(self.labels)
        self.types = np.array(self.types)
        self.cls_idx = 0

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        if self.phase == 'train':
            return img, label

        if label == 0:
            gt = torch.zeros([1, img.size(-2), img.size(-1)])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


class LOCODataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*/000.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        size = (img.size[1], img.size[0])
        img = self.transform(img)
        type = self.types[idx]
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path, type, size


class InsPLADDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
        self.transform = transform
        self.phase = phase
        # load dataset
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        tot_labels = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
                img_tot_paths.extend(img_paths)
                tot_labels.extend([0] * len(img_paths))
            else:
                if self.phase == 'train':
                    continue
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
                img_tot_paths.extend(img_paths)
                tot_labels.extend([1] * len(img_paths))

        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        img = Image.open(img_path).convert('RGB')

        img = self.transform(img)

        return img, label, img_path


class AeBADDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.phase = phase
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)
        defect_types = [i for i in defect_types if i[0] != '.']
        for defect_type in defect_types:
            if defect_type == 'good':
                domain_types = os.listdir(os.path.join(self.img_path, defect_type))
                domain_types = [i for i in domain_types if i[0] != '.']

                for domain_type in domain_types:
                    img_paths = glob.glob(os.path.join(self.img_path, defect_type, domain_type) + "/*.png")
                    img_tot_paths.extend(img_paths)
                    gt_tot_paths.extend([0] * len(img_paths))
                    tot_labels.extend([0] * len(img_paths))
                    tot_types.extend(['good'] * len(img_paths))
            else:
                domain_types = os.listdir(os.path.join(self.img_path, defect_type))
                domain_types = [i for i in domain_types if i[0] != '.']

                for domain_type in domain_types:
                    img_paths = glob.glob(os.path.join(self.img_path, defect_type, domain_type) + "/*.png")
                    gt_paths = glob.glob(os.path.join(self.gt_path, defect_type, domain_type) + "/*.png")
                    img_paths.sort()
                    gt_paths.sort()
                    img_tot_paths.extend(img_paths)
                    gt_tot_paths.extend(gt_paths)
                    tot_labels.extend([1] * len(img_paths))
                    tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if self.phase == 'train':
            return img, label
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


class MiniDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):

        self.img_path = root
        self.transform = transform
        # load dataset
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        tot_labels = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
            img_tot_paths.extend(img_paths)
            tot_labels.extend([1] * len(img_paths))

        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        try:
            img_path, label = self.img_paths[idx], self.labels[idx]
            img = Image.open(img_path).convert('RGB')
        except:
            img_path, label = self.img_paths[idx - 1], self.labels[idx - 1]
            img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return img, label


class MVTecDRAEMDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, strong_transform, phase, anomaly_source_path, anomaly_ratio=0.5,
                 size=256):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        self.strong_transform = strong_transform
        self.anomaly_ratio = anomaly_ratio
        self.size = size
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path + "/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))
                           ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def augment_image(self, image, anomaly_source_path):
        no_anomaly = random.random()
        if no_anomaly > self.anomaly_ratio:
            return image, 0
        else:
            aug = self.randAugmenter()

            perlin_scale = 6
            min_perlin_scale = 0
            anomaly_source_img = Image.open(anomaly_source_path).convert('RGB').resize((self.size, self.size))
            anomaly_source_img = np.asarray(anomaly_source_img)
            anomaly_img_augmented = aug(image=anomaly_source_img)

            perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
            perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

            perlin_noise = rand_perlin_2d_np((self.size, self.size),
                                             (perlin_scalex, perlin_scaley))
            perlin_noise = self.rot(image=perlin_noise)
            threshold = 0.5
            perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
            perlin_thr = np.expand_dims(perlin_thr, axis=2)

            img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr

            beta = random.random() * 0.7 + 0.1

            image = image.resize((self.size, self.size))
            image = np.asarray(image)
            augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (perlin_thr)
            # augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1 - msk) * image

            return Image.fromarray(np.uint8(augmented_image)), 1

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')

        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        a_img, label = self.augment_image(img, self.anomaly_source_paths[anomaly_source_idx])

        img = self.transform(img)
        a_img = self.strong_transform(a_img)

        assert img.size()[1:] == a_img.size()[1:], "image.size != a_img.size !!!"

        return img, a_img, label


class MVTecSimplexDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform

        self.simplexNoise = Simplex_CLASS()
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img_normal = self.transform(img)

        if random.random() > 0.5:
            return img_normal, img_normal
        ## simplex_noise
        size = 256
        img = img.resize((size, size))
        img = np.asarray(img)
        h_noise = np.random.randint(10, int(size // 8))
        w_noise = np.random.randint(10, int(size // 8))
        start_h_noise = np.random.randint(1, size - h_noise)
        start_w_noise = np.random.randint(1, size - w_noise)
        noise_size = (h_noise, w_noise)
        simplex_noise = self.simplexNoise.rand_3d_octaves((3, *noise_size), 6, 0.6)
        init_zero = np.zeros((256, 256, 3))
        init_zero[start_h_noise: start_h_noise + h_noise, start_w_noise: start_w_noise + w_noise,
        :] = 0.2 * simplex_noise.transpose(1, 2, 0)
        img_noise = img + init_zero * 255
        img_noise = Image.fromarray(np.uint8(img_noise))
        img_noise = self.transform(img_noise)

        return img_normal, img_noise


# ============================================================================
# --item_list 解析：把「用户写的类别列表」与「磁盘上真实存在的类别」强制对齐
# ============================================================================
# 背景：item_list 的默认值是 MVTec 的 15 类，如果拿着它去跑 VisA / Real-IAD，
# 会在 ImageFolder(root=<data_path>/carpet/train) 处抛一个语焉不详的
# FileNotFoundError（目录不存在），看不出真正原因是"类别列表不对"。
# 下面这组函数把这件事变成：能在磁盘上验证的、有明确提示的解析。
# ============================================================================

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def _has_image(folder):
    """folder 下（含子目录）是否存在至少一张图片。"""
    if not os.path.isdir(folder):
        return False
    for _, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                return True
    return False


def _item_status(data_path, item, layout):
    """检查单个类别在 data_path 下是否可用。返回 (ok, reason)。"""
    if layout == 'folder':
        # MVTec / VisA_pytorch(1cls) / 3D-ADAM：<data_path>/<item>/train/...
        d = os.path.join(data_path, item)
        if not os.path.isdir(d):
            return False, "目录不存在"
        if os.path.isdir(os.path.join(d, 'Data', 'Images')):
            # 先判这个：目录是存在的，但内部是原始 VisA 布局
            return False, ("只有 <item>/Data/Images（**原始 VisA** 布局），"
                           "需要先用 prepare_data/prepare_visa.py 转成 VisA_pytorch")
        if not _has_image(os.path.join(d, 'train')):
            return False, "缺少 <item>/train 下的图像"
        return True, ""
    if layout == 'realiad':
        if not os.path.isdir(os.path.join(data_path, 'realiad_1024', item)):
            return False, "缺少 realiad_1024/<item>"
        js = os.path.join(data_path, 'realiad_jsons', 'realiad_jsons', item + '.json')
        if not os.path.isfile(js):
            return False, "缺少 realiad_jsons/realiad_jsons/<item>.json"
        return True, ""
    raise ValueError(f"unknown layout: {layout}")


def probe_item_list(data_path, layout='folder'):
    """探测 data_path 下真实可用（且已排序）的类别名列表。"""
    if layout == 'folder':
        root = data_path
    elif layout == 'realiad':
        root = os.path.join(data_path, 'realiad_1024')
    else:
        raise ValueError(f"unknown layout: {layout}")
    if not os.path.isdir(root):
        return []
    found = []
    for name in sorted(os.listdir(root)):
        if not os.path.isdir(os.path.join(root, name)):
            continue
        ok, _ = _item_status(data_path, name, layout)
        if ok:
            found.append(name)
    return found


def resolve_item_list(data_path, item_list, layout='folder', print_fn=print):
    """把 --item_list 解析成"确实存在于磁盘上"的类别列表。

    规则（按优先级）：
      1) item_list 为 None/空，或含字面量 'auto'  -> 直接从磁盘探测
      2) item_list 每一项在磁盘上都可用           -> **原样返回**（顺序不变，
                                                    与旧行为逐位一致）
      3) 有任何一项不可用                         -> 打印警告（逐条给出原因），
                                                    改用磁盘探测结果
      4) 磁盘上什么都探测不到                     -> SystemExit，给出可操作提示

    这样 MVTec 的既有调用（15 类都在）行为完全不变；而把 VisA 的
    --data_path 配上默认（MVTec）列表时，会自动落到 VisA 的 12 类上，
    同时把"发生了什么"明确打出来，而不是抛 FileNotFoundError。
    """
    item_list = list(item_list) if item_list else []
    auto = (len(item_list) == 0) or any(str(x).lower() == 'auto' for x in item_list)

    if not os.path.isdir(data_path):
        raise SystemExit(
            f"[item_list] --data_path 不存在: {data_path}\n"
            f"           请检查路径；MVTec 应指到数据集根目录，"
            f"VisA 应指到 VisA_pytorch/1cls。"
        )

    if not auto:
        missing = []
        for it in item_list:
            ok, why = _item_status(data_path, it, layout)
            if not ok:
                missing.append((it, why))
        if not missing:
            return item_list
        if len(missing) == len(item_list):
            # 整份列表都不在（例如拿 MVTec 默认值去跑 VisA）：不逐条刷屏
            head = ', '.join(it for it, _ in missing[:5])
            print_fn(f"[item_list] 警告：请求的 {len(item_list)} 个类别在 data_path "
                     f"下**全部**不可用（例如 {head}, ...），原因：{missing[0][1]}。")
        else:
            print_fn(f"[item_list] 警告：请求的 {len(item_list)} 个类别中有 "
                     f"{len(missing)} 个在 data_path 下不可用：")
            for it, why in missing:
                print_fn(f"[item_list]   - {it}: {why}")
        print_fn(f"[item_list] data_path = {data_path}")
    else:
        print_fn("[item_list] 未显式指定（或指定为 auto），将从 data_path 自动探测类别。")

    found = probe_item_list(data_path, layout)
    if not found:
        expect = ("<data_path>/<item>/train/<good>/*.jpg" if layout == 'folder'
                  else "<data_path>/realiad_1024/<item> + realiad_jsons/.../<item>.json")
        raise SystemExit(
            f"[item_list] 在 data_path={data_path} 下探测不到任何可用类别。\n"
            f"           期望布局: {expect}\n"
            f"           请确认 --data_path 指向数据集根目录，"
            f"或显式传类别：--item_list candle capsules ..."
        )

    print_fn(f"[item_list] 自动探测到 {len(found)} 个类别: {found}")
    if not auto:
        print_fn("[item_list] 已改用探测结果（请求的列表未全部命中）。"
                 "如不符合预期，请显式传正确的 --item_list。")
    return found
