"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import os
from os.path import join as pjoin

import torch.utils.data as data
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset_all(root, dir, dataset):
    images = []
    if 'hypersim' in dataset:
        root = pjoin(root, 'hypersim')
        if dataset == 'hypersim_omnivision':
            root = pjoin(root, 'train', dir)
            images = sorted([pjoin(root, img)
                            for img in os.listdir(root) if os.path.isfile(pjoin(root, img))])
        elif dataset == 'hypersim_omnivision_ablation':
            root = pjoin(root, 'ablation', dir)
            images = sorted([pjoin(root, img)
                            for img in os.listdir(root) if os.path.isfile(pjoin(root, img))])
        else:
            root = pjoin(root, 'download_data/data')
            scenes = sorted([pjoin(root, fold, 'images')
                            for fold in os.listdir(root) if os.path.isdir(pjoin(root, fold))])
            for scene in scenes:
                if dir in ['rgb', 'base_depth', 'high_depth']:
                    preview_folders = sorted(
                        [pjoin(scene, fold) for fold in os.listdir(scene) if 'preview' in fold])
                else:
                    preview_folders = sorted(
                        [pjoin(scene, fold) for fold in os.listdir(scene) if 'geometry' in fold])

                for preview_folder in preview_folders:
                    if dir == 'rgb':
                        imgs = sorted([pjoin(preview_folder, img) for img in os.listdir(
                            preview_folder) if img.endswith('.jpg')])
                    elif dir == 'base_depth':
                        preview_folder = pjoin(
                            preview_folder, 'b_midas_v2_f_midas_v2/double_estimation')
                        imgs = sorted([pjoin(preview_folder, img) for img in os.listdir(
                            preview_folder) if img.endswith('base.png')])
                    elif dir == 'high_depth':
                        preview_folder = pjoin(
                            preview_folder, 'b_midas_v2_f_midas_v2/double_estimation')
                        imgs = sorted([pjoin(preview_folder, img) for img in os.listdir(
                            preview_folder) if img.endswith('high.png')])
                    elif dir in ['depth', 'mask']:
                        imgs = sorted([pjoin(preview_folder, img) for img in os.listdir(
                            preview_folder) if img.endswith('.hdf5')])
                    images += imgs

    elif dataset == 'tartan_air':
        root = pjoin(root, dataset, 'data')
        scenes = sorted([pjoin(root, fold)
                        for fold in os.listdir(root) if os.path.isdir(pjoin(root, fold))])
        for scene in scenes:
            scene = pjoin(scene, 'Easy')
            shots = sorted([pjoin(scene, fold)
                           for fold in os.listdir(scene) if os.path.isdir(pjoin(scene, fold))])

            for shot in shots:
                if dir in ['rgb', 'base_depth', 'high_depth']:
                    shot = pjoin(shot, 'image_left')

                    if dir == 'rgb':
                        imgs = sorted([pjoin(shot, img) for img in os.listdir(
                            shot) if img.endswith('.png')])
                    elif dir == 'base_depth':
                        shot = pjoin(
                            shot, 'b_midas_v2_f_midas_v2/double_estimation')
                        imgs = sorted([pjoin(shot, img) for img in os.listdir(
                            shot) if img.endswith('base.png')])
                    elif dir == 'high_depth':
                        shot = pjoin(
                            shot, 'b_midas_v2_f_midas_v2/double_estimation')
                        imgs = sorted([pjoin(shot, img) for img in os.listdir(
                            shot) if img.endswith('high.png')])
                else:
                    shot = pjoin(shot, 'depth_left')
                    imgs = sorted([pjoin(shot, img)
                                  for img in os.listdir(shot) if img.endswith('npy')])
                images += imgs

    elif dataset == 'sunrgbd':
        root = pjoin(root, dataset, 'SUNRGBD', 'data')
        if dir == 'rgb':
            rgb_path = pjoin(root, 'rgb')
            images = sorted([pjoin(rgb_path, img)
                            for img in os.listdir(rgb_path)])
        elif dir in ['depth', 'mask']:
            depth_path = pjoin(root, 'depth')
            images = sorted([pjoin(depth_path, img)
                            for img in os.listdir(depth_path)])
        elif dir == 'base_depth':
            depth_path = pjoin(
                root, 'bmd/b_midas_v2_f_midas_v2/double_estimation')
            images = sorted([pjoin(depth_path, img) for img in os.listdir(
                depth_path) if img.endswith('base.png')])
        elif dir == 'high_depth':
            depth_path = pjoin(
                root, 'bmd/b_midas_v2_f_midas_v2/double_estimation')
            images = sorted([pjoin(depth_path, img) for img in os.listdir(
                depth_path) if img.endswith('high.png')])

        # duplicating images to increase its frequency in dataloader
        images = images * 5

    elif dataset == 'driving_stereo':
        root = pjoin(root, dataset)
        if dir == 'rgb':
            rgb_path = pjoin(root, 'rgb')
            images = sorted([pjoin(rgb_path, img)
                            for img in os.listdir(rgb_path)])
        elif dir in ['depth', 'mask']:
            depth_path = pjoin(root, 'depth')
            images = sorted([pjoin(depth_path, img)
                            for img in os.listdir(depth_path)])
        elif dir == 'base_depth':
            depth_path = pjoin(
                root, 'bmd/b_midas_v2_f_midas_v2/double_estimation')
            images = sorted([pjoin(depth_path, img) for img in os.listdir(
                depth_path) if img.endswith('base.png')])
        elif dir == 'high_depth':
            depth_path = pjoin(
                root, 'bmd/b_midas_v2_f_midas_v2/double_estimation')
            images = sorted([pjoin(depth_path, img) for img in os.listdir(
                depth_path) if img.endswith('high.png')])

    return images


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = pjoin(root, fname)
                images.append(path)
    # return images[:min(max_dataset_size, len(images))]
    return images


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
