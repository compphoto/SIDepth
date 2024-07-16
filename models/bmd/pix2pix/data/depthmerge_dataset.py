import os
import random

import cv2
import h5py
import numpy as np
import torch
import skimage.measure

from dependencies.bmd.pix2pix.data.base_dataset import BaseDataset
from dependencies.bmd.pix2pix.data.image_folder import (make_dataset,
                                                        make_dataset_all)
from dependencies.bmd.pix2pix.util.guidedfilter import GuidedFilter
from dependencies.monodepth.dataset.hypersim_o2m import match_scalar
from dependencies.monodepth.utils.data_utils import compute_rx
from PIL import Image


def normalize(img):
    img = img * 2
    img = img - 1
    return img


def normalize01(img):
    img = img - img.min()
    img = img / img.max()
    return img

def rgb2gray(rgb):
    # Converts rgb to gray
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def resizewithpool(img, size):
    i_size = img.shape[0]
    n = int(np.floor(i_size/size))

    out = skimage.measure.block_reduce(img, (n, n), np.max)
    return out


def compute_rx(img, basesize, confidence=0.1, scale_threshold=3, whole_size_threshold=3000):
    # Returns the R_x resolution described in section 5 of the main paper.

    # Parameters:
    #    img :input rgb image
    #    basesize : size the dilation kernel which is equal to receptive field of the network.
    #    confidence: value of x in R_x; allowed percentage of pixels that are not getting any contextual cue.
    #    scale_threshold: maximum allowed upscaling on the input image ; it has been set to 3.
    #    whole_size_threshold: maximum allowed resolution. (R_max from section 6 of the main paper)

    # Returns:
    #    outputsize_scale*speed_scale :The computed R_x resolution
    #    patch_scale: K parameter from section 6 of the paper

    # speed scale parameter is to process every image in a smaller size to accelerate the R_x resolution search
    speed_scale = 32
    image_dim = int(min(img.shape[0:2]))

    gray = rgb2gray(img)
    grad = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)) + \
        np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    grad = cv2.resize(grad, (image_dim, image_dim), cv2.INTER_AREA)

    # thresholding the gradient map to generate the edge-map as a proxy of the contextual cues
    m = grad.min()
    M = grad.max()
    middle = m + (0.4 * (M - m))
    grad[grad < middle] = 0
    grad[grad >= middle] = 1

    # dilation kernel with size of the receptive field
    kernel = np.ones((int(basesize/speed_scale),
                     int(basesize/speed_scale)), np.float)
    # dilation kernel with size of the a quarter of receptive field used to compute k
    # as described in section 6 of main paper
    kernel2 = np.ones((int(basesize / (4*speed_scale)),
                      int(basesize / (4*speed_scale))), np.float)

    # Output resolution limit set by the whole_size_threshold and scale_threshold.
    threshold = min(whole_size_threshold, scale_threshold * max(img.shape[:2]))

    outputsize_scale = basesize / speed_scale
    for p_size in range(int(basesize/speed_scale), int(threshold/speed_scale), int(basesize / (2*speed_scale))):
        grad_resized = resizewithpool(grad, p_size)
        grad_resized = cv2.resize(
            grad_resized, (p_size, p_size), cv2.INTER_NEAREST)
        grad_resized[grad_resized >= 0.5] = 1
        grad_resized[grad_resized < 0.5] = 0

        dilated = cv2.dilate(grad_resized, kernel, iterations=1)
        meanvalue = (1-dilated).mean()
        if meanvalue > confidence:
            break
        else:
            outputsize_scale = p_size

    grad_region = cv2.dilate(grad_resized, kernel2, iterations=1)
    patch_scale = grad_region.mean()

    return int(outputsize_scale*speed_scale), patch_scale


def convert_distance_to_depth(npyDistance, fltFocal=886.81, intWidth=1024, intHeight=768):
    # fixing the ground-truth hypersim depth from distance to depth
    npyImageplaneX = np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth).reshape(
        1, intWidth).repeat(intHeight, 0).astype(np.float32)[:, :, None]
    npyImageplaneY = np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5,
                                 intHeight).reshape(intHeight, 1).repeat(intWidth, 1).astype(np.float32)[:, :, None]
    npyImageplaneZ = np.full([intHeight, intWidth, 1], fltFocal, np.float32)
    npyImageplane = np.concatenate(
        [npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)

    npyDepth = npyDistance / np.linalg.norm(npyImageplane, 2, 2) * fltFocal

    return npyDepth


def read_hdf5(path):
    if path.endswith('.hdf5'):
        with h5py.File(path, "r") as f:
            data = np.asarray(f['dataset'])
        data = data.astype(np.float32)
    return data


def shuffle_data(rgb, gt, base, high, mask, dset, semantics, normals):
    print(f"Shuffling data")
    print(rgb[0], gt[0], base[0], high[0], mask[0], dset[0], semantics[0], normals[0])
    data = list(zip(rgb, gt, base, high, mask, dset, semantics, normals))
    random.shuffle(data)

    if dset[0] == 'driving_stereo':
        split_value = int(len(data) * 0.2)
        data = data[:split_value]

    rgb, gt, base, high, mask, dset, semantics, normals = zip(*data)
    print(rgb[0], gt[0], base[0], high[0], mask[0], dset[0], semantics[0], normals[0])
    return rgb, gt, base, high, mask, dset, semantics, normals


class DepthMergeDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        # original data reading method
        # self.dir_outer = os.path.join(opt.dataroot, opt.phase, 'base_depth')
        # self.dir_inner = os.path.join(opt.dataroot, opt.phase, 'high_depth')
        # if opt.metric:
        #     self.dir_gtfake = os.path.join(opt.dataroot, opt.phase, 'depth')
        # else:
        #     # self.dir_gtfake = os.path.join(opt.dataroot, opt.phase, 'double_depth')
        #     self.dir_gtfake = os.path.join(opt.dataroot, opt.phase, 'double_blurred_10')
        # self.dir_rgb = os.path.join(opt.dataroot, opt.phase, 'rgb')
        # self.dir_mask = os.path.join(opt.dataroot, opt.phase, 'mask')
        # self.outer_paths = sorted(make_dataset(self.dir_outer))
        # self.inner_paths = sorted(make_dataset(self.dir_inner))
        # self.gtfake_paths = sorted(make_dataset(self.dir_gtfake))
        # self.rgb_paths = sorted(make_dataset(self.dir_rgb))
        # self.mask_paths = sorted(make_dataset(self.dir_mask))
        # ------------- v2
        self.outer_paths, self.inner_paths, self.gtfake_paths, self.rgb_paths, self.mask_paths, self.datasets, self.semantic_paths, self.normal_paths = [], [], [], [], [], [], [], []

        for dataset in opt.datasets:
            print(f"Processing {dataset}")
            outer_paths = sorted(make_dataset_all(opt.dataroot, dir='base_depth', dataset=dataset))
            inner_paths = sorted(make_dataset_all(opt.dataroot, dir='high_depth', dataset=dataset))
            gtfake_paths = sorted(make_dataset_all(opt.dataroot, dir='depth', dataset=dataset))
            rgb_paths = sorted(make_dataset_all(opt.dataroot, dir='rgb', dataset=dataset))
            mask_paths = sorted(make_dataset_all(opt.dataroot, dir='mask', dataset=dataset))
            semantic_paths = mask_paths

            normal_paths = [os.path.join('/'.join(path.split('/')[0:-1]),path.split('/')[-1][0:17],path.split('/')[-1][18:]).replace('rgb','normal') for path in rgb_paths]
            # caminfo_paths = [os.path.join('/'.join(path.split('/')[0:-1]),path.split('/')[-1][0:17],path.split('/')[-1][18:]).replace('rgb.png','point_info.json') for path in rgb_paths]

            dsets = [dataset for i in range(len(rgb_paths))]

            if dataset == 'hypersim_omnivision_ablation':
                semantic_paths = sorted(make_dataset_all(opt.dataroot, dir='semantic', dataset=dataset))

            self.outer_paths += outer_paths
            self.inner_paths += inner_paths
            self.gtfake_paths += gtfake_paths
            self.rgb_paths += rgb_paths
            self.mask_paths += mask_paths
            self.datasets += dsets
            self.semantic_paths += semantic_paths
            self.normal_paths += normal_paths

        self.rgb_paths, self.gtfake_paths, self.outer_paths, self.inner_paths, self.mask_paths, self.datasets, self.semantic_paths, self.normal_paths = shuffle_data(
            self.rgb_paths, self.gtfake_paths, self.outer_paths, self.inner_paths, self.mask_paths, self.datasets, self.semantic_paths, self.normal_paths)

        self.dataset_size = len(self.outer_paths)
        if opt.phase == 'train':
            self.isTrain = True
        else:
            self.isTrain = False

        self.opt = opt

    def __getitem__(self, index):      
        image_path = self.rgb_paths[index]
        dset = self.datasets[index]
        normalize_coef = np.float32(2 ** 16) - 1

        try:
            data_rgb = Image.open(self.rgb_paths[index])
            data_rgb = np.array(data_rgb, dtype=np.float32)
            data_rgb = data_rgb / 255.0

            data_outer = Image.open(self.outer_paths[index])
            data_outer = np.array(data_outer, dtype=np.float32)
            data_outer = data_outer / normalize_coef

            data_inner = Image.open(self.inner_paths[index])
            data_inner = np.array(data_inner, dtype=np.float32)
            data_inner = data_inner / normalize_coef

            if self.isTrain:
                if dset in ['hypersim_omnivision', 'hypersim_omnivision_ablation']:
                    data_gtfake = Image.open(self.gtfake_paths[index])
                    data_gtfake = np.array(data_gtfake, dtype=np.float32)
                    data_gtfake = convert_distance_to_depth(npyDistance=data_gtfake)
                    data_mask = np.asarray(Image.open(self.mask_paths[index])).astype(np.bool8)

                    data_surfnorm = Image.open(self.normal_paths[index])
                    data_surfnorm = np.array(data_surfnorm, dtype=np.float32)
                    data_surfnorm = data_surfnorm / 255.0

                    # normalize the surface normal to unit vec
                    data_surfnorm = data_surfnorm - 0.5
                    data_surfnorm = data_surfnorm / np.linalg.norm(data_surfnorm, axis=2, keepdims=True)

                    focal_length=torch.Tensor([886.81])
                    principal_point=torch.Tensor([768//2,1024//2])

                elif dset == 'hypersim_original':
                    data_gtfake = read_hdf5(self.gtfake_paths[index])
                    data_gtfake = convert_distance_to_depth(npyDistance=data_gtfake)
                    data_mask = np.logical_not(np.logical_or(np.isnan(data_gtfake), np.isinf(data_gtfake))).astype(np.bool8)

                elif dset == 'tartan_air':
                    data_gtfake = np.load(self.gtfake_paths[index])
                    data_mask = np.less(data_gtfake, 10000)

                elif dset == 'sunrgbd':
                    data_gtfake = Image.open(self.gtfake_paths[index])
                    data_gtfake = np.array(data_gtfake, dtype=np.float32)
                    data_mask = np.greater(data_gtfake, 0)

                    focal_length=torch.Tensor([570.342224])
                    principal_point=torch.Tensor([231.0,291.0])

                    data_surfnorm = np.ones_like(data_rgb)
                    
                elif dset == 'driving_stereo':
                    data_gtfake = Image.open(self.gtfake_paths[index])
                    data_gtfake = np.array(data_gtfake, dtype=np.float32)
                    data_mask = np.greater(data_gtfake, 0)

                # setting unkonwn depth to max depth existing in the scene to stabalize converting depth to disparity
                data_gtfake[~data_mask] = np.nan
                max_gt = np.nanmax(data_gtfake)
                data_gtfake[~data_mask] = max_gt

                # depth to disparity
                data_gtfake = 1.0 / data_gtfake

                # setting the unkown values either to min or 0
                if self.opt.min_mask:
                    data_gtfake[~data_mask] = np.nanmin(data_gtfake[data_mask])
                else:
                    data_gtfake[~data_mask] = 0

                if dset == 'hypersim_omnivision_ablation':
                    data_semantic = np.asarray(
                        Image.open(self.semantic_paths[index]))[:, :, 0:3]
                    # [174, 199, 232]: wall
                    # [152, 223, 138]: floor
                    h, w = data_gtfake.shape
                    labels = [[174, 199, 232], [152, 223, 138]]
                    wall = (data_semantic[:, :, 0] == labels[0][0]) & (
                        data_semantic[:, :, 1] == labels[0][1]) & (data_semantic[:, :, 2] == labels[0][2])
                    floor = (data_semantic[:, :, 0] == labels[1][0]) & (
                        data_semantic[:, :, 1] == labels[1][1]) & (data_semantic[:, :, 2] == labels[1][2])
                    data_semantic = np.dstack((wall, floor))  # H x W x 2
                else:
                    data_semantic = np.zeros_like(data_gtfake)

            if self.opt.crop:
                h, w, _ = data_rgb.shape

                r_size, _ = compute_rx(img=data_rgb, basesize=384, confidence=0)
                if r_size <= 384:
                    r_size = 384
                
                min_cropsize = min(h,w) * 384 / r_size
                if np.argmin(data_rgb.shape[0:2]) == 0:
                    rand_crop_h = random.randint(int(min_cropsize), h)
                    rand_crop_w = rand_crop_h
                else:
                    rand_crop_w = random.randint(int(min_cropsize), w)
                    rand_crop_h = rand_crop_w

                # ensure multiple of 32
                rand_crop_h = rand_crop_h - rand_crop_h % 32
                rand_crop_w = rand_crop_w - rand_crop_w % 32

                rand_h = random.randint(0, h - rand_crop_h)
                rand_w = random.randint(0, w - rand_crop_w)

                rand_h_offset = rand_h + rand_crop_h
                rand_w_offset = rand_w + rand_crop_w

                data_rgb = data_rgb[rand_h: rand_h_offset,
                                    rand_w:rand_w_offset]
                data_inner = data_inner[rand_h: rand_h_offset,
                                        rand_w:rand_w_offset]
                data_outer = data_outer[rand_h: rand_h_offset,
                                        rand_w:rand_w_offset]
                

                if self.isTrain:
                    data_gtfake = data_gtfake[rand_h: rand_h_offset,rand_w:rand_w_offset]
                    data_mask = data_mask[rand_h: rand_h_offset,rand_w:rand_w_offset]
                    data_semantic = data_semantic[rand_h: rand_h_offset,rand_w:rand_w_offset]
                    data_surfnorm = data_surfnorm[rand_h: rand_h_offset,rand_w:rand_w_offset]
                    # update principal point after cropping
                    principal_point = principal_point - torch.Tensor([rand_h,rand_w])


            # normalize to [0,1] / needed due to cropping
            data_inner = normalize01(data_inner)
            data_outer = normalize01(data_outer)

            # match the scale of gt and inner (high) with outer (base)            
            if self.opt.fit_scalar:
                data_inner, high_scalar = match_scalar(source=data_inner, target=data_outer, 
                                                       mask=data_mask, min_percentile=80, max_percentile=100)
                if self.opt.isTrain:
                            data_gtfake, gt_scalar = match_scalar(source=data_gtfake, target=data_outer, 
                                                                  mask=data_mask, min_percentile=80, max_percentile=100)

            # resizing images
            scale = self.opt.img_size / data_rgb.shape[0]            

            dim_h = self.opt.img_size 
            dim_w = self.opt.img_size 

            data_rgb = cv2.resize(data_rgb, (dim_w,dim_h),interpolation=cv2.INTER_LINEAR)
            data_inner = cv2.resize(data_inner, (dim_w,dim_h),interpolation=cv2.INTER_LINEAR)
            data_outer = cv2.resize(data_outer, (dim_w,dim_h),interpolation=cv2.INTER_LINEAR)
        
            data_rgb = torch.from_numpy(data_rgb).permute(2, 0, 1)
            data_outer = torch.from_numpy(data_outer).unsqueeze(0)
            data_inner = torch.from_numpy(data_inner).unsqueeze(0)

            if self.isTrain:

                data_gtfake = cv2.resize(data_gtfake, (dim_w,dim_h),interpolation=cv2.INTER_LINEAR)
                data_mask = cv2.resize(data_mask.astype('uint8'), (dim_w,dim_h),interpolation=cv2.INTER_LINEAR).astype('bool8')
                data_semantic = cv2.resize(data_semantic, (dim_w,dim_h),interpolation=cv2.INTER_NEAREST)
                data_surfnorm = cv2.resize(data_surfnorm, (dim_w,dim_h),interpolation=cv2.INTER_LINEAR)

                # update focal length and principal point after rescaling
                # resize to opt.img_size
                focal_length = focal_length * scale
                principal_point = principal_point * scale

                data_gtfake = torch.from_numpy(data_gtfake).unsqueeze(0)
                data_mask = torch.from_numpy(data_mask).unsqueeze(0)
                data_surfnorm = torch.from_numpy(data_surfnorm).permute(2, 0, 1)

                data_semantic = torch.from_numpy(data_semantic)
                if data_semantic.ndim == 3:
                    data_semantic = data_semantic.permute(2, 0, 1)
                else:
                    data_semantic = torch.unsqueeze(data_semantic, 0)

            if self.opt.outer_activation == 'tanh':
                data_gtfake = normalize(data_gtfake)
                data_outer = normalize(data_outer)
                data_inner = normalize(data_inner)
                data_rgb = normalize(data_rgb)


        except Exception as e:
            # print(f"Exception occurred for {image_path} as {e}")
            h, w = self.opt.img_size, self.opt.img_size   
            data_inner = 0.25*torch.ones((1, h, w))
            data_outer = 0.75*torch.ones((1, h, w))
            data_gtfake = 0.5*torch.ones((1, h, w))
            data_rgb = 0.3*torch.ones((3, h, w))
            data_mask = torch.ones((1, h, w))
            data_semantic = torch.ones((1, h, w))
            data_surfnorm = 1/3*torch.ones((3, h, w))
            focal_length = 512*torch.ones((1))
            principal_point = 256*torch.ones((2))

        if self.isTrain:
            return {'data_inner': data_inner, 'data_outer': data_outer,
                    'data_gtfake': data_gtfake, 'image_path': image_path,
                    'data_rgb': data_rgb, 'data_mask': data_mask.float(),
                    'data_semantic': data_semantic.float(),
                    'data_surfnorm': data_surfnorm, 'dset': dset, 'focal_length': focal_length, 'principal_point': principal_point}
        else:
            return {'data_inner': data_inner, 'data_outer': data_outer,
                    'image_path': image_path, 'data_rgb': data_rgb, 'dset': dset}

    def __len__(self):
        """Return the total number of images."""
        return self.dataset_size
