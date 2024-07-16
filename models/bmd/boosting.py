import argparse
import os
import warnings
from operator import getitem
from os.path import join as pjoin
from time import time

import cv2
import numpy as np
import torch
# AdelaiDepth
from dependencies.bmd.lib.multi_depth_model_woauxi import RelDepthModel
from dependencies.bmd.lib.net_tools import strip_prefix_if_present
# MIDAS
from dependencies.bmd.midas import utils as midas_utils
from dependencies.bmd.midas.models.dpt_depth import DPTDepthModel
from dependencies.bmd.midas.models.midas_net import MidasNet
from dependencies.bmd.midas.models.transforms import (NormalizeImage,
                                                      PrepareForNet, Resize)
# PIX2PIX : MERGE NET
# from dependencies.bmd.pix2pix.options.test_options import TestOptions
from dependencies.bmd.pix2pix.models.pix2pix4depth_model import \
    Pix2Pix4DepthModel
# Load Pix2Pix model
# OUR
from dependencies.bmd.utils import (ImageandPatchs, ImageDataset,
                                    applyGridpatch, calculateprocessingres,
                                    generatemask, getGF_fromintegral,
                                    load_pix2pix, rgb2gray)
from torchvision.transforms import Compose, transforms
from tqdm import tqdm

warnings.simplefilter('ignore', np.RankWarning)

# select device
device = torch.device("cuda")
print("device: %s" % device)

# Global variables
pix2pixmodel = None
midasmodel = None
srlnet = None
leresmodel = None
factor = None
whole_size_threshold = 3000  # R_max from the paper
# Limit for the GPU (NVIDIA RTX 2080), can be adjusted
GPU_threshold = 1600 - 32

# MAIN PART OF OUR METHOD


def run_bmd(dataset, option):

    # Load merge network
    global pix2pixmodel
    pix2pixmodel = load_pix2pix()
    pix2pixmodel.eval()

    # Decide which depth estimation network to load
    if option.depth_net == "midas2":
        midas_model_path = "./dependencies/bmd/weights/midas/model-f6b98070.pt"
        global midasmodel
        midasmodel = MidasNet(midas_model_path, non_negative=True)
        midasmodel.to(device)
        midasmodel.eval()
    elif option.depth_net == "sgrnet":
        global srlnet
        srlnet = DepthNet.DepthNet()
        srlnet = torch.nn.DataParallel(srlnet, device_ids=[0]).cuda()
        checkpoint = torch.load('./dependencies/bmd/weights/sgr/model.pth.tar')
        srlnet.load_state_dict(checkpoint['state_dict'])
        srlnet.eval()
    elif option.depth_net == "leres":
        global leresmodel
        leres_model_path = "./dependencies/bmd/weights/leres/res101.pth"
        checkpoint = torch.load(leres_model_path)
        leresmodel = RelDepthModel(backbone='resnext101')
        leresmodel.load_state_dict(strip_prefix_if_present(checkpoint['depth_model'], "module."),
                                   strict=True)
        del checkpoint
        torch.cuda.empty_cache()
        leresmodel.to(device)
        leresmodel.eval()

    # Generating required directories
    result_dir = option.pseudo_label_dir
    os.makedirs(result_dir, exist_ok=True)

    if option.save_whole_estimation:
        whole_est_outputpath = pjoin(
            option.pseudo_label_dir, 'whole_estimation')
        os.makedirs(whole_est_outputpath, exist_ok=True)

    if option.save_patches:
        patchped_est_outputpath = pjoin(
            option.pseudo_label_dir, "patch_estimation")
        os.makedirs(patchped_est_outputpath, exist_ok=True)

    # Generate mask used to smoothly blend the local pathc estimations to the base estimate.
    # It is arbitrarily large to avoid artifacts during rescaling for each crop.
    mask_org = generatemask((3000, 3000))
    mask = mask_org.copy()

    # Value x of R_x defined in the section 5 of the main paper.
    r_threshold_value = 0.2
    if option.estimation_type == "R0":
        r_threshold_value = 0
    elif option.estimation_type == "R20":
        r_threshold_value = 0.2

    # Go through all images in input directory
    print("start processing")
    for image_ind, images in tqdm(enumerate(dataset), total=len(dataset)):
        # print('processing image', image_ind, ':', images.name)

        try:
            # Load image from dataset
            img = images.rgb_image
            input_resolution = img.shape

            scale_threshold = 3  # Allows up-scaling with a scale up to 3

            # Find the best input resolution R-x. The resolution search described in section 5-double estimation of the main paper and section B of the
            # supplementary material.

            whole_image_optimal_size, patch_scale = calculateprocessingres(img, option.net_receptive_field_size,
                                                                           r_threshold_value, scale_threshold,
                                                                           whole_size_threshold)

            # print('\t wholeImage being processed in :', whole_image_optimal_size)

            # Generate the base estimate using the double estimation.
            whole_estimate = doubleestimate(img, option.net_receptive_field_size, whole_image_optimal_size,
                                            option.pix2pix_size, option.depth_net)
        except Exception as e:
            print(e)
            continue

        if option.depth_type in ["merge_estimation", "complete_estimation"]:
            if option.estimation_type in ["R0", "R20"]:
                # Save even the double estimation

                # clamp depth to 0
                whole_estimate_double = whole_estimate.copy()
                whole_estimate_double[whole_estimate_double < 0] = 0

                path = pjoin(option.double_estimation_dir, images.name)
                if option.output_resolution == 1:
                    midas_utils.write_depth(path, cv2.resize(whole_estimate_double, (input_resolution[1], input_resolution[0]),
                                                             interpolation=cv2.INTER_CUBIC), bits=2, colored=False, save_raw=False)
                else:
                    midas_utils.write_depth(
                        path, whole_estimate_double, bits=2, colored=False)

                if option.depth_type == "merge_estimation":
                    continue

        # Output double estimation if required
        if option.save_whole_estimation:
            path = pjoin(whole_est_outputpath, images.name)
            if option.output_resolution == 1:
                midas_utils.write_depth(path,
                                        cv2.resize(whole_estimate_double, (input_resolution[1], input_resolution[0]),
                                                   interpolation=cv2.INTER_CUBIC), bits=2,
                                        colored=option.colorize_results)
            else:
                midas_utils.write_depth(
                    path, whole_estimate_double, bits=2, colored=option.colorize_results)

        # Compute the multiplier described in section 6 of the main paper to make sure our initial patch can select
        # small high-density regions of the image.
        global factor
        factor = max(min(1, 4 * patch_scale *
                     whole_image_optimal_size / whole_size_threshold), 0.2)
        # print('Adjust factor is:', 1/factor)

        # Check if Local boosting is beneficial.
        if option.max_resolution < whole_image_optimal_size:
            print("No Local boosting. Specified Max Res is smaller than R20")
            path = pjoin(result_dir, images.name)
            if option.output_resolution == 1:
                midas_utils.write_depth(path,
                                        cv2.resize(whole_estimate,
                                                   (input_resolution[1],
                                                    input_resolution[0]),
                                                   interpolation=cv2.INTER_CUBIC), bits=2,
                                        colored=option.colorize_results)
            else:
                midas_utils.write_depth(path, whole_estimate, bits=2,
                                        colored=option.colorize_results)
            continue

        # Compute the default target resolution.
        if img.shape[0] > img.shape[1]:
            a = 2 * whole_image_optimal_size
            b = round(2 * whole_image_optimal_size *
                      img.shape[1] / img.shape[0])
        else:
            a = round(2 * whole_image_optimal_size *
                      img.shape[0] / img.shape[1])
            b = 2 * whole_image_optimal_size
        b = int(round(b / factor))
        a = int(round(a / factor))

        # recompute a, b and saturate to max res.
        if max(a, b) > option.max_resolution:
            # print('Default Res is higher than max-res: Reducing final resolution')
            if img.shape[0] > img.shape[1]:
                a = option.max_resolution
                b = round(option.max_resolution * img.shape[1] / img.shape[0])
            else:
                a = round(option.max_resolution * img.shape[0] / img.shape[1])
                b = option.max_resolution
            b = int(b)
            a = int(a)

        img = cv2.resize(img, (b, a), interpolation=cv2.INTER_CUBIC)

        # Extract selected patches for local refinement
        base_size = option.net_receptive_field_size*2
        patchset = generatepatchs(img, base_size)

        # print('Target resolution: ', img.shape)

        # Computing a scale in case user prompted to generate the results as the same resolution of the input.
        # Notice that our method output resolution is independent of the input resolution and this parameter will only
        # enable a scaling operation during the local patch merge implementation to generate results with the same resolution
        # as the input.
        if option.output_resolution == 1:
            mergein_scale = input_resolution[0] / img.shape[0]
            # print('Dynamicly change merged-in resolution; scale:', mergein_scale)
        else:
            mergein_scale = 1

        imageandpatchs = ImageandPatchs(
            images.name, patchset, img, mergein_scale)
        whole_estimate_resized = cv2.resize(whole_estimate, (round(img.shape[1]*mergein_scale),
                                            round(img.shape[0]*mergein_scale)), interpolation=cv2.INTER_CUBIC)
        imageandpatchs.set_base_estimate(whole_estimate_resized.copy())
        imageandpatchs.set_updated_estimate(whole_estimate_resized.copy())

        # print('\t Resulted depthmap res will be :',
        #       whole_estimate_resized.shape[:2])
        # print('patchs to process: '+str(len(imageandpatchs)))

        # Enumerate through all patches, generate their estimations and refining the base estimate.
        for patch_ind in tqdm(range(len(imageandpatchs)), total=len(imageandpatchs)):

            # Get patch information
            patch = imageandpatchs[patch_ind]  # patch object
            patch_rgb = patch['patch_rgb']  # rgb patch
            # corresponding patch from base
            patch_whole_estimate_base = patch['patch_whole_estimate_base']
            rect = patch['rect']  # patch size and location
            patch_id = patch['id']  # patch ID
            # the original size from the unscaled input
            org_size = patch_whole_estimate_base.shape
            # print('\t processing patch', patch_ind, '|', rect)

            # We apply double estimation for patches. The high resolution value is fixed to twice the receptive
            # field size of the network for patches to accelerate the process.
            patch_estimation = doubleestimate(patch_rgb, option.net_receptive_field_size, option.patch_net_size,
                                              option.pix2pix_size, option.depth_net)

            # Output patch estimation if required
            if option.save_patches:
                path = pjoin(
                    patchped_est_outputpath, imageandpatchs.name + '_{:04}'.format(patch_id))
                midas_utils.write_depth(
                    path, patch_estimation, bits=2, colored=option.colorize_results)

            patch_estimation = cv2.resize(patch_estimation, (option.pix2pix_size, option.pix2pix_size),
                                          interpolation=cv2.INTER_CUBIC)

            patch_whole_estimate_base = cv2.resize(patch_whole_estimate_base, (option.pix2pix_size, option.pix2pix_size),
                                                   interpolation=cv2.INTER_CUBIC)

            # Merging the patch estimation into the base estimate using our merge network:
            # We feed the patch estimation and the same region from the updated base estimate to the merge network
            # to generate the target estimate for the corresponding region.
            pix2pixmodel.set_input(patch_whole_estimate_base, patch_estimation)

            # Run merging network
            pix2pixmodel.test()
            visuals = pix2pixmodel.get_current_visuals()

            prediction_mapped = visuals['fake_B']
            prediction_mapped = (prediction_mapped+1)/2
            prediction_mapped = prediction_mapped.squeeze().cpu().numpy()

            mapped = prediction_mapped

            # We use a simple linear polynomial to make sure the result of the merge network would match the values of
            # base estimate
            p_coef = np.polyfit(mapped.reshape(-1),
                                patch_whole_estimate_base.reshape(-1), deg=1)
            merged = np.polyval(p_coef, mapped.reshape(-1)
                                ).reshape(mapped.shape)

            merged = cv2.resize(
                merged, (org_size[1], org_size[0]), interpolation=cv2.INTER_CUBIC)

            # Get patch size and location
            w1 = rect[0]
            h1 = rect[1]
            w2 = w1 + rect[2]
            h2 = h1 + rect[3]

            # To speed up the implementation, we only generate the Gaussian mask once with a sufficiently large size
            # and resize it to our needed size while merging the patches.
            if mask.shape != org_size:
                mask = cv2.resize(
                    mask_org, (org_size[1], org_size[0]), interpolation=cv2.INTER_LINEAR)

            tobemergedto = imageandpatchs.estimation_updated_image

            # Update the whole estimation:
            # We use a simple Gaussian mask to blend the merged patch region with the base estimate to ensure seamless
            # blending at the boundaries of the patch region.
            tobemergedto[h1:h2, w1:w2] = np.multiply(
                tobemergedto[h1:h2, w1:w2], 1 - mask) + np.multiply(merged, mask)
            imageandpatchs.set_updated_estimate(tobemergedto)

        # Output the result
        path = pjoin(result_dir, imageandpatchs.name)

        estimated_updated_image = imageandpatchs.estimation_updated_image

        # clipping the depth to the range of [0,1]
        estimated_updated_image[estimated_updated_image < 0] = 0

        if option.output_resolution == 1:
            midas_utils.write_depth(path,
                                    cv2.resize(estimated_updated_image,
                                               (input_resolution[1],
                                                input_resolution[0]),
                                               interpolation=cv2.INTER_CUBIC), bits=2, colored=option.colorize_results, save_raw=True)
        else:
            midas_utils.write_depth(
                path, estimated_updated_image, bits=2, colored=option.colorize_results, save_raw=True)

    print("finished")

# Generating local patches to perform the local refinement described in section 6 of the main paper.


def generatepatchs(img, base_size):

    # Compute the gradients as a proxy of the contextual cues.
    img_gray = rgb2gray(img)
    whole_grad = np.abs(cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)) +\
        np.abs(cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3))

    threshold = whole_grad[whole_grad > 0].mean()
    whole_grad[whole_grad < threshold] = 0

    # We use the integral image to speed-up the evaluation of the amount of gradients for each patch.
    gf = whole_grad.sum()/len(whole_grad.reshape(-1))
    grad_integral_image = cv2.integral(whole_grad)

    # Variables are selected such that the initial patch size would be the receptive field size
    # and the stride is set to 1/3 of the receptive field size.
    blsize = int(round(base_size/2))
    stride = int(round(blsize*0.75))

    # Get initial Grid
    patch_bound_list = applyGridpatch(blsize, stride, img, [0, 0, 0, 0])

    # Refine initial Grid of patches by discarding the flat (in terms of gradients of the rgb image) ones. Refine
    # each patch size to ensure that there will be enough depth cues for the network to generate a consistent depth map.
    # print("Selecting patchs ...")
    patch_bound_list = adaptiveselection(
        grad_integral_image, patch_bound_list, gf)

    # Sort the patch list to make sure the merging operation will be done with the correct order: starting from biggest
    # patch
    patchset = sorted(patch_bound_list.items(),
                      key=lambda x: getitem(x[1], 'size'), reverse=True)
    return patchset


# Adaptively select patches
def adaptiveselection(integral_grad, patch_bound_list, gf):
    patchlist = {}
    count = 0
    height, width = integral_grad.shape

    search_step = int(32/factor)

    # Go through all patches
    for c in range(len(patch_bound_list)):
        # Get patch
        bbox = patch_bound_list[str(c)]['rect']

        # Compute the amount of gradients present in the patch from the integral image.
        cgf = getGF_fromintegral(integral_grad, bbox)/(bbox[2]*bbox[3])

        # Check if patching is beneficial by comparing the gradient density of the patch to
        # the gradient density of the whole image
        if cgf >= gf:
            bbox_test = bbox.copy()
            patchlist[str(count)] = {}

            # Enlarge each patch until the gradient density of the patch is equal
            # to the whole image gradient density
            while True:

                bbox_test[0] = bbox_test[0] - int(search_step/2)
                bbox_test[1] = bbox_test[1] - int(search_step/2)

                bbox_test[2] = bbox_test[2] + search_step
                bbox_test[3] = bbox_test[3] + search_step

                # Check if we are still within the image
                if bbox_test[0] < 0 or bbox_test[1] < 0 or bbox_test[1] + bbox_test[3] >= height \
                        or bbox_test[0] + bbox_test[2] >= width:
                    break

                # Compare gradient density
                cgf = getGF_fromintegral(
                    integral_grad, bbox_test)/(bbox_test[2]*bbox_test[3])
                if cgf < gf:
                    break
                bbox = bbox_test.copy()

            # Add patch to selected patches
            patchlist[str(count)]['rect'] = bbox
            patchlist[str(count)]['size'] = bbox[2]
            count = count + 1

    # Return selected patches
    return patchlist


# Generate a double-input depth estimation
def doubleestimate(img, size1, size2, pix2pix_size, net_type):
    # Generate the low resolution estimation
    estimate1 = singleestimate(img, size1, net_type)
    # Resize to the inference size of merge network.
    estimate1 = cv2.resize(
        estimate1, (pix2pix_size, pix2pix_size), interpolation=cv2.INTER_CUBIC)

    # Generate the high resolution estimation
    estimate2 = singleestimate(img, size2, net_type)
    # Resize to the inference size of merge network.
    estimate2 = cv2.resize(
        estimate2, (pix2pix_size, pix2pix_size), interpolation=cv2.INTER_CUBIC)

    # Inference on the merge model
    pix2pixmodel.set_input(estimate1, estimate2)
    pix2pixmodel.test()
    visuals = pix2pixmodel.get_current_visuals()
    prediction_mapped = visuals['fake_B']
    prediction_mapped = (prediction_mapped+1)/2
    prediction_mapped = (prediction_mapped - torch.min(prediction_mapped)) / (
        torch.max(prediction_mapped) - torch.min(prediction_mapped))
    prediction_mapped = prediction_mapped.squeeze().cpu().numpy()

    return prediction_mapped


# Generate a single-input depth estimation
def singleestimate(img, msize, net_type):
    if msize > GPU_threshold:
        # print(" \t \t DEBUG| GPU THRESHOLD REACHED",
        #       msize, '--->', GPU_threshold)
        msize = GPU_threshold

    if net_type == "midas2":
        return estimatemidas(img, msize)
    elif net_type == "sgrnet":
        return estimatesrl(img, msize)
    elif net_type == "leres":
        return estimateleres(img, msize)


# Inference on SGRNet
def estimatesrl(img, msize):
    # SGRNet forward pass script adapted from https://github.com/KexianHust/Structure-Guided-Ranking-Loss
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_resized = cv2.resize(
        img, (msize, msize), interpolation=cv2.INTER_CUBIC).astype('float32')
    tensor_img = img_transform(img_resized)

    # Forward pass
    input_img = torch.autograd.Variable(
        tensor_img.cuda().unsqueeze(0), volatile=True)
    with torch.no_grad():
        output = srlnet(input_img)

    # Normalization
    depth = output.squeeze().cpu().data.numpy()
    min_d, max_d = depth.min(), depth.max()
    depth_norm = (depth - min_d) / (max_d - min_d)

    depth_norm = cv2.resize(
        depth_norm, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    return depth_norm

# Inference on MiDas-v2


def estimatemidas(img, msize):
    # MiDas -v2 forward pass script adapted from https://github.com/intel-isl/MiDaS/tree/v2

    transform = Compose(
        [
            Resize(
                msize,
                msize,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    img_input = transform({"image": img})["image"]

    # Forward pass
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        prediction = midasmodel.forward(sample)

    prediction = prediction.squeeze().cpu().numpy()
    prediction = cv2.resize(
        prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Normalization
    depth_min = prediction.min()
    depth_max = prediction.max()

    if depth_max - depth_min > np.finfo("float").eps:
        prediction = (prediction - depth_min) / (depth_max - depth_min)
    else:
        prediction = 0

    return prediction


def scale_torch(img):
    """
    Scale the image and output it in torch.tensor.
    :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        img = transform(img.astype(np.float32))
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    return img

# Inference on LeRes


def estimateleres(img, msize):
    # LeReS forward pass script adapted from https://github.com/aim-uofa/AdelaiDepth/tree/main/LeReS

    rgb_c = img[:, :, ::-1].copy()
    A_resize = cv2.resize(rgb_c, (msize, msize))
    img_torch = scale_torch(A_resize)[None, :, :, :]

    # Forward pass
    with torch.no_grad():
        prediction = leresmodel.inference(img_torch)

    prediction = prediction.squeeze().cpu().numpy()
    prediction = cv2.resize(
        prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    return prediction


if __name__ == "__main__":
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, required=True, help='input files directory '
                                                                    'Images can be .png .jpg .tiff')
    parser.add_argument('--pseudo_labels', type=str, required=True, help='result dir. result depth will be png.'
                        ' vides are JMPG as avi')
    parser.add_argument('--save_patches', action="store_true", default=False, required=False,
                        help='Activate to save the patch estimations')
    parser.add_argument('--save_whole_estimation', action="store_true", required=False, default=False,
                        help='Activate to save the base estimations')
    parser.add_argument('--output_resolution', type=int, default=1, required=False,
                        help='0 for results in maximum resolution 1 for resize to input size')
    parser.add_argument('--net_receptive_field_size', type=int,
                        required=False)  # Do not set the value here
    parser.add_argument('--pix2pix_size', type=int,
                        default=1024, required=False)  # Do not change it
    parser.add_argument('--depth_net', type=str, choices=["midas2", "leres", "sgrnet"], required=False,
                        help='use to select different base depth networks 0:midas 1:strurturedRL 2:LeRes')
    parser.add_argument('--colorize_results', action='store_true')
    parser.add_argument('--estimation_type', type=str,
                        default="final", choices=["R0", "R20", "final"])
    parser.add_argument('--max_resolution', type=float, default=np.inf)

    # Check for required input
    option_, _ = parser.parse_known_args()
    print(option_)

    if option_.depth_net == "sgrnet":
        from structuredrl.models import DepthNet

    # Setting each networks receptive field and setting the patch estimation resolution to twice the receptive
    # field size to speed up the local refinement as described in the section 6 of the main paper.
    if option_.depth_net == "midas2":
        option_.net_receptive_field_size = 384
        option_.patch_net_size = 2 * option_.net_receptive_field_size
    elif option_.depth_net in ["sgrnet", "leres"]:
        option_.net_receptive_field_size = 448
        option_.patch_net_size = 2 * option_.net_receptive_field_size
    else:
        assert False, 'depthNet can only be 0,1 or 2'

    # Create dataset from input images
    dataset_ = ImageDataset(option_.data_dir, 'test')

    # Run pipeline
    run_bmd(dataset_, option_)
