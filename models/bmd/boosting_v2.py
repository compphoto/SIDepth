import argparse
import os
import warnings
from operator import getitem
from os.path import join as pjoin

import cv2
import numpy as np
import torch
from dependencies.bmd.midas import utils as midas_utils
from dependencies.bmd.model_utils import get_prediction, load_model
from dependencies.bmd.utils import (ImageandPatchs, ImageDataset,
                                    applyGridpatch, calculateprocessingres,
                                    generatemask, getGF_fromintegral,
                                    load_pix2pix, rgb2gray, rmse, to_pix2pix)
from tqdm import tqdm

warnings.simplefilter('ignore', np.RankWarning)

# select device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'


class BoostDepth():
    def __init__(self, option):
        self.option = option

        # Create dataset from input images
        self.dataset = ImageDataset(self.option)

        # Limit for the GPU (NVIDIA RTX 2080), can be adjusted
        self.GPU_threshold = 1600 - 32

        self.whole_size_threshold = 3000  # R_max from the paper
        self.factor = None

    def get_merge_estimate(self, base_estimate, high_res_estimate, rgb=None):
        self.pix2pixmodel.set_input(
            base_estimate, high_res_estimate, rgb=rgb, eval=True)
        self.pix2pixmodel.test(eval=True)
        visuals = self.pix2pixmodel.get_current_visuals()
        prediction_mapped = visuals['fake_B']
        prediction_mapped = (prediction_mapped + 1) / 2
        prediction_mapped = (prediction_mapped - torch.min(prediction_mapped)) / \
            (torch.max(prediction_mapped) - torch.min(prediction_mapped))
        prediction_mapped = prediction_mapped.squeeze().cpu().numpy()
        return prediction_mapped

    def get_single_estimate(self, img, msize, type):
        """Generate a single-input depth estimation
        """
        # setting the maximum size of the input image
        msize = self.GPU_threshold if msize > self.GPU_threshold else msize

        if type == 'base':
            output = get_prediction(name=self.option.base_net,
                                    net=self.base_model,
                                    img=img,
                                    msize=msize)
        else:
            output = get_prediction(name=self.option.high_res_net,
                                    net=self.high_res_model,
                                    img=img,
                                    msize=msize)

        return output

    def get_double_estimate(self, img, high_res_size, base_image=None):
        """Generate a double-input depth estimation

        Args:
            img ([type]): [description]
            high_res_size ([type]): [description]

        Returns:
            [type]: [description]
        """

        base_size = self.option.net_receptive_field_size
        pix2pix_size = self.option.pix2pix_size

        # Generate the low resolution estimation
        if base_image is not None:
            # print("Using base image")
            base_estimate = to_pix2pix(base_image, pix2pix_size)
        else:
            base_estimate = to_pix2pix(self.get_single_estimate(
                img, base_size, 'base'), pix2pix_size)

        # Generate the high resolution estimation
        high_res_estimate = to_pix2pix(self.get_single_estimate(
            img, high_res_size, 'high'), pix2pix_size)

        # Inference on the merge model
        if self.option.rgb_input:
            input_img = to_pix2pix(img, pix2pix_size)
            merged_output = self.get_merge_estimate(
                base_estimate, high_res_estimate, input_img)
        else:
            merged_output = self.get_merge_estimate(
                base_estimate, high_res_estimate)
        return merged_output, base_estimate, high_res_estimate

    def boost_depth(self):

        # setup models
        self.pix2pixmodel = load_pix2pix(merge_net=self.option.merge_name,
                                         checkpoint_dir=self.option.merge_ckpt,
                                         epoch=self.option.merge_epoch,
                                         outer_activation=self.option.activation)

        # get model
        print(f"Loading {self.option.base_net} as the base model")
        self.base_model = load_model(
            self.option.base_net, weights=self.option.base_weights, activation=self.option.midas_activation)

        # Incase both the models are same, but with different weights (e.g., base_net=midas_v2, high_res_net=midas_v2) and trained for different purpose
        # if self.option.base_net == self.option.high_res_net:
        #     self.high_res_model = self.base_model
        # else:

        print(f"Loading {self.option.high_res_net} as the high-res model")
        self.high_res_model = load_model(
            self.option.high_res_net, weights=self.option.high_res_weights, activation=self.option.midas_activation)

        self.base_model.eval()
        self.base_model.to(DEVICE)
        self.high_res_model.eval()
        self.high_res_model.to(DEVICE)

        # eval metrics
        rmse_loss = 0

        # Generating required directories
        self.result_dir = self.option.pseudo_label_dir
        os.makedirs(self.result_dir, exist_ok=True)

        if self.option.save_whole_estimation:
            whole_est_outputpath = pjoin(
                self.option.pseudo_label_dir, 'whole_estimation')
            os.makedirs(whole_est_outputpath, exist_ok=True)

        if self.option.save_patches:
            patchped_est_outputpath = pjoin(
                self.option.pseudo_label_dir, "patch_estimation")
            os.makedirs(patchped_est_outputpath, exist_ok=True)

        # Generate mask used to smoothly blend the local pathc estimations to the base estimate.
        # It is arbitrarily large to avoid artifacts during rescaling for each crop.
        mask_org = generatemask((3000, 3000))
        mask = mask_org.copy()

        # Value x of R_x defined in the section 5 of the main paper.
        r_threshold_value = 0.2
        if self.option.estimation_type == "R0":
            r_threshold_value = 0
        elif self.option.estimation_type == "R20":
            r_threshold_value = 0.2

        # Go through all images in input directory
        for image_ind, images in tqdm(enumerate(self.dataset), total=len(self.dataset)):
            try:
                # Load image from dataset
                img = images.rgb_image
                input_resolution = img.shape

                scale_threshold = 3  # Allows up-scaling with a scale up to 3

                # Find the best input resolution R-x. The resolution search described in section 5-double estimation of the main paper and section B of the
                # supplementary material.

                whole_image_optimal_size, patch_scale = calculateprocessingres(img, self.option.net_receptive_field_size,
                                                                               r_threshold_value, scale_threshold,
                                                                               self.whole_size_threshold)

                # print(
                #     f"Estimated R20 resolution for the image: {whole_image_optimal_size}")

                # Generate the base estimate using the double estimation.
                whole_estimate, base_estimate_v1, high_estimation_v1 = self.get_double_estimate(
                    img, whole_image_optimal_size, images.base_image)

            except Exception as e:
                print(e)
                continue

            if self.option.depth_type in ["double_estimation", "complete_estimation"]:
                if self.option.estimation_type in ["R0", "R20"]:

                    # clamp depth to 0
                    whole_estimate_double = whole_estimate.copy()
                    whole_estimate_double[whole_estimate_double < 0] = 0

                    base_estimate_v1 = base_estimate_v1.copy()
                    base_estimate_v1[base_estimate_v1 < 0] = 0

                    high_estimation_v1 = high_estimation_v1.copy()
                    high_estimation_v1[high_estimation_v1 < 0] = 0

                    path = pjoin(
                        self.option.double_estimation_dir, images.name)
                    if self.option.output_resolution == 1:
                        midas_utils.write_depth(path, cv2.resize(whole_estimate_double, (input_resolution[1], input_resolution[0]),
                                                                 interpolation=cv2.INTER_NEAREST), bits=2, colored=self.option.colorize_results, save_raw=self.option.save_raw, img=img, save_pcd=self.option.save_pcd)

                        path = pjoin(
                            self.option.double_estimation_dir, images.name + '_base')
                        midas_utils.write_depth(path, cv2.resize(base_estimate_v1, (input_resolution[1], input_resolution[0]),
                                                                 interpolation=cv2.INTER_NEAREST), bits=2, colored=self.option.colorize_results, save_raw=False, img=img, save_pcd=self.option.save_pcd)
                        # path = pjoin(
                        #     self.option.double_estimation_dir, images.name)
                        # midas_utils.write_depth(path, cv2.resize(base_estimate_v1, (input_resolution[1], input_resolution[0]),
                        #                                          interpolation=cv2.INTER_NEAREST), bits=2, colored=self.option.colorize_results, save_raw=self.option.save_raw, img=img, save_pcd=self.option.save_pcd)
                        path = pjoin(
                            self.option.double_estimation_dir, images.name + '_high')
                        midas_utils.write_depth(path, cv2.resize(high_estimation_v1, (input_resolution[1], input_resolution[0]),
                                                                 interpolation=cv2.INTER_NEAREST), bits=2, colored=self.option.colorize_results, save_raw=False, img=img, save_pcd=self.option.save_pcd)
                    else:
                        midas_utils.write_depth(
                            path, whole_estimate_double, bits=2, colored=False)

                        path = pjoin(
                            self.option.double_estimation_dir, images.name + '_base')
                        midas_utils.write_depth(
                            path, base_estimate_v1, bits=2, colored=False)

                        path = pjoin(
                            self.option.double_estimation_dir, images.name + '_high')
                        midas_utils.write_depth(
                            path, high_estimation_v1, bits=2, colored=False)

                    if self.option.depth_type == "double_estimation":
                        continue

            # Output double estimation if required
            if self.option.save_whole_estimation:
                path = pjoin(whole_est_outputpath, images.name)
                if self.option.output_resolution == 1:
                    midas_utils.write_depth(path,
                                            cv2.resize(whole_estimate_double, (input_resolution[1], input_resolution[0]),
                                                       interpolation=cv2.INTER_CUBIC), bits=2,
                                            colored=self.option.colorize_results)
                else:
                    midas_utils.write_depth(
                        path, whole_estimate_double, bits=2, colored=self.option.colorize_results)

            # Compute the multiplier described in section 6 of the main paper to make sure our initial patch can select
            # small high-density regions of the image.
            global factor
            factor = max(min(1, 4 * patch_scale *
                             whole_image_optimal_size / self.whole_size_threshold), 0.2)
            print('Adjust factor is:', 1/factor)

            # Check if Local boosting is beneficial.
            if self.option.max_resolution < whole_image_optimal_size:
                print("No Local boosting. Specified Max Res is smaller than R20")
                path = pjoin(self.result_dir, images.name)
                if self.option.output_resolution == 1:
                    midas_utils.write_depth(path,
                                            cv2.resize(whole_estimate,
                                                       (input_resolution[1],
                                                        input_resolution[0]),
                                                       interpolation=cv2.INTER_CUBIC), bits=2,
                                            colored=self.option.colorize_results)
                else:
                    midas_utils.write_depth(path, whole_estimate, bits=2,
                                            colored=self.option.colorize_results)
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
            if max(a, b) > self.option.max_resolution:
                # print('Default Res is higher than max-res: Reducing final resolution')
                if img.shape[0] > img.shape[1]:
                    a = self.option.max_resolution
                    b = round(self.option.max_resolution *
                              img.shape[1] / img.shape[0])
                else:
                    a = round(self.option.max_resolution *
                              img.shape[0] / img.shape[1])
                    b = self.option.max_resolution
                b = int(b)
                a = int(a)

            print(f"Whole image optimal size (h,w ): {a, b}")

            img = cv2.resize(img, (b, a), interpolation=cv2.INTER_CUBIC)

            # Extract selected patches for local refinement
            base_size = self.option.net_receptive_field_size*2
            patchset = self.generate_patchs(img, base_size)

            # Computing a scale in case user prompted to generate the results as the same resolution of the input.
            # Notice that our method output resolution is independent of the input resolution and this parameter will only
            # enable a scaling operation during the local patch merge implementation to generate results with the same resolution
            # as the input.
            if self.option.output_resolution == 1:
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
                patch_estimation, _, _ = self.get_double_estimate(
                    patch_rgb, high_res_size=self.option.patch_net_size)

                # Output patch estimation if required
                if self.option.save_patches:
                    path = pjoin(
                        patchped_est_outputpath, imageandpatchs.name + '_{:04}'.format(patch_id))
                    midas_utils.write_depth(
                        path, patch_estimation, bits=2, colored=self.option.colorize_results)

                # patch_estimation = cv2.resize(patch_estimation, (self.option.pix2pix_size, self.option.pix2pix_size),
                #                               interpolation=cv2.INTER_CUBIC)

                # patch_whole_estimate_base = cv2.resize(patch_whole_estimate_base, (self.option.pix2pix_size, self.option.pix2pix_size),
                #                                        interpolation=cv2.INTER_CUBIC)

                patch_estimation = to_pix2pix(
                    patch_estimation, self.option.pix2pix_size)
                patch_whole_estimate_base = to_pix2pix(
                    patch_whole_estimate_base, self.option.pix2pix_size)

                # Merging the patch estimation into the base estimate using our merge network:
                # We feed the patch estimation and the same region from the updated base estimate to the merge network
                # to generate the target estimate for the corresponding region.
                if self.option.rgb_input:
                    input_img = to_pix2pix(
                        patch_rgb, self.option.pix2pix_size)
                    mapped = self.get_merge_estimate(
                        base_estimate=patch_whole_estimate_base, high_res_estimate=patch_estimation, rgb=input_img)
                else:
                    mapped = self.get_merge_estimate(
                        base_estimate=patch_whole_estimate_base, high_res_estimate=patch_estimation)

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
            path = pjoin(self.result_dir, imageandpatchs.name)

            estimated_updated_image = imageandpatchs.estimation_updated_image

            # clipping the depth to the range of [0,1]
            estimated_updated_image[estimated_updated_image < 0] = 0

            if self.option.output_resolution == 1:
                midas_utils.write_depth(path,
                                        cv2.resize(estimated_updated_image,
                                                   (input_resolution[1],
                                                    input_resolution[0]),
                                                   interpolation=cv2.INTER_CUBIC), bits=2, colored=self.option.colorize_results, save_raw=self.option.save_raw)
            else:
                midas_utils.write_depth(
                    path, estimated_updated_image, bits=2, colored=self.option.colorize_results, save_raw=self.option.save_raw)

        # # average rmse loss
        if self.option.eval_mode:
            rmse_loss = np.mean(rmse_loss) / len(self.dataset)
            print('mean rmse loss:', rmse_loss)

        print("finished")

    # Generating local patches to perform the local refinement described in section 6 of the main paper.

    def generate_patchs(self, img, base_size):

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
        patch_bound_list = self.adaptive_selection(
            grad_integral_image, patch_bound_list, gf)

        # Sort the patch list to make sure the merging operation will be done with the correct order: starting from biggest
        # patch
        patchset = sorted(patch_bound_list.items(),
                          key=lambda x: getitem(x[1], 'size'), reverse=True)
        return patchset

    # Adaptively select patches

    def adaptive_selection(self, integral_grad, patch_bound_list, gf):
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
    parser.add_argument('--backbone_net', type=str, choices=["midas", "leres", "sgrnet"], required=False,
                        help='use to select different base depth networks 0:midas 1:strurturedRL 2:LeRes')
    parser.add_argument('--colorize_results', action='store_true')
    parser.add_argument('--save_pcd', action='store_true')
    parser.add_argument('--estimation_type', type=str,
                        default="final", choices=["R0", "R20", "final"])
    parser.add_argument('--max_resolution', type=float, default=np.inf)

    # Check for required input
    option_, _ = parser.parse_known_args()
    print(option_)

    if option_.backbone_net == "sgrnet":
        from structuredrl.models import DepthNet

    # Setting each networks receptive field and setting the patch estimation resolution to twice the receptive
    # field size to speed up the local refinement as described in the section 6 of the main paper.
    if option_.backbone_net == "midas":
        option_.net_receptive_field_size = 384
        option_.patch_net_size = 2 * option_.net_receptive_field_size
    elif option_.backbone_net in ["sgrnet", "leres"]:
        option_.net_receptive_field_size = 448
        option_.patch_net_size = 2 * option_.net_receptive_field_size
    else:
        assert False, 'depthNet can only be 0,1 or 2'

    # Run pipeline
    bmd = BoostDepth(option_)
    bmd.boost_depth()
