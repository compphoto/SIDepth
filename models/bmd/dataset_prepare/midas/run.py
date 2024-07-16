"""Compute depth maps for images in the input folder.
"""
import argparse
import glob
import os

import cv2
import torch
from dependencies.bmd.dataset_prepare.midas import utils
from dependencies.bmd.dataset_prepare.midas.models.midas_net import MidasNet
from dependencies.bmd.dataset_prepare.midas.models.transforms import (
    NormalizeImage, PrepareForNet, Resize)
from dependencies.bmd.model_utils import NETWORKS
from torchvision.transforms import Compose


def generate_depth(args):
    """Run MonoDepthNN to compute depth maps.
    Args:
        args: Command line arguments.
    """

    # select device
    device = torch.device("cuda")
    print("device: %s" % device)

    # load network
    model = MidasNet(args.weights, non_negative=True)

    mean = NETWORKS[args.network]["mean"]
    std = NETWORKS[args.network]["std"]

    transform = Compose(
        [
            Resize(
                args.resolution,
                args.resolution,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=mean, std=std),
            PrepareForNet(),
        ]
    )

    model.to(device)
    model.eval()

    # get input
    img_names = glob.glob(os.path.join(args.input_dir, "*"))
    num_images = len(img_names)

    # create output folder
    os.makedirs(args.output_dir, exist_ok=True)

    print("start processing")

    for ind, img_name in enumerate(img_names):

        print("processing {} ({}/{})".format(img_name, ind + 1, num_images))

        # input
        img = utils.read_image(img_name)
        img_input = transform({"image": img})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
            prediction = model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        # output
        filename = os.path.join(
            args.output_dir, os.path.splitext(os.path.basename(img_name))[0]
        )
        utils.write_depth(filename, prediction, bits=2, colorize=args.colorize)

    print("finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input_dir', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--resolution', required=True, type=int)
    parser.add_argument('--weights', default=None, type=str)
    parser.add_argument('--colorize', action="store_true", default=False)
    args = parser.parse_args()

    if args.weights is None:
        args.weights = NETWORKS["midas_v2"]["weights"]

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    generate_depth(args)
