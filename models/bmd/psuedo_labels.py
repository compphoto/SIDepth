import os
from os.path import join as pjoin

import numpy as np
from dependencies.bmd.boosting_v2 import BoostDepth


def update_args(args):

    # BMD options
    args.save_patches = False
    args.save_whole_estimation = False
    args.output_resolution = 1
    args.net_receptive_field_size = 384
    args.pix2pix_size = 1024
    args.backbone_net = "midas"
    args.estimation_type = "R20"
    args.max_resolution = np.inf

    return args


def compute_pseudo_labels(json_data, args, json_similar_data):

    # update args
    args = update_args(args)

    if args.backbone_net == "midas":
        args.net_receptive_field_size = 384
        args.patch_net_size = 2 * args.net_receptive_field_size
    elif args.backbone_net in ["leres", "sgrnet"]:
        args.net_receptive_field_size = 448
        args.patch_net_size = 2 * args.net_receptive_field_size
    else:
        assert False, 'backbone net can only be 0,1 or 2'

    # iterate over all images in the json file and compute pseudo labels
    for dataset in json_data:

        if dataset not in args.pseudo_datasets:
            continue

        for object in json_data[dataset]:
            image_list = json_similar_data[dataset][object]["rgb"]
            args.image_list = image_list

            args.model_name = f"b_{args.base_net}_f_{args.high_res_net}"
            args.pseudo_label_dir = pjoin(
                args.output_path, args.model_name, "complete_estimation")
            os.makedirs(args.pseudo_label_dir, exist_ok=True)

            args.double_estimation_dir = pjoin(
                args.output_path, args.model_name, "double_estimation")
            os.makedirs(args.double_estimation_dir, exist_ok=True)

            # Run pipeline
            boosting = BoostDepth(args)
            boosting.boost_depth()

            if args.depth_type == "double_estimation":
                labels_list = [pjoin(args.double_estimation_dir, "{}.png".format(
                    img.split(os.sep)[-1].split('.')[0])) for img in image_list]
                json_similar_data[dataset][object][args.depth_type] = labels_list
            else:
                labels_list = [pjoin(args.pseudo_label_dir, "{}.raw".format(
                    img.split(os.sep)[-1].split('.')[0])) for img in image_list]
                json_similar_data[dataset][object][args.depth_type] = labels_list

    return json_similar_data
