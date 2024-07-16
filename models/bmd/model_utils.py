import cv2
import numpy as np
import torch
# from dependencies.bmd.lib.multi_depth_model_woauxi import RelDepthModel
# from dependencies.bmd.lib.net_tools import strip_prefix_if_present
from dependencies.bmd.midas.models.dpt_depth import DPTDepthModel
from dependencies.bmd.midas.models.midas_net import MidasNet
from dependencies.bmd.midas.models.transforms import (NormalizeImage,
                                                      PrepareForNet, Resize)
from dependencies.bmd.structuredrl.models.DepthNet import DepthNet
from torchvision.transforms import Compose, transforms

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'

NETWORKS = {
    'midas_v2': {
        'model': MidasNet,
        'weights': "./dependencies/bmd/weights/midas/model-f6b98070.pt",
        'resize_mode': 'upper_bound',
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
    'midas_v3_hybrid': {
        'model':  DPTDepthModel,
        'weights': "./dependencies/bmd/weights/midas/dpt_hybrid-midas-501f0c75.pt",
        'resize_mode': 'minimal',
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5]
    },
    'midas_v3_large': {
        'model': DPTDepthModel,
        'weights': "./dependencies/bmd/weights/midas/dpt_large-midas-2f21e586.pt",
        'resize_mode': 'minimal',
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5]
    },
    'omnidata': {
        'model': DPTDepthModel,
        'weights': "./dependencies/bmd/weights/midas/omnidata-midas-d8f8f8e0.pt",
    },
    'sgrnet': {
        'model': DepthNet,
        'weights': "./dependencies/bmd/weights/sgr/model.pth.tar",
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
    # 'leres': {
    #     'model': RelDepthModel,
    #     'weights': "./dependencies/bmd/weights/leres/res101.pth",
    # }
}


def load_model(name: str, weights: str = None, activation='relu'):
    model = NETWORKS[name]['model']
    weights = NETWORKS[name]['weights'] if weights is None else weights

    checkpoint = None
    if activation == "sigmoid":
        checkpoint = torch.load(weights, map_location=DEVICE)
        weights = None

    if name == "midas_v2":
        net = model(path=weights, activation=activation)
    elif name == "midas_v3_large":
        net = model(
            path=weights, backbone="vitl16_384", non_negative=True)
    elif name == "midas_v3_hybrid":
        net = model(
            path=weights, backbone="vitb_rn50_384", non_negative=True)
    elif name == "sgrnet":
        net = model()
        net = torch.nn.DataParallel(net, device_ids=[0])
        checkpoint = torch.load(weights)
        net.load_state_dict(checkpoint['state_dict'])
    # elif name == "leres":
    #     checkpoint = torch.load(weights)
    #     net = model(backbone='resnetx101')
    #     net.load_state_dict(strip_prefix_if_present(checkpoint['depth_model'], "module."),
    #                         strict=True)
    else:
        pass

    if activation == "sigmoid":
        if "model" in checkpoint.keys():
            print(f"Loading weights from {weights} for {name}")
            net.load_state_dict(checkpoint["model"])

    torch.cuda.empty_cache()
    net = net.to(DEVICE)
    net.eval()
    return net


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


def get_prediction(name, net, img, msize):
    prediction = None
    if name in ['midas_v2', 'midas_v3_large', 'midas_v3_hybrid']:
        resize_mode = NETWORKS[name]['resize_mode']
        transform = Compose(
            [
                Resize(
                    msize,
                    msize,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method=resize_mode,
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=NETWORKS[name]['mean'],
                               std=NETWORKS[name]['std']),
                PrepareForNet(),
            ]
        )

        img_input = transform({"image": img})["image"]

        # Forward pass
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(DEVICE).unsqueeze(0)
            prediction = net.forward(sample)

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

    elif name == 'leres':
        # https://github.com/aim-uofa/AdelaiDepth/tree/main/LeReS
        rgb_c = img[:, :, ::-1].copy()
        A_resize = cv2.resize(rgb_c, (msize, msize))
        img_torch = scale_torch(A_resize)[None, :, :, :]

        # Forward pass
        with torch.no_grad():
            prediction = net.inference(img_torch)

        prediction = prediction.squeeze().cpu().numpy()
        prediction = cv2.resize(
            prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    elif name == 'sgrnet':
        # https://github.com/KexianHust/Structure-Guided-Ranking-Loss
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
            output = net(input_img)

        # Normalization
        depth = output.squeeze().cpu().data.numpy()
        min_d, max_d = depth.min(), depth.max()
        depth_norm = (depth - min_d) / (max_d - min_d)

        prediction = cv2.resize(
            depth_norm, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    else:
        pass

    return prediction
