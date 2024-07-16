"""This module contains simple helper functions """
from __future__ import print_function

import os

import numpy as np
import torch
from PIL import Image


def tensor2im(input_image, imtype=np.uint16, outer_activation='tanh'):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        # convert it into a numpy array

        # save according to the image (depth or RGB)
        if image_tensor.size(1) == 1:
            image_numpy = image_tensor[0].cpu().float().numpy()
            
            if outer_activation == 'tanh':
                image_numpy = (image_numpy + 1) / 2.0 
            
            image_numpy = image_numpy * (2**16-1)
            image_numpy = image_numpy.astype(imtype)
        else:
            image_numpy = image_tensor[0].cpu().float().numpy()
            # image_numpy = np.transpose(image_numpy, (1, 2, 0))

            if outer_activation == 'tanh':
                image_numpy = (image_numpy + 1) / 2.0 
            
            image_numpy = image_numpy* (2**8-1)
            image_numpy = image_numpy.astype(np.uint8)
        return image_numpy

        # image_numpy = torch.squeeze(image_tensor).cpu().numpy()
        # image_numpy = (image_numpy + 1) / 2.0 * (2**16-1)
        # else:  # if it is a numpy array, do nothing
        #     image_numpy = input_image
        # return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    # hack to save an image if the batch size was higher than 1
    if image_numpy.shape[0] > 1:
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        image_pil = Image.fromarray(image_numpy)
        image_pil = image_pil.convert('RGB')
    else:
        image_numpy = image_numpy[0]
        image_pil = Image.fromarray(image_numpy)
        image_pil = image_pil.convert('I;16')

    # image_pil = Image.fromarray(image_numpy)
    # h, w, _ = image_numpy.shape
    #
    # if aspect_ratio > 1.0:
    #     image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    # if aspect_ratio < 1.0:
    #     image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)

    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
