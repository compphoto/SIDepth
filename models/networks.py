
import numpy as np

import torch
from torchvision import transforms

import cv2
from PIL import Image

from models.MiDaS.midas.midas_net import MidasNet
from models.MiDaS.midas.transforms import NormalizeImage, PrepareForNet, Resize
from models.bmd.pix2pix.models.pix2pix4depth_model import SSIToSIDepthModel, BMDMergeModel
from models.bmd.utils import calculateprocessingres as compute_rx

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class Depth():
    def __init__(self, cfg, resize_method=None):
        self.cfg = cfg
        self.resize_method = resize_method if resize_method is not None else self.cfg.resize_method
        self.normalization = NormalizeImage(
            mean=self.cfg.norm_mean, std=self.cfg.norm_std
        )

    def resize_img(self, img, width, height, rgb=False):

        if rgb:
            img = (img * 255.).astype(np.uint8)

        img = Image.fromarray(img)
        img = img.resize((width, height), Image.BILINEAR)
        img = np.asarray(img)

        if rgb:
            img = img.astype(np.float32) / 255.

        return img
    
    def evalaute(self):
        pass


class BMDSSI(Depth):
    def __init__(self, cfg, resize_method=None):
        super().__init__(cfg, resize_method)
        
        self.bb_model = MidasNet(activation='sigmoid').to(device)
        state_dict_ssi = torch.load(self.cfg.ssi_weights)
        self.bb_model.load_state_dict(state_dict_ssi)
        self.bb_model.eval()
        
        self.p2p_model = BMDMergeModel(device)
        state_dict_bmd = torch.load(self.cfg.bmd_weights)  
        print("Loading BMD weights from {}".format(self.cfg.bmd_weights))      
        self.p2p_model.netG.load_state_dict(state_dict_bmd)
        self.p2p_model.netG.eval()

    def get_transformation(self, width, height, normalize=True):
        resize = Resize(
                width=width, 
                height=height,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=self.resize_method,
                image_interpolation_method=cv2.INTER_CUBIC
            )
        
        if normalize:
            tfs = transforms.Compose([resize, self.normalization, PrepareForNet()])
        else:
            tfs = transforms.Compose([resize, PrepareForNet()])

        return tfs

    def evaluate(self, img, width=None, height=None):
        """
        The estimated map is in the disparity space.
        """

        ori_img = np.asarray(img) / 255.0
        original_height, original_width, _ = img.shape

        if width is None or height is None:
            width, height = self.cfg.width, self.cfg.height

        self.transform = self.get_transformation(width, height, normalize=True)
        img_b = self.transform({'image': ori_img})['image']
        img_b = torch.autograd.Variable(torch.from_numpy(img_b).float().unsqueeze(0).to(device))

        # get r20 size
        whole_image_optimal_size, _ = compute_rx(ori_img, self.cfg.width, 0.2, 3, 3000)

        self.transform_r20 = self.get_transformation(whole_image_optimal_size, whole_image_optimal_size, normalize=True)
        img_r20 = self.transform_r20({'image': ori_img})['image']
        img_r20 = torch.autograd.Variable(torch.from_numpy(img_r20).float().unsqueeze(0).to(device))

        with torch.no_grad():
            pred_disp = self.bb_model.forward(img_b)
            pred_disp = torch.squeeze(pred_disp)
            pred_disp = pred_disp.cpu().numpy()

            pred_disp_r20 = self.bb_model.forward(img_r20)
            pred_disp_r20 = torch.squeeze(pred_disp_r20)
            pred_disp_r20 = pred_disp_r20.cpu().numpy()
        
        # resize to 1024
        p2p_width = 1024
        p2p_height = 1024
        pred_disp = self.resize_img(pred_disp, p2p_width, p2p_height)
        pred_disp_r20 = self.resize_img(pred_disp_r20, p2p_width, p2p_height)
        
        # resize original rgb image to original size
        ori_img = self.resize_img(ori_img, p2p_width, p2p_height, rgb=True)

        # get the output
        self.p2p_model.set_input(pred_disp, pred_disp_r20)
        with torch.no_grad():
            self.p2p_model.forward()
            output = self.p2p_model.fake_B

        # change output range from [-1, 1] to [0, 1]
        output = (output + 1) / 2.0
        output = output.detach().cpu().squeeze(0).squeeze(0).numpy()

        # resize of original size
        output = self.resize_img(output, original_width, original_height)

        return output

class OursSSI(Depth):
    def __init__(self, cfg, resize_method=None):
        super().__init__(cfg, resize_method)
        
        self.model = MidasNet(activation='sigmoid').to(device)
        checkpoint = torch.load(self.cfg.ssi_weights)
        print(f"Loading checkpoint from: {self.cfg.ssi_weights}")
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def evaluate(self, img, width=None, height=None):
        """
        The estimated map is in the disparity space.
        """
        img = np.asarray(img) / 255.0
        original_height, original_width, _ = img.shape

        if width is None or height is None:
            width, height = self.cfg.width, self.cfg.height

        self.transform = transforms.Compose([
            Resize(
                width=width, 
                height=height,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=self.resize_method,
                image_interpolation_method=cv2.INTER_CUBIC
            ),
            self.normalization,
            PrepareForNet()
        ])

        img = self.transform({'image': img})['image']
        img = torch.autograd.Variable(torch.from_numpy(img).float().unsqueeze(0).to(device))
        pred_disp = self.model.forward(img)
        pred_disp = torch.squeeze(pred_disp)
        pred_disp = pred_disp.cpu().detach().numpy()

        # resize to original size
        pred_disp = self.resize_img(pred_disp, original_width, original_height)

        return pred_disp

class OursSI(Depth):
    def __init__(self, cfg, resize_method=None):
        super().__init__(cfg, resize_method)
        self.cfg = cfg
        
        self.bb_model = MidasNet(activation='sigmoid').to(device)
        state_dict_ssi = torch.load(self.cfg.ssi_weights)
        print(f"Loading SSI model weights from: {self.cfg.ssi_weights}")
        self.bb_model.load_state_dict(state_dict_ssi)
        self.bb_model.eval()

        print("==== Network Details ====")
        print(f"Size of RX: {self.cfg.rx_size}")
        print("==========================")

        self.o2m_model = SSIToSIDepthModel(device)
        state_dict_si = torch.load(self.cfg.si_weights)  
        print("Loading SI weights from {}".format(self.cfg.si_weights))      
        self.o2m_model.netG.load_state_dict(state_dict_si)
        self.o2m_model.netG.eval()

    def get_transformation(self, width, height, normalize=True):
        resize = Resize(
                width=width, 
                height=height,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=self.resize_method,
                image_interpolation_method=cv2.INTER_CUBIC
            )
        
        if normalize:
            tfs = transforms.Compose([resize, self.normalization, PrepareForNet()])
        else:
            tfs = transforms.Compose([resize, PrepareForNet()])

        return tfs

    def evaluate(self, img, width=None, height=None):
        """
        The estimated map is in the disparity space.
        """

        ori_img = np.asarray(img) / 255.0
        original_height, original_width, _ = img.shape

        if width is None or height is None:
            width, height = self.cfg.width, self.cfg.height

        # Mahdi: Testing high res
        width, height = 448, 448
        print("Base Resolution: {}x{}".format(width, height))
        
        self.transform = self.get_transformation(width, height, normalize=True)
        img_b = self.transform({'image': ori_img})['image']
        img_b = torch.autograd.Variable(torch.from_numpy(img_b).float().unsqueeze(0).to(device))

        # get r20 size
        whole_image_optimal_size, _ = compute_rx(ori_img, self.cfg.width, self.cfg.rx_size, 3, 3000)
        self.transform_r20 = self.get_transformation(whole_image_optimal_size, whole_image_optimal_size, normalize=True)
        img_r20 = self.transform_r20({'image': ori_img})['image']
        img_r20 = torch.autograd.Variable(torch.from_numpy(img_r20).float().unsqueeze(0).to(device))

        self.o2m_model.netG.to('cpu')
        self.bb_model.to(device)

        torch.cuda.empty_cache()

        with torch.no_grad():
            pred_disp = self.bb_model.forward(img_b)
            pred_disp = torch.squeeze(pred_disp)
            pred_disp = pred_disp.cpu().numpy()

            pred_disp_r20 = self.bb_model.forward(img_r20)
            pred_disp_r20 = torch.squeeze(pred_disp_r20)
            pred_disp_r20 = pred_disp_r20.cpu().numpy()
        
        # resize to metric_size
        p2p_width = int(self.cfg.metric_size)
        p2p_height = int(self.cfg.metric_size)

        pred_disp = self.resize_img(pred_disp, p2p_width, p2p_height)
        pred_disp_r20 = self.resize_img(pred_disp_r20, p2p_width, p2p_height)
        
        # resize original rgb image to original size
        ori_img = self.resize_img(ori_img, p2p_width, p2p_height, rgb=True)

        # find a scalar using lstsq method
        pred_disp = (pred_disp - np.min(pred_disp)) / (np.max(pred_disp) - np.min(pred_disp))
        scalar = np.linalg.lstsq(pred_disp_r20.reshape(-1, 1), pred_disp.reshape(-1, 1), rcond=None)[0]
        pred_disp_r20 = pred_disp_r20 * scalar
        
        # get the output
        self.bb_model.to('cpu')
        self.o2m_model.netG.to(device)
        
        torch.cuda.empty_cache()

        self.o2m_model.set_input(pred_disp, pred_disp_r20, rgb=ori_img)
        with torch.no_grad():   
            self.o2m_model.forward()
            output = self.o2m_model.fake_B
            output = output.detach().cpu().squeeze(0).squeeze(0).numpy()

        # resize of original size
        output = self.resize_img(output, original_width, original_height)
        return output
