import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import argparse
from models.config import _C as cfg
from models.networks import OursSSI, OursSI, BMDSSI
from models.bmd.midas.utils import write_depth


def read_data(img):
    img = np.asarray(Image.open(img).convert('RGB'))
    return img

def read_datalist(datapath):
    rgbs = sorted([os.path.join(datapath, x) for x in os.listdir(datapath) if x.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']])
    return rgbs

def evaluate(args):  
    
    if args.model == "SSI":
        model = BMDSSI(cfg.model_config)
    elif args.model == "SSIBase":
        model = OursSSI(cfg.model_config)
    elif args.model == "SI":
        model = OursSI(cfg.model_config)

    image_list = read_datalist(args.input_path)
    print(len(image_list), "images found.")

    for img_path in tqdm(image_list, total=len(image_list)):
       
        img = read_data(img_path)
        imagename = os.path.basename(img_path).split('.')[0]
        print("parsed image name:", imagename)

        output_path = args.output_path
        os.makedirs(output_path, exist_ok=True)
        
        base, high = None, None

        prediction = model.evaluate(img)
        
        ## normalize depth maps:
        if args.model == "SSI" or args.model == "SSIBase":
            prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min())
        elif args.model == "SI":
            prediction = prediction / prediction.max()
        else:
            raise ("Unknown network type")
        
        write_depth(os.path.join(output_path, imagename + '.png'), prediction, colored=args.colorize_depth)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model",choices = ['SSI', 'SI' , 'SSIBase'])
    parser.add_argument("-i", "--input_path",help="Input folder path - reads all 'jpg', 'png', 'jpeg' files")
    parser.add_argument("-o", "--output_path",help="Output folder path")
    parser.add_argument("--colorize_depth", action='store_true')
    args = parser.parse_args()

    evaluate(args)