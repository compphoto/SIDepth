from yacs.config import CfgNode as CN

# config
_C = CN()

# basic parameters
_C.dataroot = None
_C.name = "void"
_C.gpu_ids = [0]
_C.checkpoints_dir = "./pix2pix/checkpoints"

# model parameters
_C.model = "pix2pix4depth"
_C.input_nc = 2
_C.output_nc = 1
_C.rgb_input = False
_C.ngf = 64
_C.ndf = 64
_C.netD = "basic"
_C.netG = "unet_1024"
_C.n_layers_D = 3
_C.norm = "none"
_C.init_type = "normal"
_C.init_gain = 0.02
_C.no_dropout = False
_C.rgb_input = False

# dataset parameters
_C.dataset_mode = "depthmerge"
_C.direction = "AtoB"
_C.serial_batches = False
_C.num_threads = 4
_C.batch_size = 1
_C.load_size = 672
_C.crop_size = 672
_C.max_dataset_size = 10000
_C.preprocess = "resize_and_crop"
_C.no_flip = False
_C.display_winsize = 256

# additional parameters
# _C.epoch = "latest"  # mergenet v1 (BMD)
_C.epoch = "200"  # mergenet v2 (DIV2K dataset)
_C.load_iter = 0
_C.verbose = False
_C.suffix = ""
_C.aspect_ratio = 1.0
_C.eval = False
_C.isTrain = False
_C.num_test = 50
_C.phase = "test"
_C.sobel_kernel = 7
