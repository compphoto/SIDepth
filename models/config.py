from yacs.config import CfgNode as CN

# config
_C = CN()

_C.model_config = CN()
_C.model_config.ssi_weights = './weights/ssi_weights.pth'
_C.model_config.si_weights = './weights/si_weights.pth'
_C.model_config.height = 384
_C.model_config.width = 384
_C.model_config.norm_mean = [0.485, 0.456, 0.406]
_C.model_config.norm_std = [0.229, 0.224, 0.225]
_C.model_config.resize_method = 'upper_bound'
_C.model_config.metric_size = '1024'
_C.model_config.rx_size = 0.2 
_C.model_config.bmd_weights = './weights/bmd_weights.pth'
