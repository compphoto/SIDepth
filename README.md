## Scale-Invariant Monocular Depth Estimation via SSI Depth [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/compphoto/SIDepth/blob/main/demo.ipynb) [![Arxiv](http://img.shields.io/badge/cs.CV-arXiv-B31B1B.svg)](https://yaksoy.github.io/papers/SIG24-SI-Depth-Supp.pdf)


> S. Mahdi H. Miangoleh, Mahesh Reddy, Yağız Aksoy.
> [Main pdf](https://yaksoy.github.io/papers/SIG24-SI-Depth.pdf),
> [Supplementary pdf](https://yaksoy.github.io/papers/SIG24-SI-Depth-Supp.pdf),
> [Project Page](https://yaksoy.github.io/sidepth/). 

Proc. SIGGRAPH, 2024

[![video](figures/gitplay.jpg)](https://www.youtube.com/watch?v=R_vW6TjYiEM)



Existing methods for scale-invariant monocular depth estimation (SI MDE) often struggle due to the complexity of the task, and limited and non-diverse datasets, hindering generalizability in real-world scenarios. This is while shift-and-scale-invariant (SSI) depth estimation, simplifying the task and enabling training with abundant stereo datasets achieves high performance. We present a novel approach that leverages SSI inputs to enhance SI depth estimation, streamlining the network's role and facilitating in-the-wild generalization for SI depth estimation while only using a synthetic dataset for training. Emphasizing the generation of high-resolution details, we introduce a novel sparse ordinal loss that substantially improves detail generation in SSI MDE, addressing critical limitations in existing approaches. Through in-the-wild qualitative examples and zero-shot evaluation we substantiate the practical utility of our approach in computational photography applications, showcasing its ability to generate highly detailed SI depth maps and achieve generalization in diverse scenarios.


Try our model easily on Colab : [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/compphoto/SIDepth/blob/main/demo.ipynb)



# Inference

1. Create a python conda environment as following:

```
conda create -n ENVNAME python=3.8
pip install -r requirements.txt

conda activate ENVNAME
```

2. Download our model weights from [here](https://drive.google.com/file/d/1jbcgAkKNXxQO37iwjjWbEYCcVQVERwTc/view?usp=drive_link) and place them inside "./weights/" folder.

3. Run the code.
```
python eval.py -i PATH-TO-INPUT -o PATH-TO-OUTPUT -m SI --colorize
python eval.py -i PATH-TO-INPUT -o PATH-TO-OUTPUT -m SSI --colorize
```

** Our implementation boosts our SSI depth using [BoostingMonocularDepth](https://github.com/compphoto/BoostingMonocularDepth) by default. If you need the base estimations at the lower receptieve field resolution run the following:
```
python eval.py -i PATH-TO-INPUT -o PATH-TO-OUTPUT -m SSIBase --colorize
```

## Citation

This implementation is provided for academic use only. Please cite our paper if you use this code or any of the models.
```
@INPROCEEDINGS{miangolehSIDepth,
author={S. Mahdi H. Miangoleh and Mahesh Reddy and Ya\u{g}{\i}z Aksoy},
title={Scale-Invariant Monocular Depth Estimation via SSI Depth},
booktitle={Proc. SIGGRAPH},
year={2024},
}

```

## Credits

"./model/bmd/" is adapted from [BoostingMonocularDepth](https://github.com/compphoto/BoostingMonocularDepth) for their Boosting framework implementation.  

"./model/bmd/pix2pix" folder was adapted from the [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) repository. 

"./model/MiDaS" is adapted from [MiDaS](https://github.com/intel-isl/MiDaS/tree/v2) for their EfficientNet implementation.   
