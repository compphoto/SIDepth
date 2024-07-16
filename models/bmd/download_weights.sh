#!/bin/bash

# download midas_v3 large dpt model
# wget https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt -O ./weights/dpt_large-midas-2f21e586.pt
gdown https://drive.google.com/uc?id=1vnuhoMc6caF-buQQ4hK0CeiMk9SjwB-G -O ./weights/dpt_large-midas-2f21e586.pt

# download midas_v3 hypbird dpt model
gdown https://drive.google.com/uc?id=1dgcJEYYw1F8qirXhZxgNK8dWWz_8gZBD -O ./weights/dpt_hybrid-midas-501f0c75.pt

# download midas_v3 CNN model
wget https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt -O ./weights/midas_v21-f6b98070.pt
