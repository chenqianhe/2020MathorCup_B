!git clone https://gitee.com/paddlepaddle/PaddleGAN.git
%cd PaddleGAN/
!pip install -v -e .


import os
from PaddleGAN.ppgan.apps.realsr_predictor import RealSRPredictor

sr = RealSRPredictor()

for i in os.listdir("small2"):
    sr.run("small2/"+i)