# -*- coding: utf-8 -*-
#import cv2
import numpy as np
import time
import torch
import pdb
from collections import OrderedDict
import sys
# import onnxruntime
from models import *

net = ISNetDIS()
print(net)
net.load_state_dict(torch.load('../saved_models/IS-Net-test/gpu_itr_17800_traLoss_2.9288_traTarLoss_0.2515_valLoss_3.0427_valTarLoss_0.2362_maxF1_0.9568_mae_0.0222_time_0.024903.pth', map_location=torch.device('cpu')))
# input = torch.randn(1, 3, 1024, 1024, device='cpu')
input = torch.randn(1, 3, 640, 640, device='cpu')
torch.onnx.export(net, input, 'isnet.onnx',
                export_params=True, opset_version=11, do_constant_folding=True,
                input_names = ['input'])

