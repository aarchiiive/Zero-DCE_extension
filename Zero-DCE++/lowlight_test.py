import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F

import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
from pathlib import Path
import glob
import time

from torchprofile import profile_macs  # FLOPs 계산용 라이브러리


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def calculate_model_metrics(model, input_tensor):
    """
    Calculate FLOPs and parameter count for a given model.
    """
    model.eval()
    flops = profile_macs(model, input_tensor)
    params = sum(p.numel() for p in model.parameters())
    return flops, params

def lowlight(DCE_net, image_path, save_path):
    data_lowlight = Image.open(image_path)
    data_lowlight = data_lowlight.convert('RGB')
    data_lowlight = np.asarray(data_lowlight) / 255.0
    data_lowlight = torch.from_numpy(data_lowlight).float()
    org_h, org_w = data_lowlight.shape[:2]

    h = (data_lowlight.shape[0] // scale_factor) * scale_factor
    w = (data_lowlight.shape[1] // scale_factor) * scale_factor
    # data_lowlight = data_lowlight[0:h, 0:w, :]
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)
    data_lowlight = F.interpolate(data_lowlight, size=(h, w), mode='bilinear', align_corners=False)

    start = time.time()
    enhanced_image, params_maps = DCE_net(data_lowlight)
    end_time = time.time() - start

    # restore input size
    enhanced_image = F.interpolate(enhanced_image, size=(org_h, org_w), mode='bilinear', align_corners=False)

    torchvision.utils.save_image(enhanced_image, save_path)

if __name__ == '__main__':
    # image_dir = Path('/home/ubuntu/data/DarkFace_Train_2021/image')
    # save_dir = Path('/home/ubuntu/data/DarkFace_Train_2021/Zero-DCE++')
    # image_dir = Path('../LOD/RGB_Dark')
    # save_dir = Path('../LOD/RGB_ZeroDCE++')
    image_dir = Path('../Exdark/JPEGImages/IMGS_dark')
    save_dir = Path('../Exdark/JPEGImages/IMGS_ZeroDCE++')
    save_dir.mkdir(exist_ok=True, parents=True)
    images = sorted(image_dir.glob('*'))

    scale_factor = 12
    DCE_net = model.enhance_net_nopool(scale_factor).cuda()
    DCE_net.load_state_dict(torch.load('Exdark/Epoch99.pth'))

    # Dummy input tensor for FLOPs and parameter calculation
    # dummy_input = torch.randn(1, 3, 720, 1080).cuda()
    # flops, params = calculate_model_metrics(DCE_net, dummy_input)
    # print(f"GFLOPs: {flops / 1e9}, Parameters: {params}")

    with torch.no_grad():
        for image in images:
            print(image)
            save_path = save_dir / image.name
            lowlight(DCE_net, image, save_path)
