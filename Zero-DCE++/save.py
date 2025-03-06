import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F

import os
import time
from tqdm import tqdm

import model

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
    image_dir = Path('/home/ubuntu/data/DarkFace_Train_2021/image')
    save_dir = Path('../DarkFace_ZeroDCE++')
    # image_dir = Path('../LOD/RGB_Dark')
    # save_dir = Path('../LOD/RGB_ZeroDCE++')
    # image_dir = Path('../Exdark/JPEGImages/IMGS_dark')
    # save_dir = Path('../Exdark/JPEGImages/IMGS_ZeroDCE++')
    # save_dir = Path('../Exdark_ZeroDCE++')
    save_dir.mkdir(exist_ok=True, parents=True)
    # video_dir = Path('../videos/Exdark')
    video_dir = Path('../videos/DarkFace')
    video_dir.mkdir(exist_ok=True, parents=True)
    images = sorted(image_dir.glob('*'))

    scale_factor = 12
    DCE_net = model.enhance_net_nopool(scale_factor).cuda()
    DCE_net.eval()

    for image_index in tqdm(range(len(images))):
        image = images[image_index]
        results = []
        for i in range(100):
            # DCE_net.load_state_dict(torch.load(f'Exdark/Epoch{i}.pth'))
            DCE_net.load_state_dict(torch.load(f'DarkFace/Epoch{i}.pth'))
            with torch.no_grad():
                save_path = save_dir / f'{i}_{image.name}'
                lowlight(DCE_net, image, save_path)
                results.append(save_path)

        img = cv2.imread(save_path)
        h, w = img.shape[:2]

        # video = cv2.VideoWriter(f'{image.stem}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, (w, h))
        video = cv2.VideoWriter(video_dir / f'{image.stem}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, (w * 2, h))

        for i, result in enumerate(results):
            img = cv2.imread(str(image))
            results_img = cv2.imread(str(result))
            # print(img.shape, results_img.shape)
            img = np.hstack((img, results_img))
            # print(img.shape)
            cv2.putText(img, f'Epoch: {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            video.write(img)

        video.release()