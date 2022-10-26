import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from networks.denoising_rgb import DenoiseNeXt
from dataloaders.data_rgb import get_test_data
import utils
import lycon
from skimage import img_as_ubyte
from PIL import Image
import cv2
import time

parser = argparse.ArgumentParser(description='RGB denoising evaluation on the testing set of asus')
parser.add_argument('--input_dir', default='./datasets/asus_test/asus_test_rgb/', type=str, help='Directory of validation images')
parser.add_argument('--weights', default='./pretrained_models/denoising/sidd_rgb.pth', type=str, help='Path to weights')

args = parser.parse_args()

test_dataset = get_test_data(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False, num_workers=8, drop_last=False)

model_restoration = DenoiseNeXt()

#utils.load_checkpoint(model_restoration, args.weights)
model_restoration.load_state_dict(torch.load(args.weights))
print("===> Testing using weights: ", args.weights)

model_restoration.cuda()
model_restoration.eval()

elapsed_time = 0.0

img_name = None
result_name = None
denoised_img = None
original_img = None

with torch.no_grad():
    for i, data_test in enumerate(tqdm(test_loader), 0):
        rgb_noisy = data_test[0].cuda()
        filenames = data_test[1]

        if img_name == None:
            img_name = os.path.join('./NTU_Sky_Noise/noisy/', filenames[0][:-10]+'.jpg')
            matte_name = os.path.join('./NTU_Sky_Noise_Matte/noisy/', filenames[0][:-10]+'.png')
            result_name = os.path.join('./results/DenoiseNeXt/', filenames[0][:-10]+'.jpg')

            original_img = cv2.imread(img_name)
            denoised_img = cv2.imread(img_name)
            original_matte = cv2.imread(matte_name)

        start_time = time.time()
        rgb_restored = model_restoration(rgb_noisy)
        rgb_restored = torch.clamp(rgb_restored, 0, 1)
        elapsed_time = elapsed_time + time.time() - start_time

        rgb_restored = rgb_restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        
        start_time = time.time()
        for batch in range(len(rgb_restored)):
            cropped_img = img_as_ubyte(rgb_restored[batch])
            idxH = int(filenames[batch][-9:-7])
            idxW = int(filenames[batch][-6:-4])
            denoised_img[idxH*128:(idxH+1)*128, idxW*128:(idxW+1)*128, :] = cropped_img
        elapsed_time = elapsed_time + time.time() - start_time
        
        if (i+1) % 48 == 0:
            start_time = time.time()

            foreground = original_img.astype(float)
            background = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2RGB).astype(float)
        
            alpha = original_matte.astype(float)/255
            foreground = cv2.multiply(1.0 - alpha, foreground)
            background = cv2.multiply(alpha, background)

            result = cv2.add(background.astype('float32'), foreground.astype('float32'))
            elapsed_time = elapsed_time + time.time() - start_time

            cv2.imwrite(result_name, result)
            img_name = None
        

print(elapsed_time/53)
