import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from networks.denoising_rgb import DenoiseNet
from dataloaders.data_rgb import get_validation_data
import utils
import lycon
from skimage import img_as_ubyte
from PIL import Image

parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of asus')
parser.add_argument('--input_dir', default='./datasets/asus_train/asus_train_rgb/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./CycleISP/retrained/cropped/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/denoising/sidd_rgb.pth', type=str, help='Path to weights')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')

args = parser.parse_args()

utils.mkdir(args.result_dir)

val_dataset = get_validation_data(args.input_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=8, drop_last=False)

model_restoration = DenoiseNet()

#utils.load_checkpoint(model_restoration, args.weights)
model_restoration.load_state_dict(torch.load(args.weights))
print("===> Validation using weights: ", args.weights)

model_restoration.cuda()

#model_restoration = nn.DataParallel(model_restoration)

model_restoration.eval()


with torch.no_grad():
    psnr_val_rgb = []
    for ii, data_val in enumerate(tqdm(val_loader), 0):
        rgb_gt = data_val[0].cuda()
        rgb_noisy = data_val[1].cuda()
        filenames = data_val[2]
        rgb_restored = model_restoration(rgb_noisy)
        rgb_restored = torch.clamp(rgb_restored,0,1)
     
        psnr_val_rgb.append(utils.batch_PSNR(rgb_restored, rgb_gt, 1.))

        rgb_gt = rgb_gt.permute(0, 2, 3, 1).cpu().detach().numpy()
        rgb_noisy = rgb_noisy.permute(0, 2, 3, 1).cpu().detach().numpy()
        rgb_restored = rgb_restored.permute(0, 2, 3, 1).cpu().detach().numpy()

        if args.save_images:
            for batch in range(len(rgb_gt)):
                denoised_img = img_as_ubyte(rgb_restored[batch])
                #lycon.save(args.result_dir + filenames[batch][:-4] + '.png', denoised_img)
                Image.fromarray(denoised_img).save(args.result_dir + filenames[batch][:-4] + '.png')
            
psnr_val_rgb = sum(psnr_val_rgb)/len(psnr_val_rgb)
print("PSNR: %.2f " %(psnr_val_rgb))
