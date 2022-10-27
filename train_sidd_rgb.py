"""
## CycleISP: Real Image Restoration Via Improved Data Synthesis
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## CVPR 2020
## https://arxiv.org/abs/2003.07761
"""

import numpy as np
import os
import argparse

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from networks.denoising_rgb import DenoiseNeXt
from dataloaders.data_rgb import get_training_data, get_validation_data
import utils

parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
parser.add_argument('--input_dir', default='./datasets/asus_train/', type=str)
parser.add_argument('--weights', default='./pretrained_models/denoising/sidd_rgb.pth', type=str)

args = parser.parse_args()

train_dataset = get_training_data(args.input_dir)
val_dataset = get_validation_data(args.input_dir)
train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=False, num_workers=8, drop_last=False)
val_loader = DataLoader(dataset=val_dataset, batch_size=4, shuffle=False, num_workers=8, drop_last=False)

model_restoration = DenoiseNeXt()

#utils.load_checkpoint(model_restoration, args.weights)
#print("===> Using pretrained weights: ", args.weights)

model_restoration.cuda()

EPOCH = 65
lr = 0.0001
loss_fn = nn.L1Loss()
optimizer = torch.optim.AdamW(model_restoration.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

for epoch in range(EPOCH):
    psnr_val_rgb = []
    total_loss = 0.0
    model_restoration.train()
    for idx, data_train in enumerate(train_loader):
        optimizer.zero_grad()
        rgb_gt = data_train[0].cuda()
        rgb_noisy = data_train[1].cuda()
        filenames = data_train[2]
        rgb_restored = model_restoration(rgb_noisy)
        rgb_restored = torch.clamp(rgb_restored, 0, 1)

        loss = loss_fn(rgb_gt, rgb_restored)
        loss.backward()
        optimizer.step()
        psnr_val_rgb.append(utils.batch_PSNR(rgb_restored, rgb_gt, 1.))

        rgb_gt = rgb_gt.permute(0, 2, 3, 1).cpu().detach().numpy()
        rgb_noisy = rgb_noisy.permute(0, 2, 3, 1).cpu().detach().numpy()
        rgb_restored = rgb_restored.permute(0, 2, 3, 1).cpu().detach().numpy()

        total_loss += loss.item()
        #print('Epoch: [{0}][{1}/{2}] loss: {3}'.format(epoch+1, idx+1, len(train_loader), loss.item()))

    print('Epoch: [{0}] loss: {1}'.format(epoch+1, total_loss))
    utils.save_checkpoint(model_restoration.state_dict(), epoch)
    scheduler.step()

    for idx, data_val in enumerate(val_loader):
        optimizer.zero_grad()
        rgb_gt = data_val[0].cuda()
        rgb_noisy = data_val[1].cuda()
        rgb_restored = model_restoration(rgb_noisy)
        rgb_restored = torch.clamp(rgb_restored, 0, 1)

        psnr_val_rgb.append(utils.batch_PSNR(rgb_restored, rgb_gt, 1.))

        rgb_gt = rgb_gt.permute(0, 2, 3, 1).cpu().detach().numpy()
        rgb_noisy = rgb_noisy.permute(0, 2, 3, 1).cpu().detach().numpy()
        rgb_restored = rgb_restored.permute(0, 2, 3, 1).cpu().detach().numpy()

    psnr_val_rgb = sum(psnr_val_rgb)/len(psnr_val_rgb)
    print("PSNR: %.2f " %(psnr_val_rgb))
