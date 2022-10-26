import numpy as np
import os
import glob
from skimage import img_as_ubyte
import cv2
import random

file_names = sorted(glob.glob('./datasets/asus_train/asus_train_rgb/clean/*'))
result_dir = './datasets/asus_train/asus_train_rgb/noisy/'


def generate_synthesized_training_data():
    mean = 0

    for filename in file_names:
        sigma = random.uniform(0.01, 0.03)
        img = cv2.imread(filename)
        img = img / 255
        noise = np.random.normal(mean, sigma, img.shape)
        rgb_noisy = img + noise
        rgb_noisy = np.clip(rgb_noisy, 0, 1)

        rgb_noisy = np.uint8(rgb_noisy*255)

        noise = np.uint8(noise*255)
        
        cv2.imwrite(result_dir+filename.split('/')[-1], rgb_noisy)

def generate_sample_images():
    mean = 0

    for sigma in [0.01, 0.015, 0.02, 0.03]:
        img = cv2.imread('./datasets/asus_train/asus_train_rgb/clean/main_s21u_20220530_195557_00_00.png')
        img = img / 255
        noise = np.random.normal(mean, sigma, (128, 128, 3))
        rgb_noisy = img + noise[:128, :128, :]
        rgb_noisy = np.clip(rgb_noisy, 0, 1)

        rgb_noisy = np.uint8(rgb_noisy*255)

        noise = np.uint8(noise*255)
        
        cv2.imwrite('./sample/sample_{}.jpg'.format(sigma), rgb_noisy)

#generate_sample_images()
generate_synthesized_training_data()