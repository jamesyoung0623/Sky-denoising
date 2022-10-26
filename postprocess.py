import glob
import cv2
import numpy as np
from PIL import Image

file_names = sorted(glob.glob('./CycleISP/retrained/cropped/*'))

for idx in range(len(file_names)//768): 
    w = int(file_names[768*idx+31].split('_')[-1][:2])
    h = 0

    if w == 31:
        w = 32
        h = 24
    else:
        w = 24
        h = 32

    print(h, w)
    img = np.zeros((h*128, w*128, 3))
    #img = Image.new('RGB', (w*128, h*128))

    for i in range(h):
        for j in range(w):
            cropped_img = cv2.imread(file_names[768*idx+i*w+j])
            #cropped_img = Image.open(file_names[768*idx+i*w+j])
            cropped_img = cv2.bilateralFilter(cropped_img.astype('float32'), 7, 45, 45)

            #img.paste(im=cropped_img, box=(j*128, i*128))

            img[i*128:(i+1)*128, j*128:(j+1)*128, :] = cropped_img
    
    cv2.imwrite('./CycleISP/retrained/img/'+file_names[768*idx].split('/')[-1][:-10]+'.jpg', img)
    #img.save('./CycleISP/retrained/img/'+file_names[768*idx].split('/')[-1][:-10]+'.jpg')