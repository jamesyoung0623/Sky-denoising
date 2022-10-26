import glob
import cv2

file_names = sorted(glob.glob('./NTU_Sky_Noise_Matte/noisy/*'))

for name in file_names: 
    img = cv2.imread(name)
    print(img.shape)
    img = cv2.resize(img, (1024*(img.shape[1]//1000), 1024*(img.shape[0]//1000)))
    print(img.shape)
    cv2.imwrite(name, img)

    #for i in range(img.shape[0]//128):
    #    for j in range(img.shape[1]//128):
    #        cropped_img = img[i*128:(i+1)*128, j*128:(j+1)*128, :]
    #        cv2.imwrite('./cropped_matte/clean/' + name.split('/')[-1][:-4]+'_{:02d}_{:02d}.png'.format(i, j), cropped_img)