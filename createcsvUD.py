import numpy as np
import cv2
import os,random as rn
from PIL import Image

IMG_DIR = './UD'
files=os.listdir(IMG_DIR)
num_files=len(files)
num=0
for img in os.listdir(IMG_DIR):
 if num==2025:
  break
 index=rn.randrange(0,num_files)
# img=Image.open(IMG_DIR+'/'+files[index])
# img_array = cv2.imread(os.path.join(IMG_DIR,img))
 img_array = cv2.imread(os.path.join(IMG_DIR,files[index]), cv2.IMREAD_GRAYSCALE)
 img_array = (img_array.flatten())
 img_array  = img_array.reshape(-1, 1).T
 with open('ud_gray.csv', 'ab') as f:
  np.savetxt(f, img_array, delimiter=",")
 num=num+1

