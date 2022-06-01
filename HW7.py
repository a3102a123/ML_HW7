import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

# global variable
traing_imgs = []
testing_imgs = []

def read_dataset(folder_name):
    folder_path = os.path.join("Data",folder_name)
    file_list = os.listdir(folder_path)
    img_arr = []
    for f in file_list:
        img = cv2.imread(os.path.join(folder_path,f))
        img_arr.append(img)
    return img_arr

def load():
    global traing_imgs,testing_imgs
    traing_imgs = read_dataset("Training")
    testing_imgs = read_dataset("testing")
    

# main
load()
# plt.imshow(testing_imgs[0])
plt.show()