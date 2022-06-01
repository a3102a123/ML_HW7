import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

is_test = True
# global variable
traing_imgs = []
testing_imgs = []
img_w = 0
img_h = 0

def im_show_gray(img):
    plt.imshow(img,cmap='gray')

def read_dataset(folder_name):
    global img_w,img_h
    folder_path = os.path.join("Data",folder_name)
    file_list = os.listdir(folder_path)
    img_arr = []
    for f in file_list:
        img = cv2.imread(os.path.join(folder_path,f),-1)
        if is_test:
            img = cv2.resize(img, (20, 24), interpolation=cv2.INTER_AREA)
        if img_w == 0:
            img_h,img_w = img.shape
        img_arr.append(img.flatten())
    return np.array(img_arr)

def load():
    global traing_imgs,testing_imgs
    traing_imgs = read_dataset("Training")
    testing_imgs = read_dataset("testing")

def PCA(data):
    N = data.shape[0]
    mean = np.mean(data,axis=0)
    data = data - mean
    cov = np.cov(data.T)
    print(cov.shape)
    eigValues , eigVectors = np.linalg.eig(cov)
    index = np.argsort(eigValues)[::-1]
    W = eigVectors[:,index].astype(np.float64)
    print(W.shape)
    # im_show_gray(data[5].reshape(img_h,img_w))
    im_show_gray(W[:,-1].reshape(img_h,img_w))

# main
load()
PCA(testing_imgs)
# print(traing_imgs[0])
print("image size : {} x {} = {}".format(img_h,img_w,img_h*img_w))
print("Traing image num : ",len(traing_imgs))
print("Testing image num : ",len(testing_imgs))
# plt.imshow(testing_imgs[0],cmap='gray', vmin=0, vmax=255)
plt.show()