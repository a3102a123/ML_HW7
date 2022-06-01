import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

is_test = True
# global variable
train_imgs = []
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
    global train_imgs,testing_imgs
    train_imgs = read_dataset("Training")
    testing_imgs = read_dataset("testing")

def PCA(data,dim=None):
    N = data.shape[0]
    mean = np.mean(data,axis=0)
    data = data - mean
    cov = np.cov(data.T)
    eigValues , eigVectors = np.linalg.eig(cov)
    index = np.argsort(eigValues)[::-1]
    if dim != None:
        index = index[:dim]
    W = eigVectors[:,index].real.astype(np.float64)
    # im_show_gray(data[5].reshape(img_h,img_w))
    # im_show_gray(W[:,5].reshape(img_h,img_w))
    return W

def LDA(data,stride,class_num):
    N , d = data.shape
    mean = np.mean(data,axis=0)
    Sw = np.zeros((d,d))
    Sb = np.zeros((d,d))
    # get the mean of every image subjects
    classes_mean = np.zeros((class_num,d))
    for i in range(class_num):
        begin = i * stride
        end = begin + stride
        classes_mean[i,:] = np.mean(data[begin:end,:],axis=0)

    # the distance within class scatter
    for i in range(class_num):
        for j in range(stride):
            d = data[i*stride+j,:] - classes_mean[i,:]
            Sw += d.T @ d

    # the distance between class scatter
    for i in range(class_num):
        d = classes_mean[i,:] - mean
        Sb += stride * d.T @ d

    eigValues , eigVectors = np.linalg.eig(np.linalg.pinv(Sw)@Sb)
    index = np.argsort(eigValues)[::-1]
    W = eigVectors[:,index].real.astype(np.float64)
    return W
    # im_show_gray(W[:,1].reshape(img_h,img_w))

# main
load()
pca_W = PCA(train_imgs,train_imgs.shape[0] - 15)
train_imgs_pca = (train_imgs - np.mean(train_imgs,axis=0)) @ pca_W
print(train_imgs_pca.shape)
# plt.imshow(train_imgs_pca[0,:].reshape(img_h,img_w))
# plt.show()
lda_W = LDA(train_imgs_pca,9,15)
W = pca_W @ lda_W
plt.imshow(W[:,5].reshape(img_h,img_w))
# print(train_imgs[0])
print("image size : {} x {} = {}".format(img_h,img_w,img_h*img_w))
print("Traing image num : ",len(train_imgs))
print("Testing image num : ",len(testing_imgs))
# plt.imshow(testing_imgs[5,:].reshape(img_h,img_w),cmap='gray', vmin=0, vmax=255)
plt.show()