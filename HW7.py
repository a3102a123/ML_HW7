import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys

is_test = True
# global variable
train_imgs = []
testing_imgs = []
train_label = np.repeat(range(15),9)
testing_label = np.repeat(range(15),2)
train_mean = 0
testing_mean = 0
img_w = 0
img_h = 0

def im_show_gray(img):
    plt.imshow(img,cmap='gray')

def data_to_img(img_data):
    return img_data.reshape(img_h,img_w)

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def plot_W(W):
    plt.figure(figsize=(8, 6), dpi=120)
    plt.subplots_adjust( wspace=0.2 ,hspace=0.4)
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.axis('off')
        plt.title(str(i+1) + ' Image')
        im_show_gray(W[:,i].reshape(img_h,img_w))

def info(W):
    print("image size : {} x {} = {}".format(img_h,img_w,img_h*img_w))
    print("Traing image num : ",len(train_imgs))
    print("Testing image num : ",len(testing_imgs))
    print("W size : ",W.shape)

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
    global train_imgs,testing_imgs,train_mean,testing_mean
    train_imgs = read_dataset("Training")
    testing_imgs = read_dataset("testing")
    train_mean = np.mean(train_imgs,axis=0)
    testing_mean = np.mean(testing_imgs,axis=0)

def knn(img_data,imgs,label,k):
    N = len(label)
    dist = np.zeros((N))
    for i in range(N):
        dist[i] = np.linalg.norm(img_data - imgs[i])
    index = np.argsort(dist)
    result = label[index[:k]]
    result_l , result_c = np.unique(result,return_counts=True)
    if len(result_l) != 1 and np.max(result_c) == np.min(result_c):
        ret = label[np.argmin(dist[index[:k]])]
    else:
        ret = result_l[np.argmax(result_c)]
    print(np.unique(result,return_counts=True),ret)
    return ret


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

def reconstruct(W,mean,img_data):
    y = W.T @ img_data
    x = W @ y + mean
    x = normalize(x) * 255
    img = x.reshape(img_h,img_w)
    return img

# main
load()
pca_W = PCA(train_imgs,train_imgs.shape[0] - 15)

train_imgs_pca = (train_imgs - np.mean(train_imgs,axis=0)) @ pca_W
lda_W = LDA(train_imgs_pca,9,15)

W = pca_W @ lda_W
plot_W(W)
# for i in range(testing_imgs.shape[0]):
#     img = reconstruct(W,train_mean,testing_imgs[i])
#     knn(img.flatten(),train_imgs,train_label,3)
img = reconstruct(W,train_mean,testing_imgs[0])
plt.figure()
im_show_gray(img)
plt.figure()
im_show_gray(data_to_img(testing_imgs[0]))
info(W)

plt.show()