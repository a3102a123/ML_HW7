from dataclasses import replace
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
from scipy.spatial.distance import *

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
RBF_gamma = 0.000001
linear_gamma = 1

def im_show_gray(img):
    plt.imshow(img,cmap='gray')

def data_to_img(img_data):
    return img_data.reshape(img_h,img_w)

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# mode : 0 : RBF , 1 : linear kernel(cosine similarity)
def kernel(X,Y,gamma = 1,is_center = True,mode=0):
    if mode == 0:
        dist = cdist(X,Y,'sqeuclidean')
        K = np.exp(-gamma * dist)
    elif mode == 1:
        K = gamma* (1 - cdist(X,Y,'cosine'))
    # Center the kernel matrix.
    if is_center:
        N = K.shape[0]
        one_n = np.ones((N,N)) / N
        K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    return K

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


def PCA(data,dim=None,is_kernel=False,gamma=15,mode=0):
    N = data.shape[0]
    if is_kernel:
        cov = kernel(data,data,gamma=gamma,is_center=True,mode=mode)
        # plt.figure()
        # plt.imshow(cov)
    else:
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

def LDA(data,dim=None,stride=9,class_num=15,is_kernel=False,gamma=15,mode=0):
    N , d = data.shape
    mean = np.mean(data,axis=0)
    Sw = np.zeros((d,d))
    Sb = np.zeros((d,d))

    if is_kernel:
        K = kernel(data,data,gamma,True,mode=mode)
        Z = np.zeros_like(K)
        diag_block = np.ones((stride,stride)) / stride
        for i in range(class_num):
            Z[i*stride:(i+1)*stride,i*stride:(i+1)*stride] = diag_block
        Sw = K@K
        Sb = K@Z@K
    else:
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
            Sb += stride * (d.T @ d)

    eigValues , eigVectors = np.linalg.eig(np.linalg.pinv(Sw)@Sb)
    index = np.argsort(eigValues)[::-1]
    if dim != None:
        index = index[:dim]
    W = eigVectors[:,index].real.astype(np.float64)
    return W
    # im_show_gray(W[:,1].reshape(img_h,img_w))

def reconstruct(W,mean,img_data):
    y = W.T @ (img_data - mean)
    x = W @ y + mean
    # x = normalize(x) * 255
    img = x.reshape(img_h,img_w)
    return img

def plot_W(W,is_PCA=True):
    plt.figure(figsize=(8, 6), dpi=120)
    if is_PCA:
        plt.suptitle("Eigenface")
    else:
        plt.suptitle("Fisherface")
    plt.subplots_adjust( wspace=0.2 ,hspace=0.4)
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.axis('off')
        plt.title(str(i+1) + ' Image')
        im_show_gray(W[:,i].reshape(img_h,img_w))

    fig = plt.figure(figsize=(10, 6), dpi=120)
    plt.suptitle("Face reconstruction")
    plt.subplots_adjust( wspace=0.1 ,hspace=0.5)
    np.random.seed(512)
    idx = np.random.choice(len(testing_label), size=10,replace=False)
    for i in range(10):
        img_data = testing_imgs[idx[i]]
        plt.subplot(2,10,i + 1)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        if i == 0:
            plt.ylabel("Origin image")
        img = img_data.reshape((img_h,img_w))
        im_show_gray(img)
        plt.subplot(2,10,i + 11)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        if i == 0:
            plt.ylabel("Reconstructed image")
        img = reconstruct(W,train_mean,img_data)
        im_show_gray(img)

# predict testing data by orthogonal projection matrix W which is got by (kernel) PCA , LDA
def predict(W,is_kernel=False,gamma=15,is_PCA=True,mode=0):
    count = 0
    # testing
    N = testing_imgs.shape[0]
    for i in range(N):
        # test kernel PCA , kernel LDA
        if is_kernel:
            t_k = kernel(train_imgs,[testing_imgs[i]],is_center=False,gamma=gamma,mode=mode)
            img = W.T@t_k
            l = knn(img.flatten(),W,train_label,3)
        # test PCA , LDA
        else:
            img = reconstruct(W,train_mean,testing_imgs[i])
            l = knn(img.flatten(),train_imgs,train_label,3)

        print("True label : {} , predict label : {}".format(testing_label[i],l))
        if testing_label[i] == l :
            count += 1
            # plt.figure()
            # im_show_gray(img)
    print("Accuracy : {} / {} ({:.2f}%)".format(count,N,count / N * 100))
    if not is_kernel:
        plot_W(W,is_PCA)

def test_kernel_PCA():
    # test kernel PCA in sample data set
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=100, random_state=123)
    plt.figure()
    plt.scatter(X[y==0, 0], X[y==0, 1],
    color='red', marker='^', alpha=0.5)
    plt.scatter(X[y==1, 0], X[y==1, 1],
    color='blue', marker='o', alpha=0.5)
    plt.tight_layout()

    print("Test Kernel PCA :")
    X_kpca = PCA(X,dim=2,is_kernel=True,gamma=15,mode=0)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
    ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1],color='red', marker='^', alpha=0.5)
    ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1],color='blue', marker='o', alpha=0.5)
    ax[1].scatter(X_kpca[y==0, 0], np.zeros((50,1))+0.02,color='red', marker='^', alpha=0.5)
    ax[1].scatter(X_kpca[y==1, 0], np.zeros((50,1))-0.02,color='blue', marker='o', alpha=0.5)
    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel('PC1')
    plt.tight_layout()

# main
load()
pca_W = PCA(train_imgs,train_imgs.shape[0] - 15)

train_imgs_pca = (train_imgs - np.mean(train_imgs,axis=0)) @ pca_W
lda_W = LDA(train_imgs_pca)

W = pca_W @ lda_W
# W = PCA(train_imgs)
W = PCA(train_imgs,dim=20,is_kernel=True,gamma=linear_gamma,mode=1)
# W = LDA(train_imgs,dim=20,is_kernel=True,gamma=linear_gamma,mode=0)

# plot_W(W)

# testing
predict(W,is_kernel=True,gamma=linear_gamma,is_PCA=False,mode=1)
# img = reconstruct(W,train_mean,testing_imgs[0])
# plt.figure()
# im_show_gray(img)
# plt.figure()
# im_show_gray(data_to_img(testing_imgs[0]))
info(W)
# test_kernel_PCA()
# plt.figure()
# plt.imshow(kernel(train_imgs,train_imgs,linear_gamma,True,1))
plt.show()