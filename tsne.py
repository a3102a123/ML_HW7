#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python
# 2.7.10, and it requires a working installation of NumPy. The implementation
# comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

import numpy as np
import pylab
import matplotlib.pyplot as plt
import imageio
import os

class gif_creater:
    def __init__(self,fps=2):
        self.imgs = []
        self.fps = fps

    def append(self,Y,labels,title):
        file_path = os.path.join("Image","GIF_Image","img{}.png".format(len(self.imgs)))
        plt.figure()
        plt.title(title)
        plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
        plt.savefig(file_path)
        plt.close()
        img = plt.imread(file_path)
        self.imgs.append(img)

    def save(self,filepath):
        imageio.mimsave(filepath, self.imgs, fps=self.fps)

    def clear(self):
        self.imgs.clear()

def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def tsne(X=np.array([]),labels = np.array([]), no_dims=2, initial_dims=50, perplexity=30.0,is_symmetric=False,gif_name=None):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))
    
    gif = gif_creater()
    pre_C = -1
    if is_symmetric:
        title="Symmetric SNE"
    else:
        title="t-SNE"

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        # sum_Y is some of square of every dimension of every data point in low dimensional space(N*1)
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        
        if is_symmetric:
            num = np.exp(-np.add(np.add(num, sum_Y).T, sum_Y))
        else:
            num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))

        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        if is_symmetric:
            for i in range(n):
                dY[i, :] = np.sum(np.tile(PQ[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)
        else:
            for i in range(n):
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))
            # if pre_C == C:
            #     break
            # pre_C = C

        # every 50 iteration store result to gif iamge
        if iter % 5 == 0:
            gif.append(Y,labels,title)

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.
        # break

    # save result to gif & visualize the pairwise similarity
    gif.save(os.path.join("Image","{}.gif".format(gif_name)))
    plot_similarity(P,Q,title)

    # Return solution
    return Y , P , Q

# visualize the pairwise similarity
def plot_similarity(P,Q,title):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), dpi=120)
    plt.suptitle(title)
    ax[0].set_title("similarities in high dimensional space")
    ax[0].hist(P.flatten(),bins=100,log=True)
    ax[1].set_title("similarities in low dimensional space")
    ax[1].hist(Q.flatten(),bins=100,log=True)


if __name__ == "__main__":
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    print("Running example on 2,500 MNIST digits...")
    X = np.loadtxt("Data/mnist2500_X.txt")
    labels = np.loadtxt("Data/mnist2500_labels.txt")
    # Y , P , Q = tsne(X, labels, 2, 50, 20.0,True,"Symmetric")
    Y , P , Q = tsne(X, labels, 2, 50, 20.0,False,"t-SNE")
    # for i in range(0,15,5):
    #     print("perplexity : ",i)
    #     if i == 0:
    #         i = 3
    #     Y , P , Q = tsne(X, labels, 2, 50, i,False,"t-SNE")
    #     pylab.figure()
    #     pylab.title("t-SNE perplexity = {}".format(i))
    #     pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
    pylab.show()
