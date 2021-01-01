import numpy as np
import cv2
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt
from skimage import color, io


def grad(x):
    return np.array(np.gradient(x))


def norm(x, axis=0):
    return np.sqrt(np.sum(np.square(x), axis=axis))


def stopping_fun(x):
    return 1. / (1. + norm(grad(x)) ** 2)


img = io.imread('Y3415-P1.jpg')
img = color.rgb2gray(img)
# img = img - np.mean(img)

# Smooth the image to reduce noise and separation between noise and edge becomes clear
# img_smooth = scipy.ndimage.filters.gaussian_filter(img, sigma=0.5)

F = stopping_fun(img)


def default_phi(x):
    # Initialize surface phi at the border (5px from the border) of the image
    # i.e. 1 outside the curve, and -1 inside the curve
    phi = np.ones(x.shape[:2])
    phi[5:-5, 5:-5] = -1.
    return phi


dt = 1.
n_iter = 10
for i in range(n_iter):
    dphi = grad(phi)
    dphi_norm = norm(dphi)

    dphi_t = F * dphi_norm

    phi = phi + dt * dphi_t

cv2.imshow('Original', img)
# cv2.imshow('Smooth', img_smooth)
cv2.waitKey(0)
cv2.destroyAllWindows()
