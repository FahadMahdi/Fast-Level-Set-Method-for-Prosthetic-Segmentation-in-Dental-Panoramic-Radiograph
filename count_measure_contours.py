import cv2
import numpy as np
from matplotlib import pyplot as plt

# read image
img = cv2.imread('Y3424-P1_Threshold_cropped.jpg',0)
ret,thresh = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
height, width = img.shape
print ("height and width : ",height, width)
size = img.size
print ("size of the image in number of pixels", size)

# plot the binary image
imgplot = plt.imshow(thresh, 'gray'), plt.xticks([]), plt.yticks([])
plt.show(block=False), plt.pause(3), plt.close()
num_pixel = cv2.countNonZero(thresh)
print(num_pixel)