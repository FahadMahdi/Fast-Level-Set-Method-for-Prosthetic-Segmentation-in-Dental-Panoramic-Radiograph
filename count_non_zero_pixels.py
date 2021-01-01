import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import regionprops

# Read Image
img = cv2.imread('Y3424-P1_Threshold_cropped.jpg', 0)

# Calculating total pixels in the image
height, width = img.shape
print("Image Size : ", height, width)
size = img.size
print("Total number of pixels", size)

# Applying filter to reduce noise # blurred_img = cv2.GaussianBlur(img, (5, 5), 1)
# cv2.imshow('Blurred Y3424-P1', blurred_img) # cv2.waitKey(2000) # cv2.destroyAllWindows()

# Number of contour calculation
tmp = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours = tmp[0] if len(tmp) == 2 else tmp[1]

#contour_area = cv2.contourArea(contours[0])
#cont_perimeter = cv2.arcLength(contours[0], True)

#print('Contour area of ', contour_area)
#print('Contour perimeter of ', cont_perimeter)

# Draw contours
cv2.drawContours(img, contours, 3, (0, 255, 0), 3)
cv2.imshow('img', img)

cv2.waitKey(2000)
cv2.destroyAllWindows()

# Number of nonzero pixels in the binary image
num_pixel = cv2.countNonZero(img)
print('Total number of non zero pixels', num_pixel)
print('Percentage of non zero pixels in the binary image %', num_pixel/size*100)