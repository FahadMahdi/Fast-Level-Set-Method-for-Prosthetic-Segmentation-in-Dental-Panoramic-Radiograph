import cv2
from matplotlib import pyplot as plt
import image
# Read Images
filename = 'Y3415-P1.jpg'
img = cv2.imread(filename, 0)
print(img)

print(img.shape)
# crop the image
height = img.shape[0] #print(height) #height_crop = 0.3*height #print(height_crop)
                      # #height_crop = int(height_crop) #print(height_crop)
width = img.shape[1]
crop_img = img[ int(height*0.3):int(height*0.9),int(width*0.1):int(width*0.9)]
print(crop_img)

# Simple Threshold
retval, thresholding = cv2.threshold(crop_img,225,255,cv2.THRESH_BINARY)
retval, thresholding2 = cv2.threshold(crop_img,225,255,cv2.THRESH_BINARY_INV)
retval, thresholding3 = cv2.threshold(crop_img,225,255,cv2.THRESH_TRUNC)
retval, thresholding4 = cv2.threshold(crop_img,225,255,cv2.THRESH_TOZERO)
retval, thresholding5 = cv2.threshold(crop_img,225,255,cv2.THRESH_TOZERO_INV)
#plt.hist(thresholding.ravel(),256,[0,256])
#plt.show()

cv2.imshow('Original',img)
cv2.imshow('Original_cropped',crop_img)
cv2.imshow('Threshold_Original_cropped',thresholding)
cv2.imshow('Threshold2_Original_cropped',thresholding2)
cv2.imshow('Threshold3_Original_cropped',thresholding3)
cv2.imshow('Threshold4_Original_cropped',thresholding4)
cv2.imshow('Threshold5_Original_cropped',thresholding5)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('Y3415-P1_cropped.jpg',crop_img)
cv2.imwrite('Y3415-P1_Threshold_cropped.jpg',thresholding)