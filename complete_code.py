import cv2
import csv
import numpy as np
from matplotlib import pyplot as plt

for i in range(3):
    with open('file_path.csv') as file:
        filepath = csv.reader(file)

        count = 0

        for row in filepath:
            print(row[i])
            # read image
            img = cv2.imread(row[0])
            cv2.imshow('Original Image', img)
            cv2.waitKey(1)
            cv2.destroyAllWindows()

        # Convert BGR image to Gray

            imgray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # crop the image

            height = img.shape[0]
            width = img.shape[1]
            crop_img = imgray[int(height * 0.3):int(height * 0.9), int(width * 0.1):int(width * 0.9)]

        # Simple Threshold
            retval, thresholding = cv2.threshold(crop_img, 225, 255, cv2.THRESH_BINARY)

        # Number of nonzero pixels in the binary image
            num_pixel = cv2.countNonZero(thresholding)

            height, width = crop_img.shape
            print("height and width : ", height, width)
            size = crop_img.size
            print("size of the image in number of pixels", size)

        # Find contours from cropped image

            contours, hierarchy = cv2.findContours(thresholding, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            #print(contours)

        # Convert gray image to RGB

            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2RGB)
            thresholding = cv2.cvtColor(thresholding, cv2.COLOR_GRAY2RGB)

        # Draw contours

            crop_img = cv2.drawContours(crop_img, contours, -1, (0, 0, 255), 2)
            thresholding = cv2.drawContours(thresholding, contours, -1, (0, 0, 255), 2)

        # Show contours
            cv2.imshow('Threshold Image', thresholding)
            cv2.imshow('Cropped Image', crop_img)

            cv2.waitKey(1)
            cv2.destroyAllWindows()

    # Save Image

            # with open('save_path.csv') as save_file:
                # savepath = csv.reader(save_file)

            # for row in filepath:
            cv2.imwrite(row[i] + 'p.jpg', crop_img)

            # Print number of nonzero pixels in the binary image and percentage of it

            print('Total number of non zero pixels', num_pixel)
            print('Percentage of non zero pixels in the binary image %', num_pixel/size*100)
        break