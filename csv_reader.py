import csv
import cv2


for i in range(2):
    with open('file_path.csv') as file:
        filepath = csv.reader(file)

        count = 0

        for row in filepath:
            print(row[i])
            img = cv2.imread(row[0])
            cv2.imshow('Original Image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    with open('save_path.csv') as save_file:
        savepath = csv.reader(save_file)

        for row in savepath:
            cv2.imwrite(row[i]+'.jpg', img)
    break
