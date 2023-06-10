import cv2
import os
import numpy as np


imgb = cv2.imread('../../ITRI_dataset/camera_info/lucid_cameras_x00/gige_100_b_hdr_mask.png', cv2.IMREAD_GRAYSCALE)

kernel = np.ones((5,5), np.uint8) 

imgb = cv2.dilate(imgb, kernel, iterations = 1) #膨脹mask的範圍，能夠殺更多點


h,w = imgb.shape


file = open('b_mask.txt','w')

for i in range(h):
    for j in range(w):
        if imgb[i][j] == 255:
            temp = [i,j]
            file.write(str(j)) 
            file.write(" ")
            file.write(str(i))
            file.write('\n')
            #print (i,j)
file.close()


imgf = cv2.imread('../../ITRI_dataset/camera_info/lucid_cameras_x00/gige_100_f_hdr_mask.png', cv2.IMREAD_GRAYSCALE)
h,w = imgf.shape
imgf = cv2.dilate(imgf, kernel, iterations = 1)


file = open('f_mask.txt','w')

for i in range(h):
    for j in range(w):
        if imgf[i][j] == 255:
            temp = [i,j]
            file.write(str(j)) 
            file.write(" ")
            file.write(str(i))
            file.write('\n')
            #print (i,j)
file.close()

imgfl = cv2.imread('../../ITRI_dataset/camera_info/lucid_cameras_x00/gige_100_fl_hdr_mask.png', cv2.IMREAD_GRAYSCALE)
h,w = imgfl.shape
imgfl = cv2.dilate(imgfl, kernel, iterations = 1)


file = open('fl_mask.txt','w')

for i in range(h):
    for j in range(w):
        if imgfl[i][j] == 255:
            temp = [i,j]
            file.write(str(j)) 
            file.write(" ")
            file.write(str(i))
            file.write('\n')
            #print (i,j)
file.close()

imgfr = cv2.imread('../../ITRI_dataset/camera_info/lucid_cameras_x00/gige_100_fr_hdr_mask.png', cv2.IMREAD_GRAYSCALE)
h,w = imgfr.shape

imgfr = cv2.dilate(imgfr, kernel, iterations = 1)

file = open('fr_mask.txt','w')

for i in range(h):
    for j in range(w):
        if imgfr[i][j] == 255:
            temp = [i,j]
            file.write(str(j)) 
            file.write(" ")
            file.write(str(i))
            file.write('\n')
            #print (i,j)
file.close()