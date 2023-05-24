import numpy as np 
import cv2



def img_prcess(img):
    h,w,c = img.shape
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)
    # thresh = cv2.dilate(thresh,None)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # approx = cv2.approxPolyDP(contours[0], 3, True)
    
    # img2 = np.zeros((h,w),np.uint8)
    # cv2.polylines(img2, [approx], True, (0, 0, 255), 2)
    # img2 = cv2.drawContours(thresh, contours, 3, (0,255,0), 10)
    # cv2.imshow('test',img2)
    # cv2.waitKey(0)
    
  
    return gray


def find_corner(img_ls):
    imgs_corners = []
    img_num = len(img_ls)
    for i in range(img_num):
        processed = img_prcess(img_ls[i])
        dst = cv2.cornerHarris(processed,2,3,0.04)
        #print(dst.shape)
        # cv2.imshow('test2',dst)
        # cv2.waitKey(0)
        dst = cv2.dilate(dst,None) # corner
        img_ls[i][dst>0.01*dst.max()]=[0,0,255]

        dst = cv2.cornerHarris(processed,2,3,0.04)
        dst = cv2.dilate(dst,None)
        ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
        dst = np.uint8(dst)
        # find centroids
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        # define the criteria to stop and refine the corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(processed,np.float32(centroids),(5,5),(-1,-1),criteria)
        imgs_corners.append(corners)
        #print(corners)
                # reg = gray[i]
        # print(reg.shape)
        # cv2.imshow('test',img_ls[i])
        # cv2.waitKey(0)
    return img_ls, imgs_corners

def combine_crop(img,corner_img,coord_aug):  ########for visualize only
    img_num = len(corner_img)
    for i in range(img_num):
        h,w,c = corner_img[i].shape
        img[coord_aug[i][1]:coord_aug[i][1]+h,coord_aug[i][0]:coord_aug[i][0]+w] = corner_img[i][:,:]
    cv2.imshow('test',img)
    cv2.waitKey(0)
    return img

def mask():
    mask =0
