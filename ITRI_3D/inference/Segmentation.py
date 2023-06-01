import numpy as np
import pandas as pd
import cv2


def Crop(img,target):
    box  = np.zeros((len(target.index),4),np.int64)
    crop = []
    cls_id = []
    score = []
    origins = []
    # print(box.shape)
    for i in target.index:
        box[i,:] = target.loc[i,:3].astype(np.int64)
        crop.append(img[box[i,1]:box[i,3],box[i,0]:box[i,2]])
        cls_id.append(target.loc[i,4])
        score.append(target.loc[i,5])
        origins.append([box[i,0],box[i,1]])
        # reg = crop[i]
        # print(reg.shape)
        # cv2.imshow(str(cls_id[i])+' '+str(score[i]),reg)
        # cv2.waitKey(0)
    return crop, cls_id, score, origins

def threshold(score):
    #####Not done
    if score < 0:
        del score
