import numpy as np
import pandas as pd
import cv2


def Crop(img,target,threshold):
    box  = np.zeros((len(target.index),4),np.int64)
    
    crop = []
    cls_id = []
    score = []
    origins = []
    # print(box.shape)
    for i in target.index:
        sc = target.loc[i,5]
        if sc < threshold:
            continue
        h = target.loc[i,2] - target.loc[i,0] 
        w = target.loc[i,3] - target.loc[i,1] 
        if (w < 20):
            continue
        if (h < 20):
            continue

        box[i,:] = target.loc[i,:3].astype(np.int64)
        crop.append(img[box[i,1]:box[i,3],box[i,0]:box[i,2]])
        cls_id.append(target.loc[i,4])

        score.append(target.loc[i,5])

        origins.append([box[i,0],box[i,1]])
        # reg = crop[i]
        # print(reg.shape)
        # cv2.imshow(str(cls_id[i])+' '+str(score[i]),reg)
        # cv2.waitKey(0)
    #print(len(crop))
    return crop, cls_id, score, origins

def threshold(score):
    #####Not done
    if score < 0:
        del score
