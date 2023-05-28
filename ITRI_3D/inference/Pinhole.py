import numpy as np
import pandas as pd
import dataset.read_cam_params as cam
import tool.pose_transform as EQ
import inference.Segmentation as seg
import inference.get_corners as gcor
import dataset.cfg as cfg
import cv2
import os
import pickle



def trans_3Dto2D(coord,camera_m,extrintic_m):
    cor = np.array([coord]).transpose(1,0)
    T = np.matmul(camera_m,extrintic_m)
    Ans = np.matmul(T,cor)
    return Ans

def train_2Dto3D():
    print('Nonthing')

def extrinsic_metrix(cam_type):
    pos_info = cfg.launch_file()
    if cam_type == 'f':
        rpy = EQ.q2rpy(pos_info[3][3:])
        coord = pos_info[3][0:3]
    elif cam_type == 'fl':
        rpy = EQ.q2rpy(pos_info[1][3:])
        coord = pos_info[1][0:3]
    elif cam_type == 'fr':
        rpy = EQ.q2rpy(pos_info[0][3:])
        coord = pos_info[0][0:3]
    elif cam_type == 'fb':
        rpy = EQ.q2rpy(pos_info[2][3:])
        coord = pos_info[2][0:3]
    else: print('Not a camera!')
    
    dd = [EQ.rad2deg(_) for i,_ in enumerate(rpy)] 
    new_psoe_info = np.concatenate([coord,dd],axis=0)
    print(new_psoe_info.shape)  
    
    tt  =EQ.coord_aug(coord,dim=3)
    rm = EQ.rotation_mx(dd)
    trm = EQ.tranformation_mx(rm,tt)
    return trm
    

def intrinsic_metrix(int_root):
    int_m = cam.read_camera_info(int_root)
    proj_m = int_m[3]
    cam_m = int_m[0]
    intri = np.matmul(cam_m,proj_m)
    print(intri.shape)
    return intri

    



    
    

