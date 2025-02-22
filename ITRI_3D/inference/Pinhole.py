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





def trans_3Dto2D(inti_m,extri_m,D3_ls):
    D3_arr = np.array([np.concatenate([D3_ls,[1]],axis=0)]).astype(np.float64)
    D3_arr = D3_arr.transpose(1,0)
    print(D3_arr)
    PHM = np.matmul(inti_m,extri_m)
    D2_pt = np.matmul(PHM,D3_arr)
    # print('Nonthing')
    return D2_pt.astype(np.int64)

def trans_2Dto3D_v3(proj_m,extri_m,D2_ls):
    D2_arr = np.array([np.concatenate([D2_ls,[1]],axis=0)]).astype(np.float64)
    D2_arr = D2_arr.transpose(1,0)
    print('2D pt\n',D2_arr)
    print('Proj\n',proj_m)
    new_proj_m = proj_m
    new_proj_m[2,2] = 1
    Tf_m = np.matmul(new_proj_m,extri_m)
    new_rot = Tf_m[:,:3]
    new_trp = np.array([Tf_m[:,3]],np.float64).transpose(1,0)
    print(new_rot)
    print(new_trp)
    new_D2_arr = D2_arr - new_trp
    lu, piv = lu_factor(new_rot)
    D3_pt = lu_solve((lu, piv), new_D2_arr)
    D3_pt[2] = D3_pt[2]+1.63
    return D3_pt

def trans_2Dto3D_v2(proj_m,extri_m,D2_ls):
    D2_arr = np.array([np.concatenate([D2_ls,[1]],axis=0)]).astype(np.float64)
    D2_arr = D2_arr.transpose(1,0)
    # print('2D pt\n',D2_arr)
    # print('Proj\n',proj_m)
    new_proj_m = proj_m
    new_proj_m[2,2] = 1
    Tf_m = np.matmul(new_proj_m,extri_m)
    new_rot = Tf_m[:,:3]
    new_trp = np.array([Tf_m[:,3]],np.float64).transpose(1,0)
    # print(new_rot)
    # print(new_trp)
    new_D2_arr = D2_arr - new_trp
    D3_pt = np.matmul(np.linalg.inv(new_rot),new_D2_arr)
    D3_pt[2] = D3_pt[2]+1.63
    return D3_pt

def trans_2Dto3D(proj_m,extri_m,D2_ls):
    D2_arr = np.array([np.concatenate([D2_ls,[1]],axis=0)]).astype(np.float64)
    D2_arr = D2_arr.transpose(1,0)
    print('2D pt\n',D2_arr)
    ############# mm to m  ####10^-3
    # cam_m = cam_m * 10**(-3)
    # cam_m[2,2] = 1
    # print('CAM\n',cam_m)
    new_proj_m = proj_m # * 10**(-3)
    new_proj_m[2,2] = 1
    print('Proj\n',new_proj_m)
    ############
    # left = np.matmul(np.linalg.inv(cam_m),D2_arr)
    ######### discomp 3*4 .to 3*3+3*1
    Tf_m = np.matmul(new_proj_m,extri_m)
    print('TF\n',Tf_m)

    print('Inv TF\n',np.linalg.pinv(Tf_m))
    D3_pt = np.matmul(np.linalg.pinv(Tf_m),D2_arr)
    D3_pt[2] = D3_pt[2]+1.63
    return D3_pt

def extrinsic_metrix(cam_type): #
    pos_info = cfg.launch_file()
    
    
    rpy = EQ.q2rpy(pos_info[0][3:])
    coord = pos_info[0][0:3]
    coord[2] = coord[2]
    f_t  =EQ.coord_aug(coord,dim=3)
    f_rm = EQ.rotation_mx(rpy)
    f_trm = EQ.tranformation_mx(f_rm,f_t)
    if cam_type == 'fl':
        fl_rpy = EQ.q2rpy(pos_info[2][3:])
        fl_coord = pos_info[2][0:3]
        fl_t  =EQ.coord_aug(fl_coord,dim=3)
        fl_rm = EQ.rotation_mx(fl_rpy)
        fl_trm = EQ.tranformation_mx(fl_rm,fl_t)
        trm = np.matmul(f_trm,fl_trm)
    elif cam_type == 'fr':
        fr_rpy = EQ.q2rpy(pos_info[1][3:])
        fr_coord = pos_info[1][0:3]
        fr_t  =EQ.coord_aug(fr_coord,dim=3)
        fr_rm = EQ.rotation_mx(fr_rpy)
        fr_trm = EQ.tranformation_mx(fr_rm,fr_t)
        trm = np.matmul(f_trm,fr_trm)
    elif cam_type == 'fb':
        fl_rpy = EQ.q2rpy(pos_info[2][3:])
        fl_coord = pos_info[2][0:3]
        fl_t  =EQ.coord_aug(fl_coord,dim=3)
        fl_rm = EQ.rotation_mx(fl_rpy)
        fl_trm = EQ.tranformation_mx(fl_rm,fl_t)
        mid_trm = np.matmul(f_trm,fl_trm)
        
        fb_rpy = EQ.q2rpy(pos_info[3][3:])
        fb_coord = pos_info[3][0:3]
        fb_t  =EQ.coord_aug(fb_coord,dim=3)
        fb_rm = EQ.rotation_mx(fb_rpy)
        fb_trm = EQ.tranformation_mx(fb_rm,fb_t)
        trm = np.matmul(mid_trm,fb_trm)
    
    if cam_type == 'f':
        trm =f_trm
        
    ############ rule
    # zb = +1.63
    #base -> f -> fr
    #base -> f -> fl
    #base -> f -> fl -> fb
    ############ rule

    
    
    
    # print('Ext\n',trm)
    return trm
    

def intrinsic_metrix(int_root):
    int_m = cam.read_camera_info(int_root)
    proj_m = int_m[3]
    cam_m = int_m[0]
    intri = np.matmul(cam_m,proj_m)
    # print('Proj as instr\n',proj_m)
    return proj_m

    



    
    

