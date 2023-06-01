import pandas as pd
import dataset.read_cam_params as cam
import tool.pose_transform as EQ
import inference.Segmentation as seg
import inference.get_corners as gcor
import inference.Pinhole as PH
import dataset.cfg as cfg
import numpy as np
import os
import tool.PKL_out as pkl

int_root = r'C:\Users\ataraxia\Documents\AV_LAB\NTU_CV\finals\ITRI_3D\ITRI_dataset\camera_info\lucid_cameras_x00\gige_100_fl_hdr_camera_info.yaml'

int_matrix_ls = PH.intrinsic_metrix(int_root)
cam_sym = 'fl'
trfm = PH.extrinsic_metrix(cam_sym)
# print(trfm)
fn = cam_sym +'_Ex_metrix' + '.pkl'
# pkl.pkl(fn,trfm)
PHM = np.matmul(int_matrix_ls[0],trfm)
# ans = PH.trans_3Dto2D(int_matrix,trfm,[13.5683,-82.6988,44.6411])
# print(ans*10**(-6))
TT = cam.read_camera_info(int_root)
ans = PH.trans_2Dto3D(int_matrix_ls[1],int_matrix_ls[2],trfm,[500,500])
print(ans)

