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
int_matrix = PH.intrinsic_metrix(int_root)
cam_sym = 'fb'
trfm = PH.extrinsic_metrix(cam_sym)
fn = cam_sym +'_Ex_metrix' + '.pkl'
pkl.pkl(fn,trfm)
trfm = np.vstack([trfm,np.array([0, 0, 0, 1]).astype(np.float64)])
print(trfm)
PHM = np.matmul(int_matrix,trfm)
print(PHM)

