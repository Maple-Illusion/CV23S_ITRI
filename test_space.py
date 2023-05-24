import argparse
import pandas as pd
import tool.read_cam_params as cam
import tool.pose_transform as EQ
import inference.Segmentation as seg
import inference.get_corners as gcor
import dataset.cfg as cfg
import cv2
import os

sp = cfg.launch_file()
# print(len(sp))

rpy = EQ.q2rpy(sp[1][3:])
# print(rpy)

dd = [EQ.rad2deg(_) for i,_ in enumerate(rpy)]
tp = [EQ.coord_aug(ls=n[0:3],dim=3) for b,n in enumerate(sp)]
print(dd)
rm = EQ.rotation_mx(dd)
print(rm)
tt  =EQ.coord_aug(sp[1][0:3],dim=3)
print(tt)
trm = EQ.tranformation_mx(rm,tt)
print(trm)
ocrr  = [50,50,50]
pos =EQ.trans_3Dto2D(ocrr,trm)