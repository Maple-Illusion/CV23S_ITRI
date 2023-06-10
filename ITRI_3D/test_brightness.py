import argparse
import pandas as pd
import tool.read_cam_params as cam
import inference.Segmentation as seg
import inference.get_corners as gcor
import cv2
import os
import numpy as np
import inference.Pinhole as PH
import csv
from tqdm import tqdm


if __name__ == '__main__':
    paser = argparse.ArgumentParser(prog='main',description='Data preparation')
    paser.add_argument('--camera_info',type=str,default=r'../../ITRI_dataset/camera_info/lucid_cameras_x00')
    paser.add_argument('--cam_launch',type=str,default=r'../../ITRI_dataset/camera_info/lucid_cameras_x00/camera_extrinsic_static_tf.launch')
    paser.add_argument('--data_folder',type=str,default=r'../../ITRI_DLC/test2/dataset')
    #paser.add_argument('--data_folder',type=str,default=r'D:\CV\ITRI_DLC\ITRI_DLC\test2\dataset')
    # paser.add_argument('--data_folder',type=str,default=r'D:\CV\ITRI_dataset\ITRI_dataset\seq2\dataset\1681116127_676490974')
    paser.add_argument('--cam_num',type=int,default=1)
    paser.add_argument('--mask_folder',type=str,default=r'./')
    paser.add_argument('--output',type=str,default=r'../../ITRI_DLC/test2/dataset')
    # paser.add_argument('--target',type=str,default=r'')
    args = paser.parse_args()
    dir_list = os.listdir(args.data_folder)
    n = 0
    fr_mask= np.loadtxt(os.path.join(args.mask_folder,'fr_mask.txt'), dtype=int)
    fl_mask= np.loadtxt(os.path.join(args.mask_folder,'fl_mask.txt'), dtype=int)
    f_mask= np.loadtxt(os.path.join(args.mask_folder,'f_mask.txt'), dtype=int)
    b_mask= np.loadtxt(os.path.join(args.mask_folder,'b_mask.txt'), dtype=int)

    
    for img_folder in tqdm(sorted(dir_list)):
    ################################### data preparation
        
        img_root = os.path.join(args.data_folder,img_folder,'raw_image.jpg')
        #img_root = r'D:\CV\ITRI_dataset\ITRI_dataset\seq2\dataset\1681116129_479546307\raw_image.jpg'
        camara_type = pd.read_csv(os.path.join(args.data_folder,img_folder,'camera.csv'),header=None,encoding='utf-8')
        camara_type = camara_type.iloc[0][0]
        camara_type = camara_type.replace('/lucid_cameras_x00/', '')
        camara_type = camara_type + '_camera_info.yaml'
        cam_root = os.path.join(args.camera_info,camara_type)

        #print(img_root)
        #marker = pd.read_csv(r'D:\CV\ITRI_dataset\ITRI_dataset\seq2\dataset\1681116129_479546307\detect_road_marker.csv',header=None,encoding='utf-8')

        try:
            marker = pd.read_csv(os.path.join(args.data_folder,img_folder,'detect_road_marker.csv'),header=None,encoding='utf-8')
        except pd.errors.EmptyDataError:
            print("No data") 
        marker = marker.astype('float')
        marker = marker.clip(lower=0)

        #marker.iloc[:,[3]]=marker.iloc[:,[3]].clip(0, 926)
        # print(marker)
        car_pos_matrix = pd .read_csv(os.path.join(args.data_folder,img_folder,'initial_pose.csv'),header=None,encoding='utf-8')
        img = cv2.imread(img_root)
        # print(img.shape)
        
        threshold = 0.1
        ################################### Segmentation
        crop_img, obj_id, prob, origin = seg.Crop(img,marker,threshold)
        #for i in range(len(crop_img)):
        #    cv2.imshow('d',crop_img[i])
        #    cv2.waitKey(0)
        cma_params = cam.read_camera_info(cam_root)
        print(img_folder)
        ################################### Get corners
        corner_imgs, imgs_corners= gcor.find_corner1(crop_img,origin)  # imgs_corners 為四張圖f,fr,fl,b 的corner
        
        
        if img_folder == '1681114212_639755979':
            for i in corner_imgs:
                cv2.imshow('test',i)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        #if corner_imgs != []:
        #    cv2.imshow('test',corner_imgs[-1])
        #    cv2.waitKey(0)
        #    cv2.destroyAllWindows()
        # print(imgs_corners[0])
        #imgs_corners[4]= imgs_corners[4].astype(int)







        ###########check and combine points
        
        # new = gcor.combine_crop(img,corner_imgs,origin)

        ################################### Pinhole model


