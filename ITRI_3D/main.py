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
    paser.add_argument('--camera_info',type=str,default=r'D:\CV\ITRI_dataset\ITRI_dataset\camera_info\lucid_cameras_x00')
    paser.add_argument('--cam_launch',type=str,default=r'D:\CV\ITRI_dataset\ITRI_dataset\camera_info\lucid_cameras_x00\camera_extrinsic_static_tf.launch')
    paser.add_argument('--data_folder',type=str,default=r'D:\CV\ITRI_DLC\ITRI_DLC\test2\dataset')
    paser.add_argument('--data_folder2',type=str,default=r'D:\CV\ITRI_DLC2\ITRI_DLC2\test2\new_init_pose')

    # paser.add_argument('--data_folder',type=str,default=r'D:\CV\ITRI_dataset\ITRI_dataset\seq2\dataset\1681116127_676490974')
    paser.add_argument('--cam_num',type=int,default=1)
    paser.add_argument('--mask_folder',type=str,default= r'D:\CV\CV23S_ITRI-main_2\CV23S_ITRI-main\ITRI_3D\mask_txt')
    paser.add_argument('--output',type=str,default=r'D:\CV\ITRI_DLC\ITRI_DLC\test2\dataset')
    # paser.add_argument('--target',type=str,default=r'')
    args = paser.parse_args()
    dir_list = os.listdir(args.data_folder2)
    n = 0
    fr_mask= np.loadtxt(os.path.join(args.mask_folder,'fr_mask.txt'), dtype=int)
    fl_mask= np.loadtxt(os.path.join(args.mask_folder,'fl_mask.txt'), dtype=int)
    f_mask= np.loadtxt(os.path.join(args.mask_folder,'f_mask.txt'), dtype=int)
    b_mask= np.loadtxt(os.path.join(args.mask_folder,'b_mask.txt'), dtype=int)

    
    for img_folder in tqdm(dir_list):
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

        marker = pd.read_csv(os.path.join(args.data_folder,img_folder,'detect_road_marker.csv'),header=None,encoding='utf-8')
        marker = marker.astype('float')
        marker = marker.clip(lower=0)

        #marker.iloc[:,[3]]=marker.iloc[:,[3]].clip(0, 926)
        # print(marker)
        car_pos_matrix = pd .read_csv(os.path.join(args.data_folder2,img_folder,'initial_pose.csv'),header=None,encoding='utf-8')
        img = cv2.imread(img_root)
        # print(img.shape)
        
        threshold = 0
        ################################### Segmentation
        crop_img, obj_id, prob, origin = seg.Crop(img,marker,threshold)
        cma_params = cam.read_camera_info(cam_root)

        ################################### Get corners
        corner_imgs, imgs_corners= gcor.find_corner(crop_img,origin)  # imgs_corners 為四張圖f,fr,fl,b 的corner

        # print(imgs_corners[0])
        #imgs_corners[4]= imgs_corners[4].astype(int)


        def setdiff2d_set(arr1, arr2):
            set1 = set(map(tuple, arr1))
            set2 = set(map(tuple, arr2))
            return np.array(list(set1 - set2))
        
        #print(fr_mask)
        # Convert the arrays to tuples
        new_corners = []

        

        for i in range(len(imgs_corners)):
            if camara_type == 'gige_100_fr_hdr_camera_info.yaml':
                # print(fr_mask.shape)
                new_corner = gcor.mask_diff(imgs_corners[i],fr_mask)
                new_corners.append(new_corner)
                
                outpath = os.path.join(args.output,img_folder,'output.csv')
                with open(outpath, 'w', newline='') as csvfile:

                    for point in new_corner:

                        int_root = os.path.join(args.camera_info,camara_type)
                        int_matrix_ls = PH.intrinsic_metrix(int_root)
                        cam_sym = 'fr'
                        trfm = PH.extrinsic_metrix(cam_sym)
                        ans = PH.trans_2Dto3D_v2(int_matrix_ls,trfm,[point[0],point[1]])

                        # 建立 CSV 檔寫入器
                        writer = csv.writer(csvfile)
                        # 寫入一列資料
                        writer.writerow([ans[0][0], ans[1][0], ans[2][0]])                     

            elif camara_type == 'gige_100_fl_hdr_camera_info.yaml':
                new_corner = gcor.mask_diff(imgs_corners[i],fl_mask)
                new_corners.append(new_corner)
                outpath = os.path.join(args.output,img_folder,'output.csv')
                with open(outpath, 'w', newline='') as csvfile:

                    for point in new_corner:

                        int_root = os.path.join(args.camera_info,camara_type)
                        int_matrix_ls = PH.intrinsic_metrix(int_root)
                        cam_sym = 'fl'
                        trfm = PH.extrinsic_metrix(cam_sym)
                        ans = PH.trans_2Dto3D_v2(int_matrix_ls,trfm,[point[0],point[1]])

                        # 建立 CSV 檔寫入器
                        writer = csv.writer(csvfile)
                        # 寫入一列資料
                        writer.writerow([ans[0][0], ans[1][0], ans[2][0]])        



            elif camara_type == 'gige_100_f_hdr_camera_info.yaml':
                new_corner = gcor.mask_diff(imgs_corners[i],f_mask)
                new_corners.append(new_corner)
                outpath = os.path.join(args.output,img_folder,'output.csv')
                with open(outpath, 'w', newline='') as csvfile:

                    for point in new_corner:

                        int_root = os.path.join(args.camera_info,camara_type)
                        int_matrix_ls = PH.intrinsic_metrix(int_root)
                        cam_sym = 'f'
                        trfm = PH.extrinsic_metrix(cam_sym)
                        ans = PH.trans_2Dto3D_v2(int_matrix_ls,trfm,[point[0],point[1]])

                        # 建立 CSV 檔寫入器
                        writer = csv.writer(csvfile)
                        # 寫入一列資料
                        writer.writerow([ans[0][0], ans[1][0], ans[2][0]])        

            elif camara_type == 'gige_100_b_hdr_camera_info.yaml':
                new_corner = gcor.mask_diff(imgs_corners[i],b_mask)
                new_corners.append(new_corner)
                outpath = os.path.join(args.output,img_folder,'output.csv')
                with open(outpath, 'w', newline='') as csvfile:

                    for point in new_corner:

                        int_root = os.path.join(args.camera_info,camara_type)
                        int_matrix_ls = PH.intrinsic_metrix(int_root)
                        cam_sym = 'fb'
                        trfm = PH.extrinsic_metrix(cam_sym)
                        ans = PH.trans_2Dto3D_v2(int_matrix_ls,trfm,[point[0],point[1]])

                        # 建立 CSV 檔寫入器
                        writer = csv.writer(csvfile)
                        # 寫入一列資料
                        writer.writerow([ans[0][0], ans[1][0], ans[2][0]])        

        # # print(len(new_corners))
        # img2= cv2.imread(img_root)

        # point_size = 1
        # point_color = (0, 0, 255) # red
        # thickness = 3
        
        # #print(asd)
        # for i in range(len(new_corners)):

        #     for point in new_corners[i]:
        #         x_l = point[0]
        #         y_l = point[1]
        #         #print(x_l,y_l)
        #         cv2.circle(img2, (x_l,y_l), point_size, point_color, thickness)
        # cv2.imshow('test',img2)
        # cv2.waitKey(0)






        ###########check and combine points
        
        # new = gcor.combine_crop(img,corner_imgs,origin)

        ################################### Pinhole model

