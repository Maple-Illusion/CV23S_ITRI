import argparse
import pandas as pd
import tool.read_cam_params as cam
import inference.Segmentation as seg
import inference.get_corners as gcor
import cv2
import os






if __name__ == '__main__':
    paser = argparse.ArgumentParser(prog='main',description='Data preparation')
    paser.add_argument('--camera_info',type=str,default=r'C:\Users\ataraxia\Documents\AV_LAB\NTU_CV\finals\ITRI_dataset\camera_info\lucid_cameras_x00\gige_100_b_hdr_camera_info.yaml')
    paser.add_argument('--cam_launch',type=str,default=r'C:\Users\ataraxia\Documents\AV_LAB\NTU_CV\finals\ITRI_dataset\camera_info\lucid_cameras_x00\camera_extrinsic_static_tf.launch')
    paser.add_argument('--data_folder',type=str,default=r'C:\Users\ataraxia\Documents\AV_LAB\NTU_CV\finals\ITRI_dataset\seq2\dataset\1681116127_215363553')
    paser.add_argument('--cam_num',type=int,default=1)
    # paser.add_argument('--target',type=str,default=r'')
    args = paser.parse_args()
    


    ################################### data preparation
    cam_root = args.camera_info
    img_root = os.path.join(args.data_folder,'raw_image.jpg')
    marker = pd.read_csv(os.path.join(args.data_folder,'detect_road_marker.csv'),header=None,encoding='utf-8')
    
    car_pos_matrix = pd.read_csv(os.path.join(args.data_folder,'initial_pose.csv'),header=None,encoding='utf-8')
    img = cv2.imread(img_root)
    # print(img.shape)
    h, w, c = img.shape
    
    # print(crop_img[0][:])
    # print(h,w)
    # cv2.imshow('test',crop_img[0])
    # cv2.waitKey(0)

    ################################### Segmentation
    crop_img, obj_id, prob, origin = seg.Crop(img,marker)
    cma_params = cam.read_camera_info(cam_root)


    ################################### Get corners
    corner_imgs, imgs_corners= gcor.find_corner(crop_img)  # imgs_corners 為四張圖f,fr,fl,b 的corner
    

    ###########check and combine points
    new = gcor.combine_crop(img,corner_imgs,origin)

    ################################### Pinhole model

