import argparse
import pandas as pd
import tool.read_cam_params as cam
import inference.Segmentation as seg
import inference.get_corners as gcor
import dataset.cfg as cfg
import cv2
import os






if __name__ == '__main__':
    paser = argparse.ArgumentParser(prog='main',description='Data preparation')
    paser.add_argument('--camera_info',type=str,default=r'C:\Users\ataraxia\Documents\AV_LAB\NTU_CV\finals\ITRI_3D\ITRI_dataset\camera_info\lucid_cameras_x00\gige_100_fl_hdr_camera_info.yaml')
    paser.add_argument('--cam_launch',type=str,default=r'C:\Users\ataraxia\Documents\AV_LAB\NTU_CV\finals\ITRI_3D\ITRI_dataset\camera_info\lucid_cameras_x00\camera_extrinsic_static_tf.launch')
    paser.add_argument('--data_folder',type=str,default=r'C:\Users\ataraxia\Documents\AV_LAB\NTU_CV\finals\ITRI_3D\ITRI_dataset\seq1\dataset\1681710717_532211005')
    paser.add_argument('--cam_type',type=str,default='fisheye')
    paser.add_argument('--cam_num',type=int,default=1)
    # paser.add_argument('--target',type=str,default=r'')
    args = paser.parse_args()
    


    ################################### data preparation
    
    cam_root = args.camera_info
    img_root = os.path.join(args.data_folder,'raw_image.jpg')
    marker = pd.read_csv(os.path.join(args.data_folder,'detect_road_marker.csv'),header=None,encoding='utf-8')
    
    car_pos_matrix = pd.read_csv(os.path.join(args.data_folder,'initial_pose.csv'),header=None,encoding='utf-8')
    img = cv2.imread(img_root)
    h, w, c = img.shape
    
    cam_params = cam.read_camera_info(cam_root)
    print(cam_params[0])
    if args.cam_type == 'fisheye':
        new = img
        cv2.fisheye.undistortImage(img,new,cam_params[0],cam_params[1],(h,w))
        # print('done')
    # print(img.shape)
    cv2.imshow('good',img)
    cma_tf_ls = cfg.launch_file()
    

    # print(crop_img[0][:])
    # print(h,w)
    
    # cv2.waitKey(0)
    ################################### Preprocess

    ################################### Segmentation
    crop_img, obj_id, prob, origin = seg.Crop(img,marker)
    


    ################################### Get corners
    img_corner = gcor.find_corner(crop_img)
    

    ###########check and combine points
    new = gcor.combine_crop(img,img_corner,origin)

    ################################### Pinhole model

