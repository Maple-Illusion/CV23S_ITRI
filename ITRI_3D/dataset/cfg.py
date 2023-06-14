import os
import numpy as np

def launch_file():
    cam_1 = ['tf_main_camera__gige_100_fr_hdr','0.559084 0.0287952 -0.0950537 -0.0806252 0.607127 0.0356452 0.789699','$(arg main_camera_frame_id) /lucid_cameras_x00/gige_100_fr_hdr']
    cam_2 = ['tf_main_camera__gige_100_fl_hdr','-0.564697 0.0402756 -0.028059 -0.117199 -0.575476 -0.0686302 0.806462','$(arg main_camera_frame_id) /lucid_cameras_x00/gige_100_fl_hdr']
    cam_3 = ['velo2cam_tf__gige_100_fl_hdr_gige_100_fr_hdr_mix','-1.2446 0.21365 -0.91917 0.074732 -0.794 -0.10595 0.59393','$(arg main_camera_frame_id) /lucid_cameras_x00/gige_100_b_hdr']
    car_base = ['tf_main_camera__base_link_tmp','0.0 0.0 0.0 -0.5070558775462676 0.47615311808704197 -0.4812773544166568 0.5334272708696808','/base_link $(arg main_camera_frame_id)']
    ######Filter 
    tf_ls = []
    cam_ls = [car_base,cam_1,cam_2,cam_3]
    for i,_ in enumerate(cam_ls):
        info = np.array(_[1].split(' '))
        tf_ls.append(info.astype(np.float64))
        # print(info)

    
    return tf_ls