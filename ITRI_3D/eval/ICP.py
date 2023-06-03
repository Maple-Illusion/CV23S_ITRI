import os, sys, argparse, csv, copy
import numpy as np
import open3d as o3d
from glob import glob


def ICP(source, target, threshold, init_pose, iteration=30):
    # implement iterative closet point and return transformation matrix
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_pose,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iteration)
    )
    print(reg_p2p)
    print(reg_p2p.transformation)
    return reg_p2p.transformation

def csv_reader(filename):
    # read csv file into numpy array
    data = np.loadtxt(filename, delimiter=',')
    return data

def numpy2pcd(arr):
    # turn numpy array into open3d point cloud format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr)
    return pcd


if __name__ == '__main__':

    path = r'D:\CV\ITRI_DLC\ITRI_DLC\test1\dataset'
    f = open(r'D://CV/ITRI_DLC/ITRI_DLC/test1//localization_timestamp.txt')

    with open(r"D://CV/ITRI_DLC/ITRI_DLC/test1//pridict.txt", 'w', newline='') as csvfile:

        for line in f.readlines():
            line=line.strip() 
            sub_map = path + '\\'+ line +'\sub_map.csv'
            #print(sub_map)
            output = path + '\\'+  line +'\output.csv'
            initial_pose = path + '\\'+  line +'\initial_pose.csv'

            #path_name = './seq1/dataset/1681710717_541398178'

            # Target point cloud
            target = csv_reader(sub_map)
            target_pcd = numpy2pcd(target)

            # Source point cloud
            #TODO: Read your point cloud here#
            source = csv_reader(output)
            source_pcd = numpy2pcd(source)

            # Initial pose
            init_pose = csv_reader(initial_pose)

            # Implement ICP
            transformation = ICP(source_pcd, target_pcd, threshold=0.02, init_pose=init_pose)
            pred_x = transformation[0,3]
            pred_y = transformation[1,3]
            #print(pred_x, pred_y)
            # 建立 CSV 檔寫入器
            writer = csv.writer(csvfile)
            # 寫入一列資料
            writer.writerow([pred_x,pred_y])     
        f.close
