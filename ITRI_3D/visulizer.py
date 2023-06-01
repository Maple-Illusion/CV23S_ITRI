import open3d as o3d
import numpy as np
import pandas as pd

root = r'C:\Users\ataraxia\Documents\AV_LAB\NTU_CV\finals\ITRI_3D\ITRI_dataset\seq1\dataset\1681710717_532211005\sub_map.csv'
df = pd.read_csv(root,header=None)
print(df)
new = np.array(df,np.float64)
print(new)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(new)

o3d.visualization.draw_geometries([pcd])