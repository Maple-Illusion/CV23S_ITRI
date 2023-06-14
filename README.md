# CV23S_ITRI
## 3D Reconstruction from Road Marker Feature Points
### 外系拼裝車
> * M11130413蘇子揚
> * R10943147李彥緯
> * M11152024謝維鈞
> * M11103441廖柏丞



# Environment install
* `conda install python=3.8`
* `pip install -r requirements.txt`

# 預測點雲

`python main.py --camera_info <camera_info_dir> --cam_launch <cam_launch_dir> --data_folder <data_folder_dir> --data_folder2 <data_folder2_dir> --mask_folder <mask_folder_dir> --output <output_dir>`


* **camera_info_dir**: Directory to camera_info\lucid_cameras_x00
* **cam_launch_dir**: Directory to the camera lunch檔位置 (camera_info\lucid_cameras_x00\camera_extrinsic_static_tf.launch)
* **data_folder_dir**: Directory to dlc1 test data (ITRI_DLC\ITRI_DLC\test2\dataset)
* **data_folder2**: Directory to dlc1 test data (ITRI_DLC2\ITRI_DLC2\test2\new_init_pose)
* **mask_folder_dir**: 使用get_mask_coor.py 計算出來的 車體mask位置，檔案名稱分別為fr_mask.txt、fl_mask.txt、f_mask.txt、b_mask.txt，請將位置倒到此檔案位置
* **output_dir**: output 預測點雲的位置


# 計算ICP

python ICP_v2.py 

需替換以下位置
* **第34行path:  ITRI_DLC\test2\dataset 
* **第35行path2:  ITRI_DLC2\test2\new_init_pose
* **第36行timestamp:  ITRI_DLC2/test2/localization_timestamp.txt
* **第37行output_file = output_dir

Note: 
1. output file後須把檔案中分割符號 , 換成空格" "已符合繳交格式 
2. 再跑test2時最後會有兩個file因為 main.py 沒有算出output.csv 導致無法計算，遇到這個問題時，請使用timestamp 相近的output.csv 補上就可以跑了
> file如下
* test2\dataset\1681114223_302607000 換成 test2\dataset\1681114223_282975987 的output.csv
* test2\dataset\1681114223_338684533 換成 test2\dataset\1681114223_356937246 的output.csv
