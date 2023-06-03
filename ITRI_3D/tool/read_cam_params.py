import numpy as np
import yaml


def read_camera_info(root):
    with open(root,'r',encoding="utf-8") as F:
        data = yaml.load(F,Loader=yaml.FullLoader)
        # print(yaml.dump(data))
        # print(type(data))
        # print(data)
        # print(data['camera_matrix']['data'])
        CamM  = np.array(data['camera_matrix']['data'])
        disC = np.array(data['distortion_coefficients']['data'])
        RectiM =  np.array(data['rectification_matrix']['data'])
        ProjM = np.array(data['projection_matrix']['data'])
    return [CamM.reshape(data['camera_matrix']['rows'],data['camera_matrix']['cols']), disC.reshape(data['distortion_coefficients']['rows'],data['distortion_coefficients']['cols']), 
            RectiM.reshape(data['rectification_matrix']['rows'],data['rectification_matrix']['cols']), ProjM.reshape(data['projection_matrix']['rows'],data['projection_matrix']['cols'])]


