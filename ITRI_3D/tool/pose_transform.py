import numpy as np
import math


# def rpy2q(angle):


def q2rpy(complex):
    q = complex ## x y z w
    # print(q)
    R = np.arctan(2*(q[1]*q[2])/(1-2*(np.power(q[0],2)+np.power(q[1],2))))
    P = np.arcsin(2*(q[0]*q[2]-q[3]*q[1]))
    Y = np.arctan(2*(q[1]*q[2])/(1-2*(np.power(q[1],2)+np.power(q[2],2))))
    RPY = [R,P,Y]
    return RPY

def rad2deg(radian):
    degree = radian * (180/math.pi)
    return degree

def rotation_mx(ls):
    r = ls[0]
    p = ls[1]
    y = ls[2]
    Rm = [np.cos(y)*np.cos(p), -np.sin(y)*np.cos(r)+np.cos(y)*np.sin(r), np.sin(y)*np.sin(r)+np.cos(y)*np.cos(r)*np.sin(p),
          np.sin(y)*np.cos(p), np.cos(y)*np.cos(r)+np.sin(r)*np.sin(p)*np.sin(y), -np.cos(y)*np.sin(r)+np.sin(p)*np.sin(y)*np.cos(r),
          -np.sin(p), np.cos(p)*np.sin(r), np.cos(p)*np.cos(r)]
    return np.array(Rm).reshape(3,3)
def coord_aug(ls,dim = 2):
    dis = np.ones((1,dim),np.float64)
    dis[0][0:dim] = np.array(ls)
    # print(dis)
    # dis = np.hstack(,1,axis=0)
    # dis = np.concatenate(ls,[1],axis=0)
    dis = dis.astype(np.float64)
    
    return dis.transpose(1,0)
    

def tranformation_mx(rotate_mx,trans_mx):
    # reg = np.zeros((1,3),np.float64)
    # reg[0][2] = 1.
    # rt_m_aug = np.vstack(rotate_mx,reg)
    TFM = np.hstack([rotate_mx,trans_mx])
    return TFM