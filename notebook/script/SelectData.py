# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 20:55:51 2022

@author: LiCheng_Xu
"""
import numpy as np

def cosine_similarity_manual(x,y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return np.round(num / denom,decimals=8)
    
def get_base_index(react_desc,threshold,init_idx=None,rand_seed=None):
    np.random.seed(rand_seed)
    #degree_thred = 20
    if init_idx == None:
        init_idx = np.random.randint(0,len(react_desc))
    base_index = [init_idx]
    diff_deg = np.arccos(np.array([cosine_similarity_manual(react_desc[init_idx],
               react_desc[i]) for i in range(len(react_desc))]))/np.pi * 180

    diff_index = np.where(diff_deg>threshold)[0]
    rand_idx = np.random.randint(0,len(diff_index))
    base_index.append(diff_index[rand_idx])

    while True:
        rest_react_desc = react_desc[diff_index]
        diff_deg = np.arccos(np.array([cosine_similarity_manual(rest_react_desc[rand_idx],
                    rest_react_desc[i]) for i in range(len(rest_react_desc))]))/np.pi*180.0
        diff_index = diff_index[np.where(diff_deg>threshold)[0]]

        try:
            rand_idx = np.random.randint(0,len(diff_index))
            base_index.append(diff_index[rand_idx])
        except:
            break
        if len(diff_index) <= 1:
            break
    return base_index

def get_selected_index(point_desc,pot_sel_desc,method='euclidean',threshold=2):
    method = method.lower()
    assert method in ['euclidean','cosine'], 'method should be euclidean or cosine'
    if method == 'euclidean':
        dist = np.linalg.norm(point_desc - pot_sel_desc,axis=1)   ## 空间中的某个点到剩余已知数据的距离
        sel_idx = np.where(dist < threshold)[0]
    elif method == 'cosine':
        angle = np.arccos(np.array([cosine_similarity_manual(point_desc,
                                    pot_sel_desc[i].reshape(1,-1)) 
                    for i in range(len(pot_sel_desc))]))/np.pi*180
        sel_idx = np.where(angle < threshold)[0]
    return sel_idx    