# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 21:00:31 2022

@author: LiCheng_Xu
"""
import numpy as np
from .TargetTransformation import ddG2ee
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.model_selection import KFold

def std_error(truth,pred):
    return np.sqrt(np.sum((truth - pred)**2)/(len(truth)-1))    
    
def genCountMap(_smi_set,synthetic_space,point_pred_map,point_error_map,species='TDG'):

    pred_space_ee = []
    pred_space_ddG = []
    pred_space_error = []
    for i in range(len(point_pred_map)):
        pred_ddG = point_pred_map[i]
        pred_error = point_error_map[i]
        pred_ee = ddG2ee(pred_ddG,60+273.15)
        pred_space_ddG.append(pred_ddG)
        pred_space_ee.append(pred_ee)
        pred_space_error.append(pred_error)

    _count_map = {}
    _up_count_map = {}
    _down_count_map = {}
    for _smi in _smi_set:
        _count_map[_smi] = {0.1:0,0.2:0,0.3:0,0.4:0,0.5:0,
                                  0.6:0,0.7:0,0.8:0,0.9:0,1:0}
        _up_count_map[_smi] = {0.1:0,0.2:0,0.3:0,0.4:0,0.5:0,
                                     0.6:0,0.7:0,0.8:0,0.9:0,1:0}
        _down_count_map[_smi] = {0.1:0,0.2:0,0.3:0,0.4:0,0.5:0,
                                       0.6:0,0.7:0,0.8:0,0.9:0,1:0}
    for i in range(len(synthetic_space)):
        _smi = synthetic_space.iloc[i][species]
        tmp_ee = pred_space_ee[i]
        tmp_ddG = pred_space_ddG[i]
        tmp_error = pred_space_error[i]
        tmp_ee_up = ddG2ee(tmp_ddG+tmp_error,60+273.15)
        tmp_ee_down = ddG2ee(tmp_ddG-tmp_error,60+273.15)

        for th in _count_map[_smi]:
            if tmp_ee < th:
                _count_map[_smi][th] += 1
                break

        for th in _up_count_map[_smi]:
            if tmp_ee_up < th:
                _up_count_map[_smi][th] += 1
                break

        for th in _down_count_map[_smi]:
            if tmp_ee_down < th:
                _down_count_map[_smi][th] += 1
                break
    _ave_count_map = {}
    for smi in _count_map:
        _ave_count_map[smi] = {}
        for key in _count_map[smi]:
            ave = int((_count_map[smi][key] + _up_count_map[smi][key] + _down_count_map[smi][key])/3)
            _ave_count_map[smi][key] = ave
    return _count_map,_up_count_map,_down_count_map,_ave_count_map    
    
def vis_distribution(ave_count_map,sel_smi_color_map,title=''):
    plt.figure(figsize=(14,5))
    x = np.array([10,20,30,40,50,60,70,80,90,100])
    x_smooth = np.linspace(x.min(), x.max(), 100)
    for smi in sel_smi_color_map:
        y_ave = np.array([ave_count_map[smi][key] for key in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]])
        y_ave_smooth = make_interp_spline(x,y_ave)(x_smooth)
        y_ave_smooth = np.where(y_ave_smooth>0,y_ave_smooth,0)
        plt.plot(x_smooth, y_ave_smooth,c=sel_smi_color_map[smi],alpha=0.9)
        plt.fill_between(x_smooth,y_ave_smooth,np.zeros(len(y_ave_smooth)),color=sel_smi_color_map[smi],alpha=0.1)
    plt.xticks([10,20,30,40,50,60,70,80,90,100],['<10','10-20','20-30','30-40','40-50',
                                                 '50-60','60-70','70-80','80-90','>90'],fontsize=14)
    plt.yticks([0,10000,20000,30000,40000],['0','10000','20000','30000','40000'],fontsize=14)
    plt.xlabel('ee (%)',fontsize=16)
    plt.ylabel('Count',fontsize=16)
    plt.tick_params(bottom='on',left='on')
    plt.title(title,fontsize=16)
    plt.tight_layout()
    plt.show()
    
def DeltaLearningPrediction(base_x,rest_x,space_x,base_y,rest_y,base_model,specific_model,base_model_only_point_idx,
                            selidx2idxs_map,k_fold_num,random_seed=2022):
    
    val_p = []
    val_Y = []
    kfold = KFold(n_splits=k_fold_num,shuffle=True,random_state=random_seed)
    for fit_idx,val_idx in kfold.split(base_x):
        fit_x,fit_y = base_x[fit_idx],base_y[fit_idx]
        val_x,val_y = base_x[val_idx],base_y[val_idx]
        base_model.fit(fit_x,fit_y)
        val_p.append(base_model.predict(val_x))
        val_Y.append(val_y)
    val_p = np.concatenate(val_p)
    val_y = np.concatenate(val_Y)
    base_error = std_error(val_y,val_p)
    point_error_map = {idx:base_error for idx in base_model_only_point_idx}   ## 给出空间每个点的预测误差，以此作为置信度

    base_model.fit(base_x,base_y)
    points_x = space_x[base_model_only_point_idx]
    points_p = base_model.predict(points_x)
    point_pred_map = {idx:points_p[i] for i,idx in enumerate(base_model_only_point_idx)}   ## Check


    for j,selidx in enumerate(selidx2idxs_map):

        idxs = selidx2idxs_map[selidx]
        sel_x = rest_x[list(selidx)]
        sel_y = rest_y[list(selidx)]

        val_p = []
        val_Y = []
        if len(sel_x) > k_fold_num:
            for fit_idx,val_idx in kfold.split(sel_x):
                fit_x,fit_y = sel_x[fit_idx],sel_y[fit_idx]
                val_x,val_y = sel_x[val_idx],sel_y[val_idx]
                fit_p = base_model.predict(fit_x)
                fit_d = fit_y - fit_p
                specific_model.fit(fit_x,fit_d)
                val_p.append(base_model.predict(val_x)+specific_model.predict(val_x))
                val_Y.append(val_y)
            val_p = np.concatenate(val_p)
            val_y = np.concatenate(val_Y)
            error = std_error(val_y,val_p)
            if error > base_error:
                for idx in idxs:
                    point_error_map[idx] = base_error
                points_x = space_x[idxs]
                points_p = base_model.predict(points_x)
                for i,idx in enumerate(idxs):
                    point_pred_map[idx] = points_p[i]
            else:
                for idx in idxs:
                    point_error_map[idx] = error
                sel_p = base_model.predict(sel_x)
                sel_d = sel_y - sel_p
                specific_model.fit(sel_x,sel_d)
                points_x = space_x[idxs]
                points_p = base_model.predict(points_x) + specific_model.predict(points_x)
                for i,idx in enumerate(idxs):
                    point_pred_map[idx] = points_p[i]
        else:
            points_x = space_x[idxs]
            points_p = base_model.predict(points_x)
            for i,idx in enumerate(idxs):
                point_pred_map[idx] = points_p[i]
                point_error_map[idx] = base_error
    sorted_point_error_map = {}
    for i in range(len(point_error_map)):
        sorted_point_error_map[i] = point_error_map[i]
    sorted_point_pred_map = {}
    for i in range(len(point_pred_map)):
        sorted_point_pred_map[i] = point_pred_map[i]

    return sorted_point_pred_map,sorted_point_error_map    
    
