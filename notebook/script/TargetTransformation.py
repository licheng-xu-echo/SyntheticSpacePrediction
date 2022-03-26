# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 20:54:13 2022

@author: LiCheng_Xu
"""
import numpy as np
import matplotlib.pyplot as plt
def box_cox_trans(x,lambda_):
    '''
    Box-Cox Transformation

    Parameters
    ----------
    x : ndarray
        DESCRIPTION.
    lambda_ : float
        DESCRIPTION.

    Returns
    -------
    ndarray
        DESCRIPTION.

    '''
    if lambda_ != 0:
        return (np.power(x,lambda_)-1)/lambda_
    else:
        return np.log(x)

def de_box_cox_trans(x,lambda_):
    
    if lambda_ != 0:
        return np.power((1+lambda_*x),1/lambda_)
    else:
        return np.exp(np.power(x,lambda_))
    
def log_trans(x):
    '''
    Logarithmic Transformation

    Parameters
    ----------
    x : ndarray
        DESCRIPTION.

    Returns
    -------
    ndarray
        DESCRIPTION.

    '''
    
    return np.log((1-x)/(1+x))

def de_log_trans(x):
    return (1-np.exp(x))/(1+np.exp(x))

def ee2ddG(ee,T):
    '''
    Transformation from ee to ΔΔG
    Parameters
    ----------
    ee : ndarray
        Enantiomeric excess.
    T : ndarray or float
        Temperature (K).

    Returns
    -------
    ddG : ndarray
        ΔΔG (kcal/mol).
    '''
    
    ddG = np.abs(8.314 * T * np.log((1-ee)/(1+ee)))  # J/mol
    ddG = ddG/1000/4.18            # kcal/mol
    return ddG

def ddG2ee(ddG,T):
    '''
    Transformation from ΔΔG to ee. 
    Parameters
    ----------
    ddG : ndarray
        ΔΔG (kcal/mol).
    T : ndarray or float
        Temperature (K).

    Returns
    -------
    ee : ndarray
        Absolute value of enantiomeric excess.
    '''
    
    ddG = ddG*1000*4.18
    ee = (1-np.exp(ddG/(8.314*T)))/(1+np.exp(ddG/(8.314*T)))
    return np.abs(ee)
def vis_transformed_result(transformed_target,algorithm,color):
    grouped_value = {0.1:0,0.2:0,0.3:0,0.4:0,0.5:0,0.6:0,0.7:0,0.8:0,0.9:0,1:0}
    for target in transformed_target:
        for th in grouped_value:
            if target < th:
                grouped_value[th] += 1
                break

    x = np.array(list(grouped_value.keys()))
    y = np.array([grouped_value[item] for item in grouped_value])
    skew_score = np.abs(np.mean(transformed_target)-np.median(transformed_target))
    plt.figure(figsize=(10,5))
    plt.bar(x,y,width=0.05,color=color)
    plt.xlabel("y",fontsize=21)
    plt.ylabel("Count",fontsize=21)
    plt.xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"],fontsize=19)
    y_range = list(range(0,max(y)//10 * 10 + 20,10))
    plt.yticks(y_range,list(map(str,y_range)),fontsize=19)
    plt.text(0.05,y_range[-2]+5,"Skew Score: %.3f"%skew_score,fontsize=19)
    plt.title('%s Transferred Target Distribution'%algorithm,fontsize=21)
    plt.tick_params(left='on',bottom='on')
    plt.tight_layout()
