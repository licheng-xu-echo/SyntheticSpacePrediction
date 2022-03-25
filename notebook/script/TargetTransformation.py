# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 20:54:13 2022

@author: LiCheng_Xu
"""
import numpy as np

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

