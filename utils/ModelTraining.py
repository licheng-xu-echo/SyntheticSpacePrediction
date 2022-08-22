# -*- coding: utf-8 -*-
"""

@author: Li-Cheng Xu
"""
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset,DataLoader
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error,r2_score
from scipy.stats import pearsonr
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


random_seed = 2022

class TorchDataset(Dataset):
    def __init__(self,desc,target):
        self.desc = torch.tensor(desc,dtype=torch.float32)
        self.target = torch.tensor(target,dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.target)
    def __getitem__(self,idx):
        d,t = self.desc[idx],self.target[idx]
        return d,t
class NeuralNetwork(nn.Module):
    def __init__(self,input_size,node_num=50,hidden_layer_num=1,output_size=1):
        super(NeuralNetwork, self).__init__()
        #self.flatten = nn.Flatten()
        hidden_layers = []
        for i in range(hidden_layer_num):
            hidden_layers.append(nn.Linear(node_num,node_num))
            hidden_layers.append(nn.ReLU())

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size,node_num),
            nn.ReLU(),
            *hidden_layers,
            nn.Linear(node_num, output_size))

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
def init_model(model_name,desc_name,best_params,random_seed=None):
    assert model_name == 'RF' or model_name == 'ET' or model_name == 'KNN' or\
            model_name == 'DT' or model_name == 'SVR' or model_name == 'KRR' or\
            model_name == 'XGB', 'Not support this ML model %s'%model_name
    if model_name == 'RF':
        model = RandomForestRegressor(n_jobs=-1,random_state=random_seed,
                                      max_depth=best_params[(model_name,desc_name)]['max_depth'],
                                      n_estimators=best_params[(model_name,desc_name)]['n_estimators'])
    elif model_name == 'ET':
        model = ExtraTreesRegressor(n_jobs=-1,random_state=random_seed,
                                    max_depth=best_params[(model_name,desc_name)]['max_depth'],
                                    n_estimators=best_params[(model_name,desc_name)]['n_estimators'])
    elif model_name == 'KNN':
        model = KNeighborsRegressor(n_jobs=-1,
                                    leaf_size=best_params[(model_name,desc_name)]['leaf_size'],
                                    n_neighbors=best_params[(model_name,desc_name)]['n_neighbors'])
    elif model_name == 'DT':
        model = DecisionTreeRegressor(random_state=random_seed,
                                      max_depth=best_params[(model_name,desc_name)]['max_depth'],
                                      min_samples_split=best_params[(model_name,desc_name)]['min_samples_split'])
    elif model_name == 'SVR':
        model = SVR(gamma=best_params[(model_name,desc_name)]['gamma'],
                    kernel=best_params[(model_name,desc_name)]['kernel'])
    elif model_name == 'KRR':
        model = KernelRidge(gamma=best_params[(model_name,desc_name)]['gamma'])
    elif model_name == 'XGB':
        model = XGBRegressor(n_estimators=best_params[(model_name,desc_name)]['n_estimators'],
                             max_depth=best_params[(model_name,desc_name)]['max_depth'],
                             n_jobs=-1,random_state=random_seed)
    return model

def oos_fit_pred(model,train_x,train_y,oos_x,oos_y,n=10,sel_index=None):
    
    if np.all(sel_index != None):
        train_x = train_x[:,sel_index]
        oos_x = oos_x[:,sel_index]
        
    oos_Pred = []
    # To increase robustness, model training procedure is repeat n times
    for try_ in range(n):
        model.fit(train_x,train_y)
        oos_p_ = model.predict(oos_x)
        oos_Pred.append(oos_p_)
    oos_p = np.mean(oos_Pred,axis=0)

    oos_r2 = r2_score(oos_y,oos_p)
    oos_pearson_r,_ = pearsonr(oos_y,oos_p)
    oos_mae = mean_absolute_error(oos_y,oos_p)
    return oos_p,oos_mae,oos_r2,oos_pearson_r

def cv_fit_pred(model,train_val_x,train_val_y,cv,n=10,sel_index=None):
    
    if np.all(sel_index != None):
        train_val_x = train_val_x[:,sel_index]
        
    all_test_y = []
    all_test_p = []
    
    for train_idx,test_idx in cv.split(train_val_x):
        train_x,test_x = train_val_x[train_idx],train_val_x[test_idx]
        train_y,test_y = train_val_y[train_idx],train_val_y[test_idx]

        test_P = []
        # To increase robustness, model training procedure is repeat n times
        for try_ in range(n): 
            model.fit(train_x,train_y)
            test_p = model.predict(test_x)
            test_P.append(test_p)
        test_p = np.mean(test_P,axis=0)
        all_test_p.append(test_p)
        all_test_y.append(test_y)
    cv_test_p = np.concatenate(all_test_p)
    cv_test_y = np.concatenate(all_test_y)
    cv_mae = mean_absolute_error(cv_test_y,cv_test_p)
    cv_r2 = r2_score(cv_test_y,cv_test_p)
    cv_pearson_r,_ = pearsonr(cv_test_y,cv_test_p)
    
    return cv_test_y,cv_test_p,cv_mae,cv_r2,cv_pearson_r

def feature_selection(model,train_val_x,train_val_y,cv):
    selector = RFECV(model, step=1, min_features_to_select=1,cv=cv, n_jobs=-1)
    selector = selector.fit(train_val_x, train_val_y)
    sel_index = np.where(selector.support_==True)[0]
    return sel_index

def cv_fit_pred_NN(train_val_x,train_val_y,epoch,node_num,cv,random_seed):
    cv_target = []
    cv_pred = []
    loss_fn = nn.MSELoss().to(device) ## MAE
    learning_rate = 1e-3
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    layer_num, learning_rate, batch_size, patient_epoch = 1, 1e-3, 2, 50

    for train_idx,test_idx in cv.split(train_val_x):
        best_r2 = 0
        epoch_ = 0
        train_x,test_x = train_val_x[train_idx],train_val_x[test_idx]
        train_y,test_y = train_val_y[train_idx],train_val_y[test_idx]
        train_dataset = TorchDataset(train_x,train_y)
        train_dataloader = DataLoader(train_dataset,shuffle=True,batch_size=batch_size)

        model = NeuralNetwork(train_val_x.shape[1],node_num,layer_num-1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for i in range(epoch):
            epoch_ += 1
            for data in train_dataloader:
                pred = model(data[0].to(device))
                loss = loss_fn(pred,data[1].to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            test_p = model(torch.tensor(test_x).to(device))
            test_p = test_p.detach().cpu().numpy().reshape(-1)

            r2 = r2_score(test_y,test_p)
            if best_r2 < r2:
                torch.save(model.state_dict(),'./torch_model.pth')
                best_r2 = r2
                epoch_ = 0
            if epoch_ >= patient_epoch:
                break

        best_model = NeuralNetwork(train_val_x.shape[1],node_num,layer_num-1).to(device)
        best_model.load_state_dict(torch.load('./torch_model.pth'))
        test_p = best_model(torch.tensor(test_x).to(device))
        test_p = test_p.detach().cpu().numpy().reshape(-1)
        cv_pred.append(test_p)
        cv_target.append(test_y)
    cv_pred = np.concatenate(cv_pred)
    cv_target = np.concatenate(cv_target)
    cv_mae = mean_absolute_error(cv_target,cv_pred)
    cv_r2 = r2_score(cv_target,cv_pred)
    cv_pearson_r,_ = pearsonr(cv_target,cv_pred)
    return cv_target,cv_pred,cv_mae,cv_r2,cv_pearson_r

def oos_fit_pred_NN(train_x,train_y,oos_x,oos_y,epoch,node_num,random_seed):
    loss_fn = nn.MSELoss().to(device)
    learning_rate = 1e-3
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    layer_num, learning_rate, batch_size, patient_epoch = 1, 1e-3, 2, 50
    
    best_r2 = 0

    train_dataset = TorchDataset(train_x,train_y)
    train_dataloader = DataLoader(train_dataset,shuffle=True,batch_size=batch_size)

    model = NeuralNetwork(train_x.shape[1],node_num,layer_num-1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for i in range(epoch):

        for data in train_dataloader:
            pred = model(data[0].to(device))
            loss = loss_fn(pred,data[1].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_p = model(torch.tensor(train_x).to(device))
        train_p = train_p.detach().cpu().numpy().reshape(-1)

        r2 = r2_score(train_y,train_p)

        if best_r2 < r2:
            torch.save(model.state_dict(),'./torch_model.pth')
            best_r2 = r2


    best_model = NeuralNetwork(train_x.shape[1],node_num,layer_num-1).to(device)
    best_model.load_state_dict(torch.load('./torch_model.pth'))
    oos_p = best_model(torch.tensor(oos_x).to(device))
    oos_p = oos_p.detach().cpu().numpy().reshape(-1)
    oos_r2 = r2_score(oos_y,oos_p)
    oos_mae = mean_absolute_error(oos_y,oos_p)
    oos_pearson_r,_ = pearsonr(oos_y,oos_p)

    return oos_p,oos_mae,oos_r2,oos_pearson_r
def model_evaluation(train_val_desc,train_val_target,
                         oos_desc,oos_target,best_params,
                         model_name,desc_name,random_seed=random_seed):
    
    assert model_name == 'RF' or model_name == 'ET' or model_name == 'KNN' or\
            model_name == 'DT' or model_name == 'SVR' or model_name == 'KRR' or\
            model_name == 'XGB' or model_name == 'NN', 'Not support this ML model %s'%model_name
    
    sel_index = None
    cv = KFold(n_splits=10, shuffle=True, random_state=random_seed)
    if model_name != 'NN':
        if model_name == 'ET':

            model = ExtraTreesRegressor(n_jobs=-1,random_state=random_seed,
                                        max_depth=best_params[(model_name,desc_name)]['max_depth'],
                                        n_estimators=best_params[(model_name,desc_name)]['n_estimators'])
            sel_index = feature_selection(model,train_val_desc,train_val_target,cv)
        elif model_name == 'RF':
            model = RandomForestRegressor(n_jobs=-1,random_state=random_seed,
                                          max_depth=best_params[(model_name,desc_name)]['max_depth'],
                                          n_estimators=best_params[(model_name,desc_name)]['n_estimators'])
            sel_index = feature_selection(model,train_val_desc,train_val_target,cv)
        elif model_name == 'DT':
            model = DecisionTreeRegressor(random_state=random_seed,
                                          max_depth=best_params[(model_name,desc_name)]['max_depth'],
                                          min_samples_split=best_params[(model_name,desc_name)]['min_samples_split'])
            sel_index = feature_selection(model,train_val_desc,train_val_target,cv)
        elif model_name == 'XGB':
            model = XGBRegressor(n_estimators=best_params[(model_name,desc_name)]['n_estimators'],
                                 max_depth=best_params[(model_name,desc_name)]['max_depth'],
                                 n_jobs=-1,random_state=random_seed)
            sel_index = feature_selection(model,train_val_desc,train_val_target,cv)

        elif model_name == 'SVR':
            model = SVR(gamma=best_params[(model_name,desc_name)]['gamma'],
                        kernel=best_params[(model_name,desc_name)]['kernel'])

        elif model_name == 'KRR':
            model = KernelRidge(gamma=best_params[(model_name,desc_name)]['gamma'])

        cv_test_y,cv_test_p,cv_mae,cv_r2,cv_pearson_r = cv_fit_pred(model,train_val_desc,train_val_target,
                                                                    cv,n=10,sel_index=sel_index)

        oos_pred,oos_mae,oos_r2,oos_pearson_r = oos_fit_pred(model,train_val_desc,train_val_target,
                                                    oos_desc,oos_target,n=10,sel_index=sel_index)
        
    else:
        epoch = best_params[(model_name,desc_name)]['epoch']
        node_num = best_params[(model_name,desc_name)]['node_num']
        cv_test_y,cv_test_p,cv_mae,cv_r2,cv_pearson_r = cv_fit_pred_NN(train_val_desc,train_val_target,
                                                                     epoch,node_num,cv,random_seed)
        
        oos_pred,oos_mae,oos_r2,oos_pearson_r = oos_fit_pred_NN(train_val_desc,train_val_target,
                                                             oos_desc,oos_target,epoch,node_num,random_seed)
    return cv_test_y,cv_test_p,cv_mae,cv_r2,cv_pearson_r,\
            oos_target,oos_pred,oos_mae,oos_r2,oos_pearson_r
















