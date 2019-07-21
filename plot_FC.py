# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:10:15 2019

@author: Administrator
"""
#####################
import scipy.io as sio
import seaborn as sns
import scipy.interpolate
import matplotlib.pyplot as plt
import numpy as np
###############################################
path1='E:/target/FC_data/target8/repeat/pte/'
response=path1+('PTE_theta_avr')
response_data=sio.loadmat(response)
A_data=response_data['data']
data=np.mean(A_data,0)
######################
f, ax = plt.subplots(figsize=(6,5))
mask = np.zeros_like(data)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(data, mask=mask,vmin=-0.01, vmax=np.max(data), square=True,  cmap=plt.cm.jet)
    plt.xticks(np.arange(27), ('Fp1', 'Fp2', 'F3', 'F4', 'C3','C4', 'P3', 'P4', 'O1', 'O2','F7', 'F8','T7', 'T8', 'P7','P8', 'Fz', 'Cz', 'Pz', 
               'FC1','FC2', 'CP1', 'CP2', 'FC5', 'FC6','CP5', 'CP6'),rotation=90,fontsize=8)
    plt.yticks(np.arange(27), ('Fp1', 'Fp2', 'F3', 'F4', 'C3','C4', 'P3', 'P4', 'O1', 'O2','F7', 'F8', 'T7', 'T8', 'P7','P8', 'Fz', 'Cz', 'Pz', 
               'FC1','FC2', 'CP1', 'CP2', 'FC5', 'FC6','CP5', 'CP6'),rotation=360,fontsize=8)  
plt.xlabel('To', fontsize=14)
plt.ylabel('From', fontsize=14)
plt.savefig("C:/Users/Administrator/Desktop/End_paper/target/FC_data/"+'PTE_theta_T8_repeat.png',dpi=1800)
plt.show()
###############################################
path1='E:/target/FC_data/target8/switch/pte/'
response=path1+('PTE_theta_avr')
response_data=sio.loadmat(response)
A_data=response_data['data']
data=np.mean(A_data,0)
######################
f, ax = plt.subplots(figsize=(6,5))
mask = np.zeros_like(data)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(data, mask=mask,vmin=-0.01, vmax=np.max(data),square=True,linewidths=.5,cmap=plt.cm.jet)
    plt.xticks(np.arange(27), ('Fp1', 'Fp2', 'F3', 'F4', 'C5','C4', 'P3', 'P4', 'O1', 'O2','F7', 'F8','T7', 'T8', 'P7','P8', 'Fz', 'Cz', 'Pz', 
               'FC1','FC2', 'CP1', 'CP2', 'FC5', 'FC6','CP5', 'CP6'),rotation=90,fontsize=8)
    plt.yticks(np.arange(27), ('Fp1', 'Fp2', 'F3', 'F4', 'C5','C4', 'P3', 'P4', 'O1', 'O2','F7', 'F8', 'T7', 'T8', 'P7','P8', 'Fz', 'Cz', 'Pz', 
               'FC1','FC2', 'CP1', 'CP2', 'FC5', 'FC6','CP5', 'CP6'),rotation=360,fontsize=8)  
plt.xlabel('To', fontsize=14)
plt.ylabel('From', fontsize=14)    
plt.savefig("C:/Users/Administrator/Desktop/End_paper/target/FC_data/"+'PTE_theta_T8_switch.png',dpi=1800)
plt.show()

#
#sns.heatmap(corr, mask=mask, cmap=plt.cm.jet,vmin=0.01,vmax=.03, center=0,
#            square=True, linewidths=.5, cbar_kws={"shrink": .5})

