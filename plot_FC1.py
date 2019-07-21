# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 08:37:55 2019

@author: Administrator
"""

from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io as sio
sns.set(style="white")
import matplotlib
#############################################################################################################
#path1='E:/Inhibition/inhibition_data/FC_data/switch/high_low/pte/'
#response=path1+('PTE_theta_avr')
#response_data=sio.loadmat(response)
#A_data=response_data['data']
#corr=np.mean(A_data,0)
#names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3','C4', 'P3', 'P4', 'O1', 'O2','F7', 'F8','T7', 'T8', 'P7','P8', 'Fz', 'Cz', 'Pz', 
#               'FC1','FC2', 'CP1', 'CP2', 'FC5', 'FC6','CP5', 'CP6']
## plot correlation matrix
#f, ax = plt.subplots(figsize=(12,10))
#norm = matplotlib.colors.Normalize(vmin=-0.01, vmax=np.max(corr))
#ticks = np.arange(0,27,1)
#ax.set_xticks(ticks)
#ax.set_yticks(ticks)
#ax.set_xticklabels(names,fontsize=14,rotation=315)
#ax.set_yticklabels(names,fontsize=14)
#plt.imshow(corr, interpolation="nearest", origin="upper",cmap=plt.cm.jet,norm=norm)
#plt.colorbar()
#plt.xlabel('To', fontsize=24)
#plt.ylabel('From', fontsize=24)    
#plt.savefig("C:/Users/Administrator/Desktop/End_paper/inhibition/FC_data/"+'PTE_theta_H_L_switch.png',dpi=600)
#plt.show()
############################################################################################################
#path1='E:/Inhibition/inhibition_data/FC_data/repeat/high_low/pte/'
#response=path1+('PTE_theta_avr')
#response_data=sio.loadmat(response)
#A_data=response_data['data']
#corr=np.mean(A_data,0)
#names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3','C4', 'P3', 'P4', 'O1', 'O2','F7', 'F8','T7', 'T8', 'P7','P8', 'Fz', 'Cz', 'Pz', 
#               'FC1','FC2', 'CP1', 'CP2', 'FC5', 'FC6','CP5', 'CP6']
## plot correlation matrix
#f, ax = plt.subplots(figsize=(12,10))
#norm = matplotlib.colors.Normalize(vmin=-0.01, vmax=np.max(corr))
#ticks = np.arange(0,27,1)
#ax.set_xticks(ticks)
#ax.set_yticks(ticks)
#ax.set_xticklabels(names,fontsize=14,rotation=315)
#ax.set_yticklabels(names,fontsize=14)
#plt.imshow(corr, interpolation="nearest", origin="upper",cmap=plt.cm.jet,norm=norm)
#plt.colorbar()
#plt.xlabel('To', fontsize=24)
#plt.ylabel('From', fontsize=24)    
#plt.savefig("C:/Users/Administrator/Desktop/End_paper/inhibition/FC_data/"+'PTE_theta_H_L_repeat.png',dpi=600)
#plt.show()


############################################################################################################
path1='E:/cue/FC_data/'
response=path1+('Label')
response_data=sio.loadmat(response)
A_data=response_data['Label']
corr=A_data
names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3','C4', 'P3', 'P4', 'O1', 'O2','F7', 'F8','T7', 'T8', 'P7','P8', 'Fz', 'Cz', 'Pz', 
               'FC1','FC2', 'CP1', 'CP2', 'FC5', 'FC6','CP5', 'CP6']
# plot correlation matrix
f, ax = plt.subplots(figsize=(12,10))
norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max(corr))
ticks = np.arange(0,27,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names,fontsize=14,rotation=315)
ax.set_yticklabels(names,fontsize=14)
plt.imshow(corr, interpolation="nearest", origin="upper",cmap=plt.cm.jet,norm=norm)
plt.colorbar()
plt.xlabel('To', fontsize=24)
plt.ylabel('From', fontsize=24)    
plt.savefig("C:/Users/Administrator/Desktop/End_paper/cue/FC_data/"+'PTE_theta_difference.png',dpi=600)
plt.show()




