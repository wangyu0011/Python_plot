# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 10:10:46 2019

@author: Administrator
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import scipy.io as sio
import pandas as pd
from PIL import Image,ImageDraw,ImageFont # 导入模块

font=ImageFont.truetype(r'C:/Users/Administrator/Desktop/End_paper/msyh.ttf',300)

#channels=['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T7','T8','P7','P8','Fz','Cz','Pz','IO','FC1',
#          'FC2','CP1','CP2','FC5','FC6','CP5','CP6','FT9','FT10','F1','F2','C1','C2','P1','P2','AF3','AF4','FC3','FC4',
#          'CP3','CP4','PO3','PO4','F5','F6','C5','C6','P5','P6','AF7','AF8','FT7','FT8','TP7','TP8','PO7','PO8','Fpz',
#          'CPz','POz','Oz']

#channels=['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T7','T8','P7','P8','Fz','Cz','Pz','FC1',
#          'FC2','CP1','CP2','FC5','FC6','CP5','CP6','FT9','FT10','F1','F2','C1','C2','P1','P2','AF3','AF4','FC3','FC4',
#          'CP3','CP4','PO3','PO4','F5','F6','C5','C6','P5','P6','AF7','AF8','FT7','FT8','TP7','TP8','PO7','PO8','Fpz',
#          'CPz','POz','Oz']

channels=['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T7','T8','P7','P8','Fz','Cz','Pz','FC1',
          'FC2','CP1','CP2','FC5','FC6','CP5','CP6','F1','F2','C1','C2','P1','P2','AF3','AF4','FC3','FC4',
          'CP3','CP4','PO3','PO4','F5','F6','C5','C6','P5','P6','AF7','AF8','FT7','FT8','TP7','TP8','PO7','PO8','Fpz',
          'CPz','POz','Oz']
path='E:/target/time_feature/'
S910_beta_freq=path+('P200_repeat_T1_le')
beta_freq=sio.loadmat(S910_beta_freq)
repeat_T1=beta_freq['DATA2']
S910_beta_time=path+('P200_repeat_T2_T8_le')
beta_time=sio.loadmat(S910_beta_time)
repeat_T28=beta_time['DATA']
S910_beta_freq=path+('P200_switch_T1_le')
beta_freq=sio.loadmat(S910_beta_freq)
switch_T1=beta_freq['DATA2']
S910_beta_time=path+('P200_switch_T2_T8_le')
beta_time=sio.loadmat(S910_beta_time)
switch_T28=beta_time['DATA']
#for i in np.arange(59):
i=4
plt.figure(figsize=(6,6),facecolor='none',edgecolor='none')
city=['Pre','Pos','HC']
pos = np.arange(len(city))
bar_width = 0.35
T1_mean=[np.mean(repeat_T1[:,i]),np.mean(repeat_T28[:,i]),np.mean(repeat_T28[:,i])]
T_mean=[T1_mean]
T1_std=[np.std(repeat_T1[:,i])/11.4018,np.std(repeat_T28[:,i])/11.4018,np.std(repeat_T28[:,i])/11.4018]
T_std=[T1_std]
plt.bar(pos[1],T1_mean[1],bar_width,yerr=T1_std[1],color='blue',edgecolor='black')
plt.bar(pos[0],T1_mean[0],bar_width,yerr=T1_std[0],color='red',edgecolor='black')
plt.bar(pos[2],T1_mean[2],bar_width,yerr=T1_std[2],color='green',edgecolor='black')
plt.style.use('ggplot')
plt.xticks(pos, city)
plt.tick_params(labelsize=13)
plt.xlabel('Stimulus', fontsize=16)
plt.ylabel('Latency', fontsize=16)
plt.ylim(np.min(T_mean)-np.max(T_std),np.max(T_mean)+np.max(T_std)+np.min(T_std))
plt.savefig("C:/Users/Administrator/Desktop/P200_result/new_bar/"+'T1_T28_P200_latency_bar-'+str(channels[i])+'.png',dpi=600)#+elec2[num-1]  #循环保存图片,bbox_inches='tight'
