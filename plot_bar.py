#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 15:02:33 2018

@author: wangyu
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
for i in np.arange(59):
    plt.figure(figsize=(6,6),facecolor='none',edgecolor='none')
    city=['Target1','Target2-8']
    Gender=['Repeat','Switch']
    pos = np.arange(len(city))
    bar_width = 0.35
    T1_mean=[np.mean(repeat_T1[:,i]),np.mean(repeat_T28[:,i])]
    T28_mean=[np.mean(switch_T1[:,i]),np.mean(switch_T28[:,i])]
    T_mean=[T1_mean,T28_mean]
    T1_std=[np.std(repeat_T1[:,i])/11.4018,np.std(repeat_T28[:,i])/11.4018]
    T28_std=[np.std(switch_T1[:,i])/11.4018,np.std(switch_T28[:,i])/11.4018]
    T_std=[T1_std,T28_std]
    plt.bar(pos,T1_mean,bar_width,yerr=T1_std,color='blue',edgecolor='black')
    plt.bar(pos+bar_width,T28_mean,bar_width,yerr=T28_std,color='red',edgecolor='black')
    plt.style.use('ggplot')
    plt.xticks(pos, city)
    plt.tick_params(labelsize=13)
    plt.xlabel('Stimulus', fontsize=16)
    plt.ylabel('Latency', fontsize=16)
    plt.ylim(np.min(T_mean)-np.max(T_std),np.max(T_mean)+np.max(T_std)+np.min(T_std))
    plt.legend(Gender, fontsize=16)
    plt.savefig("C:/Users/Administrator/Desktop/End_paper/target/ERP/"+'T1_T28_P200_latency_bar-'+str(channels[i])+'.png',dpi=600)#+elec2[num-1]  #循环保存图片,bbox_inches='tight'

S910_beta_freq=path+('P300_repeat_T1_le')
beta_freq=sio.loadmat(S910_beta_freq)
repeat_T1=beta_freq['DATA2']
S910_beta_time=path+('P300_repeat_T2_T8_le')
beta_time=sio.loadmat(S910_beta_time)
repeat_T28=beta_time['DATA']
S910_beta_freq=path+('P300_switch_T1_le')
beta_freq=sio.loadmat(S910_beta_freq)
switch_T1=beta_freq['DATA2']
S910_beta_time=path+('P300_switch_T2_T8_le')
beta_time=sio.loadmat(S910_beta_time)
switch_T28=beta_time['DATA']
for i in np.arange(59):
    plt.figure(figsize=(6,6),facecolor='none',edgecolor='none')
    city=['Target1','Target2-8']
    Gender=['Repeat','Switch']
    pos = np.arange(len(city))
    bar_width = 0.35
    T1_mean=[np.mean(repeat_T1[:,i]),np.mean(repeat_T28[:,i])]
    T28_mean=[np.mean(switch_T1[:,i]),np.mean(switch_T28[:,i])]
    T_mean=[T1_mean,T28_mean]
    T1_std=[np.std(repeat_T1[:,i])/11.4018,np.std(repeat_T28[:,i])/11.4018]
    T28_std=[np.std(switch_T1[:,i])/11.4018,np.std(switch_T28[:,i])/11.4018]
    T_std=[T1_std,T28_std]
    plt.bar(pos,T1_mean,bar_width,yerr=T1_std,color='blue',edgecolor='black')
    plt.bar(pos+bar_width,T28_mean,bar_width,yerr=T28_std,color='red',edgecolor='black')
    plt.style.use('ggplot')
    plt.xticks(pos, city)
    plt.tick_params(labelsize=13)
    plt.xlabel('Stimulus', fontsize=16)
    plt.ylabel('Latency', fontsize=16)
    plt.ylim(np.min(T_mean)-np.max(T_std),np.max(T_mean)+np.max(T_std)+np.min(T_std))
    plt.legend(Gender, fontsize=16)
    plt.savefig("C:/Users/Administrator/Desktop/End_paper/target/ERP/"+'T1_T28_P300_latency_bar-'+str(channels[i])+'.png',dpi=600)#+elec2[num-1]  #循环保存图片,bbox_inches='tight'

S910_beta_freq=path+('N200_repeat_T1_le')
beta_freq=sio.loadmat(S910_beta_freq)
repeat_T1=beta_freq['DATA2']
S910_beta_time=path+('N200_repeat_T2_T8_le')
beta_time=sio.loadmat(S910_beta_time)
repeat_T28=beta_time['DATA']
S910_beta_freq=path+('N200_switch_T1_le')
beta_freq=sio.loadmat(S910_beta_freq)
switch_T1=beta_freq['DATA2']
S910_beta_time=path+('N200_switch_T2_T8_le')
beta_time=sio.loadmat(S910_beta_time)
switch_T28=beta_time['DATA']
for i in np.arange(59):
    plt.figure(figsize=(6,6),facecolor='none',edgecolor='none')
    city=['Target1','Target2-8']
    Gender=['Repeat','Switch']
    pos = np.arange(len(city))
    bar_width = 0.35
    T1_mean=[np.mean(repeat_T1[:,i]),np.mean(repeat_T28[:,i])]
    T28_mean=[np.mean(switch_T1[:,i]),np.mean(switch_T28[:,i])]
    T_mean=[T1_mean,T28_mean]
    T1_std=[np.std(repeat_T1[:,i])/11.4018,np.std(repeat_T28[:,i])/11.4018]
    T28_std=[np.std(switch_T1[:,i])/11.4018,np.std(switch_T28[:,i])/11.4018]
    T_std=[T1_std,T28_std]
    plt.bar(pos,T1_mean,bar_width,yerr=T1_std,color='blue',edgecolor='black')
    plt.bar(pos+bar_width,T28_mean,bar_width,yerr=T28_std,color='red',edgecolor='black')
    plt.style.use('ggplot')
    plt.xticks(pos, city)
    plt.tick_params(labelsize=13)
    plt.xlabel('Stimulus', fontsize=16)
    plt.ylabel('Latency', fontsize=16)
    plt.ylim(np.min(T_mean)-np.max(T_std),np.max(T_mean)+np.max(T_std)+np.min(T_std))
    plt.legend(Gender, fontsize=16)
    plt.savefig("C:/Users/Administrator/Desktop/End_paper/target/ERP/"+'T1_T28_N200_latency_bar-'+str(channels[i])+'.png',dpi=600)#+elec2[num-1]  #循环保存图片,bbox_inches='tight'

S910_beta_freq=path+('P200_repeat_T1_am')
beta_freq=sio.loadmat(S910_beta_freq)
repeat_T1=beta_freq['DATA1']
S910_beta_time=path+('P200_repeat_T2_T8_am')
beta_time=sio.loadmat(S910_beta_time)
repeat_T28=beta_time['DATA']
S910_beta_freq=path+('P200_switch_T1_am')
beta_freq=sio.loadmat(S910_beta_freq)
switch_T1=beta_freq['DATA1']
S910_beta_time=path+('P200_switch_T2_T8_am')
beta_time=sio.loadmat(S910_beta_time)
switch_T28=beta_time['DATA']
for i in np.arange(59):
    plt.figure(figsize=(6,6),facecolor='none',edgecolor='none')
    city=['Target1','Target2-8']
    Gender=['Repeat','Switch']
    pos = np.arange(len(city))
    bar_width = 0.35
    T1_mean=[np.mean(repeat_T1[:,i]),np.mean(repeat_T28[:,i])]
    T28_mean=[np.mean(switch_T1[:,i]),np.mean(switch_T28[:,i])]
    T_mean=[T1_mean,T28_mean]
    T1_std=[np.std(repeat_T1[:,i])/11.4018,np.std(repeat_T28[:,i])/11.4018]
    T28_std=[np.std(switch_T1[:,i])/11.4018,np.std(switch_T28[:,i])/11.4018]
    T_std=[T1_std,T28_std]
    plt.bar(pos,T1_mean,bar_width,yerr=T1_std,color='blue',edgecolor='black')
    plt.bar(pos+bar_width,T28_mean,bar_width,yerr=T28_std,color='red',edgecolor='black')
    plt.style.use('ggplot')
    plt.xticks(pos, city)
    plt.tick_params(labelsize=13)
    plt.xlabel('Stimulus', fontsize=16)
    plt.ylabel('Amplitude', fontsize=16)
    plt.ylim(np.min(T_mean)-np.max(T_std),np.max(T_mean)+np.max(T_std)+np.min(T_std))
    plt.legend(Gender, fontsize=16)
    plt.savefig("C:/Users/Administrator/Desktop/End_paper/target/ERP/"+'T1_T28_P200_amplitude_bar-'+str(channels[i])+'.png',dpi=600)#+elec2[num-1]  #循环保存图片,bbox_inches='tight'

S910_beta_freq=path+('P300_repeat_T1_am')
beta_freq=sio.loadmat(S910_beta_freq)
repeat_T1=beta_freq['DATA1']
S910_beta_time=path+('P300_repeat_T2_T8_am')
beta_time=sio.loadmat(S910_beta_time)
repeat_T28=beta_time['DATA']
S910_beta_freq=path+('P300_switch_T1_am')
beta_freq=sio.loadmat(S910_beta_freq)
switch_T1=beta_freq['DATA1']
S910_beta_time=path+('P300_switch_T2_T8_am')
beta_time=sio.loadmat(S910_beta_time)
switch_T28=beta_time['DATA']
for i in np.arange(59):
    plt.figure(figsize=(6,6),facecolor='none',edgecolor='none')
    city=['Target1','Target2-8']
    Gender=['Repeat','Switch']
    pos = np.arange(len(city))
    bar_width = 0.35
    T1_mean=[np.mean(repeat_T1[:,i]),np.mean(repeat_T28[:,i])]
    T28_mean=[np.mean(switch_T1[:,i]),np.mean(switch_T28[:,i])]
    T_mean=[T1_mean,T28_mean]
    T1_std=[np.std(repeat_T1[:,i])/11.4018,np.std(repeat_T28[:,i])/11.4018]
    T28_std=[np.std(switch_T1[:,i])/11.4018,np.std(switch_T28[:,i])/11.4018]
    T_std=[T1_std,T28_std]
    plt.bar(pos,T1_mean,bar_width,yerr=T1_std,color='blue',edgecolor='black')
    plt.bar(pos+bar_width,T28_mean,bar_width,yerr=T28_std,color='red',edgecolor='black')
    plt.style.use('ggplot')
    plt.xticks(pos, city)
    plt.tick_params(labelsize=13)
    plt.xlabel('Stimulus', fontsize=16)
    plt.ylabel('Amplitude', fontsize=16)
    plt.ylim(np.min(T_mean)-np.max(T_std),np.max(T_mean)+np.max(T_std)+np.min(T_std))
    plt.legend(Gender, fontsize=16)
    plt.savefig("C:/Users/Administrator/Desktop/End_paper/target/ERP/"+'T1_T28_P300_amplitude_bar-'+str(channels[i])+'.png',dpi=600)#+elec2[num-1]  #循环保存图片,bbox_inches='tight'













S910_beta_freq=path+('N200_repeat_T1_am')
beta_freq=sio.loadmat(S910_beta_freq)
repeat_T1=beta_freq['DATA1']
S910_beta_time=path+('N200_repeat_T2_T8_am')
beta_time=sio.loadmat(S910_beta_time)
repeat_T28=beta_time['DATA']
S910_beta_freq=path+('N200_switch_T1_am')
beta_freq=sio.loadmat(S910_beta_freq)
switch_T1=beta_freq['DATA1']
S910_beta_time=path+('N200_switch_T2_T8_am')
beta_time=sio.loadmat(S910_beta_time)
switch_T28=beta_time['DATA']
for i in np.arange(59):
    plt.figure(figsize=(6,6),facecolor='none',edgecolor='none')
    city=['Target1','Target2-8']
    Gender=['Repeat','Switch']
    pos = np.arange(len(city))
    bar_width = 0.35
    T1_mean=[np.mean(repeat_T1[:,i]),np.mean(repeat_T28[:,i])]
    T28_mean=[np.mean(switch_T1[:,i]),np.mean(switch_T28[:,i])]
    T_mean=[T1_mean,T28_mean]
    T1_std=[np.std(repeat_T1[:,i])/11.4018,np.std(repeat_T28[:,i])/11.4018]
    T28_std=[np.std(switch_T1[:,i])/11.4018,np.std(switch_T28[:,i])/11.4018]
    T_std=[T1_std,T28_std]
    plt.bar(pos,T1_mean,bar_width,yerr=T1_std,color='blue',edgecolor='black')
    plt.bar(pos+bar_width,T28_mean,bar_width,yerr=T28_std,color='red',edgecolor='black')
    plt.style.use('ggplot')
    plt.xticks(pos, city)
    plt.tick_params(labelsize=13)
    plt.xlabel('Stimulus', fontsize=16)
    plt.ylabel('Amplitude', fontsize=16)
    plt.ylim(np.min(T_mean)-np.max(T_std),np.max(T_mean)+np.max(T_std)+np.min(T_std))
    plt.legend(Gender, fontsize=16)
    plt.savefig("C:/Users/Administrator/Desktop/End_paper/target/ERP/"+'T1_T28_N200_amplitude_bar-'+str(channels[i])+'.png',dpi=600)#+elec2[num-1]  #循环保存图片,bbox_inches='tight'










