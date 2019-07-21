# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 21:05:39 2019

@author: Administrator
"""
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
names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3','C4', 'P3', 'P4', 'O1', 'O2','F7', 'F8','T7', 'T8', 'P7','P8', 'Fz', 'Cz', 'Pz', 
               'FC1','FC2', 'CP1', 'CP2', 'FC5', 'FC6','CP5', 'CP6']
#channels=['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T7','T8','P7','P8','Fz','Cz','Pz','IO','FC1',
#          'FC2','CP1','CP2','FC5','FC6','CP5','CP6','FT9','FT10','F1','F2','C1','C2','P1','P2','AF3','AF4','FC3','FC4',
#          'CP3','CP4','PO3','PO4','F5','F6','C5','C6','P5','P6','AF7','AF8','FT7','FT8','TP7','TP8','PO7','PO8','Fpz',
#          'CPz','POz','Oz']
#channels=['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T7','T8','P7','P8','Fz','Cz','Pz','FC1',
#          'FC2','CP1','CP2','FC5','FC6','CP5','CP6','FT9','FT10','F1','F2','C1','C2','P1','P2','AF3','AF4','FC3','FC4',
#          'CP3','CP4','PO3','PO4','F5','F6','C5','C6','P5','P6','AF7','AF8','FT7','FT8','TP7','TP8','PO7','PO8','Fpz',
#          'CPz','POz','Oz']
#channels=['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T7','T8','P7','P8','Fz','Cz','Pz','FC1',
#          'FC2','CP1','CP2','FC5','FC6','CP5','CP6','F1','F2','C1','C2','P1','P2','AF3','AF4','FC3','FC4',
#          'CP3','CP4','PO3','PO4','F5','F6','C5','C6','P5','P6','AF7','AF8','FT7','FT8','TP7','TP8','PO7','PO8','Fpz',
#          'CPz','POz','Oz']
path1='E:/target/FC_data/target1/repeat/pte/'
response=path1+('PTE_theta_avr')
response_data=sio.loadmat(response)
repeat_T1=response_data['data']
path1='E:/target/FC_data/target1/switch/pte/'
response=path1+('PTE_theta_avr')
response_data=sio.loadmat(response)
switch_T1=response_data['data']
path1='E:/target/FC_data/target2/repeat/pte/'
response=path1+('PTE_theta_avr')
response_data=sio.loadmat(response)
repeat_T2=response_data['data']
path1='E:/target/FC_data/target2/switch/pte/'
response=path1+('PTE_theta_avr')
response_data=sio.loadmat(response)
switch_T2=response_data['data']
path1='E:/target/FC_data/target3/repeat/pte/'
response=path1+('PTE_theta_avr')
response_data=sio.loadmat(response)
repeat_T3=response_data['data']
path1='E:/target/FC_data/target3/switch/pte/'
response=path1+('PTE_theta_avr')
response_data=sio.loadmat(response)
switch_T3=response_data['data']
path1='E:/target/FC_data/target4/repeat/pte/'
response=path1+('PTE_theta_avr')
response_data=sio.loadmat(response)
repeat_T4=response_data['data']
path1='E:/target/FC_data/target4/switch/pte/'
response=path1+('PTE_theta_avr')
response_data=sio.loadmat(response)
switch_T4=response_data['data']

path1='E:/target/FC_data/target5/repeat/pte/'
response=path1+('PTE_theta_avr')
response_data=sio.loadmat(response)
repeat_T5=response_data['data']
path1='E:/target/FC_data/target5/switch/pte/'
response=path1+('PTE_theta_avr')
response_data=sio.loadmat(response)
switch_T5=response_data['data']
path1='E:/target/FC_data/target6/repeat/pte/'
response=path1+('PTE_theta_avr')
response_data=sio.loadmat(response)
repeat_T6=response_data['data']
path1='E:/target/FC_data/target6/switch/pte/'
response=path1+('PTE_theta_avr')
response_data=sio.loadmat(response)
switch_T6=response_data['data']
path1='E:/target/FC_data/target7/repeat/pte/'
response=path1+('PTE_theta_avr')
response_data=sio.loadmat(response)
repeat_T7=response_data['data']
path1='E:/target/FC_data/target7/switch/pte/'
response=path1+('PTE_theta_avr')
response_data=sio.loadmat(response)
switch_T7=response_data['data']
path1='E:/target/FC_data/target8/repeat/pte/'
response=path1+('PTE_theta_avr')
response_data=sio.loadmat(response)
repeat_T8=response_data['data']
path1='E:/target/FC_data/target8/switch/pte/'
response=path1+('PTE_theta_avr')
response_data=sio.loadmat(response)
switch_T8=response_data['data']

for i in np.arange(27):
    for j in np.arange(27):
        plt.figure(figsize=(12,6),facecolor='none',edgecolor='none')
        city=['T1','T2','T3','T4','T5','T6','T7','T8']
        Gender=['Repeat','Switch']
        pos = np.arange(len(city))
        bar_width = 0.35
        T1_mean=[np.mean(repeat_T1[:,i,j]),np.mean(repeat_T2[:,i,j]),np.mean(repeat_T3[:,i,j]),np.mean(repeat_T4[:,i,j]),
                 np.mean(repeat_T5[:,i,j]),np.mean(repeat_T6[:,i,j]),np.mean(repeat_T7[:,i,j]),np.mean(repeat_T8[:,i,j])]
        T28_mean=[np.mean(switch_T1[:,i,j]),np.mean(switch_T2[:,i,j]),np.mean(switch_T3[:,i,j]),np.mean(switch_T4[:,i,j]),
                  np.mean(switch_T5[:,i,j]),np.mean(switch_T6[:,i,j]),np.mean(switch_T7[:,i,j]),np.mean(switch_T8[:,i,j])]
        T_mean=[T1_mean,T28_mean]
        T1_std=[np.std(repeat_T1[:,i,j])/11.4018,np.std(repeat_T2[:,i,j])/11.4018,np.std(repeat_T3[:,i,j])/11.4018,np.std(repeat_T4[:,i,j])/11.4018,
                np.std(repeat_T5[:,i,j])/11.4018,np.std(repeat_T6[:,i,j])/11.4018,np.std(repeat_T7[:,i,j])/11.4018,np.std(repeat_T8[:,i,j])/11.4018]
        T28_std=[np.std(switch_T1[:,i,j])/11.4018,np.std(switch_T2[:,i,j])/11.4018,np.std(switch_T3[:,i,j])/11.4018,np.std(switch_T4[:,i,j])/11.4018,
                 np.std(switch_T5[:,i,j])/11.4018,np.std(switch_T6[:,i,j])/11.4018,np.std(switch_T7[:,i,j])/11.4018,np.std(switch_T8[:,i,j])/11.4018]
        T_std=[T1_std,T28_std]
        plt.bar(pos,T1_mean,bar_width,yerr=T1_std,color='blue',edgecolor='black')
        plt.bar(pos+bar_width,T28_mean,bar_width,yerr=T28_std,color='red',edgecolor='black')
        plt.style.use('ggplot')
        plt.xticks(pos, city)
        plt.tick_params(labelsize=13)
        plt.xlabel('Stimulus', fontsize=16)
        plt.ylabel('PSD(dB)', fontsize=16)
        plt.ylim(np.min(T_mean)-np.max(T_std),np.max(T_mean)+np.max(T_std)+np.min(T_std))
        plt.legend(Gender, fontsize=16)
        plt.savefig("C:/Users/Administrator/Desktop/End_paper/target/FC_data/"+'PTE_theta_T1-T8-'+str(names[i])+'-to-'+str(names[j])+'.png',dpi=600)#+elec2[num-1]  #循环保存图片,bbox_inches='tight'
    
    
path1='E:/target/FC_data/target1/repeat/pte/'
response=path1+('PTE_alpha_avr')
response_data=sio.loadmat(response)
repeat_T1=response_data['data']
path1='E:/target/FC_data/target1/switch/pte/'
response=path1+('PTE_alpha_avr')
response_data=sio.loadmat(response)
switch_T1=response_data['data']
path1='E:/target/FC_data/target2/repeat/pte/'
response=path1+('PTE_alpha_avr')
response_data=sio.loadmat(response)
repeat_T2=response_data['data']
path1='E:/target/FC_data/target2/switch/pte/'
response=path1+('PTE_alpha_avr')
response_data=sio.loadmat(response)
switch_T2=response_data['data']
path1='E:/target/FC_data/target3/repeat/pte/'
response=path1+('PTE_alpha_avr')
response_data=sio.loadmat(response)
repeat_T3=response_data['data']
path1='E:/target/FC_data/target3/switch/pte/'
response=path1+('PTE_alpha_avr')
response_data=sio.loadmat(response)
switch_T3=response_data['data']
path1='E:/target/FC_data/target4/repeat/pte/'
response=path1+('PTE_theta_avr')
response_data=sio.loadmat(response)
repeat_T4=response_data['data']
path1='E:/target/FC_data/target4/switch/pte/'
response=path1+('PTE_alpha_avr')
response_data=sio.loadmat(response)
switch_T4=response_data['data']

path1='E:/target/FC_data/target5/repeat/pte/'
response=path1+('PTE_alpha_avr')
response_data=sio.loadmat(response)
repeat_T5=response_data['data']
path1='E:/target/FC_data/target5/switch/pte/'
response=path1+('PTE_alpha_avr')
response_data=sio.loadmat(response)
switch_T5=response_data['data']
path1='E:/target/FC_data/target6/repeat/pte/'
response=path1+('PTE_alpha_avr')
response_data=sio.loadmat(response)
repeat_T6=response_data['data']
path1='E:/target/FC_data/target6/switch/pte/'
response=path1+('PTE_alpha_avr')
response_data=sio.loadmat(response)
switch_T6=response_data['data']
path1='E:/target/FC_data/target7/repeat/pte/'
response=path1+('PTE_alpha_avr')
response_data=sio.loadmat(response)
repeat_T7=response_data['data']
path1='E:/target/FC_data/target7/switch/pte/'
response=path1+('PTE_alpha_avr')
response_data=sio.loadmat(response)
switch_T7=response_data['data']
path1='E:/target/FC_data/target8/repeat/pte/'
response=path1+('PTE_alpha_avr')
response_data=sio.loadmat(response)
repeat_T8=response_data['data']
path1='E:/target/FC_data/target8/switch/pte/'
response=path1+('PTE_alpha_avr')
response_data=sio.loadmat(response)
switch_T8=response_data['data']

for i in np.arange(27):
    for j in np.arange(27):
        plt.figure(figsize=(12,6),facecolor='none',edgecolor='none')
        city=['T1','T2','T3','T4','T5','T6','T7','T8']
        Gender=['Repeat','Switch']
        pos = np.arange(len(city))
        bar_width = 0.35
        T1_mean=[np.mean(repeat_T1[:,i,j]),np.mean(repeat_T2[:,i,j]),np.mean(repeat_T3[:,i,j]),np.mean(repeat_T4[:,i,j]),
                 np.mean(repeat_T5[:,i,j]),np.mean(repeat_T6[:,i,j]),np.mean(repeat_T7[:,i,j]),np.mean(repeat_T8[:,i,j])]
        T28_mean=[np.mean(switch_T1[:,i,j]),np.mean(switch_T2[:,i,j]),np.mean(switch_T3[:,i,j]),np.mean(switch_T4[:,i,j]),
                  np.mean(switch_T5[:,i,j]),np.mean(switch_T6[:,i,j]),np.mean(switch_T7[:,i,j]),np.mean(switch_T8[:,i,j])]
        T_mean=[T1_mean,T28_mean]
        T1_std=[np.std(repeat_T1[:,i,j])/11.4018,np.std(repeat_T2[:,i,j])/11.4018,np.std(repeat_T3[:,i,j])/11.4018,np.std(repeat_T4[:,i,j])/11.4018,
                np.std(repeat_T5[:,i,j])/11.4018,np.std(repeat_T6[:,i,j])/11.4018,np.std(repeat_T7[:,i,j])/11.4018,np.std(repeat_T8[:,i,j])/11.4018]
        T28_std=[np.std(switch_T1[:,i,j])/11.4018,np.std(switch_T2[:,i,j])/11.4018,np.std(switch_T3[:,i,j])/11.4018,np.std(switch_T4[:,i,j])/11.4018,
                 np.std(switch_T5[:,i,j])/11.4018,np.std(switch_T6[:,i,j])/11.4018,np.std(switch_T7[:,i,j])/11.4018,np.std(switch_T8[:,i,j])/11.4018]
        T_std=[T1_std,T28_std]
        plt.bar(pos,T1_mean,bar_width,yerr=T1_std,color='blue',edgecolor='black')
        plt.bar(pos+bar_width,T28_mean,bar_width,yerr=T28_std,color='red',edgecolor='black')
        plt.style.use('ggplot')
        plt.xticks(pos, city)
        plt.tick_params(labelsize=13)
        plt.xlabel('Stimulus', fontsize=16)
        plt.ylabel('PSD(dB)', fontsize=16)
        plt.ylim(np.min(T_mean)-np.max(T_std),np.max(T_mean)+np.max(T_std)+np.min(T_std))
        plt.legend(Gender, fontsize=16)
        plt.savefig("C:/Users/Administrator/Desktop/End_paper/target/FC_data/"+'PTE_alpha_T1-T8-'+str(names[i])+'-to-'+str(names[j])+'.png',dpi=600)#+elec2[num-1]  #循环保存图片,bbox_inches='tight'
      
#path1='E:/Inhibition/inhibition_data/FC_data/repeat/high_high/pte/'
#response=path1+('PTE_alpha_avr')
#response_data=sio.loadmat(response)
#repeat_H_H=response_data['data']
#path1='E:/Inhibition/inhibition_data/FC_data/switch/high_high/pte/'
#response=path1+('PTE_alpha_avr')
#response_data=sio.loadmat(response)
#switch_H_H=response_data['data']
#path1='E:/Inhibition/inhibition_data/FC_data/repeat/high_low/pte/'
#response=path1+('PTE_alpha_avr')
#response_data=sio.loadmat(response)
#repeat_H_L=response_data['data']
#path1='E:/Inhibition/inhibition_data/FC_data/switch/high_low/pte/'
#response=path1+('PTE_alpha_avr')
#response_data=sio.loadmat(response)
#switch_H_L=response_data['data']
#path1='E:/Inhibition/inhibition_data/FC_data/repeat/low_high/pte/'
#response=path1+('PTE_alpha_avr')
#response_data=sio.loadmat(response)
#repeat_L_H=response_data['data']
#path1='E:/Inhibition/inhibition_data/FC_data/switch/low_high/pte/'
#response=path1+('PTE_alpha_avr')
#response_data=sio.loadmat(response)
#switch_L_H=response_data['data']
#path1='E:/Inhibition/inhibition_data/FC_data/repeat/low_low/pte/'
#response=path1+('PTE_alpha_avr')
#response_data=sio.loadmat(response)
#repeat_L_L=response_data['data']
#path1='E:/Inhibition/inhibition_data/FC_data/switch/low_low/pte/'
#response=path1+('PTE_alpha_avr')
#response_data=sio.loadmat(response)
#switch_L_L=response_data['data']
#for i in np.arange(27):
#    for j in np.arange(27):
#        plt.figure(figsize=(12,6),facecolor='none',edgecolor='none')
#        city=['H_H','H_L','L_H','L_L']
#        Gender=['Repeat','Switch']
#        pos = np.arange(len(city))
#        bar_width = 0.35
#        T1_mean=[np.mean(repeat_H_H[:,i,j]),np.mean(repeat_H_L[:,i,j]),np.mean(repeat_L_H[:,i,j]),np.mean(repeat_L_L[:,i,j])]
#        T28_mean=[np.mean(switch_H_H[:,i,j]),np.mean(switch_H_L[:,i,j]),np.mean(switch_L_H[:,i,j]),np.mean(switch_L_L[:,i,j])]
#        T_mean=[T1_mean,T28_mean]
#        T1_std=[np.std(repeat_H_H[:,i,j])/11.4018,np.std(repeat_H_L[:,i,j])/11.4018,np.std(repeat_L_H[:,i,j])/11.4018,np.std(repeat_L_L[:,i,j])/11.4018]
#        T28_std=[np.std(switch_H_H[:,i,j])/11.4018,np.std(switch_H_L[:,i,j])/11.4018,np.std(switch_L_H[:,i,j])/11.4018,np.std(switch_L_L[:,i,j])/11.4018]
#        T_std=[T1_std,T28_std]
#        plt.bar(pos,T1_mean,bar_width,yerr=T1_std,color='blue',edgecolor='black')
#        plt.bar(pos+bar_width,T28_mean,bar_width,yerr=T28_std,color='red',edgecolor='black')
#        plt.style.use('ggplot')
#        plt.xticks(pos, city)
#        plt.tick_params(labelsize=13)
#        plt.xlabel('Stimulus', fontsize=16)
#        plt.ylabel('PSD(dB)', fontsize=16)
#        plt.ylim(np.min(T_mean)-np.max(T_std),np.max(T_mean)+np.max(T_std)+np.min(T_std))
#        plt.legend(Gender, fontsize=16)
#        plt.savefig("C:/Users/Administrator/Desktop/End_paper/inhibition/FC_data/"+'PTE_alpha_inhibition-'+str(names[i])+'-to-'+str(names[j])+'.png',dpi=600)#+elec2[num-1]  #循环保存图片,bbox_inches='tight'
#    
#


























#path='E:/target/PSD_data/target1/repeat/'
#S910_beta_freq=path+('Theta_PSD')
#beta_freq=sio.loadmat(S910_beta_freq)
#repeat_T1=np.mean(beta_freq['DATA'],2)
#
#path='E:/target/PSD_data/target2/repeat/'
#S910_beta_time=path+('Theta_PSD')
#beta_time=sio.loadmat(S910_beta_time)
#repeat_T2=np.mean(beta_time['DATA'],2)
#
#path='E:/target/PSD_data/target3/repeat/'
#S910_beta_freq=path+('Theta_PSD')
#beta_freq=sio.loadmat(S910_beta_freq)
#repeat_T3=np.mean(beta_freq['DATA'],2)
#
#path='E:/target/PSD_data/target4/repeat/'
#S910_beta_time=path+('Theta_PSD')
#beta_time=sio.loadmat(S910_beta_time)
#repeat_T4=np.mean(beta_time['DATA'],2)
#
#path='E:/target/PSD_data/target5/repeat/'
#S910_beta_freq=path+('Theta_PSD')
#beta_freq=sio.loadmat(S910_beta_freq)
#repeat_T5=np.mean(beta_freq['DATA'],2)
#
#path='E:/target/PSD_data/target6/repeat/'
#S910_beta_time=path+('Theta_PSD')
#beta_time=sio.loadmat(S910_beta_time)
#repeat_T6=np.mean(beta_time['DATA'],2)
#
#path='E:/target/PSD_data/target7/repeat/'
#S910_beta_freq=path+('Theta_PSD')
#beta_freq=sio.loadmat(S910_beta_freq)
#repeat_T7=np.mean(beta_freq['DATA'],2)
#
#path='E:/target/PSD_data/target8/repeat/'
#S910_beta_time=path+('Theta_PSD')
#beta_time=sio.loadmat(S910_beta_time)
#repeat_T8=np.mean(beta_time['DATA'],2)
#repeat_T9=(repeat_T2+repeat_T3+repeat_T4+repeat_T5+repeat_T6+repeat_T7+repeat_T8)/7
####################################################
####################################################
#path='E:/target/PSD_data/target1/switch/'
#S910_beta_freq=path+('Theta_PSD')
#beta_freq=sio.loadmat(S910_beta_freq)
#switch_T1=np.mean(beta_freq['DATA'],2)
#
#path='E:/target/PSD_data/target2/switch/'
#S910_beta_time=path+('Theta_PSD')
#beta_time=sio.loadmat(S910_beta_time)
#switch_T2=np.mean(beta_time['DATA'],2)
#
#path='E:/target/PSD_data/target3/switch/'
#S910_beta_freq=path+('Theta_PSD')
#beta_freq=sio.loadmat(S910_beta_freq)
#switch_T3=np.mean(beta_freq['DATA'],2)
#
#path='E:/target/PSD_data/target4/switch/'
#S910_beta_time=path+('Theta_PSD')
#beta_time=sio.loadmat(S910_beta_time)
#switch_T4=np.mean(beta_time['DATA'],2)
#
#path='E:/target/PSD_data/target5/switch/'
#S910_beta_freq=path+('Theta_PSD')
#beta_freq=sio.loadmat(S910_beta_freq)
#switch_T5=np.mean(beta_freq['DATA'],2)
#
#path='E:/target/PSD_data/target6/switch/'
#S910_beta_time=path+('Theta_PSD')
#beta_time=sio.loadmat(S910_beta_time)
#switch_T6=np.mean(beta_time['DATA'],2)
#
#path='E:/target/PSD_data/target7/switch/'
#S910_beta_freq=path+('Theta_PSD')
#beta_freq=sio.loadmat(S910_beta_freq)
#switch_T7=np.mean(beta_freq['DATA'],2)
#
#path='E:/target/PSD_data/target8/switch/'
#S910_beta_time=path+('Theta_PSD')
#beta_time=sio.loadmat(S910_beta_time)
#switch_T8=np.mean(beta_time['DATA'],2)
#switch_T9=(switch_T2+switch_T3+switch_T4+switch_T5+switch_T6+switch_T7+switch_T8)/7
####################################################
#for i in np.arange(59):
#    plt.figure(figsize=(6,6),facecolor='none',edgecolor='none')
#    city=['T1','T2-T8']
#    Gender=['Repeat','Switch']
#    pos = np.arange(len(city))
#    bar_width = 0.35
#    T1_mean=[np.mean(repeat_T1[:,i]),np.mean(repeat_T9[:,i])]
#    T28_mean=[np.mean(switch_T1[:,i]),np.mean(switch_T9[:,i])]
#    T_mean=[T1_mean,T28_mean]
#    T1_std=[np.std(repeat_T1[:,i])/11.4018,np.std(repeat_T9[:,i])/11.4018]
#    T28_std=[np.std(switch_T1[:,i])/11.4018,np.std(switch_T9[:,i])/11.4018]
#    T_std=[T1_std,T28_std]
#    plt.bar(pos,T1_mean,bar_width,yerr=T1_std,color='blue',edgecolor='black')
#    plt.bar(pos+bar_width,T28_mean,bar_width,yerr=T28_std,color='red',edgecolor='black')
#    plt.style.use('ggplot')
#    plt.xticks(pos, city)
#    plt.tick_params(labelsize=13)
#    plt.xlabel('Stimulus', fontsize=16)
#    plt.ylabel('PSD(dB)', fontsize=16)
#    plt.ylim(np.min(T_mean)-np.max(T_std),np.max(T_mean)+np.max(T_std)+np.min(T_std))
#    plt.legend(Gender, fontsize=16)
#    plt.savefig("C:/Users/Administrator/Desktop/End_paper/target/PSD/"+'T1-T28_Theta_PSD_bar-'+str(channels[i])+'.png',dpi=600)#+elec2[num-1]  #循环保存图片,bbox_inches='tight'
#



###################################################
#for i in np.arange(59):
#    plt.figure(figsize=(12,6),facecolor='none',edgecolor='none')
#    city=['T1','T2','T3','T4','T5','T6','T7','T8']
#    Gender=['Repeat','Switch']
#    pos = np.arange(len(city))
#    bar_width = 0.35
#    T1_mean=[np.mean(repeat_T1[:,i]),np.mean(repeat_T2[:,i]),np.mean(repeat_T3[:,i]),np.mean(repeat_T4[:,i]),
#             np.mean(repeat_T5[:,i]),np.mean(repeat_T6[:,i]),np.mean(repeat_T7[:,i]),np.mean(repeat_T8[:,i])]
#    T28_mean=[np.mean(switch_T1[:,i]),np.mean(switch_T2[:,i]),np.mean(switch_T3[:,i]),np.mean(switch_T4[:,i]),
#              np.mean(switch_T5[:,i]),np.mean(switch_T6[:,i]),np.mean(switch_T7[:,i]),np.mean(switch_T8[:,i])]
#    T_mean=[T1_mean,T28_mean]
#    T1_std=[np.std(repeat_T1[:,i])/11.4018,np.std(repeat_T2[:,i])/11.4018,np.std(repeat_T3[:,i])/11.4018,np.std(repeat_T4[:,i])/11.4018,
#            np.std(repeat_T5[:,i])/11.4018,np.std(repeat_T6[:,i])/11.4018,np.std(repeat_T7[:,i])/11.4018,np.std(repeat_T8[:,i])/11.4018]
#    T28_std=[np.std(switch_T1[:,i])/11.4018,np.std(switch_T2[:,i])/11.4018,np.std(switch_T3[:,i])/11.4018,np.std(switch_T4[:,i])/11.4018,
#             np.std(switch_T5[:,i])/11.4018,np.std(switch_T6[:,i])/11.4018,np.std(switch_T7[:,i])/11.4018,np.std(switch_T8[:,i])/11.4018]
#    T_std=[T1_std,T28_std]
#    plt.bar(pos,T1_mean,bar_width,yerr=T1_std,color='blue',edgecolor='black')
#    plt.bar(pos+bar_width,T28_mean,bar_width,yerr=T28_std,color='red',edgecolor='black')
#    plt.style.use('ggplot')
#    plt.xticks(pos, city)
#    plt.tick_params(labelsize=13)
#    plt.xlabel('Stimulus', fontsize=16)
#    plt.ylabel('PSD(dB)', fontsize=16)
#    plt.ylim(np.min(T_mean)-np.max(T_std),np.max(T_mean)+np.max(T_std)+np.min(T_std))
#    plt.legend(Gender, fontsize=16)
#    plt.savefig("C:/Users/Administrator/Desktop/End_paper/target/PSD/"+'T1-T8_Alpha_PSD_bar-'+str(channels[i])+'.png',dpi=600)#+elec2[num-1]  #循环保存图片,bbox_inches='tight'

#path='E:/Inhibition/inhibition_data/PSD_data/repeat/H_H/'
#S910_beta_freq=path+('Alpha_PSD')
#beta_freq=sio.loadmat(S910_beta_freq)
#repeat_H_H=np.mean(beta_freq['DATA'],2)
#
#path='E:/Inhibition/inhibition_data/PSD_data/repeat/H_L/'
#S910_beta_time=path+('Alpha_PSD')
#beta_time=sio.loadmat(S910_beta_time)
#repeat_H_L=np.mean(beta_time['DATA'],2)
#
#path='E:/Inhibition/inhibition_data/PSD_data/repeat/L_H/'
#S910_beta_freq=path+('Alpha_PSD')
#beta_freq=sio.loadmat(S910_beta_freq)
#repeat_L_H=np.mean(beta_freq['DATA'],2)
#
#path='E:/Inhibition/inhibition_data/PSD_data/repeat/L_L/'
#S910_beta_time=path+('Alpha_PSD')
#beta_time=sio.loadmat(S910_beta_time)
#repeat_L_L=np.mean(beta_time['DATA'],2)
#
#path='E:/Inhibition/inhibition_data/PSD_data/switch/H_H/'
#S910_beta_freq=path+('Alpha_PSD')
#beta_freq=sio.loadmat(S910_beta_freq)
#switch_H_H=np.mean(beta_freq['DATA'],2)
#
#path='E:/Inhibition/inhibition_data/PSD_data/switch/H_L/'
#S910_beta_time=path+('Alpha_PSD')
#beta_time=sio.loadmat(S910_beta_time)
#switch_H_L=np.mean(beta_time['DATA'],2)
#
#path='E:/Inhibition/inhibition_data/PSD_data/switch/L_H/'
#S910_beta_freq=path+('Alpha_PSD')
#beta_freq=sio.loadmat(S910_beta_freq)
#switch_L_H=np.mean(beta_freq['DATA'],2)
#
#path='E:/Inhibition/inhibition_data/PSD_data/switch/L_L/'
#S910_beta_time=path+('Alpha_PSD')
#beta_time=sio.loadmat(S910_beta_time)
#switch_L_L=np.mean(beta_time['DATA'],2)
####################################################
#for i in np.arange(59):
#    plt.figure(figsize=(10,6),facecolor='none',edgecolor='none')
#    city=['H-H','H-L','L-H','L-L']
#    Gender=['Repeat','Switch']
#    pos = np.arange(len(city))
#    bar_width = 0.35
#    T1_mean=[np.mean(repeat_H_H[:,i]),np.mean(repeat_H_L[:,i]),np.mean(repeat_L_H[:,i]),np.mean(repeat_L_L[:,i])]
#    T28_mean=[np.mean(switch_H_H[:,i]),np.mean(switch_H_L[:,i]),np.mean(switch_L_H[:,i]),np.mean(switch_L_L[:,i])]
#    T_mean=[T1_mean,T28_mean]
#    
#    T1_std=[np.std(repeat_H_H[:,i])/11.4018,np.std(repeat_H_L[:,i])/11.4018,np.std(repeat_L_H[:,i])/11.4018,np.std(repeat_L_L[:,i])/11.4018]
#    T28_std=[np.std(switch_H_H[:,i])/11.4018,np.std(switch_H_L[:,i])/11.4018,np.std(switch_L_H[:,i])/11.4018,np.std(switch_L_L[:,i])/11.4018]
#    T_std=[T1_std,T28_std]
#    plt.bar(pos,T1_mean,bar_width,yerr=T1_std,color='blue',edgecolor='black')
#    plt.bar(pos+bar_width,T28_mean,bar_width,yerr=T28_std,color='red',edgecolor='black')
#    plt.style.use('ggplot')
#    plt.xticks(pos, city)
#    plt.tick_params(labelsize=13)
#    plt.xlabel('Stimulus', fontsize=16)
#    plt.ylabel('PSD(dB)', fontsize=16)
#    plt.ylim(np.min(T_mean)-np.max(T_std),np.max(T_mean)+np.max(T_std)+np.min(T_std))
#    plt.legend(Gender, fontsize=16)
#    plt.xlabel('Stimulus', fontsize=16)
#    plt.ylabel('PSD(dB)', fontsize=16)
#    plt.savefig("C:/Users/Administrator/Desktop/End_paper/inhibition/PSD/"+'Inhibition_Alpha_PSD_bar-'+str(channels[i])+'.png',dpi=600)#+elec2[num-1]  #循环保存图片,bbox_inches='tight'
