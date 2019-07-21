"""
==============
System Monitor
==============

"""
import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import scipy.io as sio
import pandas as pd
from PIL import Image,ImageDraw,ImageFont # 导入模块
names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3','C4', 'P3', 'P4', 'O1', 'O2','F7', 'F8','T7', 'T8', 'P7','P8', 'Fz', 'Cz', 'Pz', 
               'FC1','FC2', 'CP1', 'CP2', 'FC5', 'FC6','CP5', 'CP6']
path1='E:/cue/FC_data/repeat/pte/'
response=path1+('PTE_alpha_avr')
response_data=sio.loadmat(response)
repeat=response_data['data']
path1='E:/cue/FC_data/switch/pte/'
response=path1+('PTE_alpha_avr')
response_data=sio.loadmat(response)
switch=response_data['data']
for i in np.arange(27):
    for j in np.arange(27):
        pos = np.arange(2)
        fig, ax = plt.subplots(figsize=(8,6))
        ind =[0,1]
        T1_mean=[np.mean(repeat[:,i,j]),np.mean(switch[:,i,j])]
        T1_std=[np.std(repeat[:,i,j])/11.4018,np.std(switch[:,i,j])/11.4018]
        # show the figure, but do not block
        plt.style.use('ggplot')
        plt.bar(ind[0], T1_mean[0],yerr=T1_std[0],color='red',edgecolor='black',label='Pos')
        plt.bar(ind[1], T1_mean[1],yerr=T1_std[1],color='blue',edgecolor='black',label='Pos')
        plt.ylim(np.min(T1_mean)-np.max(T1_std),np.max(T1_mean)+np.max(T1_std))
        plt.xlabel('Stimulus', fontsize=16)
        plt.ylabel('PTE', fontsize=16)
        plt.xticks(pos, ['Repeat','Switch'])
        plt.savefig("C:/Users/Administrator/Desktop/End_paper/cue/FC_data/"+'PTE_alpha_cue-'+str(names[i])+'-to-'+str(names[j])+'.png',dpi=600)#+elec2[num-1]  #循环保存图片,bbox_inches='tight'
        plt.show()


#font=ImageFont.truetype(r'C:/Users/Administrator/Desktop/End_paper/msyh.ttf',300)
#channels=['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T7','T8','P7','P8','Fz','Cz','Pz','FC1',
#          'FC2','CP1','CP2','FC5','FC6','CP5','CP6','F1','F2','C1','C2','P1','P2','AF3','AF4','FC3','FC4',
#          'CP3','CP4','PO3','PO4','F5','F6','C5','C6','P5','P6','AF7','AF8','FT7','FT8','TP7','TP8','PO7','PO8','Fpz',
#          'CPz','POz','Oz']
#path='E:/cue/PSD_data/repeat/'
#S910_beta_time=path+('Alpha_PSD')
#beta_time=sio.loadmat(S910_beta_time)
#repeat=np.mean(beta_time['DATA'],2)
#
#path='E:/cue/PSD_data/switch/'
#S910_beta_freq=path+('Alpha_PSD')
#beta_freq=sio.loadmat(S910_beta_freq)
#switch=np.mean(beta_freq['DATA'],2)
#
#for i in np.arange(59):
#    pos = np.arange(2)
#    fig, ax = plt.subplots()
#    ind =[0,1]
#    T1_mean=[np.mean(repeat[:,i]),np.mean(switch[:,i])]
#    T1_std=[np.std(repeat[:,i])/11.4018,np.std(switch[:,i])/11.4018]
#    # show the figure, but do not block
#    plt.style.use('ggplot')
#    plt.bar(ind[0], T1_mean[0],yerr=T1_std[0],color='red',edgecolor='black',label='Pos')
#    plt.bar(ind[1], T1_mean[1],yerr=T1_std[1],color='blue',edgecolor='black',label='Pos')
#    plt.ylim(np.min(T1_mean)-np.max(T1_std),np.max(T1_mean)+np.max(T1_std))
#    plt.xlabel('Stimulus', fontsize=16)
#    plt.ylabel('PSD(dB)', fontsize=16)
#    plt.xticks(pos, ['repeat','switch'])
#    plt.savefig("C:/Users/Administrator/Desktop/End_paper/cue/PSD/"+'Cue_Alpha_PSD_bar-'+str(channels[i])+'.png',dpi=600)#+elec2[num-1]  #循环保存图片,bbox_inches='tight'
#    plt.show()