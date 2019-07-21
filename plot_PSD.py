# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 10:12:24 2019

@author: Administrator
"""
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

font2 = {'family' : 'Times New Roman','weight' : 'normal','size'   : 15,}
font1 = {'family' : 'Times New Roman','weight' : 'normal','size'   : 12,}

channels=['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T7','T8','P7','P8','Fz','Cz','Pz','FC1',
          'FC2','CP1','CP2','FC5','FC6','CP5','CP6','F1','F2','C1','C2','P1','P2','AF3','AF4','FC3','FC4',
          'CP3','CP4','PO3','PO4','F5','F6','C5','C6','P5','P6','AF7','AF8','FT7','FT8','TP7','TP8','PO7','PO8','Fpz',
          'CPz','POz','Oz']

path1='E:/cue/PSD_data/switch/'
path_hon=path1+'PSD.mat'
data_hon=sio.loadmat(path_hon)
T1_repeat= data_hon['DATA']
path1='E:/cue/PSD_data/repeat/'
path_hon=path1+'PSD.mat'
data_hon=sio.loadmat(path_hon)
T2_repeat= data_hon['DATA']
for i in np.arange(59):
    series1=np.mean(T1_repeat,0)
    series2=np.mean(T2_repeat,0)
    x=[4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    plt.style.use('ggplot')
    plt.plot(x, series1[i,0:17],linestyle='-', color="#FF0000",  linewidth=2,)
    plt.plot(x, series2[i,0:17],linestyle='-', color="#0000FF",  linewidth=2,)             
    plt.title(channels[i])
    plt.xlabel('Frequency(Hz)))')
    plt.ylabel('PSD(dB)')
    plt.legend(['Switch','Repeat'])
    plt.savefig("C:/Users/Administrator/Desktop/End_paper/cue/PSD/"+'Cue_-'+str(channels[i])+'.png',dpi=600)
    plt.show()












































#path1='E:/Inhibition/inhibition_data/PSD_data/repeat/H_H/'
#path_hon=path1+'PSD.mat'
#data_hon=sio.loadmat(path_hon)
#T1_repeat= data_hon['DATA']
#path1='E:/Inhibition/inhibition_data/PSD_data/repeat/H_L/'
#path_hon=path1+'PSD.mat'
#data_hon=sio.loadmat(path_hon)
#T2_repeat= data_hon['DATA']
#path1='E:/Inhibition/inhibition_data/PSD_data/repeat/L_H/'
#path_hon=path1+'PSD.mat'
#data_hon=sio.loadmat(path_hon)
#T3_repeat= data_hon['DATA']
#path1='E:/Inhibition/inhibition_data/PSD_data/repeat/L_L/'
#path_hon=path1+'PSD.mat'
#data_hon=sio.loadmat(path_hon)
#T4_repeat= data_hon['DATA']
#for i in np.arange(59):
#    series1=np.mean(T1_repeat,0)
#    series2=np.mean(T2_repeat,0)
#    series3=np.mean(T3_repeat,0)
#    series4=np.mean(T4_repeat,0)
#    x=[4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
#    plt.style.use('ggplot')
#    plt.plot(x, series1[i,0:17],linestyle='-', color="#FF0000",  linewidth=2,)
#    plt.plot(x, series2[i,0:17],linestyle='-', color="#0000FF",  linewidth=2,)
#    plt.plot(x, series3[i,0:17],linestyle='-', color="#00FF00",  linewidth=2,)
#    plt.plot(x, series4[i,0:17],linestyle='-', color="#542437",  linewidth=2,)                  
#    plt.title(channels[i])
#    plt.xlabel('Frequency(Hz)))')
#    plt.ylabel('PSD(dB)')
#    plt.legend(['Repeat-H_H','Repeat-H_L','Repeat-L_H','Repeat-L_L'])
#    plt.savefig("C:/Users/Administrator/Desktop/End_paper/inhibition/PSD/"+'Repeat_Inhibition_-'+str(channels[i])+'.png',dpi=600)
#    plt.show()

#path1='E:/target/PSD_data/target1/switch/'
#path_hon=path1+'PSD.mat'
#data_hon=sio.loadmat(path_hon)
#T1_repeat= data_hon['DATA']
#path1='E:/target/PSD_data/target2/switch/'
#path_hon=path1+'PSD.mat'
#data_hon=sio.loadmat(path_hon)
#T2_repeat= data_hon['DATA']
#path1='E:/target/PSD_data/target3/switch/'
#path_hon=path1+'PSD.mat'
#data_hon=sio.loadmat(path_hon)
#T3_repeat= data_hon['DATA']
#path1='E:/target/PSD_data/target4/switch/'
#path_hon=path1+'PSD.mat'
#data_hon=sio.loadmat(path_hon)
#T4_repeat= data_hon['DATA']
#path1='E:/target/PSD_data/target5/switch/'
#path_hon=path1+'PSD.mat'
#data_hon=sio.loadmat(path_hon)
#T5_repeat= data_hon['DATA']
#path1='E:/target/PSD_data/target6/switch/'
#path_hon=path1+'PSD.mat'
#data_hon=sio.loadmat(path_hon)
#T6_repeat= data_hon['DATA']
#path1='E:/target/PSD_data/target7/switch/'
#path_hon=path1+'PSD.mat'
#data_hon=sio.loadmat(path_hon)
#T7_repeat= data_hon['DATA']
#path1='E:/target/PSD_data/target8/switch/'
#path_hon=path1+'PSD.mat'
#data_hon=sio.loadmat(path_hon)
#T8_repeat= data_hon['DATA']
#for i in np.arange(59):
#    series1=np.mean(T1_repeat,0)
#    series2=np.mean(T2_repeat,0)
#    series3=np.mean(T3_repeat,0)
#    series4=np.mean(T4_repeat,0)
#    series5=np.mean(T5_repeat,0)
#    series6=np.mean(T6_repeat,0)
#    series7=np.mean(T7_repeat,0)
#    series8=np.mean(T8_repeat,0)
#    series9=(series2+series3+series4+series5+series6+series7+series8)/7
#    x=[4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
#    plt.style.use('ggplot')
#    plt.plot(x, series1[i,0:17],linestyle='-', color="#FF0000",  linewidth=2,)
#    plt.plot(x, series9[i,0:17],linestyle='-', color="#0000FF",  linewidth=2,)
##    plt.plot(x, series3[i,0:17],linestyle='-', color="#00FF00",  linewidth=2,)
##    plt.plot(x, series4[i,0:17],linestyle='-', color="#542437",  linewidth=2,)         
##    plt.plot(x, series5[i,0:17],linestyle='-', color="#ECD078",  linewidth=2,)
##    plt.plot(x, series6[i,0:17],linestyle='-', color="#D95B43",  linewidth=2,)
##    plt.plot(x, series7[i,0:17],linestyle='-', color="#542437",  linewidth=2,)
##    plt.plot(x, series8[i,0:17],linestyle='-', color="#53777A",  linewidth=2,)                
#    plt.title(channels[i])
#    plt.xlabel('Frequency(Hz)))')
#    plt.ylabel('PSD(dB)')
#    plt.legend(['T1-switch','T2-switch','T3-switch','T4-switch','T5-switch','T6-switch','T7-switch','T8-switch'])
#    plt.savefig("C:/Users/Administrator/Desktop/End_paper/target/PSD/"+'Switch_T1-T28-'+str(channels[i])+'.png',dpi=600)
#    plt.show()
