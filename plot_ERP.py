# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 18:20:12 2019
@author: Administrator
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import matplotlib.patches as patches
import os
repeat_names=list()
path1='E:/target/Scout/target1/switch/'
path2='E:/target/Scout/target2/switch/'
path3='E:/target/Scout/target3/switch/'
path4='E:/target/Scout/target4/switch/'
path5='E:/target/Scout/target5/switch/'
path6='E:/target/Scout/target6/switch/'
path7='E:/target/Scout/target7/switch/'
path8='E:/target/Scout/target8/switch/'
dirs=os.listdir(path1)
k=0
for i in dirs:
    if os.path.splitext(i)[1]=='.mat':
        repeat_names.append(i)
switch_names=list()
dirs=os.listdir(path2)
k=0
for i in dirs:
    if os.path.splitext(i)[1]=='.mat':
        switch_names.append(i) 
######################
for i in np.arange(148):
    response=path1+repeat_names[i]
    response_data=sio.loadmat(response)
    repeat_T1=response_data['DATA']
    response=path2+switch_names[i]
    response_data=sio.loadmat(response)
    repeat_T2=response_data['DATA']
    response=path3+repeat_names[i]
    response_data=sio.loadmat(response)
    repeat_T3=response_data['DATA']
    response=path4+switch_names[i]
    response_data=sio.loadmat(response)
    repeat_T4=response_data['DATA']
    response=path5+repeat_names[i]
    response_data=sio.loadmat(response)
    repeat_T5=response_data['DATA']
    response=path6+switch_names[i]
    response_data=sio.loadmat(response)
    repeat_T6=response_data['DATA']
    response=path7+repeat_names[i]
    response_data=sio.loadmat(response)
    repeat_T7=response_data['DATA']
    response=path8+switch_names[i]
    response_data=sio.loadmat(response)
    repeat_T8=response_data['DATA']
    A_data=np.mean(repeat_T1[:,75:675],0)
    B_data=np.mean(repeat_T2[:,75:675],0)
    C_data=np.mean(repeat_T3[:,75:675],0)
    D_data=np.mean(repeat_T4[:,75:675],0)
    E_data=np.mean(repeat_T5[:,75:675],0)
    F_data=np.mean(repeat_T6[:,75:675],0)
    H_data=np.mean(repeat_T7[:,75:675],0)
    G_data=np.mean(repeat_T8[:,75:675],0)
    
    plt.figure(figsize=(12,6),facecolor='none',edgecolor='none')
    plt.plot([-200, 1000], [0, 0], color='k', linewidth=1, linestyle="-")
    plt.plot([0, 0], [np.min(A_data),np.max(B_data)], color='k', linewidth=1, linestyle="-")
    plt.xlabel('Time/ms',fontsize=20)
    plt.ylabel('ERP/uV',fontsize=20)
    plt.tick_params(labelsize=20)
    sig_x_mat=range(-200,1000,2)
    plt.Rectangle((9,0),9,9,color='red')
    plt.style.use('ggplot')
    plt.plot(sig_x_mat, A_data, linestyle='-', color="#0000FF",  linewidth=2,label='Switch_T1')   
    plt.plot(sig_x_mat, B_data, linestyle='-', color="#FF0000",  linewidth=2,label='Switch_T2')  
    plt.plot(sig_x_mat, C_data, linestyle='-', color="#00FF00",  linewidth=2,label='Switch_T3')   
    plt.plot(sig_x_mat, D_data, linestyle='-', color="#542437",  linewidth=2,label='Switch_T4')           
    plt.plot(sig_x_mat, E_data, linestyle='-', color="#ECD078",  linewidth=2,label='Switch_T5')   
    plt.plot(sig_x_mat, F_data, linestyle='-', color="#D95B43",  linewidth=2,label='Switch_T6')  
    plt.plot(sig_x_mat, H_data, linestyle='-', color="#542437",  linewidth=2,label='Switch_T7')   
    plt.plot(sig_x_mat, G_data, linestyle='-', color="#53777A",  linewidth=2,label='Switch_T8')              
    plt.legend(fontsize=20,loc=1)   
    plt.savefig("C:/Users/Administrator/Desktop/End_paper/target/source_data/"+'Switch'+str(repeat_names[i])+'.png',dpi=600)#+elec2[num-1]  #循环保存图片,bbox_inches='tight'
    plt.show()  



#import matplotlib.pyplot as plt
#import numpy as np
#import scipy.io as sio
#import matplotlib.patches as patches
#import os
#repeat_names=list()
#path1='E:/cue/Scout_data/repeat/'
#dirs=os.listdir(path1)
#k=0
#for i in dirs:
#    if os.path.splitext(i)[1]=='.mat':
#        repeat_names.append(i)
#
#switch_names=list()
#path2='E:/cue/Scout_data/switch/'
#dirs=os.listdir(path2)
#k=0
#for i in dirs:
#    if os.path.splitext(i)[1]=='.mat':
#        switch_names.append(i) 
#######################
#for i in np.arange(148):
#    response=path1+repeat_names[i]
#    response_data=sio.loadmat(response)
#    repeat=response_data['DATA']
#    response=path2+switch_names[i]
#    response_data=sio.loadmat(response)
#    switch=response_data['DATA']
#    A_data=np.mean(repeat[:,75:675],0)
#    B_data=np.mean(switch[:,75:675],0)
#    plt.figure(figsize=(12,6),facecolor='none',edgecolor='none')
#    plt.plot([-200, 1000], [0, 0], color='k', linewidth=1, linestyle="-")
#    plt.plot([0, 0], [np.min(A_data),np.max(B_data)], color='k', linewidth=1, linestyle="-")
#    plt.xlabel('Time/ms',fontsize=20)
#    plt.ylabel('ERP/uV',fontsize=20)
#    plt.tick_params(labelsize=20)
#    sig_x_mat=range(-200,1000,2)
#    plt.Rectangle((9,0),9,9,color='red')
#    plt.style.use('ggplot')
#    plt.plot(sig_x_mat, A_data, linestyle='-', color="#0000FF",  linewidth=2,label='Repeat')   
#    plt.plot(sig_x_mat, B_data, linestyle='-', color="#FF0000",  linewidth=2,label='Switch')        
#    plt.legend(fontsize=20,loc=1)   
#    plt.savefig("C:/Users/Administrator/Desktop/End_paper/cue/source_data/"+str(repeat_names[i])+'.png',dpi=600)#+elec2[num-1]  #循环保存图片,bbox_inches='tight'
#    plt.show()  
    
    
#from matplotlib import pyplot as plt
#from matplotlib import style
#import scipy.io as sio
#import seaborn as sns
#import numpy as np
#import pandas as pd
#style.use('ggplot')
#x =np.arange(-200,1000,2)
################################################
#path1='E:/cue/switch_data/'
#response=path1+('cue_switch')
#response_data=sio.loadmat(response)
#DATA=response_data['DATA']
################################################
#path1='E:/cue/repeat_data/'
#response=path1+('cue_repeat')
#response_data=sio.loadmat(response)
#DATA1=response_data['DATA']
## can plot specifically, after just showing the defaults:
#i=16
#plt.plot(x,DATA[i,74:674],linewidth=2)
#plt.plot(x,DATA1[i,74:674],linewidth=2)
#ax = plt.gca()
#ax.spines["right"].set_color("none")
#ax.spines["top"].set_color("none")
##将左边框放到x=0的位置，将下边框放大y=0的位置
#ax.spines["top"].set_position(("data", 0))
#ax.spines["right"].set_position(("data", 0))
#plt.ylabel('Amplitude/uV')
#plt.xlabel('Time/ms')
#plt.show()
#channels=['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T7','T8','P7','P8','Fz','Cz','Pz','IO','FC1',
#          'FC2','CP1','CP2','FC5','FC6','CP5','CP6','FT9','FT10','F1','F2','C1','C2','P1','P2','AF3','AF4','FC3','FC4',
#          'CP3','CP4','PO3','PO4','F5','F6','C5','C6','P5','P6','AF7','AF8','FT7','FT8','TP7','TP8','PO7','PO8','Fpz',
#          'CPz','POz','Oz']

#channels=['Fp1','Fpz','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4','T8','CP5','CP1','CP2','CP6',
#          'P7','P3','Pz','P4','P8','POz','O1','O2','AF7','AF3','AF4','AF8','F5','F1','F2','F6','FC3','FCz','FC4','C5','C1','C2','C6',
#          'CP3','CP4','P5','P1','P2','P6','PO5','PO3','PO4','PO6','FT7','FT8','TP7','TP8','PO7','PO8','Oz']
######################
#path1='D:/Emotion/New2/result/ERP_result/'
#response=path1+('A_Positive')
#response_data=sio.loadmat(response)
#DATA=response_data['DATA']
#######################
#path1='D:/Emotion/New2/result/ERP_result/'
#response=path1+('B_Positive')
#response_data=sio.loadmat(response)
#DATA1=response_data['DATA']
########################
#path1='D:/Emotion/New2/result/ERP_result/'
#response=path1+('H_Positive')
#response_data=sio.loadmat(response)
#DATA2=response_data['DATA']
#######################
#path1='E:/target/target4/switch_data/'
#response=path1+('T4_switch')
#response_data=sio.loadmat(response)
#DATA3=response_data['DATA']
######################
#path1='E:/target/target5/switch_data/'
#response=path1+('T5_switch')
#response_data=sio.loadmat(response)
#DATA4=response_data['DATA']
#######################
#path1='E:/target/target6/switch_data/'
#response=path1+('T6_switch')
#response_data=sio.loadmat(response)
#DATA5=response_data['DATA']
#######################
#path1='E:/target/target7/switch_data/'
#response=path1+('T7_switch')
#response_data=sio.loadmat(response)
#DATA6=response_data['DATA']
#######################
#path1='E:/target/target8/switch_data/'
#response=path1+('T8_switch')
#response_data=sio.loadmat(response)
#DATA7=response_data['DATA']
######################
#path1='E:/target/'
#response=path1+('T2_8_switch')
#response_data=sio.loadmat(response)
#DATA1=response_data['DATA']
###############################################
###############################################
###############################################
#path1='E:/Inhibition/inhibition_data/result_data/'
#response=path1+('repeat_H_H')
#response_data=sio.loadmat(response)
#DATA=np.mean(response_data['Data'],1)
#######################
#path1='E:/Inhibition/inhibition_data/result_data/'
#response=path1+('repeat_H_L')
#response_data=sio.loadmat(response)
#DATA1=np.mean(response_data['Data'],1)
#######################
#path1='E:/Inhibition/inhibition_data/result_data/'
#response=path1+('repeat_L_H')
#response_data=sio.loadmat(response)
#DATA2=np.mean(response_data['Data'],1)
#######################
#path1='E:/Inhibition/inhibition_data/result_data/'
#response=path1+('repeat_L_L')
#response_data=sio.loadmat(response)
#DATA3=np.mean(response_data['Data'],1)
#for i in np.arange(61):
#    plt.figure(figsize=(12,6),facecolor='none',edgecolor='none')
#    
#    plt.plot([-200, 1000], [0, 0], color='k', linewidth=1, linestyle="-")
#    plt.plot([0, 0], [-3, 4], color='k', linewidth=1, linestyle="-")
#    plt.xlabel('Time/ms',fontsize=20)
#    plt.ylabel('ERP/uV',fontsize=20)
#    plt.tick_params(labelsize=20)
#    # 导入数据
#    A_data=DATA[i,:]
#    B_data=DATA1[i,:]
#    C_data=DATA2[i,:]
##    D_data=DATA3[i,74:674]
##    E_data=DATA4[i,74:674]
##    F_data=DATA5[i,74:674]
##    H_data=DATA6[i,74:674]
##    G_data=DATA7[i,74:674]
#    sig_x_mat=range(-200,1000,1)
#    plt.Rectangle((9,0),9,9,color='red')
#    #绘制折线
#    #plt.plot(sig_x_mat, sig_S910, linestyle='-', color="#FF851B", linewidth=5)
##    plt.grid()
#    plt.style.use('ggplot')
#    plt.plot(sig_x_mat, A_data, linestyle='-', color="#0000FF",  linewidth=2,label='Pos')   
#    plt.plot(sig_x_mat, B_data, linestyle='-', color="#FF0000",  linewidth=2,label='Pre')  
#    plt.plot(sig_x_mat, C_data, linestyle='-', color="#00FF00",  linewidth=2,label='HC')   
##    plt.plot(sig_x_mat, D_data, linestyle='-', color="#542437",  linewidth=2,label='repeat_L_L')  
##    plt.plot(sig_x_mat, E_data, linestyle='-', color="#ECD078",  linewidth=2,label='Target5-switch')   
##    plt.plot(sig_x_mat, F_data, linestyle='-', color="#D95B43",  linewidth=2,label='Target6-switch')  
##    plt.plot(sig_x_mat, H_data, linestyle='-', color="#542437",  linewidth=2,label='Target7-switch')   
##    plt.plot(sig_x_mat, G_data, linestyle='-', color="#53777A",  linewidth=2,label='Target8-switch')           
#    plt.legend(fontsize=20,loc=1)         
#    ## add legend    
##    ##  P300
##    alpha=0.3 # 锐化系数，（0,1.0）
##    plt.bar(370,2, color='0.5', width=100, edgecolor=None, alpha=alpha,label='None') 
##    ## 画坐标轴直线
##    ####  P100
##    alpha=0.3 # 锐化系数，（0,1.0）
##    plt.bar(280,-1.5, color='0.5', width=80, edgecolor=None, alpha=alpha,label='None')
##    ####  P200
##    alpha=0.3 # 锐化系数，（0,1.0）
##    plt.bar(180,2, color='0.5', width=80, edgecolor=None, alpha=alpha,label='None')
#    plt.savefig("C:/Users/Administrator/Desktop/P200_result/ERP/"+'Positive-'+str(channels[i])+'.png',dpi=600)#+elec2[num-1]  #循环保存图片,bbox_inches='tight'
#













