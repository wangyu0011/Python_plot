# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 21:46:30 2019

@author: Administrator
"""

import scipy.io as sio
import seaborn as sns
import scipy.interpolate
import matplotlib.pyplot as plt
import numpy as np
Gender=['Repeat','Switch']


path1='F:/Emotion/After/ROI/Negative/'
response=path1+('LPP1_alpha')
response_data=sio.loadmat(response)
A_data=response_data['DATA']
path1='F:/Emotion/Before/ROI/Negative/'
response=path1+('LPP1_alpha')
response_data=sio.loadmat(response)
B_data=response_data['DATA']
path1='F:/Emotion/HC/ROI/Negative/'
response=path1+('LPP1_alpha')
response_data=sio.loadmat(response)
H_data=response_data['DATA']
channels=['Fp1','Fpz','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4','T8','CP5','CP1','CP2','CP6',
          'P7','P3','Pz','P4','P8','POz','O1','O2','AF7','AF3','AF4','AF8','F5','F1','F2','F6','FC3','FCz','FC4','C5','C1','C2','C6',
          'CP3','CP4','P5','P1','P2','P6','PO5','PO3','PO4','PO6','FT7','FT8','TP7','TP8','PO7','PO8','Oz']
##for i in np.arange(61):
#i=1
#plt.style.use('ggplot')
#y=[np.mean(B_data[i,:]),np.mean(A_data[i,:]),np.mean(H_data[i,:])]
#dy=[np.std(B_data[i,:])/4.4721,np.std(A_data[i,:])/4.4721,np.std(H_data[i,:])/4.4721]
##plt.errorbar(x,y,yerr=dy,fmt='o',ecolor='r',color='b',elinewidth=2,capsize=4)
#plt.ylim(np.min(y)-np.max(dy),np.max(y)+np.max(dy)+np.max(dy)+np.max(dy))
#
#plt.errorbar([3,4,5],y,yerr=dy,fmt='-',ecolor='r',elinewidth=5,ms=5,mfc='blue',capsize=10)
#plt.errorbar([3.2,4.2,5.2],y,yerr=dy,fmt='-',ecolor='b',elinewidth=5,ms=5,mfc='blue',capsize=10)
#
#plt.legend(Gender, fontsize=16)
#plt.xticks([3,4,5],('Pre','Pos','Hc'))
#plt.ylabel('ERSP/dB',fontsize=17)
#plt.tick_params(labelsize=14)
##    plt.savefig('F:/Emotion/'+'A_Negative_LPP_alpha_'+str(channels[i])+'.png',dpi=1800)
#plt.show()


channels=['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T7','T8','P7','P8','Fz','Cz','Pz','FC1',
          'FC2','CP1','CP2','FC5','FC6','CP5','CP6','F1','F2','C1','C2','P1','P2','AF3','AF4','FC3','FC4',
          'CP3','CP4','PO3','PO4','F5','F6','C5','C6','P5','P6','AF7','AF8','FT7','FT8','TP7','TP8','PO7','PO8','Fpz',
          'CPz','POz','Oz']
#for i in np.arange(61):
i=1
plt.style.use('ggplot')
y=[np.mean(B_data[i,:]),np.mean(A_data[i,:]),np.mean(H_data[i,:])]
dy=[np.std(B_data[i,:])/4.4721,np.std(A_data[i,:])/4.4721,np.std(H_data[i,:])/4.4721]
#plt.errorbar(x,y,yerr=dy,fmt='o',ecolor='r',color='b',elinewidth=2,capsize=4)
plt.ylim(np.min(y)-np.max(dy),np.max(y)+np.max(dy)+np.max(dy)+np.max(dy))

plt.errorbar([3,4,5],y,yerr=dy,fmt='-',ecolor='r',elinewidth=5,ms=5,mfc='blue',capsize=10)
plt.errorbar([3.2,4.2,5.2],y,yerr=dy,fmt='-',ecolor='b',elinewidth=5,ms=5,mfc='blue',capsize=10)

plt.legend(Gender, fontsize=16)
plt.xticks([3,4,5],('Pre','Pos','Hc'))
plt.ylabel('ERSP/dB',fontsize=17)
plt.tick_params(labelsize=14)
plt.savefig("C:/Users/Administrator/Desktop/End_paper/target/ROI/"+'T1_T28_P200_latency_bar-'+str(channels[i])+'.png',dpi=600)
plt.show()





















































