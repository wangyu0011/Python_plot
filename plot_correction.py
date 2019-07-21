# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 08:15:33 2019

@author: Administrator
"""

import scipy.io as sio
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
font2 = {'family' : 'Times New Roman','weight' : 'normal','size'   : 20,}
font1 = {'family' : 'Times New Roman','weight' : 'normal','size'   : 12,}
channels=['Fp1','Fpz','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4','T8',
          'CP5','CP1','CP2','CP6','P7','P3','Pz','P4','P8','POz','O1','O2','AF7','AF3','AF4','AF8','F5',
          'F1','F2','F6','FC3','FC4','C5','C1','C2','C6','CP3','CP4','P5','P1','P2','P6','PO5','PO3','PO4',
          'PO6','FT7','FT8','TP7','TP8','PO7','PO8','POz','Oz']
###  RT
path1='D:/Emotion/New2/new_result/AFter/ERSP/ROI/Negative/'
response=path1+('P200_alpha')
response_data=sio.loadmat(response)
DATA=response_data['DATA']
### ERP component timefeature
path2='D:/Emotion/New2/time_feature/'
P170_am=path2+('A_Negative_P200_le')
P170_am_data=sio.loadmat(P170_am)
P170=P170_am_data['Data'].T
for i in np.arange(61):
    g=sns.jointplot(DATA[i,:], P170[i,:], kind="reg")
    g = g.plot_joint(plt.scatter,color="r", s=40, edgecolor="black")
    g = g.plot_marginals(sns.distplot, kde=True, color="b")
    g = g.annotate(stats.pearsonr)
    g=g.set_axis_labels(xlabel="P200-alpha", ylabel="P200-Latency")
    g.savefig('A_Nagative_P200_alpha_Latency_'+str(channels[i])+'.png',dpi=600)   
path1='D:/Emotion/New2/new_result/AFter/ERSP/ROI/Negative/'
response=path1+('P200_theta')
response_data=sio.loadmat(response)
DATA=response_data['DATA']
### ERP component timefeature
path2='D:/Emotion/New2/time_feature/'
P170_am=path2+('A_Negative_P200_le')
P170_am_data=sio.loadmat(P170_am)
P170=P170_am_data['Data'].T
for i in np.arange(61):
    g=sns.jointplot(DATA[i,:], P170[i,:], kind="reg")
    g = g.plot_joint(plt.scatter,color="r", s=40, edgecolor="black")
    g = g.plot_marginals(sns.distplot, kde=True, color="b")
    g = g.annotate(stats.pearsonr)
    g=g.set_axis_labels(xlabel="P200-theta", ylabel="P200-Latency")
    g.savefig('A_Nagative_P200_theta_Latency_'+str(channels[i])+'.png',dpi=600)   
########################################################################################################
path1='D:/Emotion/New2/new_result/AFter/ERSP/ROI/Neutral/'
response=path1+('P200_alpha')
response_data=sio.loadmat(response)
DATA=response_data['DATA']
### ERP component timefeature
path2='D:/Emotion/New2/time_feature/'
P170_am=path2+('A_Neutral_P200_le')
P170_am_data=sio.loadmat(P170_am)
P170=P170_am_data['Data'].T
for i in np.arange(61):
    g=sns.jointplot(DATA[i,:], P170[i,:], kind="reg")
    g = g.plot_joint(plt.scatter,color="r", s=40, edgecolor="black")
    g = g.plot_marginals(sns.distplot, kde=True, color="b")
    g = g.annotate(stats.pearsonr)
    g=g.set_axis_labels(xlabel="P200-alpha", ylabel="P200-Latency")
    g.savefig('A_Neutral_P200_alpha_Latency_'+str(channels[i])+'.png',dpi=600)   
path1='D:/Emotion/New2/new_result/AFter/ERSP/ROI/Neutral/'
response=path1+('P200_theta')
response_data=sio.loadmat(response)
DATA=response_data['DATA']
### ERP component timefeature
path2='D:/Emotion/New2/time_feature/'
P170_am=path2+('A_Neutral_P200_le')
P170_am_data=sio.loadmat(P170_am)
P170=P170_am_data['Data'].T
for i in np.arange(61):
    g=sns.jointplot(DATA[i,:], P170[i,:], kind="reg")
    g = g.plot_joint(plt.scatter,color="r", s=40, edgecolor="black")
    g = g.plot_marginals(sns.distplot, kde=True, color="b")
    g = g.annotate(stats.pearsonr)
    g=g.set_axis_labels(xlabel="P200-theta", ylabel="P200-Latency")
    g.savefig('A_Neutral_P200_theta_Latency_'+str(channels[i])+'.png',dpi=600)   
#########################################################################################################
path1='D:/Emotion/New2/new_result/AFter/ERSP/ROI/Positive/'
response=path1+('P200_alpha')
response_data=sio.loadmat(response)
DATA=response_data['DATA']
### ERP component timefeature
path2='D:/Emotion/New2/time_feature/'
P170_am=path2+('A_Positive_P200_le')
P170_am_data=sio.loadmat(P170_am)
P170=P170_am_data['Data'].T
for i in np.arange(61):
    g=sns.jointplot(DATA[i,:], P170[i,:], kind="reg")
    g = g.plot_joint(plt.scatter,color="r", s=40, edgecolor="black")
    g = g.plot_marginals(sns.distplot, kde=True, color="b")
    g = g.annotate(stats.pearsonr)
    g=g.set_axis_labels(xlabel="P200-alpha", ylabel="P200-Latency")
    g.savefig('A_Positive_P200_alpha_Latency_'+str(channels[i])+'.png',dpi=600)   
path1='D:/Emotion/New2/new_result/AFter/ERSP/ROI/Positive/'
response=path1+('P200_theta')
response_data=sio.loadmat(response)
DATA=response_data['DATA']
### ERP component timefeature
path2='D:/Emotion/New2/time_feature/'
P170_am=path2+('A_Positive_P200_le')
P170_am_data=sio.loadmat(P170_am)
P170=P170_am_data['Data'].T
for i in np.arange(61):
    g=sns.jointplot(DATA[i,:], P170[i,:], kind="reg")
    g = g.plot_joint(plt.scatter,color="r", s=40, edgecolor="black")
    g = g.plot_marginals(sns.distplot, kde=True, color="b")
    g = g.annotate(stats.pearsonr)
    g=g.set_axis_labels(xlabel="P200-theta", ylabel="P200-Latency")
    g.savefig('A_Positive_P200_theta_Latency_'+str(channels[i])+'.png',dpi=600)   
#########################################################################################################
path1='D:/Emotion/New2/new_result/Before/ERSP/ROI/Negative/'
response=path1+('P200_alpha')
response_data=sio.loadmat(response)
DATA=response_data['DATA']
### ERP component timefeature
path2='D:/Emotion/New2/time_feature/'
P170_am=path2+('B_Negative_P200_le')
P170_am_data=sio.loadmat(P170_am)
P170=P170_am_data['Data'].T
for i in np.arange(61):
    g=sns.jointplot(DATA[i,:], P170[i,:], kind="reg")
    g = g.plot_joint(plt.scatter,color="r", s=40, edgecolor="black")
    g = g.plot_marginals(sns.distplot, kde=True, color="b")
    g = g.annotate(stats.pearsonr)
    g=g.set_axis_labels(xlabel="P200-alpha", ylabel="P200-Latency")
    g.savefig('B_Nagative_P200_alpha_Latency_'+str(channels[i])+'.png',dpi=600)   
path1='D:/Emotion/New2/new_result/Before/ERSP/ROI/Negative/'
response=path1+('P200_theta')
response_data=sio.loadmat(response)
DATA=response_data['DATA']
### ERP component timefeature
path2='D:/Emotion/New2/time_feature/'
P170_am=path2+('B_Negative_P200_le')
P170_am_data=sio.loadmat(P170_am)
P170=P170_am_data['Data'].T
for i in np.arange(61):
    g=sns.jointplot(DATA[i,:], P170[i,:], kind="reg")
    g = g.plot_joint(plt.scatter,color="r", s=40, edgecolor="black")
    g = g.plot_marginals(sns.distplot, kde=True, color="b")
    g = g.annotate(stats.pearsonr)
    g=g.set_axis_labels(xlabel="P200-theta", ylabel="P200-Latency")
    g.savefig('B_Nagative_P200_theta_Latency_'+str(channels[i])+'.png',dpi=600)   
########################################################################################################
path1='D:/Emotion/New2/new_result/Before/ERSP/ROI/Neutral/'
response=path1+('P200_alpha')
response_data=sio.loadmat(response)
DATA=response_data['DATA']
### ERP component timefeature
path2='D:/Emotion/New2/time_feature/'
P170_am=path2+('B_Neutral_P200_le')
P170_am_data=sio.loadmat(P170_am)
P170=P170_am_data['Data'].T
for i in np.arange(61):
    g=sns.jointplot(DATA[i,:], P170[i,:], kind="reg")
    g = g.plot_joint(plt.scatter,color="r", s=40, edgecolor="black")
    g = g.plot_marginals(sns.distplot, kde=True, color="b")
    g = g.annotate(stats.pearsonr)
    g=g.set_axis_labels(xlabel="P200-alpha", ylabel="P200-Latency")
    g.savefig('B_Neutral_P200_alpha_Latency_'+str(channels[i])+'.png',dpi=600)   
path1='D:/Emotion/New2/new_result/Before/ERSP/ROI/Neutral/'
response=path1+('P200_theta')
response_data=sio.loadmat(response)
DATA=response_data['DATA']
### ERP component timefeature
path2='D:/Emotion/New2/time_feature/'
P170_am=path2+('B_Neutral_P200_le')
P170_am_data=sio.loadmat(P170_am)
P170=P170_am_data['Data'].T
for i in np.arange(61):
    g=sns.jointplot(DATA[i,:], P170[i,:], kind="reg")
    g = g.plot_joint(plt.scatter,color="r", s=40, edgecolor="black")
    g = g.plot_marginals(sns.distplot, kde=True, color="b")
    g = g.annotate(stats.pearsonr)
    g=g.set_axis_labels(xlabel="P200-theta", ylabel="P200-Latency")
    g.savefig('B_Neutral_P200_theta_Latency_'+str(channels[i])+'.png',dpi=600)   
#########################################################################################################
path1='D:/Emotion/New2/new_result/Before/ERSP/ROI/Positive/'
response=path1+('P200_alpha')
response_data=sio.loadmat(response)
DATA=response_data['DATA']
### ERP component timefeature
path2='D:/Emotion/New2/time_feature/'
P170_am=path2+('B_Positive_P200_le')
P170_am_data=sio.loadmat(P170_am)
P170=P170_am_data['Data'].T
for i in np.arange(61):
    g=sns.jointplot(DATA[i,:], P170[i,:], kind="reg")
    g = g.plot_joint(plt.scatter,color="r", s=40, edgecolor="black")
    g = g.plot_marginals(sns.distplot, kde=True, color="b")
    g = g.annotate(stats.pearsonr)
    g=g.set_axis_labels(xlabel="P200-alpha", ylabel="P200-Latency")
    g.savefig('B_Positive_P200_alphB_Latency_'+str(channels[i])+'.png',dpi=600)   
path1='D:/Emotion/New2/new_result/Before/ERSP/ROI/Positive/'
response=path1+('P200_theta')
response_data=sio.loadmat(response)
DATA=response_data['DATA']
### ERP component timefeature
path2='D:/Emotion/New2/time_feature/'
P170_am=path2+('B_Positive_P200_le')
P170_am_data=sio.loadmat(P170_am)
P170=P170_am_data['Data'].T
for i in np.arange(61):
    g=sns.jointplot(DATA[i,:], P170[i,:], kind="reg")
    g = g.plot_joint(plt.scatter,color="r", s=40, edgecolor="black")
    g = g.plot_marginals(sns.distplot, kde=True, color="b")
    g = g.annotate(stats.pearsonr)
    g=g.set_axis_labels(xlabel="P200-theta", ylabel="P200-Latency")
    g.savefig('B_Positive_P200_thetB_Latency_'+str(channels[i])+'.png',dpi=600)   
##############################################################################################################
path1='D:/Emotion/New2/new_result/HC/ERSP/ROI/Negative/'
response=path1+('P200_alpha')
response_data=sio.loadmat(response)
DATA=response_data['DATA']
### ERP component timefeature
path2='D:/Emotion/New2/time_feature/'
P170_am=path2+('H_Negative_P200_le')
P170_am_data=sio.loadmat(P170_am)
P170=P170_am_data['Data'].T
for i in np.arange(61):
    g=sns.jointplot(DATA[i,:], P170[i,:], kind="reg")
    g = g.plot_joint(plt.scatter,color="r", s=40, edgecolor="black")
    g = g.plot_marginals(sns.distplot, kde=True, color="b")
    g = g.annotate(stats.pearsonr)
    g=g.set_axis_labels(xlabel="P200-alpha", ylabel="P200-Latency")
    g.savefig('H_Nagative_P200_alphH_Latency_'+str(channels[i])+'.png',dpi=600)   
path1='D:/Emotion/New2/new_result/HC/ERSP/ROI/Negative/'
response=path1+('P200_theta')
response_data=sio.loadmat(response)
DATA=response_data['DATA']
### ERP component timefeature
path2='D:/Emotion/New2/time_feature/'
P170_am=path2+('H_Negative_P200_le')
P170_am_data=sio.loadmat(P170_am)
P170=P170_am_data['Data'].T
for i in np.arange(61):
    g=sns.jointplot(DATA[i,:], P170[i,:], kind="reg")
    g = g.plot_joint(plt.scatter,color="r", s=40, edgecolor="black")
    g = g.plot_marginals(sns.distplot, kde=True, color="b")
    g = g.annotate(stats.pearsonr)
    g=g.set_axis_labels(xlabel="P200-theta", ylabel="P200-Latency")
    g.savefig('H_Nagative_P200_thetH_Latency_'+str(channels[i])+'.png',dpi=600)   
########################################################################################################
path1='D:/Emotion/New2/new_result/HC/ERSP/ROI/Neutral/'
response=path1+('P200_alpha')
response_data=sio.loadmat(response)
DATA=response_data['DATA']
### ERP component timefeature
path2='D:/Emotion/New2/time_feature/'
P170_am=path2+('H_Neutral_P200_le')
P170_am_data=sio.loadmat(P170_am)
P170=P170_am_data['Data'].T
for i in np.arange(61):
    g=sns.jointplot(DATA[i,:], P170[i,:], kind="reg")
    g = g.plot_joint(plt.scatter,color="r", s=40, edgecolor="black")
    g = g.plot_marginals(sns.distplot, kde=True, color="b")
    g = g.annotate(stats.pearsonr)
    g=g.set_axis_labels(xlabel="P200-alpha", ylabel="P200-Latency")
    g.savefig('H_Neutral_P200_alphH_Latency_'+str(channels[i])+'.png',dpi=600)   
path1='D:/Emotion/New2/new_result/HC/ERSP/ROI/Neutral/'
response=path1+('P200_theta')
response_data=sio.loadmat(response)
DATA=response_data['DATA']
### ERP component timefeature
path2='D:/Emotion/New2/time_feature/'
P170_am=path2+('H_Neutral_P200_le')
P170_am_data=sio.loadmat(P170_am)
P170=P170_am_data['Data'].T
for i in np.arange(61):
    g=sns.jointplot(DATA[i,:], P170[i,:], kind="reg")
    g = g.plot_joint(plt.scatter,color="r", s=40, edgecolor="black")
    g = g.plot_marginals(sns.distplot, kde=True, color="b")
    g = g.annotate(stats.pearsonr)
    g=g.set_axis_labels(xlabel="P200-theta", ylabel="P200-Latency")
    g.savefig('H_Neutral_P200_thetH_Latency_'+str(channels[i])+'.png',dpi=600)   
#########################################################################################################
path1='D:/Emotion/New2/new_result/HC/ERSP/ROI/Positive/'
response=path1+('P200_alpha')
response_data=sio.loadmat(response)
DATA=response_data['DATA']
### ERP component timefeature
path2='D:/Emotion/New2/time_feature/'
P170_am=path2+('H_Positive_P200_le')
P170_am_data=sio.loadmat(P170_am)
P170=P170_am_data['Data'].T
for i in np.arange(61):
    g=sns.jointplot(DATA[i,:], P170[i,:], kind="reg")
    g = g.plot_joint(plt.scatter,color="r", s=40, edgecolor="black")
    g = g.plot_marginals(sns.distplot, kde=True, color="b")
    g = g.annotate(stats.pearsonr)
    g=g.set_axis_labels(xlabel="P200-alpha", ylabel="P200-Latency")
    g.savefig('H_Positive_P200_alphH_Latency_'+str(channels[i])+'.png',dpi=600)   
path1='D:/Emotion/New2/new_result/HC/ERSP/ROI/Positive/'
response=path1+('P200_theta')
response_data=sio.loadmat(response)
DATA=response_data['DATA']
### ERP component timefeature
path2='D:/Emotion/New2/time_feature/'
P170_am=path2+('H_Positive_P200_le')
P170_am_data=sio.loadmat(P170_am)
P170=P170_am_data['Data'].T
for i in np.arange(61):
    g=sns.jointplot(DATA[i,:], P170[i,:], kind="reg")
    g = g.plot_joint(plt.scatter,color="r", s=40, edgecolor="black")
    g = g.plot_marginals(sns.distplot, kde=True, color="b")
    g = g.annotate(stats.pearsonr)
    g=g.set_axis_labels(xlabel="P200-theta", ylabel="P200-Latency")
    g.savefig('H_Positive_P200_thete_Latency_'+str(channels[i])+'.png',dpi=600)   







    