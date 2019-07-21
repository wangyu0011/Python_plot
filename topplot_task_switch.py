# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 19:51:33 2019

@author: Administrator
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import sample
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.connectivity import spectral_connectivity
from mne.viz import circular_layout, plot_connectivity_circle
import time
import os.path as op
from mne.datasets import testing
from mne import Epochs, io, pick_types
from mne.event import define_target_events
import scipy.io as sio
import matplotlib.patches as patches
from mne.time_frequency import tfr_morlet, psd_multitaper
from mne.datasets import somato
from mne.stats import permutation_t_test
import pandas as pd
#from matplotlib.font_manager import FontProperties
from mne.time_frequency import (tfr_multitaper, tfr_stockwell,
                                tfr_array_morlet)
here = os.path.dirname(os.path.abspath(__file__))
fname ='E:/clear_wy/1_61.set'
###########                  load Raw data
event_id = {"S  1": 1, "S  2": 2, "S  3": 3 }  # must be specified for str events
#eog = {"IO","FT9"} eog=eog, 
raw = io.eeglab.read_raw_eeglab(fname,preload=True,
                                event_id=event_id)
font2 = {'family' : 'Times New Roman','weight' : 'normal','size'   : 20,}
font1 = {'family' : 'Times New Roman','weight' : 'normal','size'   : 12,}
picks = pick_types(raw.info, eeg=True)
events = mne.find_events(raw)
picks_epoch = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')
events_epoch = mne.find_events(raw)
tmin, tmax = -0.349, 1
event_id_negative={"S  2": 2 }
epochs_negative = mne.Epochs(raw, events_epoch, event_id_negative, tmin, tmax, picks=picks_epoch,preload=False)
evoked_negative=epochs_negative.average()

times=[0.360]

path1='E:/target/result_data/'
path_mat_S910=path1+('T2-8_repeat_N200')
data_S910=sio.loadmat(path_mat_S910)
DATA1=data_S910['Data']

times=[0.14]

A=np.mean(DATA1,1)
Data1=np.zeros([61,1350])
for p in np.arange(1350):
    for q in np.arange(61):
        Data1[q,p]=A[q]
vmax =np.max(Data1)   #[:,int(((times[0]+0.35)/1.35)*1350)]
vmin =np.min(Data1)   #[:,int(((times[0]+0.35)/1.35)*1350)]
evoked_negative.data=Data1/1000000; 
fig=evoked_negative.plot_topomap(times=times,vmax=vmax,vmin=vmin,cmap=plt.cm.jet,colorbar=True,show=False,size=2,ch_type='eeg')
fig.savefig('T2-8_repeat_N200.png',dpi=1800)




























    
    
#path1='F:/Emotion/Before/ROI/Negative/'
#path_mat_S910=path1+('P200_alpha')
#data_S910=sio.loadmat(path_mat_S910)
#DATA1=data_S910['DATA']
#
#path1='F:/Emotion/Before/ROI/Neutral/'
#path_mat_S910=path1+('P200_alpha')
#data_S910=sio.loadmat(path_mat_S910)
#DATA2=data_S910['DATA']
#
#path1='F:/Emotion/Before/ROI/Positive/'
#path_mat_S910=path1+('P200_alpha')
#data_S910=sio.loadmat(path_mat_S910)
#DATA3=data_S910['DATA']
#times=[0.14]
#
#A=np.mean(DATA1,1)
#Data1=np.zeros([61,1350])
#for p in np.arange(1350):
#    for q in np.arange(61):
#        Data1[q,p]=A[q]
#times=[0]
#
#A=np.mean(DATA2,1)
#Data2=np.zeros([61,1350])
#for p in np.arange(1350):
#    for q in np.arange(61):
#        Data2[q,p]=A[q]
#times=[0]
#
#A=np.mean(DATA3,1)
#Data3=np.zeros([61,1350])
#for p in np.arange(1350):
#    for q in np.arange(61):
#        Data3[q,p]=A[q]
#times=[0]
#
#vmax =np.max(Data1)   #[:,int(((times[0]+0.35)/1.35)*1350)]
#vmin =np.min(Data1)   #[:,int(((times[0]+0.35)/1.35)*1350)]
#
#evoked_negative.data=Data1/1000000; 
#fig=evoked_negative.plot_topomap(times=times,vmax=vmax,vmin=vmin,cmap=plt.cm.jet,colorbar=True,show=False,size=2,ch_type='eeg')
#fig.savefig('Before_Negative_P200_alpha.png',dpi=1800)
#
#vmax =np.max(Data2)   #[:,int(((times[0]+0.35)/1.35)*1350)]
#vmin =np.min(Data2)   #[:,int(((times[0]+0.35)/1.35)*1350)]
#evoked_negative.data=Data2/1000000; 
#fig=evoked_negative.plot_topomap(times=times,vmax=vmax,vmin=vmin,cmap=plt.cm.jet,colorbar=True,show=False,size=2,ch_type='eeg')
#fig.savefig('Before_Neutral_P200_alpha.png',dpi=1800)
#
#vmax =np.max(Data3)   #[:,int(((times[0]+0.35)/1.35)*1350)]
#vmin =np.min(Data3)   #[:,int(((times[0]+0.35)/1.35)*1350)]
#evoked_negative.data=Data3/1000000; 
#fig=evoked_negative.plot_topomap(times=times,vmax=vmax,vmin=vmin,cmap=plt.cm.jet,colorbar=True,show=False,size=2,ch_type='eeg')
#fig.savefig('Before_Positive_P200_alpha.png',dpi=1800)

#path1='F:/Emotion/Before/ROI/Negative/'
#path_mat_S910=path1+('P200_theta')
#data_S910=sio.loadmat(path_mat_S910)
#DATA1=data_S910['DATA']
#
#path1='F:/Emotion/Before/ROI/Neutral/'
#path_mat_S910=path1+('P200_theta')
#data_S910=sio.loadmat(path_mat_S910)
#DATA2=data_S910['DATA']
#
#path1='F:/Emotion/Before/ROI/Positive/'
#path_mat_S910=path1+('P200_theta')
#data_S910=sio.loadmat(path_mat_S910)
#DATA3=data_S910['DATA']
#times=[0.14]
#
#A=np.mean(DATA1,1)
#Data1=np.zeros([61,1350])
#for p in np.arange(1350):
#    for q in np.arange(61):
#        Data1[q,p]=A[q]
#times=[0]
#A=np.mean(DATA2,1)
#Data2=np.zeros([61,1350])
#for p in np.arange(1350):
#    for q in np.arange(61):
#        Data2[q,p]=A[q]
#times=[0]
#A=np.mean(DATA3,1)
#Data3=np.zeros([61,1350])
#for p in np.arange(1350):
#    for q in np.arange(61):
#        Data3[q,p]=A[q]
#times=[0]
#vmax =np.max(Data1)   #[:,int(((times[0]+0.35)/1.35)*1350)]
#vmin =np.min(Data1)   #[:,int(((times[0]+0.35)/1.35)*1350)]
#evoked_negative.data=Data1/1000000; 
#fig=evoked_negative.plot_topomap(times=times,vmax=vmax,vmin=vmin,cmap=plt.cm.jet,colorbar=True,show=False,size=2,ch_type='eeg')
#fig.savefig('Before_Negative_P200_theta.png',dpi=1800)
#
#vmax =np.max(Data2)   #[:,int(((times[0]+0.35)/1.35)*1350)]
#vmin =np.min(Data2)   #[:,int(((times[0]+0.35)/1.35)*1350)]
#evoked_negative.data=Data2/1000000; 
#fig=evoked_negative.plot_topomap(times=times,vmax=vmax,vmin=vmin,cmap=plt.cm.jet,colorbar=True,show=False,size=2,ch_type='eeg')
#fig.savefig('Before_Neutral_P200_theta.png',dpi=1800)
#
#vmax =np.max(Data3)   #[:,int(((times[0]+0.35)/1.35)*1350)]
#vmin =np.min(Data3)   #[:,int(((times[0]+0.35)/1.35)*1350)]
#evoked_negative.data=Data3/1000000; 
#fig=evoked_negative.plot_topomap(times=times,vmax=vmax,vmin=vmin,cmap=plt.cm.jet,colorbar=True,show=False,size=2,ch_type='eeg')
#fig.savefig('Before_Positive_P200_theta.png',dpi=1800)

#path1='F:/Emotion/Before/ROI/Negative/'
#path_mat_S910=path1+('LPP1_alpha')
#data_S910=sio.loadmat(path_mat_S910)
#DATA1=data_S910['DATA']
#
#path1='F:/Emotion/Before/ROI/Neutral/'
#path_mat_S910=path1+('LPP1_alpha')
#data_S910=sio.loadmat(path_mat_S910)
#DATA2=data_S910['DATA']
#
#path1='F:/Emotion/Before/ROI/Positive/'
#path_mat_S910=path1+('LPP1_alpha')
#data_S910=sio.loadmat(path_mat_S910)
#DATA3=data_S910['DATA']
#times=[0.14]
#
#A=np.mean(DATA1,1)
#Data1=np.zeros([61,1350])
#for p in np.arange(1350):
#    for q in np.arange(61):
#        Data1[q,p]=A[q]
#times=[0]
#
#A=np.mean(DATA2,1)
#Data2=np.zeros([61,1350])
#for p in np.arange(1350):
#    for q in np.arange(61):
#        Data2[q,p]=A[q]
#times=[0]
#
#A=np.mean(DATA3,1)
#Data3=np.zeros([61,1350])
#for p in np.arange(1350):
#    for q in np.arange(61):
#        Data3[q,p]=A[q]
#times=[0]
#
#vmax =np.max(Data1)   #[:,int(((times[0]+0.35)/1.35)*1350)]
#vmin =np.min(Data1)   #[:,int(((times[0]+0.35)/1.35)*1350)]
#
#evoked_negative.data=Data1/1000000; 
#fig=evoked_negative.plot_topomap(times=times,vmax=vmax,vmin=vmin,cmap=plt.cm.jet,colorbar=True,show=False,size=2,ch_type='eeg')
#fig.savefig('Before_Negative_LPP1_alpha.png',dpi=1800)
#
#vmax =np.max(Data2)   #[:,int(((times[0]+0.35)/1.35)*1350)]
#vmin =np.min(Data2)   #[:,int(((times[0]+0.35)/1.35)*1350)]
#evoked_negative.data=Data2/1000000; 
#fig=evoked_negative.plot_topomap(times=times,vmax=vmax,vmin=vmin,cmap=plt.cm.jet,colorbar=True,show=False,size=2,ch_type='eeg')
#fig.savefig('Before_Neutral_LPP1_alpha.png',dpi=1800)
#
#vmax =np.max(Data3)   #[:,int(((times[0]+0.35)/1.35)*1350)]
#vmin =np.min(Data3)   #[:,int(((times[0]+0.35)/1.35)*1350)]
#evoked_negative.data=Data3/1000000; 
#fig=evoked_negative.plot_topomap(times=times,vmax=vmax,vmin=vmin,cmap=plt.cm.jet,colorbar=True,show=False,size=2,ch_type='eeg')
#fig.savefig('Before_Positive_LPP1_alpha.png',dpi=1800)
#
#path1='F:/Emotion/Before/ROI/Negative/'
#path_mat_S910=path1+('LPP1_theta')
#data_S910=sio.loadmat(path_mat_S910)
#DATA1=data_S910['DATA']
#
#path1='F:/Emotion/Before/ROI/Neutral/'
#path_mat_S910=path1+('LPP1_theta')
#data_S910=sio.loadmat(path_mat_S910)
#DATA2=data_S910['DATA']
#
#path1='F:/Emotion/Before/ROI/Positive/'
#path_mat_S910=path1+('LPP1_theta')
#data_S910=sio.loadmat(path_mat_S910)
#DATA3=data_S910['DATA']
#times=[0.14]
#
#A=np.mean(DATA1,1)
#Data1=np.zeros([61,1350])
#for p in np.arange(1350):
#    for q in np.arange(61):
#        Data1[q,p]=A[q]
#times=[0]
#
#A=np.mean(DATA2,1)
#Data2=np.zeros([61,1350])
#for p in np.arange(1350):
#    for q in np.arange(61):
#        Data2[q,p]=A[q]
#times=[0]
#
#A=np.mean(DATA3,1)
#Data3=np.zeros([61,1350])
#for p in np.arange(1350):
#    for q in np.arange(61):
#        Data3[q,p]=A[q]
#times=[0]
#
#vmax =np.max(Data1)   #[:,int(((times[0]+0.35)/1.35)*1350)]
#vmin =np.min(Data1)   #[:,int(((times[0]+0.35)/1.35)*1350)]
#
#evoked_negative.data=Data1/1000000; 
#fig=evoked_negative.plot_topomap(times=times,vmax=vmax,vmin=vmin,cmap=plt.cm.jet,colorbar=True,show=False,size=2,ch_type='eeg')
#fig.savefig('Before_Negative_LPP1_theta.png',dpi=1800)
#
#vmax =np.max(Data2)   #[:,int(((times[0]+0.35)/1.35)*1350)]
#vmin =np.min(Data2)   #[:,int(((times[0]+0.35)/1.35)*1350)]
#evoked_negative.data=Data2/1000000; 
#fig=evoked_negative.plot_topomap(times=times,vmax=vmax,vmin=vmin,cmap=plt.cm.jet,colorbar=True,show=False,size=2,ch_type='eeg')
#fig.savefig('Before_Neutral_LPP1_theta.png',dpi=1800)
#
#vmax =np.max(Data3)   #[:,int(((times[0]+0.35)/1.35)*1350)]
#vmin =np.min(Data3)   #[:,int(((times[0]+0.35)/1.35)*1350)]
#evoked_negative.data=Data3/1000000; 
#fig=evoked_negative.plot_topomap(times=times,vmax=vmax,vmin=vmin,cmap=plt.cm.jet,colorbar=True,show=False,size=2,ch_type='eeg')
#fig.savefig('Before_Positive_LPP1_theta.png',dpi=1800)
#  
    
    
    
    
    



