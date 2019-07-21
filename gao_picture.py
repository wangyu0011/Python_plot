#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:18:34 2019

@author: wangyu
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
from mne.time_frequency import (tfr_multitaper, tfr_stockwell,
                                tfr_array_morlet)
from scipy import stats
import scipy.stats as ss
path='/home/wangyu/Documents/plot_figure/'
#############                    load time and frequency
path_1 = path+'adj_diff_third_4_5'
DATA= sio.loadmat(path_1)
data_PLV1=DATA['a'].T

path_2 = path+'adj_diff_third_2_5'
DATA= sio.loadmat(path_2)
data_PLV2=DATA['a'].T

label_name =['O2','O1','OZ','PZ','P4','CP4','P8','C4','TP8','T8','P7','P3','CP3','CPZ','CZ','FC4','FT8','TP7','C3','FCZ','FZ','F4','F8','T7','FT7','FC3','F3','FP2','F7','FP1']
################### S1234 
node_order =['O2','O1','OZ','PZ','P4','CP4','P8','C4','TP8','T8','P7','P3','CP3','CPZ','CZ','FC4','FT8','TP7','C3','FCZ','FZ','F4','F8','T7','FT7','FC3','F3','FP2','F7','FP1']
node_angles = circular_layout(label_name, node_order, start_pos=90,
                              group_boundaries=[0, len(label_name) / 2])
label_colors =[(0.09803921568627451, 0.39215686274509803, 0.1568627450980392, 1.0),
               (0.39215686274509803, 0.09803921568627451, 0.0, 1.0),
               (0.8627450980392157, 0.0784313725490196, 0.39215686274509803, 1.0),
               (0.39215686274509803, 0.0, 0.39215686274509803, 1.0),
               (0.39215686274509803, 0.0, 0.39215686274509803, 1.0),
               (0.8627450980392157, 0.0784313725490196, 0.39215686274509803, 1.0),
               (0.39215686274509803, 0.09803921568627451, 0.0, 1.0),
               (0.09803921568627451, 0.39215686274509803, 0.1568627450980392, 1.0),
               (0.00803921568627451, 0.39215686274509803, 0.1568627450980392, 1.0),
               (0.09803921568627451, 0.09215686274509803, 0.1568627450980392, 1.0),
               (0.39803921568627451, 0.39215686274509803, 0.1568627450980392, 1.0),
               (0.09803921568627451, 0.39215686274509803, 0.0068627450980392, 1.0),
               (0.39803921568627451, 0.39215686274509803, 0.0068627450980392, 1.0),
               (0.39215686274509803, 0.39803921568627451, 0.0, 1.0),
               (0.00215686274509803, 0.39803921568627451, 0.0, 1.0),
               (0.0, 0.39803921568627451, 0.4, 1.0),
               (0.0, 0.39803921568627451, 0.4, 1.0),
               (0.4, 0.0, 0.4, 1.0),
               (0.4, 0.2, 0.4, 1.0),
               (0.4, 0.4, 0.0, 1.0),
               (0.4, 0.2, 0.0, 1.0),
               (0.2, 0.0, 0.4, 1.0),
               (0.2, 0.2, 0.4, 1.0),
               (0.2, 0.4, 0.0, 1.0),
               (0.4, 0.0, 0.0, 1.0),
               (0.8, 0.2, 0.0, 1.0),
               (0.8, 0.0, 0.4, 1.0),
               (0.2, 0.8, 0.4, 1.0),
               (0.2, 0.8, 0.0, 1.0),
               (0.4, 0.0, 0.8, 1.0)]
fig = plt.figure(num=None, figsize=(9,9), facecolor='black')
plot_connectivity_circle(data_PLV1, label_name, n_lines=12,facecolor='black',textcolor='white',vmin=np.min(data_PLV1),vmax=np.max(data_PLV1),
                     node_angles=node_angles,linewidth=3,node_colors=label_colors,fig=fig,padding=6, fontsize_colorbar=15,colorbar_size=0.7, 
                     colorbar_pos=(-0.3, 0.5),colormap='hot',fontsize_title=12, fontsize_names=15)
plt.show()
fig.savefig('adj_diff_third_4.png',dpi=1800,facecolor='black')

