# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 20:34:27 2019

@author: Administrator
"""

import scipy.io as sio
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate


channels=['Fp1','Fpz','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4','T8','CP5','CP1','CP2','CP6',
          'P7','P3','Pz','P4','P8','POz','O1','O2','AF7','AF3','AF4','AF8','F5','F1','F2','F6','FC3','FCz','FC4','C5','C1','C2','C6',
          'CP3','CP4','P5','P1','P2','P6','PO5','PO3','PO4','PO6','FT7','FT8','TP7','TP8','PO7','PO8','Oz']
for n in ['Negative','Neutral','Positive']:
    for j in ['A','B','H']:
        for i in np.arange(61):
            path1='D:/Emotion/New2/ERSP/'
            response=path1+(str(j)+'_'+str(n)+str(i+1))
            response_data=sio.loadmat(response)
            A_data=response_data['DATA']
            plt.grid('off')
            plt.imshow(A_data, vmin=A_data.min(), vmax=A_data.max(), origin='lower',cmap='jet',aspect='auto',
                   extent=[-70, 1020,4,30])
            plt.xlabel('Time/ms',fontsize=17)
            plt.ylabel('ERSP/dB',fontsize=17)
            plt.tick_params(labelsize=14)
            cbar=plt.colorbar()
            font_size = 14 # Adjust as appropriate.
            cbar.ax.tick_params(labelsize=font_size)
            plt.savefig("D:/Emotion/New2/ERSP/result/"+str(j)+'_'+str(n)+'-'+str(channels[i])+'.png',dpi=600)
            plt.show()









path1='D:/Emotion/New2/ERSP/'
response=path1+('A_Negative_Pz')
response_data=sio.loadmat(response)
A_data=response_data['DATA']
path1='D:/Emotion/New2/ERSP/'
response=path1+('B_Negative_Pz')
response_data=sio.loadmat(response)
B_data=response_data['DATA']
path1='D:/Emotion/New2/ERSP/'
response=path1+('B_Negative_Pz')
response_data=sio.loadmat(response)
H_data=response_data['DATA']
plt.grid()
plt.imshow(A_data, vmin=A_data.min(), vmax=A_data.max(), origin='lower',cmap='jet',aspect='auto',
           extent=[-70, 1020,4,30])
plt.xlabel('Time/ms',fontsize=17)
plt.ylabel('ERSP/dB',fontsize=17)
plt.tick_params(labelsize=14)
cbar=plt.colorbar()
font_size = 14 # Adjust as appropriate.
cbar.ax.tick_params(labelsize=font_size)
plt.savefig("D:/Emotion/New2/ERSP/"+'A_Negative-Pz.png',dpi=1800)
plt.show()






plt.grid()
plt.imshow(B_data, vmin=B_data.min(), vmax=B_data.max(), origin='lower',cmap='jet',aspect='auto',
           extent=[-70, 1020,4,30])
plt.xlabel('Time/ms',fontsize=17)
plt.ylabel('ERSP/dB',fontsize=17)
plt.tick_params(labelsize=14)
cbar=plt.colorbar()
font_size = 14 # Adjust as appropriate.
cbar.ax.tick_params(labelsize=font_size)
plt.savefig("D:/Emotion/New2/ERSP/"+'B_Negative-Pz.png',dpi=1800)
plt.show()

plt.grid()
plt.imshow(H_data, vmin=H_data.min(), vmax=H_data.max(), origin='lower',cmap='jet',aspect='auto',
           extent=[-70, 1020,4,30])
plt.xlabel('Time/ms',fontsize=17)
plt.ylabel('ERSP/dB',fontsize=17)
plt.tick_params(labelsize=14)
cbar=plt.colorbar()
font_size = 14 # Adjust as appropriate.
cbar.ax.tick_params(labelsize=font_size)
plt.savefig("D:/Emotion/New2/ERSP/"+'H_Negative-Pz.png',dpi=1800)
plt.show()

path1='D:/Emotion/New2/ERSP/'
response=path1+('A_Negative_Cz')
response_data=sio.loadmat(response)
A_data=response_data['DATA']
path1='D:/Emotion/New2/ERSP/'
response=path1+('B_Negative_Cz')
response_data=sio.loadmat(response)
B_data=response_data['DATA']
path1='D:/Emotion/New2/ERSP/'
response=path1+('B_Negative_Cz')
response_data=sio.loadmat(response)
H_data=response_data['DATA']
plt.grid()
plt.imshow(A_data, vmin=A_data.min(), vmax=A_data.max(), origin='lower',cmap='jet',aspect='auto',
           extent=[-70, 1020,4,30])
plt.xlabel('Time/ms',fontsize=17)
plt.ylabel('ERSP/dB',fontsize=17)
plt.tick_params(labelsize=14)
cbar=plt.colorbar()
font_size = 14 # Adjust as appropriate.
cbar.ax.tick_params(labelsize=font_size)
plt.savefig("D:/Emotion/New2/ERSP/"+'A_Negative-Cz.png',dpi=1800)
plt.show()

plt.grid()
plt.imshow(B_data, vmin=B_data.min(), vmax=B_data.max(), origin='lower',cmap='jet',aspect='auto',
           extent=[-70, 1020,4,30])
plt.xlabel('Time/ms',fontsize=17)
plt.ylabel('ERSP/dB',fontsize=17)
plt.tick_params(labelsize=14)
cbar=plt.colorbar()
font_size = 14 # Adjust as appropriate.
cbar.ax.tick_params(labelsize=font_size)
plt.savefig("D:/Emotion/New2/ERSP/"+'B_Negative-Cz.png',dpi=1800)
plt.show()

plt.grid()
plt.imshow(H_data, vmin=H_data.min(), vmax=H_data.max(), origin='lower',cmap='jet',aspect='auto',
           extent=[-70, 1020,4,30])
plt.xlabel('Time/ms',fontsize=17)
plt.ylabel('ERSP/dB',fontsize=17)
plt.tick_params(labelsize=14)
cbar=plt.colorbar()
font_size = 14 # Adjust as appropriate.
cbar.ax.tick_params(labelsize=font_size)
plt.savefig("D:/Emotion/New2/ERSP/"+'H_Negative-Cz.png',dpi=1800)
plt.show()

path1='D:/Emotion/New2/ERSP/'
response=path1+('A_Negative_Fz')
response_data=sio.loadmat(response)
A_data=response_data['DATA']
path1='D:/Emotion/New2/ERSP/'
response=path1+('B_Negative_Fz')
response_data=sio.loadmat(response)
B_data=response_data['DATA']
path1='D:/Emotion/New2/ERSP/'
response=path1+('B_Negative_Fz')
response_data=sio.loadmat(response)
H_data=response_data['DATA']
plt.grid()
plt.imshow(A_data, vmin=A_data.min(), vmax=A_data.max(), origin='lower',cmap='jet',aspect='auto',
           extent=[-70, 1020,4,30])
plt.xlabel('Time/ms',fontsize=17)
plt.ylabel('ERSP/dB',fontsize=17)
plt.tick_params(labelsize=14)
cbar=plt.colorbar()
font_size = 14 # Adjust as appropriate.
cbar.ax.tick_params(labelsize=font_size)
plt.savefig("D:/Emotion/New2/ERSP/"+'A_Negative-Fz.png',dpi=1800)
plt.show()

plt.grid()
plt.imshow(B_data, vmin=B_data.min(), vmax=B_data.max(), origin='lower',cmap='jet',aspect='auto',
           extent=[-70, 1020,4,30])
plt.xlabel('Time/ms',fontsize=17)
plt.ylabel('ERSP/dB',fontsize=17)
plt.tick_params(labelsize=14)
cbar=plt.colorbar()
font_size = 14 # Adjust as appropriate.
cbar.ax.tick_params(labelsize=font_size)
plt.savefig("D:/Emotion/New2/ERSP/"+'B_Negative-Fz.png',dpi=1800)
plt.show()

plt.grid()
plt.imshow(H_data, vmin=H_data.min(), vmax=H_data.max(), origin='lower',cmap='jet',aspect='auto',
           extent=[-70, 1020,4,30])
plt.xlabel('Time/ms',fontsize=17)
plt.ylabel('ERSP/dB',fontsize=17)
plt.tick_params(labelsize=14)
cbar=plt.colorbar()
font_size = 14 # Adjust as appropriate.
cbar.ax.tick_params(labelsize=font_size)
plt.savefig("D:/Emotion/New2/ERSP/"+'H_Negative-Fz.png',dpi=1800)
plt.show()

response=path1+('A_Neutral_Pz')
response_data=sio.loadmat(response)
A_data=response_data['DATA']
path1='D:/Emotion/New2/ERSP/'
response=path1+('B_Neutral_Pz')
response_data=sio.loadmat(response)
B_data=response_data['DATA']
path1='D:/Emotion/New2/ERSP/'
response=path1+('B_Neutral_Pz')
response_data=sio.loadmat(response)
H_data=response_data['DATA']
plt.grid()
plt.imshow(A_data, vmin=A_data.min(), vmax=A_data.max(), origin='lower',cmap='jet',aspect='auto',
           extent=[-70, 1020,4,30])
plt.xlabel('Time/ms',fontsize=17)
plt.ylabel('ERSP/dB',fontsize=17)
plt.tick_params(labelsize=14)
cbar=plt.colorbar()
font_size = 14 # Adjust as appropriate.
cbar.ax.tick_params(labelsize=font_size)
plt.savefig("D:/Emotion/New2/ERSP/"+'A_Neutral-Pz.png',dpi=1800)
plt.show()

plt.grid()
plt.imshow(B_data, vmin=B_data.min(), vmax=B_data.max(), origin='lower',cmap='jet',aspect='auto',
           extent=[-70, 1020,4,30])
plt.xlabel('Time/ms',fontsize=17)
plt.ylabel('ERSP/dB',fontsize=17)
plt.tick_params(labelsize=14)
cbar=plt.colorbar()
font_size = 14 # Adjust as appropriate.
cbar.ax.tick_params(labelsize=font_size)
plt.savefig("D:/Emotion/New2/ERSP/"+'B_Neutral-Pz.png',dpi=1800)
plt.show()

plt.grid()
plt.imshow(H_data, vmin=H_data.min(), vmax=H_data.max(), origin='lower',cmap='jet',aspect='auto',
           extent=[-70, 1020,4,30])
plt.xlabel('Time/ms',fontsize=17)
plt.ylabel('ERSP/dB',fontsize=17)
plt.tick_params(labelsize=14)
cbar=plt.colorbar()
font_size = 14 # Adjust as appropriate.
cbar.ax.tick_params(labelsize=font_size)
plt.savefig("D:/Emotion/New2/ERSP/"+'H_Neutral-Pz.png',dpi=1800)
plt.show()

path1='D:/Emotion/New2/ERSP/'
response=path1+('A_Neutral_Cz')
response_data=sio.loadmat(response)
A_data=response_data['DATA']
path1='D:/Emotion/New2/ERSP/'
response=path1+('B_Neutral_Cz')
response_data=sio.loadmat(response)
B_data=response_data['DATA']
path1='D:/Emotion/New2/ERSP/'
response=path1+('B_Neutral_Cz')
response_data=sio.loadmat(response)
H_data=response_data['DATA']
plt.grid()
plt.imshow(A_data, vmin=A_data.min(), vmax=A_data.max(), origin='lower',cmap='jet',aspect='auto',
           extent=[-70, 1020,4,30])
plt.xlabel('Time/ms',fontsize=17)
plt.ylabel('ERSP/dB',fontsize=17)
plt.tick_params(labelsize=14)
cbar=plt.colorbar()
font_size = 14 # Adjust as appropriate.
cbar.ax.tick_params(labelsize=font_size)
plt.savefig("D:/Emotion/New2/ERSP/"+'A_Neutral-Cz.png',dpi=1800)
plt.show()

plt.grid()
plt.imshow(B_data, vmin=B_data.min(), vmax=B_data.max(), origin='lower',cmap='jet',aspect='auto',
           extent=[-70, 1020,4,30])
plt.xlabel('Time/ms',fontsize=17)
plt.ylabel('ERSP/dB',fontsize=17)
plt.tick_params(labelsize=14)
cbar=plt.colorbar()
font_size = 14 # Adjust as appropriate.
cbar.ax.tick_params(labelsize=font_size)
plt.savefig("D:/Emotion/New2/ERSP/"+'B_Neutral-Cz.png',dpi=1800)
plt.show()

plt.grid()
plt.imshow(H_data, vmin=H_data.min(), vmax=H_data.max(), origin='lower',cmap='jet',aspect='auto',
           extent=[-70, 1020,4,30])
plt.xlabel('Time/ms',fontsize=17)
plt.ylabel('ERSP/dB',fontsize=17)
plt.tick_params(labelsize=14)
cbar=plt.colorbar()
font_size = 14 # Adjust as appropriate.
cbar.ax.tick_params(labelsize=font_size)
plt.savefig("D:/Emotion/New2/ERSP/"+'H_Neutral-Cz.png',dpi=1800)
plt.show()

path1='D:/Emotion/New2/ERSP/'
response=path1+('A_Neutral_Fz')
response_data=sio.loadmat(response)
A_data=response_data['DATA']
path1='D:/Emotion/New2/ERSP/'
response=path1+('B_Neutral_Fz')
response_data=sio.loadmat(response)
B_data=response_data['DATA']
path1='D:/Emotion/New2/ERSP/'
response=path1+('B_Neutral_Fz')
response_data=sio.loadmat(response)
H_data=response_data['DATA']
plt.grid()
plt.imshow(A_data, vmin=A_data.min(), vmax=A_data.max(), origin='lower',cmap='jet',aspect='auto',
           extent=[-70, 1020,4,30])
plt.xlabel('Time/ms',fontsize=17)
plt.ylabel('ERSP/dB',fontsize=17)
plt.tick_params(labelsize=14)
cbar=plt.colorbar()
font_size = 14 # Adjust as appropriate.
cbar.ax.tick_params(labelsize=font_size)
plt.savefig("D:/Emotion/New2/ERSP/"+'A_Neutral-Fz.png',dpi=1800)
plt.show()

plt.grid()
plt.imshow(B_data, vmin=B_data.min(), vmax=B_data.max(), origin='lower',cmap='jet',aspect='auto',
           extent=[-70, 1020,4,30])
plt.xlabel('Time/ms',fontsize=17)
plt.ylabel('ERSP/dB',fontsize=17)
plt.tick_params(labelsize=14)
cbar=plt.colorbar()
font_size = 14 # Adjust as appropriate.
cbar.ax.tick_params(labelsize=font_size)
plt.savefig("D:/Emotion/New2/ERSP/"+'B_Neutral-Fz.png',dpi=1800)
plt.show()

plt.grid()
plt.imshow(H_data, vmin=H_data.min(), vmax=H_data.max(), origin='lower',cmap='jet',aspect='auto',
           extent=[-70, 1020,4,30])
plt.xlabel('Time/ms',fontsize=17)
plt.ylabel('ERSP/dB',fontsize=17)
plt.tick_params(labelsize=14)
cbar=plt.colorbar()
font_size = 14 # Adjust as appropriate.
cbar.ax.tick_params(labelsize=font_size)
plt.savefig("D:/Emotion/New2/ERSP/"+'H_Neutral-Fz.png',dpi=1800)
plt.show()

response=path1+('A_Positive_Pz')
response_data=sio.loadmat(response)
A_data=response_data['DATA']
path1='D:/Emotion/New2/ERSP/'
response=path1+('B_Positive_Pz')
response_data=sio.loadmat(response)
B_data=response_data['DATA']
path1='D:/Emotion/New2/ERSP/'
response=path1+('B_Positive_Pz')
response_data=sio.loadmat(response)
H_data=response_data['DATA']
plt.grid()
plt.imshow(A_data, vmin=A_data.min(), vmax=A_data.max(), origin='lower',cmap='jet',aspect='auto',
           extent=[-70, 1020,4,30])
plt.xlabel('Time/ms',fontsize=17)
plt.ylabel('ERSP/dB',fontsize=17)
plt.tick_params(labelsize=14)
cbar=plt.colorbar()
font_size = 14 # Adjust as appropriate.
cbar.ax.tick_params(labelsize=font_size)
plt.savefig("D:/Emotion/New2/ERSP/"+'A_Positive-Pz.png',dpi=1800)
plt.show()

plt.grid()
plt.imshow(B_data, vmin=B_data.min(), vmax=B_data.max(), origin='lower',cmap='jet',aspect='auto',
           extent=[-70, 1020,4,30])
plt.xlabel('Time/ms',fontsize=17)
plt.ylabel('ERSP/dB',fontsize=17)
plt.tick_params(labelsize=14)
cbar=plt.colorbar()
font_size = 14 # Adjust as appropriate.
cbar.ax.tick_params(labelsize=font_size)
plt.savefig("D:/Emotion/New2/ERSP/"+'B_Positive-Pz.png',dpi=1800)
plt.show()

plt.grid()
plt.imshow(H_data, vmin=H_data.min(), vmax=H_data.max(), origin='lower',cmap='jet',aspect='auto',
           extent=[-70, 1020,4,30])
plt.xlabel('Time/ms',fontsize=17)
plt.ylabel('ERSP/dB',fontsize=17)
plt.tick_params(labelsize=14)
cbar=plt.colorbar()
font_size = 14 # Adjust as appropriate.
cbar.ax.tick_params(labelsize=font_size)
plt.savefig("D:/Emotion/New2/ERSP/"+'H_Positive-Pz.png',dpi=1800)
plt.show()

path1='D:/Emotion/New2/ERSP/'
response=path1+('A_Positive_Cz')
response_data=sio.loadmat(response)
A_data=response_data['DATA']
path1='D:/Emotion/New2/ERSP/'
response=path1+('B_Positive_Cz')
response_data=sio.loadmat(response)
B_data=response_data['DATA']
path1='D:/Emotion/New2/ERSP/'
response=path1+('B_Positive_Cz')
response_data=sio.loadmat(response)
H_data=response_data['DATA']
plt.grid()
plt.imshow(A_data, vmin=A_data.min(), vmax=A_data.max(), origin='lower',cmap='jet',aspect='auto',
           extent=[-70, 1020,4,30])
plt.xlabel('Time/ms',fontsize=17)
plt.ylabel('ERSP/dB',fontsize=17)
plt.tick_params(labelsize=14)
cbar=plt.colorbar()
font_size = 14 # Adjust as appropriate.
cbar.ax.tick_params(labelsize=font_size)
plt.savefig("D:/Emotion/New2/ERSP/"+'A_Positive-Cz.png',dpi=1800)
plt.show()

plt.grid()
plt.imshow(B_data, vmin=B_data.min(), vmax=B_data.max(), origin='lower',cmap='jet',aspect='auto',
           extent=[-70, 1020,4,30])
plt.xlabel('Time/ms',fontsize=17)
plt.ylabel('ERSP/dB',fontsize=17)
plt.tick_params(labelsize=14)
cbar=plt.colorbar()
font_size = 14 # Adjust as appropriate.
cbar.ax.tick_params(labelsize=font_size)
plt.savefig("D:/Emotion/New2/ERSP/"+'B_Positive-Cz.png',dpi=1800)
plt.show()

plt.grid()
plt.imshow(H_data, vmin=H_data.min(), vmax=H_data.max(), origin='lower',cmap='jet',aspect='auto',
           extent=[-70, 1020,4,30])
plt.xlabel('Time/ms',fontsize=17)
plt.ylabel('ERSP/dB',fontsize=17)
plt.tick_params(labelsize=14)
cbar=plt.colorbar()
font_size = 14 # Adjust as appropriate.
cbar.ax.tick_params(labelsize=font_size)
plt.savefig("D:/Emotion/New2/ERSP/"+'H_Positive-Cz.png',dpi=1800)
plt.show()

path1='D:/Emotion/New2/ERSP/'
response=path1+('A_Positive_Fz')
response_data=sio.loadmat(response)
A_data=response_data['DATA']
path1='D:/Emotion/New2/ERSP/'
response=path1+('B_Positive_Fz')
response_data=sio.loadmat(response)
B_data=response_data['DATA']
path1='D:/Emotion/New2/ERSP/'
response=path1+('B_Positive_Fz')
response_data=sio.loadmat(response)
H_data=response_data['DATA']
plt.grid()
plt.imshow(A_data, vmin=A_data.min(), vmax=A_data.max(), origin='lower',cmap='jet',aspect='auto',
           extent=[-70, 1020,4,30])
plt.xlabel('Time/ms',fontsize=17)
plt.ylabel('ERSP/dB',fontsize=17)
plt.tick_params(labelsize=14)
cbar=plt.colorbar()
font_size = 14 # Adjust as appropriate.
cbar.ax.tick_params(labelsize=font_size)
plt.savefig("D:/Emotion/New2/ERSP/"+'A_Positive-Fz.png',dpi=1800)
plt.show()

plt.grid()
plt.imshow(B_data, vmin=B_data.min(), vmax=B_data.max(), origin='lower',cmap='jet',aspect='auto',
           extent=[-70, 1020,4,30])
plt.xlabel('Time/ms',fontsize=17)
plt.ylabel('ERSP/dB',fontsize=17)
plt.tick_params(labelsize=14)
cbar=plt.colorbar()
font_size = 14 # Adjust as appropriate.
cbar.ax.tick_params(labelsize=font_size)
plt.savefig("D:/Emotion/New2/ERSP/"+'B_Positive-Fz.png',dpi=1800)
plt.show()

plt.grid()
plt.imshow(H_data, vmin=H_data.min(), vmax=H_data.max(), origin='lower',cmap='jet',aspect='auto',
           extent=[-70, 1020,4,30])
plt.xlabel('Time/ms',fontsize=17)
plt.ylabel('ERSP/dB',fontsize=17)
plt.tick_params(labelsize=14)
cbar=plt.colorbar()
font_size = 14 # Adjust as appropriate.
cbar.ax.tick_params(labelsize=font_size)
plt.savefig("D:/Emotion/New2/ERSP/"+'H_Positive-Fz.png',dpi=1800)
plt.show()

