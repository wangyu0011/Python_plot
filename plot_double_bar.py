import matplotlib as mpl
import seaborn as sns
import pandas as pd
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

font2 = {'family' : 'Times New Roman','weight' : 'normal','size'   : 15,}
font1 = {'family' : 'Times New Roman','weight' : 'normal','size'   : 12,}

path1='D:/Emotion/New2/new_result/AFter/ERSP/ROI/Negative/'
path_hon=path1+'P200_theta.mat'
data_hon=sio.loadmat(path_hon)
A_Negative= data_hon['DATA']
path1='D:/Emotion/New2/new_result/AFter/ERSP/ROI/Positive/'
path_hon=path1+'P200_theta.mat'
data_hon=sio.loadmat(path_hon)
A_Positive= data_hon['DATA']
path1='D:/Emotion/New2/new_result/AFter/ERSP/ROI/Neutral/'
path_hon=path1+'P200_theta.mat'
data_hon=sio.loadmat(path_hon)
A_Neutral= data_hon['DATA']

path1='D:/Emotion/New2/new_result/Before/ERSP/ROI/Negative/'
path_hon=path1+'P200_theta.mat'
data_hon=sio.loadmat(path_hon)
B_Negative= data_hon['DATA']
path1='D:/Emotion/New2/new_result/Before/ERSP/ROI/Positive/'
path_hon=path1+'P200_theta.mat'
data_hon=sio.loadmat(path_hon)
B_Positive= data_hon['DATA']
path1='D:/Emotion/New2/new_result/Before/ERSP/ROI/Neutral/'
path_hon=path1+'P200_theta.mat'
data_hon=sio.loadmat(path_hon)
B_Neutral= data_hon['DATA']

path1='D:/Emotion/New2/new_result/HC/ERSP/ROI/Negative/'
path_hon=path1+'P200_theta.mat'
data_hon=sio.loadmat(path_hon)
H_Negative= data_hon['DATA']
path1='D:/Emotion/New2/new_result/HC/ERSP/ROI/Positive/'
path_hon=path1+'P200_theta.mat'
data_hon=sio.loadmat(path_hon)
H_Positive= data_hon['DATA']
path1='D:/Emotion/New2/new_result/HC/ERSP/ROI/Neutral/'
path_hon=path1+'P200_theta.mat'
data_hon=sio.loadmat(path_hon)
H_Neutral= data_hon['DATA']
channels=['Fp1','Fpz','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4','T8','CP5','CP1','CP2','CP6',
          'P7','P3','Pz','P4','P8','POz','O1','O2','AF7','AF3','AF4','AF8','F5','F1','F2','F6','FC3','FCz','FC4','C5','C1','C2','C6',
          'CP3','CP4','P5','P1','P2','P6','PO5','PO3','PO4','PO6','FT7','FT8','TP7','TP8','PO7','PO8','Oz']
# width of the bars
barWidth = 0.2
for i in np.arange(61):
    after_mean=(np.mean(A_Negative[i,:]),np.mean(A_Positive[i,:]),np.mean(A_Neutral[i,:]))
    after_std=(np.std(A_Negative[i,:])/11.4018,np.std(A_Positive[i,:])/11.4018,np.std(A_Neutral[i,:])/11.4018)
    
    before_mean=(np.mean(B_Negative[i,:]),np.mean(B_Positive[i,:]),np.mean(B_Neutral[i,:]))
    before_std=(np.std(B_Negative[i,:])/11.4018,np.std(B_Positive[i,:])/11.4018,np.std(B_Neutral[i,:])/11.4018)
    
    hc_mean=(np.mean(H_Negative[i,:]),np.mean(H_Positive[i,:]),np.mean(H_Neutral[i,:]))
    hc_std=(np.std(H_Negative[i,:])/11.4018,np.std(H_Positive[i,:])/11.4018,np.std(H_Neutral[i,:])/11.4018)
    
    data_mean=[after_mean,before_mean,hc_mean]
    data_std=[after_std,before_std,hc_std]
    r1 = np.arange(len(hc_mean))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth+ barWidth for x in r1]
    plt.style.use('ggplot') 
    # Create before bars
    plt.bar(r1, before_mean, width = barWidth, color = 'red', edgecolor = 'black', yerr=before_std, capsize=7, label='Pre')
    # Create after bars
    plt.bar(r2, after_mean, width = barWidth, color = 'blue', edgecolor = 'black', yerr=after_std, capsize=7, label='Pos')
    # Create hc bars
    plt.bar(r3, hc_mean, width = barWidth, color = 'green', edgecolor = 'black', yerr=hc_std, capsize=7, label='HC')
    plt.ylim(np.min(data_mean)-np.max(data_std),np.max(data_mean)+np.max(data_std)+np.max(data_std)+np.max(data_std)+np.max(data_std)+np.max(data_std)+np.max(data_std))##
    # general layout
    plt.xticks([r + barWidth for r in range(len(hc_mean))], ['Negative', 'Positive', 'Neutral'])
    plt.tick_params(labelsize=14)
    plt.ylabel('ERSP(db)',fontsize=14)
#    plt.legend(fontsize=14)
    plt.legend(loc='upper center',fontsize=14,ncol=3)
    plt.savefig("C:/Users/Administrator/Desktop/P200_result/new_bar/"+'bar-'+str(channels[i])+'.png',dpi=600)
    plt.show()





























#path_hon=path+'target1_switch_P200_tm.mat'
#data_hon=sio.loadmat(path_hon)
#T1_switch_P200_tm= data_hon['P200_tm']
#
#path_hon=path+'target2_switch_P200_tm.mat'
#data_hon=sio.loadmat(path_hon)
#T2_switch_P200_tm= data_hon['P200_tm']
#
#path_hon=path+'target3_switch_P200_tm.mat'
#data_hon=sio.loadmat(path_hon)
#T3_switch_P200_tm= data_hon['P200_tm']
#
#path_hon=path+'target4_switch_P200_tm.mat'
#data_hon=sio.loadmat(path_hon)
#T4_switch_P200_tm= data_hon['P200_tm']
#
#path_hon=path+'target5_switch_P200_tm.mat'
#data_hon=sio.loadmat(path_hon)
#T5_switch_P200_tm= data_hon['P200_tm']
#
#path_hon=path+'target6_switch_P200_tm.mat'
#data_hon=sio.loadmat(path_hon)
#T6_switch_P200_tm= data_hon['P200_tm']
#
#path_hon=path+'target7_switch_P200_tm.mat'
#data_hon=sio.loadmat(path_hon)
#T7_switch_P200_tm= data_hon['P200_tm']
#
#path_hon=path+'target8_switch_P200_tm.mat'
#data_hon=sio.loadmat(path_hon)
#T8_switch_P200_tm= data_hon['P200_tm']
#
#path_hon=path+'target1_repeat_P200_tm.mat'
#data_hon=sio.loadmat(path_hon)
#T1_repeat_P200_tm= data_hon['P200_tm']
#
#path_hon=path+'target2_repeat_P200_tm.mat'
#data_hon=sio.loadmat(path_hon)
#T2_repeat_P200_tm= data_hon['P200_tm']
#
#path_hon=path+'target3_repeat_P200_tm.mat'
#data_hon=sio.loadmat(path_hon)
#T3_repeat_P200_tm= data_hon['P200_tm']
#
#path_hon=path+'target4_repeat_P200_tm.mat'
#data_hon=sio.loadmat(path_hon)
#T4_repeat_P200_tm= data_hon['P200_tm']
#
#path_hon=path+'target5_repeat_P200_tm.mat'
#data_hon=sio.loadmat(path_hon)
#T5_repeat_P200_tm= data_hon['P200_tm']
#
#path_hon=path+'target6_repeat_P200_tm.mat'
#data_hon=sio.loadmat(path_hon)
#T6_repeat_P200_tm= data_hon['P200_tm']
#
#path_hon=path+'target7_repeat_P200_tm.mat'
#data_hon=sio.loadmat(path_hon)
#T7_repeat_P200_tm= data_hon['P200_tm']
#
#path_hon=path+'target8_repeat_P200_tm.mat'
#data_hon=sio.loadmat(path_hon)
#T8_repeat_P200_tm= data_hon['P200_tm']
#
#means_repeat = (np.mean(T1_repeat_P200_tm[1,:]), np.mean(T2_repeat_P200_tm[1,:]),np.mean(T3_repeat_P200_tm[1,:]),
#                np.mean(T4_repeat_P200_tm[1,:]),np.mean(T5_repeat_P200_tm[1,:]),np.mean(T6_repeat_P200_tm[1,:]),
#                np.mean(T7_repeat_P200_tm[1,:]),np.mean(T8_repeat_P200_tm[1,:]),)
#std_repeat = (np.std(T1_repeat_P200_tm[1,:])/10,np.std(T1_repeat_P200_tm[1,:])/10,np.std(T1_repeat_P200_tm[1,:])/10,
#           np.std(T1_repeat_P200_tm[1,:])/10,np.std(T1_repeat_P200_tm[1,:])/10,np.std(T1_repeat_P200_tm[1,:])/10,
#           np.std(T1_repeat_P200_tm[1,:])/10,np.std(T1_repeat_P200_tm[1,:])/10)
#
#means_switch = (np.mean(T1_switch_P200_tm[1,:]), np.mean(T2_switch_P200_tm[1,:]),np.mean(T3_switch_P200_tm[1,:]),
#                np.mean(T4_switch_P200_tm[1,:]),np.mean(T5_switch_P200_tm[1,:]),np.mean(T6_switch_P200_tm[1,:]),
#                np.mean(T7_switch_P200_tm[1,:]),np.mean(T8_switch_P200_tm[1,:]),)
#std_switch = (np.std(T1_switch_P200_tm[1,:])/10,np.std(T1_switch_P200_tm[1,:])/10,np.std(T1_switch_P200_tm[1,:])/10,
#           np.std(T1_switch_P200_tm[1,:])/10,np.std(T1_switch_P200_tm[1,:])/10,np.std(T1_switch_P200_tm[1,:])/10,
#           np.std(T1_switch_P200_tm[1,:])/10,np.std(T1_switch_P200_tm[1,:])/10)
#means_repeat = (np.mean(T1_repeat_P200_tm[1,:]), np.mean(T2_repeat_P200_tm[1,:]),np.mean(T3_repeat_P200_tm[1,:]),
#                np.mean(T4_repeat_P200_tm[1,:]),np.mean(T5_repeat_P200_tm[1,:]),np.mean(T6_repeat_P200_tm[1,:]),
#                np.mean(T7_repeat_P200_tm[1,:]),np.mean(T8_repeat_P200_tm[1,:]),)
#std_repeat = (np.std(T1_repeat_P200_tm[1,:])/10,np.std(T1_repeat_P200_tm[1,:])/10,np.std(T1_repeat_P200_tm[1,:])/10,
#           np.std(T1_repeat_P200_tm[1,:])/10,np.std(T1_repeat_P200_tm[1,:])/10,np.std(T1_repeat_P200_tm[1,:])/10,
#           np.std(T1_repeat_P200_tm[1,:])/10,np.std(T1_repeat_P200_tm[1,:])/10)
#
#means_switch = (np.mean(T1_switch_P200_tm[1,:]), np.mean(T2_switch_P200_tm[1,:]),np.mean(T3_switch_P200_tm[1,:]),
#                np.mean(T4_switch_P200_tm[1,:]),np.mean(T5_switch_P200_tm[1,:]),np.mean(T6_switch_P200_tm[1,:]),
#                np.mean(T7_switch_P200_tm[1,:]),np.mean(T8_switch_P200_tm[1,:]),)
#std_switch = (np.std(T1_switch_P200_tm[1,:])/10,np.std(T1_switch_P200_tm[1,:])/10,np.std(T1_switch_P200_tm[1,:])/10,
#           np.std(T1_switch_P200_tm[1,:])/10,np.std(T1_switch_P200_tm[1,:])/10,np.std(T1_switch_P200_tm[1,:])/10,
#           np.std(T1_switch_P200_tm[1,:])/10,np.std(T1_switch_P200_tm[1,:])/10)



