#encoding:utf-8
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sio
import seaborn
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"/home/wangyu/Documents/make figure/Plot_bar/data/msyhbd.ttf", size=6)
path='/home/wangyu/Documents/make figure/Plot_bar/linechart_bar_fourteen_electrodes/data/'


#绘制图中的间隔线
def add_line_figure():
    #间隔线的颜色
    color_all = ['white', 'black', ]
    color_number = 0
    x_y_lineposition_start = 0.5
    rows_cols = 24
    for i in range(rows_cols):
        ax.axvline(x=x_y_lineposition_start, ymin=0, ymax=1, linewidth=0.2, color=color_all[color_number])
        ax.axhline(y=x_y_lineposition_start, xmin=0, xmax=1, linewidth=0.2, color=color_all[color_number])
        x_y_lineposition_start = x_y_lineposition_start + 1

#set x and y ticks name
def add_ticks_xy(row):
    # add xticks and yticks
    elec_positon = range(row)
    # elec_name = ['6L', '6R', '7L', '7R', '8L', '8R', '9L', '9R', '10L', '10R', '13L', '13R', '24L', '24R', '32L', '32R',
    #              '40L', '40R', '44L', '44R', '46L', '46R', '47L', '47R']
    elec_name = ['O2', 'O1', 'OZ', 'PZ', 'P4', 'CP4', 'P8', 'C4', 'TP8', 'T8',
    'P7' ,'P3', 'CP3' ,'CPZ','CZ' ,'FC4' ,'FT8' ,'TP7' ,'C3', 'FCZ' ,
    'FZ', 'F4', 'F8', 'T7', 'FT7', 'FC3', 'F3', 'FP2', 'F7', 'FP1']

    plt.xticks(elec_positon, elec_name, rotation=60, fontsize=10)
    plt.yticks(elec_positon, elec_name, fontsize=10)
    # set x y ticks position
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('top')

# 归一化数据，行归一化，归一化到（0,1）
def DATA_Normalized(data):
    row1 = np.size(data, axis=0)
    data_normalized = np.zeros((1, row1))
    for row_s in range(row1):
        data_row = data[row_s, :]
        data_max_min = np.max(data_row) - np.min(data_row)
        data_row = (data_row - np.min(data_row)) / data_max_min
        data_normalized = np.vstack((data_normalized, data_row))
    data_normalized = data_normalized[1:, :]
    return data_normalized

# mathshow data and add colorbar
def matshow_data(data_mat,j):
    vmin=np.min(data_mat)
    #the color value min
    vmin=vmin-0.1
    #the color value max
    vmax=np.max(data_mat)
    vmax=vmax+0.5
    #matshow
    # cax = ax.matshow(data_mat,cmap='seismic',vmin=vmin,vmax=vmax)#
    cax = ax.matshow(data_mat,cmap = matplotlib.cm.jet,interpolation='nearest')#cmap = matplotlib.cm.jet,interpolation='nearest'
    #将matshow生成热力图设置为颜色渐变条
    if j in [1,2]:
        fig.colorbar(cax)

# main function
if __name__ == '__main__':
    path_raw='/home/wangyu/Documents/wangyu-mne/data/'

    for i in range(1,4):
        # set figure size and subplots_adjust
        fig = plt.figure(figsize=(8.7, 6.15))
        fig.subplots_adjust(left=0.0, bottom=0.01, right=1.0, top=0.90)  # , wspace=0.155, hspace=0.43
        ax = fig.add_subplot(111)
        # DATA
        if i == 1:
            path_2 = path_raw+'MI_norm_avg_diff'
            data = sio.loadmat(path_2)
            data = data['MI_norm_avg_diff']
            [row,col]=np.shape(data)
            # plt.ylabel('MI_norm_avg_diff')

        if i == 2:
            path_2 = path_raw+'MI_norm_avg_hon'
            data = sio.loadmat(path_2)
            data = data['MI_norm_avg']
            [row, col] = np.shape(data)
            # plt.ylabel('MI_norm_avg_hon')

        if i == 3:
            path_2 = path_raw + 'MI_norm_avg_lie'
            data = sio.loadmat(path_2)
            data = data['MI_norm_avg']
            [row, col] = np.shape(data)
            # plt.ylabel('MI_norm_avg_lie')

        # 归一化数据
        # data_normalized =DATA_Normalized(data)

        # 调用matshow函数,绘制矩阵色块图，添加colorbar
        matshow_data(data,i)

        #调用函数，绘制图中的间隔线
        add_line_figure()
        # plt.grid(True)

        #添加 xticks yticks
        add_ticks_xy(row)

        # 循环保存图片
        if i == 1:
            plt.savefig('.\图\matrix_MI_norm_avg_diff.png', dpi=1200)
        if i == 2:
            plt.savefig('.\图\matrix_MI_norm_avg_hon.png', dpi=1200)
        if i == 3:
            plt.savefig('.\图\matrix_MI_norm_avg_lie.png', dpi=1200)
    plt.show()