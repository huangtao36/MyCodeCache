# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import re
from mpl_toolkits.axes_grid1 import host_subplot
import numpy as np
import os

'''
问题：在本地用matplotlib绘图可以，但是在ssh远程绘图的时候会报错　RuntimeError: Invalid DISPLAY variable
原因：matplotlib的默认backend是TkAgg，而FltkAgg, GTK, GTKAgg, GTKCairo, TkAgg , Wx or WxAgg这几个backend都要求有GUI图形界面的，所以在ssh操作的时候会报错．
'''
plt.switch_backend('agg')

'''
只需修改folder,即是loss文件所在文件夹，
之后程序会在此文件夹下自动创建一个loss_figure,
用于存储loss图像
'''



def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
        
'''
获取loss数据存储到集合中
输入：文件夹路径
输出：数据列表，（嵌套列表）
'''

def get_loss_data(file_dir):
    data = []
    data_list = []
    fp = open(file_dir, 'r')
    for ln in fp:
        if 'epoch: ' in ln:
            # eopch
            epoch_data = re.findall(r'epoch: \b\d+\b', ln)
            epoch = int(epoch_data[0].strip(' ')[7:])
            data.append(epoch)
            # iters
            iters_data = re.findall(r'iters: \b\d+\b', ln)
            iters = int(iters_data[0].strip(' ')[7:])
            data.append(iters)
            # iRMSE
            iRMSE_data = re.findall(r'iRMSE: \b\S+\b', ln)
            iRMSE = float(iRMSE_data[0].strip(' ')[7:])
            data.append(iRMSE)
            # iMAE
            iMAE_data = re.findall(r'iMAE: \b\S+\b', ln)
            iMAE = float(iMAE_data[0].strip(' ')[6:])
            data.append(iMAE)
            # RMSE
            RMSE_data = re.findall(r'RMSE_m: \b\S+\b', ln)
            RMSE = float(RMSE_data[0].strip(' ')[8:])
            data.append(RMSE)
            # MAE
            MAE_data = re.findall(r'MAE_m: \b\S+\b', ln)
            MAE = float(MAE_data[0].strip(' ')[7:])
            data.append(MAE)

            data_list.append(data)
        data = []
    fp.close()
    return data_list

'''
获取每轮loss平均值
输入:数据列表，loss选择，训练的数据量
输出:loss的avg列表
'''
def get_all_avg_loss(data, which):
    Sum = 0
    avg_list = []
    count = 0
    epoch = data[-1][0]
    iters = data[-1][1]

    if which == 'iRMSE':
        for var in data:
            Sum += var[2]   # iRMSE在列表中的第3位
            count += 1
            if count % iters == 0:
                avg_list.append(Sum / iters)
                Sum = 0
    elif which == 'iMAE':
        for var in data:
            Sum += var[3]   # iMAE在列表中的第4位
            count += 1
            if count % iters == 0:
                avg_list.append(Sum / iters)
                Sum = 0
    elif which == 'RMSE':
        for var in data:
            Sum += var[4]  # RMSE在列表中的第5位
            count += 1
            if count % iters == 0:
                avg_list.append(Sum / iters)
                Sum = 0
    elif which == 'MAE':
        for var in data:
            Sum += var[5]  # MAE在列表中的第6位
            count += 1
            if count % iters == 0:
                avg_list.append(Sum / iters)
                Sum = 0
    # print(avg_list)

    return avg_list, epoch


'''
画图函数
输入：选择要画的是哪个数据、保存的name(str)、输入图片数量
输出：保存为id.png
'''
def draw_error_figure(which, opt):

    folder = os.path.join(opt.result_root_dir,  opt.variable, opt.variable_value, opt.phase)
    file = os.path.join(folder, 'train_errors_count.txt')

    datalist = get_loss_data(file)  # 获得原始数据列表
    avg, epoch = get_all_avg_loss(datalist, which)  # 获得计算后的平均值
    # print(avg)

    host = host_subplot(1, 1, 1)  # 分几个图，选择第几个图
    plt.subplots_adjust(right=0.8)   # 限定右边界

    host.set_xlabel("epoch")
    host.set_ylabel("%s error" % which)

    p1, = host.plot(avg, label="train: %s error" % which)

    host.legend(loc=1)  # 标签距离边界位置

    host.axis["left"].label.set_color(p1.get_color())

    host.set_xlim([-1, epoch])    # x轴范围
    host.set_ylim([0., 2 * np.average(avg)])  # y轴范围

    plt.draw()

    expr_dir = os.path.join(folder, 'error_figure')
    mkdirs(expr_dir)
    filename = which
    plt.savefig('./%s/%s.png' % (expr_dir, filename))

    plt.close('all')


