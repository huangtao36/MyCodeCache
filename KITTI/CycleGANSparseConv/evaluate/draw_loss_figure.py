# -*- coding:utf-8 -*-
from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.pyplot as plt
import numpy as np
import os
import re

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
            arr = re.findall(r'epoch: \b\d+\b', ln)
            epoch_data = int(arr[0].strip(' ')[7:])
            data.append(epoch_data)

            # iters
            arr1 = re.findall(r'iters: \b\d+\b', ln)
            iters_data = int(arr1[0].strip(' ')[7:])
            data.append(iters_data)

            #D_A
            arr2 = re.findall(r'D_A: \b\S+\b', ln)
            D_A_data = float(arr2[0].strip(' ')[5:])
            data.append(D_A_data)

            #G_A
            arr3 = re.findall(r'G_A: \b\S+\b', ln)
            G_A_data = float(arr3[0].strip(' ')[5:])
            data.append(G_A_data)

            #Cyc_A
            arr4 = re.findall(r'c_A: \b\S+\b', ln)
            Cyc_A_data = float(arr4[0].strip(' ')[5:])
            data.append(Cyc_A_data)

            #D_B
            arr5 = re.findall(r'D_B: \b\S+\b', ln)
            D_B_data = float(arr5[0].strip(' ')[5:])
            data.append(D_B_data)

            #G_B
            arr6 = re.findall(r'G_B: \b\S+\b', ln)
            G_A_data = float(arr6[0].strip(' ')[5:])
            data.append(G_A_data)

            #Cyc_B
            arr7 = re.findall(r'c_B: \b\S+\b', ln)
            Cyc_B_data = float(arr7[0].strip(' ')[5:])
            data.append(Cyc_B_data)

            # Fake_A_all_loss
            arr10 = re.findall(r'A_all_loss: \b\S+\b', ln)
            Fake_A_data = float(arr10[0].strip(' ')[12:])
            data.append(Fake_A_data)

            # Fake_B_all_loss
            arr11 = re.findall(r'B_all_loss: \b\S+\b', ln)
            Fake_B_data = float(arr11[0].strip(' ')[12:])
            data.append(Fake_B_data)

            # Sparse_C
            arr12 = re.findall(r'Sparse_C: \b\S+\b', ln)
            Sparse_C = float(arr12[0].strip(' ')[10:])
            data.append(Sparse_C)

            # loss_G
            arr13 = re.findall(r'loss_G: \b\S+\b', ln)
            loss_G = float(arr13[0].strip(' ')[8:])
            data.append(loss_G)

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
    # print(epoch)
    # print(iters)
    # 'D_A', 'G_A', 'Cyc_A', 'D_B', 'G_B', 'Cyc_B', 'idt_A', 'idt_B', 
    # 'Fake_A_all_loss', 'Fake_B_all_loss','Sparse_C', 'loss_G'
                
    if which == 'D_A':
        for var in data:
            Sum += var[2]  # D_A在列表中的第3位
            count += 1
            if count % iters == 0:
                avg_list.append(Sum / iters)
                Sum = 0
    elif which == 'G_A':
        for var in data:
            Sum += var[3]   # G_A在列表中的第4位
            count += 1
            if count % iters == 0:
                avg_list.append(Sum / iters)
                Sum = 0
    elif which == 'Cyc_A':
        for var in data:
            Sum += var[4]*85  # Cyc_A在列表中的第5位
            count += 1
            if count % iters == 0:
                avg_list.append(Sum / iters)
                Sum = 0
    elif which == 'D_B':
        for var in data:
            Sum += var[5]  # D_B在列表中的第6位
            count += 1
            if count % iters == 0:
                avg_list.append(Sum / iters)
                Sum = 0
    elif which == 'G_B':
        for var in data:
            Sum += var[6]  # G_B在列表中的第7位
            count += 1
            if count % iters == 0:
                avg_list.append(Sum / iters)
                Sum = 0
    elif which == 'Cyc_B':
        for var in data:
            Sum += var[7]*85  # Cyc_B在列表中的第8位
            count += 1
            if count % iters == 0:
                avg_list.append(Sum / iters)
                Sum = 0
    # 'Fake_A_all_loss', 'Fake_B_all_loss','Sparse_C', 'loss_G'
    elif which == 'Fake_A_all_loss':
        for var in data:
            Sum += var[8]*85  # idt_B在列表中的第10位
            count += 1
            if count % iters == 0:
                avg_list.append(Sum / iters)
                Sum = 0
    elif which == 'Fake_B_all_loss':
        for var in data:
            Sum += var[9]*85  # idt_B在列表中的第10位
            count += 1
            if count % iters == 0:
                avg_list.append(Sum / iters)
                Sum = 0
    elif which == 'Sparse_C':
        for var in data:
            Sum += var[10]*85  # idt_B在列表中的第10位
            count += 1
            if count % iters == 0:
                avg_list.append(Sum / iters)
                Sum = 0
    elif which == 'loss_G':
        for var in data:
            Sum += var[11]  # idt_B在列表中的第10位
            count += 1
            if count % iters == 0:
                avg_list.append(Sum / iters)
                Sum = 0

    return avg_list, epoch


'''
    画图函数
    输入：选择要画的是哪个数据、保存的name(str)、输入图片数量
    输出：保存为id.png
'''


def draw_loss_figure(which, opt):

    folder = os.path.join(opt.result_root_dir, opt.variable, opt.variable_value, opt.phase)
    file = os.path.join(folder, 'loss_log.txt')

    datalist = get_loss_data(file)  # 获得原始数据列表
    avg, epoch = get_all_avg_loss(datalist, which)  # 获得计算后的平均值

    host = host_subplot(1, 1, 1)  # 分几个图，选择第几个图
    plt.subplots_adjust(right=0.8)   # 限定右边界

    host.set_xlabel("epoch")
    host.set_ylabel("%s loss" % which)

    p1, = host.plot(avg, label="train cycleGANs %s loss" % which)

    host.legend(loc=1)  # 标签距离边界位置

    host.axis["left"].label.set_color(p1.get_color())

    host.set_xlim([-1, epoch])    # x轴范围
    host.set_ylim([0., 2 * np.average(avg)])  # y轴范围

    plt.draw()
    expr_dir = os.path.join(folder, 'loss_figure')
    # print(expr_dir)
    mkdirs(expr_dir)

    filename = which  # + '_' + str(epoch)

    plt.savefig('%s/%s.png' % (expr_dir, filename))
    plt.close('all')
    # plt.show()
