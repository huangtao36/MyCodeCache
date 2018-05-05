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
            
            # Fake_B_all_loss
            arr2 = re.findall(r'Sparse_C: \b\S+\b', ln)
            sparse_c = float(arr2[0].strip(' ')[10:])
            data.append(sparse_c)

            # Sparse_C
            arr3 = re.findall(r'fake_b_loss: \b\S+\b', ln)
            fake_b_loss = float(arr3[0].strip(' ')[13:])
            data.append(fake_b_loss)

            # loss_G
            arr4 = re.findall(r'loss_G: \b\S+\b', ln)
            loss_g = float(arr4[0].strip(' ')[8:])
            data.append(loss_g)

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
    sum_data = 0
    avg_list = []
    count = 0
    epoch = data[-1][0]
    iters = data[-1][1]
                
    if which == 'Sparse_C':
        for var in data:
            sum_data += var[2]*85
            count += 1
            if count % iters == 0:
                avg_list.append(sum_data / iters)
                sum_data = 0

    elif which == 'fake_b_loss':
        for var in data:
            sum_data += var[3]*85
            count += 1
            if count % iters == 0:
                avg_list.append(sum_data / iters)
                sum_data = 0

    elif which == 'loss_G':
        for var in data:
            sum_data += var[4]
            count += 1
            if count % iters == 0:
                avg_list.append(sum_data / iters)
                sum_data = 0

    return avg_list, epoch


'''
    画图函数
    输入：选择要画的是哪个数据、保存的name(str)、输入图片数量
    输出：保存为id.png
'''


def draw_loss_figure(which, opt):

    folder = os.path.join(opt.result_root_dir, opt.variable, opt.variable_value, opt.phase)
    file = os.path.join(folder, 'loss_log.txt')

    data_list = get_loss_data(file)                     # 获得原始数据列表
    avg, epoch = get_all_avg_loss(data_list, which)     # 获得计算后的平均值

    host = host_subplot(1, 1, 1)        # 分几个图，选择第几个图
    plt.subplots_adjust(right=0.8)      # 限定右边界

    host.set_xlabel("epoch")
    host.set_ylabel("%s loss" % which)

    p1, = host.plot(avg, label="train cycleGANs %s loss" % which)

    host.legend(loc=1)                  # 标签距离边界位置

    host.axis["left"].label.set_color(p1.get_color())

    host.set_xlim([-1, epoch])                  # x轴范围
    host.set_ylim([0., 2 * np.average(avg)])    # y轴范围

    plt.draw()
    expr_dir = os.path.join(folder, 'loss_figure')
    # print(expr_dir)
    mkdirs(expr_dir)

    filename = which  # + '_' + str(epoch)

    plt.savefig('%s/%s.png' % (expr_dir, filename))
    plt.close('all')
    # plt.show()
