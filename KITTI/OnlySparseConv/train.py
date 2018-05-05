# -*- coding:utf-8 -*-
import time
import ntpath
from config import TrainOptions
from evaluate.train_error_figure import draw_error_figure
from evaluate.draw_loss_figure import draw_loss_figure
from models.model import Model
from utilSet.visualizer import Visualizer, save_opt
from data.dataset import DataLoader

opt = TrainOptions().parse()
save_opt(opt)

data_loader = DataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)

model = Model()
model.initialize(opt)
visualizer = Visualizer(opt)

if __name__ == '__main__':

    total_steps = 0

    for epoch in range(1, opt.epoch + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):

            visualizer.reset()
            total_steps += 1
            epoch_iter += 1

            model.set_input(data)
            model.optimize_parameters()

            img_path = model.get_image_paths()
            short_path = ntpath.basename(''.join(img_path))

            '''
            # 决定每轮输出显示哪一张图片
            '''
            if epoch_iter == opt.display_num:
                visualizer.display_current_results(model.get_current_visuals(), epoch, short_path, True)

            '''
            # 计算loss值
            '''
            errors = model.get_current_errors()
            visualizer.print_current_errors(epoch, epoch_iter, short_path, errors)

            depth_errors = model.get_depth_errors()
            visualizer.print_depth_errors(epoch, epoch_iter, short_path, depth_errors)

        if epoch % 5 == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.epoch, time.time() - epoch_start_time))
            model.save('latest')
            model.save(epoch)

        model.update_learning_rate()  # 更新学习率

        '''
        # 画loss图
        '''
        loss_item = ['fake_b_loss', 'Sparse_C', 'loss_G']
        print("------Draw loss figure !------")
        for i in range(len(loss_item)):
            draw_loss_figure(loss_item[i], opt)

        '''
        # 画误差图
        '''
        error_item = ['RMSE', 'MAE']
        print("------Draw depth evaluate figure !------")
        for i in range(len(error_item)):
            draw_error_figure(error_item[i], opt)
