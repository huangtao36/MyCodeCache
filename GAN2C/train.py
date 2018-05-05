# -*- coding:utf-8 -*-
import time
from config import TrainOptions
from models.cycle_gan_model import CycleGANModel
from utilSet.visualizer import Visualizer, save_opt
from data.dataset import DataLoader

opt = TrainOptions().parse()
save_opt(opt)
data_loader = DataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)

model = CycleGANModel()
model.initialize(opt)
visualizer = Visualizer(opt)

if __name__ == '__main__':
    total_steps = 0
    for epoch in range(1, opt.epoch + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            visualizer.reset()
            total_steps += 1
            epoch_iter += 1
            model.set_input(data)
            model.optimize_parameters()

            if epoch_iter == opt.display_num:
                visualizer.display_current_results(model.get_current_visuals(), epoch, True)

            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)

        if epoch % 5 == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.epoch, time.time() - epoch_start_time))
            model.save('latest')
            model.save(epoch)

        model.update_learning_rate()
