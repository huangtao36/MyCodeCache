# -*- coding:utf-8 -*-

import os
from utilSet import html
from models.model import CycleGANModel
from utilSet.visualizer import Visualizer
from config import TestOptions
from data.dataset import DataLoader
import ntpath

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = DataLoader(opt)
dataset = data_loader.load_data()
model = CycleGANModel()
model.initialize(opt)
visualizer = Visualizer(opt)

if __name__ == '__main__':
    root_dir = os.path.join(opt.result_root_dir, opt.variable)
    web_dir = os.path.join(root_dir, opt.variable_value, opt.phase)
    webpage = html.HTML(web_dir, 'Experiment = GAN2C, Phase = test, Epoch = latest')
    # test
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()

        img_path = model.get_image_paths()
        print('process image... %s' % img_path)
        visualizer.save_images(webpage, visuals, img_path)

        short_path = ntpath.basename(''.join(img_path))
        test_depth_errors = model.get_depth_errors()
        visualizer.test_depth_errors(i + 1, short_path, test_depth_errors)

    webpage.save()
