import os
import ntpath
import time
from utilSet import util
from utilSet import html

def save_opt(opt):
    args = vars(opt)
    expr_dir = os.path.join(opt.result_root_dir, opt.variable, opt.variable_value)
    util.mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(args.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')

class Visualizer:
    def __init__(self, opt):
        self.display_id = 1
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = 256
        self.opt = opt
        self.saved = False

        self.root_dir = os.path.join(opt.result_root_dir, opt.variable)

        if self.use_html:
            self.web_dir = os.path.join(self.root_dir, opt.variable_value, opt.phase)
            self.img_dir = os.path.join(self.web_dir, 'images')
            util.mkdirs([self.web_dir, self.img_dir])

        self.log_name = os.path.join(self.root_dir, opt.variable_value, opt.phase, 'loss_log.txt')


    def reset(self):
        self.saved = False

    def display_current_results(self, visuals, epoch, save_result):
        if self.use_html and (save_result or not self.saved):
            self.saved = True
            for label, image_numpy in visuals.items():
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)
            webpage = html.HTML(self.web_dir, 'Experiment name', re_flesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %-2d, iters: %-4d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)
        message += '\n'
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
