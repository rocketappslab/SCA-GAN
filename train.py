import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import numpy as np
import os
import pickle
from collections import OrderedDict
import torchvision
import datetime

opt = TrainOptions().parse()

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)
model = create_model(opt)
visualizer = Visualizer(opt)

info_dir = os.path.join(opt.checkpoints_dir, opt.name)
infoname = '%s.pkl' % (opt.which_epoch)
infoname = os.path.join(info_dir, infoname)
if opt.continue_train and os.path.exists(infoname):
    print('Loaded epoch and total_steps')
    file = open(infoname, 'rb')
    info = pickle.load(file)
    file.close()
    epoch_count = info['epoch']
    total_steps = info['total_steps']
else:
    epoch_count = opt.epoch_count
    total_steps = 0

print("Start epoch: ", epoch_count)

for steps in range(epoch_count - 1):
    for scheduler in model.schedulers:
        scheduler.step()

stat_errors = OrderedDict([('count', 0)])
# Count start time
prev_time = time.time()
total_epoch = opt.niter + opt.niter_decay + 1

for epoch in range(epoch_count, total_epoch):
    epoch_start_time = time.time()
    epoch_iter = 0

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        iters_done = (epoch - 1) * (dataset_size / opt.batchSize) + i + 1
        visualizer.reset()
        total_steps += 1
        epoch_iter += 1
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        # Determine approximate time left
        iters_left = (total_epoch - 1) * (dataset_size / opt.batchSize) - iters_done
        time_left = datetime.timedelta(seconds=iters_left * (time.time() - prev_time))
        prev_time = time.time()
        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            visualizer.print_current_errors(epoch, epoch_iter, errors, time_left)

    model.save('latest', epoch + 1, total_steps)
    # save epoch model
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save(epoch, epoch + 1, total_steps)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()
