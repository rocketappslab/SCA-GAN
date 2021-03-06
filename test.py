import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer

from util import html
import time
import datetime
opt = TestOptions().parse()

opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))

webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

print(opt.how_many)
print(len(dataset))

model = model.eval()
print(model.training)

# Count start time
prev_time = time.time()
opt.how_many = 999999
# test
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()

    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    img_path = [img_path]
    print(img_path)
    visualizer.save_images(webpage, visuals, img_path)

    # Determine approximate time left
    iters_left = (len(dataset) / opt.batchSize) - (i + 1)
    time_left = datetime.timedelta(seconds=iters_left * (time.time() - prev_time))
    prev_time = time.time()

    print(' process %d/%d img ... %s' % (i, opt.how_many, time_left))
webpage.save()




