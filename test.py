
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html, util
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
import torch
from tqdm import tqdm
import cv2


def process_img(img, opt):
    # if opt.action is not None:
    #     action_label = torch.tensor(opt.action)

    A = Image.open(img).convert('RGB')

    transform_params = get_params(opt, A.size)
    A_transform = get_transform(opt, transform_params, grayscale=False)

    A = A_transform(A)
    #return A, action_label
    inp_data = {'A': A.unsqueeze(0), 'B': A.unsqueeze(0)}

    # if opt.action is not None:
    #     inp_data['action'] = action_label.unsqueeze(0)

    return inp_data

def get_results(opt, model, img_path):
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    if model == None:
        model = create_model(opt)      # create a model given opt.model and other options
        model.setup(opt)               # regular setup: load and print networks; create schedulers
        model.eval()

    if img_path == None:
        img_path = opt.img_path

    # for i in range(opt.num_frames):
    # if not os.path.exists(img_path+'/0000.jpg'):
    #     continue
    inp_frame_path = img_path+'/'+f'{0:04}'+'.jpg'
    data = process_img(inp_frame_path, opt)
    model.set_input(data)  # unpack data from data loader
    model.test()           # run inference
    visuals = model.get_current_visuals()  # get image results

    img = visuals['fake_B']
    # img = visuals['foreground_change']

    im = util.tensor2im(img)
    for i in range(opt.num_frames - 1):
        image_name = f'{(i+1):04}'+'.jpg'
        save_path = os.path.join(img_path, image_name)
        h, w, _ = im.shape
        img = im[:,:,i*3:i*3+3]
        print(img.shape)
        util.save_image(img[:,:,], save_path)

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)       # hard-code some parameters for test
    print('Generating Results....')
    folder_gen = os.path.join(opt.dataroot, 'test', 'Happiness')
    videos = os.listdir(folder_gen)
    for v in tqdm(videos):
        path = os.path.join(opt.dataroot, 'test', 'Happiness', v)
        get_results(opt, model, path)



# action_labels = {
#     'jack': 0,
#     'jump': 1,
#     'pjump': 2,
#     'run': 3,
#     'side': 4,
#     'skip': 5,
#     'walk': 6,
#     'wave1': 7,
#     'wave2': 8
# }

# if __name__ == '__main__':
#     opt = TestOptions().parse()  # get test options
#     # hard-code some parameters for test
#     opt.num_threads = 0   # test code only supports num_threads = 1
#     opt.batch_size = 1    # test code only supports batch_size = 1
#     opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
#     opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
#     opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
#     dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
#     model = create_model(opt)      # create a model given opt.model and other options
#     model.setup(opt)               # regular setup: load and print networks; create schedulers
#     # create a website
#     web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
#     webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
#     # test with eval mode. This only affects layers like batchnorm and dropout.
#     # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
#     # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
#     if opt.eval:
#         model.eval()
#     for i, data in enumerate(dataset):
#         if i >= opt.num_test:  # only apply our model to opt.num_test images.
#             break
#         model.set_input(data)  # unpack data from data loader
#         model.test()           # run inference
#         visuals = model.get_current_visuals()  # get image results
#         img_path = model.get_image_paths()     # get image paths
#         if i % 5 == 0:  # save images to an HTML file
#             print('processing (%04d)-th image... %s' % (i, img_path))
#         save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
#     webpage.save()  # save the HTML
