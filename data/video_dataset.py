import os.path
import torch
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2

class VideoDataset(BaseDataset):
    """A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.
    """

    def __init__(self, opt):
        folder = os.path.join(opt.dataroot, 'train')
        self.scheduled_sampling = opt.scheduled_sampling
        if opt.scheduled_sampling:
            folder_gen = os.path.join(opt.dataroot, 'generated')

        self.clip_len = opt.num_frames

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = opt.load_size
        self.resize_width = opt.load_size
        self.crop_size = opt.crop_size

        if opt.dataset == 'action':
            self.action_labels = {
                'jack': 7,
                'jump': 8,
                'pjump': 2,
                'run': 3,
                'side': 4,
                'skip': 5,
                'walk': 6,
                'wave1': 0,
                'wave2': 1
            }
        elif opt.dataset == 'UTDAL':
            self.action_labels = {
                'Happiness': 1,
                'Anger': 2,
                'Disgust': 3,
                'Fear': 4,
                'Sadness': 5,
                'Surprise': 0,
            }

        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(self.action_labels[label])

        if opt.scheduled_sampling:
            self.fnames_gen = []
            for label in sorted(os.listdir(folder_gen)):
                for fname in os.listdir(os.path.join(folder_gen, label)):
                    self.fnames_gen.append(os.path.join(folder_gen, label, fname))

        assert len(labels) == len(self.fnames)
        print('Number of {} videos: {:d}'.format('training', len(self.fnames)))
        self.label_array = np.array(labels, dtype=int)

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading and preprocessing.
        buffer = self.load_frames(self.fnames[index])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        buffer_A = buffer[0:-1]     #input sequence A
        buffer_B = buffer[1:]       #target sequence B (next frame)
        labels = np.array(self.label_array[index])
        buffer_A = self.normalize_convert(buffer_A)
        buffer_B = self.normalize_convert(buffer_B)

        return (buffer_A, buffer_B, torch.from_numpy(labels))

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir) if img != '.DS_Store'])
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), dtype = np.uint8)
        name_buffer = []
        for i, frame_name in enumerate(frames):
            frame = Image.open(frame_name).convert('RGB').resize((self.resize_height, self.resize_width))
            buffer[i] = frame
            name_buffer.append(frame_name)
        return buffer

    def crop(self, buffer, clip_len, crop_size):

        time_index = 0

        # Randomly select start indices in order to crop the video
        height_index = 0
        width_index = 0

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer

    def normalize_convert(self, buffer):
        T = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        buffer_trans = torch.empty((self.clip_len-1, 3, self.crop_size, self.crop_size))
        for i, frame in enumerate(buffer):
            frame_t = T(frame)
            buffer_trans[i] = frame_t

        return buffer_trans
