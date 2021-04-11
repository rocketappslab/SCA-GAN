import os.path
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
from PIL import ImageOps
import PIL
import random
import pandas as pd
import numpy as np
import torch
from scipy import signal

class KeyDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_P = os.path.join(opt.dataroot, opt.phase) #person images
        self.dir_K = os.path.join(opt.dataroot, opt.phase + 'K') #keypoints
        self.dir_E = os.path.join(opt.dataroot, opt.phase + 'E') #edge

        self.init_categories(opt.pairLst)
        self.transform_rgb, self.transform_edge = get_transform(opt)

    def init_categories(self, pairLst):
        pairs_file_train = pd.read_csv(pairLst)
        self.size = len(pairs_file_train)
        self.pairs = []
        print('Loading data pairs ...')
        for i in range(self.size):
            pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            self.pairs.append(pair)

        print('Loading data pairs finished ...')

    def __getitem__(self, index):
        if self.opt.phase == 'train':
            index = random.randint(0, self.size-1)

        P1_name, P2_name = self.pairs[index]
        P1_path = os.path.join(self.dir_P, P1_name) # person 1
        BP1_path = os.path.join(self.dir_K, P1_name + '.npy') # bone of person 1
        EP1_path = os.path.join(self.dir_E, 'diff', P1_name.replace('.jpg', '.png').replace('fashion', ''))  # bone of person 1

        # person 2 and its bone
        P2_path = os.path.join(self.dir_P, P2_name) # person 2
        BP2_path = os.path.join(self.dir_K, P2_name + '.npy') # bone of person 2
        EP2_path = os.path.join(self.dir_E, 'diff', P2_name.replace('.jpg', '.png').replace('fashion', ''))  # bone of person 1

        if self.opt.phase == 'test':
            EP1_path = EP1_path.replace('testE', 'trainE')
            EP2_path = EP2_path.replace('testE', 'trainE')

        P1_img = Image.open(P1_path).convert('RGB')
        P2_img = Image.open(P2_path).convert('RGB')
        P1_edge = Image.open(EP1_path).convert('L')
        P2_edge = Image.open(EP2_path).convert('L')
        P1_edge = PIL.ImageOps.invert(P1_edge)
        P2_edge = PIL.ImageOps.invert(P2_edge)

        BP1_img = np.load(BP1_path) # h, w, c
        BP2_img = np.load(BP2_path)

        # use flip
        if self.opt.phase == 'train' and self.opt.use_flip:
            # print ('use_flip ...')
            flip_random = random.uniform(0,1)
            
            if flip_random > 0.5:
                # print('fliped ...')
                P1_img = P1_img.transpose(Image.FLIP_LEFT_RIGHT)
                P2_img = P2_img.transpose(Image.FLIP_LEFT_RIGHT)
                P1_edge = P1_edge.transpose(Image.FLIP_LEFT_RIGHT)
                P2_edge = P2_edge.transpose(Image.FLIP_LEFT_RIGHT)

                BP1_img = np.array(BP1_img[:, ::-1, :]) # flip
                BP2_img = np.array(BP2_img[:, ::-1, :]) # flip

        person_edge = np.array(P1_edge.copy())
        BEP2, masked_edge_c1 = crop_shift(sp=BP1_img.copy(), tp=BP2_img.copy(), person=person_edge,
                                          radius=self.opt.crop_radius, mask=False)

        BP1 = torch.from_numpy(BP1_img).float()  # h, w, c
        BP1 = BP1.transpose(2, 0)  # c,w,h
        BP1 = BP1.transpose(2, 1)  # c,h,w

        BP2 = torch.from_numpy(BP2_img).float()
        BP2 = BP2.transpose(2, 0)  # c,w,h
        BP2 = BP2.transpose(2, 1)  # c,h,w

        P1 = self.transform_rgb(P1_img)
        P2 = self.transform_rgb(P2_img)

        EP1 = self.transform_edge(P1_edge)
        EP2 = self.transform_edge(P2_edge)
        BEP2 = self.transform_edge(BEP2)
        masked_edge_c1 = self.transform_edge(masked_edge_c1)

        return {'P1': P1, 'BP1': BP1, 'P2': P2, 'BP2': BP2, 'EP1': EP1, 'EP2': EP2,
                'BEP2': BEP2, 'BEP2c1': masked_edge_c1,
                'P1_path': P1_name, 'P2_path': P2_name}
                

    def __len__(self):
        if self.opt.phase == 'train':
            return 4000
        elif self.opt.phase == 'test':
            return self.size

    def name(self):
        return 'KeyDataset'

    def split_name(self,str,type):
        list = []
        list.append(type)
        if (str[len('fashion'):len('fashion') + 2] == 'WO'):
            lenSex = 5
        else:
            lenSex = 3
        list.append(str[len('fashion'):len('fashion') + lenSex])
        idx = str.rfind('id0')
        list.append(str[len('fashion') + len(list[1]):idx])
        id = str[idx:idx + 10]
        list.append(id[:2]+'_'+id[2:])
        pose = str[idx + 10:]
        list.append(pose[:4]+'_'+pose[4:])

        head = ''
        for path in list:
            head = os.path.join(head, path)
        return head

def map_to_cord(pose_map, threshold=0.1):
    all_peaks = [[] for i in range(18)]
    pose_map = pose_map[..., :18]

    y, x, z = np.where(np.logical_and(pose_map == pose_map.max(axis = (0, 1)),
                                     pose_map > threshold))
    for x_i, y_i, z_i in zip(x, y, z):
        all_peaks[z_i].append([x_i, y_i])

    x_values = []
    y_values = []

    for i in range(18):
        if len(all_peaks[i]) != 0:
            x_values.append(all_peaks[i][0][0])
            y_values.append(all_peaks[i][0][1])
        else:
            x_values.append(-1)
            y_values.append(-1)

    return np.concatenate([np.expand_dims(y_values, -1), np.expand_dims(x_values, -1)], axis=1)


def GaussainKernel(kernlen=10, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


def crop_shift(sp, tp, person, radius=10, size=(256, 176), mask=True):
    s_cord = map_to_cord(sp)
    t_cord = map_to_cord(tp)
    r = radius
    plates = np.ones((size[0], size[1], 18), dtype='float32')
    plate = np.ones((size[0], size[1]), dtype='float32')
    for i in range(18):
        sx = s_cord[i][1]
        sy = s_cord[i][0]
        tx = t_cord[i][1]
        ty = t_cord[i][0]
        if not (sx < 0 or tx < 0):
            # source crop
            sy_start = 0 if sy - r < 0 else sy - r
            sy_end = size[0] if sy + r > size[0] else sy + r
            sx_start = 0 if sx - r < 0 else sx - r
            sx_end = size[1] if sx + r > size[1] else sx + r

            crop_height = sy_end - sy_start
            crop_width = sx_end - sx_start

            ty_start = 0 if (ty - int((crop_height)/2)) < 0 else (ty - int((crop_height)/2))
            ty_end = size[0] if (ty_start+crop_height) > size[0] else (ty_start+crop_height)
            tx_start = 0 if (tx - int((crop_width)/2)) < 0 else (tx - int((crop_width)/2))
            tx_end = size[1] if (tx_start + crop_width) > size[1] else (tx_start + crop_width)

            # target crop
            crop = person[sy_start:sy_end, sx_start:sx_end]

            if mask:
                M = GaussainKernel(kernlen=radius*2, std=4)
                M = np.resize(M, (crop.shape))
                crop = crop * M
            crop = np.resize(crop, ((ty_end-ty_start), (tx_end-tx_start)))
            # if ty_end - ty_start < crop.shape[0] and tx_end - tx_start < crop.shape[1]:
            #     plates[:, :, i][ty_start:ty_end, tx_start:tx_end] = crop
            #     plate[ty_start:ty_end, tx_start:tx_end] = crop
            # elif ty_end - ty_start < crop.shape[0]:
            #     pass
            # elif tx_end - tx_start < crop.shape[1]:
            #     pass
            # else:
            plates[:, :, i][ty_start:ty_end, tx_start:tx_end] = crop
            plate[ty_start:ty_end, tx_start:tx_end] = crop
            # plates[:, :, i][ty - r:(ty + r), tx - r:(tx + r)] = crop
            # plate[ty - r:(ty + r), tx - r:(tx + r)] = crop

    return 255.0-plates, 255.0-plate
