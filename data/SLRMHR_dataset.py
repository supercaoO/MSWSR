import os
import numpy as np
import torch.utils.data as data

from data import common


class SLRMHRDataset(data.Dataset):
    '''
    Read LR and HR images in train and eval phases.
    '''

    def name(self):
        return common.find_benchmark(self.opt['dataroot_LR'])

    def __init__(self, opt):
        super(SLRMHRDataset, self).__init__()
        self.opt = opt
        self.train = (opt['phase'] == 'train')
        self.split = 'train' if self.train else 'test'
        self.scale = self.opt['scale']
        self.paths_HR, self.paths_LR = [], None

        # change the length of train dataset (influence the number of iterations in each epoch)
        self.repeat = 2

        # read image list from image/binary files
        self.paths_LR = common.get_image_paths(self.opt['data_type'], os.path.join(self.opt['dataroot_LR']))
        assert self.paths_LR, '[Error] LR paths are empty.'
        for scale in self.scale:
            paths_HR = common.get_image_paths(self.opt['data_type'],
                                              os.path.join(self.opt['dataroot_HR'], 'x' + str(scale)))
            if paths_HR:
                assert len(paths_HR) == len(self.paths_LR), \
                    '[Error] x%s HR: [%d] and LR: [%d] have different number of images.' % (
                        scale, len(paths_HR), len(self.paths_LR))
            self.paths_HR.append(paths_HR)

    def __getitem__(self, idx):
        lr, hr, lr_path, hr_path = self._load_file(idx)
        if self.train:
            lr, hr = self._get_patch(lr, hr)
        lr_tensor, hr_tensor = common.np2Tensor_mhr([lr, hr], self.opt['rgb_range'])
        return {'LR': lr_tensor, 'HR': hr_tensor, 'LR_path': lr_path, 'HR_path': hr_path}

    def __len__(self):
        if self.train:
            return len(self.paths_LR) * self.repeat
        else:
            return len(self.paths_LR)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.paths_LR)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        lr_path = self.paths_LR[idx]
        lr = common.read_img(lr_path, self.opt['data_type'])
        hr = []
        hr_path = []
        for i in range(len(self.scale)):
            hr_path_ = self.paths_HR[i][idx]
            hr_path.append(hr_path_)
            hr.append(common.read_img(hr_path_, self.opt['data_type']))

        return lr, hr, lr_path, hr_path

    def _get_patch(self, lr, hr):

        LR_size = self.opt['LR_size']
        # random crop and augment
        lr, hr = common.get_patch_mhr(
            lr, hr, LR_size, self.scale)
        lr, hr = common.augment_mhr([lr, hr])
        lr = common.add_noise(lr, self.opt['noise'])

        return lr, hr
