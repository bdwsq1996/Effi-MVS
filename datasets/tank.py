from torch.utils.data import Dataset
from datasets.data_io import *
import os
import numpy as np
import cv2
from collections import defaultdict
from PIL import Image
import torch
from torchvision import transforms as T
import math


class MVSDataset(Dataset):
    def __init__(self, datapath, n_views=3, ndepths=192, img_wh=(1920, 1056), split='intermediate', scan=['Family']):

        self.stages = 4
        self.datapath = datapath
        self.img_wh = img_wh
        self.input_scans = scan

        self.split = split
        self.build_metas()
        self.n_views = n_views
        self.ndepths = ndepths

    def build_metas(self):
        self.metas = []
        if self.split == 'intermediate':
            self.scans = self.input_scans
            self.image_sizes = {'Family': (1920, 1080),
                                'Francis': (1920, 1080),
                                'Horse': (1920, 1080),
                                'Lighthouse': (2048, 1080),
                                'M60': (2048, 1080),
                                'Panther': (2048, 1080),
                                'Playground': (1920, 1080),
                                'Train': (1920, 1080),
                                'Auditorium': (1920, 1080),
                                'Ballroom': (1920, 1080),
                                'Courtroom': (1920, 1080),
                                'Museum': (1920, 1080),
                                'Palace': (1920, 1080),
                                'Temple': (1920, 1080)
                                }

        elif self.split == 'advanced':
            self.scans = self.input_scans
            self.image_sizes = {'Auditorium': (1920, 1080),
                                'Ballroom': (1920, 1080),
                                'Courtroom': (1920, 1080),
                                'Museum': (1920, 1080),
                                'Palace': (1920, 1080),
                                'Temple': (1920, 1080)}

        for scan in self.scans:
            if scan in ['Family', 'Francis', 'Horse', 'Lighthouse','M60', 'Panther', 'Playground', 'Train']:
                split = 'intermediate'
            else:
                split = 'advanced'
            with open(os.path.join(self.datapath, split, scan, 'pair.txt')) as f:
                num_viewpoint = int(f.readline())
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    if len(src_views) != 0:
                        self.metas += [(scan, -1, ref_view, src_views, split)]

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))


        depth_min = float(lines[11].split()[0])
        depth_max = float(lines[11].split()[3])

        return intrinsics, extrinsics, depth_min, depth_max

    def read_img(self, filename, imsize):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        np_img = cv2.resize(np_img, imsize, interpolation=cv2.INTER_LINEAR)
        # print(np_img.shape)
        return np_img


    def center_img(self, img):  # this is very important for batch normalization
        var = np.var(img, axis=(0, 1), keepdims=True)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        return (img - mean) / (np.sqrt(var) + 0.00000001)

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        scan, _, ref_view, src_views, split = self.metas[idx]
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.n_views - 1]
        img_w, img_h = self.image_sizes[scan]


        # depth = None

        imgs = []
        depth_values = None
        proj_matrices = []

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, split, scan, f'images/{vid:08d}.jpg')
            proj_mat_filename = os.path.join(self.datapath, split, scan, f'cams/{vid:08d}_cam.txt')

            img = self.read_img(img_filename, (1920, 1056))
            intrinsics, extrinsics, depth_min_, depth_max_ = self.read_cam_file(proj_mat_filename)

            intrinsics[0] *= self.img_wh[0] / img_w
            intrinsics[1] *= self.img_wh[1] / img_h
            imgs.append(img)

            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)
            if i == 0:  # reference view
                disp_min = 1 / depth_max_
                disp_max = 1 / depth_min_
                depth_values = np.linspace(disp_min, disp_max, self.ndepths, dtype=np.float32)


        # imgs: N*3*H0*W0, N is number of images
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)
        stage1_pjmats = proj_matrices.copy()
        stage1_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.125
        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.25
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.5
        proj_matrices_ms = {
            "stage1": torch.from_numpy(stage1_pjmats.copy()).contiguous().float(),
            "stage2": torch.from_numpy(stage2_pjmats.copy()).contiguous().float(),
            "stage3": torch.from_numpy(stage3_pjmats.copy()).contiguous().float(),
            "stage4": torch.from_numpy(proj_matrices.copy()).contiguous().float(),
        }

        imgs = torch.from_numpy(imgs.copy()).contiguous().float()
        depth_values = torch.from_numpy(depth_values.copy()).contiguous().float()
        return {"imgs": imgs,
                "proj_matrices": proj_matrices_ms,
                "depth_values": depth_values,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"}

