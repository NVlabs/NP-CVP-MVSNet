# Dataloader for the DTU dataset in MVSNet format.
# From https://github.com/JiayuYANG/CVP-MVSNet
# by: Jiayu Yang
# date: 2020-01-28

from dataset.utils import *
from dataset.dataPaths import *
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import random
from random import randrange

class MVSDataset(Dataset):
    def __init__(self, args):
        super(MVSDataset, self).__init__()

        self.args = args
        self.data_root = self.args.dataset_root
        self.scan_list_file = getScanListFile(self.data_root,self.args.mode)
        self.pair_list_file = getPairListFile(self.data_root,self.args.mode,self.args.vselection)
        print("Initiating dataloader for pre-processed DTU dataset.")
        print("Using dataset:"+self.data_root+self.args.mode+"/")

        self.metas = self.build_list(self.args.mode)
        print("Dataloader initialized.")

    def build_list(self,mode):

        metas = []

        # Read scan list
        scan_list = readScanList(self.scan_list_file,self.args.mode)

        # Read pairs list
        for scan in scan_list:
            with open(self.pair_list_file) as f:
                num_viewpoint = int(f.readline())
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    if mode=="train":
                        for light_idx in range(7):
                            metas.append((scan, ref_view, src_views, light_idx))
                    else:
                        metas.append((scan, ref_view, src_views, 3))

        print("Done. metas:"+str(len(metas)))
        return metas

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, ref_view, src_views, light_idx = meta

        assert self.args.nsrc <= len(src_views)

        src_idx_list = []

        if self.args.nsrc >0:
            src_idx_list += src_views[:self.args.nsrc]

        if self.args.nbadsrc >0:
            src_idx_list += src_views[-self.args.nbadsrc::]

        if self.args.random_src > 0:
            src_idx_list += random.sample(src_views,self.args.random_src)

        ref_img = []
        src_imgs = []
        ref_depths = []
        ref_depth_mask = []
        ref_intrinsics = []
        src_intrinsics = []
        ref_extrinsics = []
        src_extrinsics = []
        depth_min = []
        depth_max = []

        ## 1. Read images
        # ref image
        ref_img_file = getImageFile(self.data_root,self.args.mode,scan,ref_view,light_idx)
        ref_img = read_img_with_size(ref_img_file,self.args.imgsize)
        # src images
        for src_idx in src_idx_list:
            if self.args.random_light == 1:
                light_idx = randrange(7)
            src_img_file = getImageFile(self.data_root,self.args.mode,scan,src_idx,light_idx)
            src_img = read_img_with_size(src_img_file,self.args.imgsize)

            src_imgs.append(src_img)

        ## 2. Read camera parameters
        cam_file = getCameraFile(self.data_root,self.args.mode,ref_view)
        ref_intrinsics, ref_extrinsics, depth_min, depth_max = read_cam_file_with_size(cam_file,self.args.imgsize)
        for src_idx in src_idx_list:
            cam_file = getCameraFile(self.data_root,self.args.mode,src_idx)
            intrinsics, extrinsics, depth_min_tmp, depth_max_tmp = read_cam_file_with_size(cam_file,self.args.imgsize)
            src_intrinsics.append(intrinsics)
            src_extrinsics.append(extrinsics)

        if self.args.mode == 'train' or self.args.eval_read_gt_depth:
            depth_file = getDepthFile(self.data_root,self.args.mode,scan,ref_view)
            ref_depth = read_depth_with_size(depth_file,self.args.imgsize)

        # Orgnize output and return
        sample = {}
        sample["ref_img"] = np.moveaxis(np.array(ref_img),2,0)
        sample["src_imgs"] = np.moveaxis(np.array(src_imgs),3,1)
        sample["ref_depth_mask"] = np.array(ref_depth_mask)
        sample["ref_intrinsics"] = np.array(ref_intrinsics)
        sample["src_intrinsics"] = np.array(src_intrinsics)
        sample["ref_extrinsics"] = np.array(ref_extrinsics)
        sample["src_extrinsics"] = np.array(src_extrinsics)
        sample["depth_min"] = depth_min
        sample["depth_max"] = depth_max

        if self.args.mode == 'train' or self.args.eval_read_gt_depth:
            sample["ref_depth"] = ref_depth.astype('float32')

        if self.args.mode != "train":
            sample["filename"] = scan + '/{}/' + '{:0>8}'.format(ref_view) + "{}"

        return sample
