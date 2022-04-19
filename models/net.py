# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
import torch.nn as nn
from models.modules import *
from torchsparse import SparseTensor
from torchsparse import nn as spnn
from random import randrange

# Feature pyramid
class FeaturePyramid_FPN_4level(nn.Module):
    def __init__(self, feature_ch):
        super(FeaturePyramid_FPN_4level, self).__init__()

        ch = feature_ch

        # H x W
        self.conv0 = nn.Sequential(
            ConvBnReLU(3,ch[0], kernel_size=3, stride=1),
            ConvBnReLU(ch[0],ch[0], kernel_size=3, stride=1),
            ConvBnReLU(ch[0],ch[0], kernel_size=3, stride=1),
            ConvBnReLU(ch[0],ch[0], kernel_size=3, stride=1),
            ConvBnReLU(ch[0],ch[0], kernel_size=3, stride=1)
        )

        # H/2 x W/2
        self.conv1 = nn.Sequential(
            ConvBnReLU(ch[0],ch[1], kernel_size=5, stride=2, pad=2),
            ConvBnReLU(ch[1],ch[1], kernel_size=3, stride=1),
            ConvBnReLU(ch[1],ch[1], kernel_size=3, stride=1)
        )

        # H/4 x W/4
        self.conv2 = nn.Sequential(
            ConvBnReLU(ch[1],ch[2], kernel_size=5, stride=2, pad=2),
            ConvBnReLU(ch[2],ch[2], kernel_size=3, stride=1),
            ConvBnReLU(ch[2],ch[2], kernel_size=3, stride=1)
        )

        # H/8 x W/8
        self.conv3 = nn.Sequential(
            ConvBnReLU(ch[2],ch[3], kernel_size=5, stride=2, pad=2),
            ConvBnReLU(ch[3],ch[3], kernel_size=3, stride=1),
            ConvBnReLU(ch[3],ch[3], kernel_size=3, stride=1)
        )

        # H/4 x W/4
        self.conv3up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
            ConvBnReLU(ch[3],ch[3], kernel_size=3, stride=1),
            ConvBnReLU(ch[3],ch[2], kernel_size=3, stride=1)
        )

        # H/2 x W/2
        self.conv2up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
            ConvBnReLU(ch[2],ch[2], kernel_size=3, stride=1),
            ConvBnReLU(ch[2],ch[1], kernel_size=3, stride=1)
        )

        # H x W
        self.conv1up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
            ConvBnReLU(ch[1],ch[1], kernel_size=3, stride=1),
            ConvBnReLU(ch[1],ch[0], kernel_size=3, stride=1)
        )

        self.conv_out0 = nn.Sequential(
            ConvBnReLU(ch[0],ch[0], kernel_size=3, stride=1),
            ConvBnReLU(ch[0],ch[0], kernel_size=3, stride=1),
            nn.Conv2d(ch[0],ch[0], kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.conv_out1 = nn.Sequential(
            ConvBnReLU(ch[1],ch[1], kernel_size=3, stride=1),
            ConvBnReLU(ch[1],ch[1], kernel_size=3, stride=1),
            nn.Conv2d(ch[1],ch[1], kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.conv_out2 = nn.Sequential(
            ConvBnReLU(ch[2],ch[2], kernel_size=3, stride=1),
            ConvBnReLU(ch[2],ch[2], kernel_size=3, stride=1),
            nn.Conv2d(ch[2],ch[2], kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.conv_out3 = nn.Sequential(
            ConvBnReLU(ch[3],ch[3], kernel_size=3, stride=1),
            ConvBnReLU(ch[3],ch[3], kernel_size=3, stride=1),
            nn.Conv2d(ch[3],ch[3], kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, img):

        conv0 = self.conv0(img) # H x W
        conv1 = self.conv1(conv0) # H/2 x W/2
        conv2 = self.conv2(conv1) # H/4 x W/4
        conv3 = self.conv3(conv2) # H/8 x W/8

        conv2o = self.conv3up(conv3) + conv2 # H/4 x W/4
        conv1o = self.conv2up(conv2o) + conv1 # H/2 x W/2
        conv0o = self.conv1up(conv1o) + conv0 # H x W

        f0 = self.conv_out0(conv0o)
        f1 = self.conv_out1(conv1o)
        f2 = self.conv_out2(conv2o)
        f3 = self.conv_out3(conv3)

        c0 = None
        c1 = None
        c2 = None
        c3 = None

        return [[f0,c0],[f1,c1],[f2,c2],[f3,c3]]

class FeaturePyramid_FPN_3level(nn.Module):
    def __init__(self, feature_ch, init_context_ch):
        super(FeaturePyramid_FPN_3level, self).__init__()

        ch = feature_ch
        self.init_context_ch = init_context_ch

        # H x W
        self.conv0 = nn.Sequential(
            ConvBnReLU(3,ch[0], kernel_size=3, stride=1),
            ConvBnReLU(ch[0],ch[0], kernel_size=3, stride=1),
            ConvBnReLU(ch[0],ch[0], kernel_size=3, stride=1),
            ConvBnReLU(ch[0],ch[0], kernel_size=3, stride=1),
            ConvBnReLU(ch[0],ch[0], kernel_size=3, stride=1)
        )

        # H/2 x W/2
        self.conv1 = nn.Sequential(
            ConvBnReLU(ch[0],ch[1], kernel_size=5, stride=2, pad=2),
            ConvBnReLU(ch[1],ch[1], kernel_size=3, stride=1),
            ConvBnReLU(ch[1],ch[1], kernel_size=3, stride=1)
        )

        # H/4 x W/4
        self.conv2 = nn.Sequential(
            ConvBnReLU(ch[1],ch[2], kernel_size=5, stride=2, pad=2),
            ConvBnReLU(ch[2],ch[2], kernel_size=3, stride=1),
            ConvBnReLU(ch[2],ch[2], kernel_size=3, stride=1)
        )

        # H/2 x W/2
        self.conv2up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
            ConvBnReLU(ch[2],ch[2], kernel_size=3, stride=1),
            ConvBnReLU(ch[2],ch[1], kernel_size=3, stride=1)
        )

        # H x W
        self.conv1up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
            ConvBnReLU(ch[1],ch[1], kernel_size=3, stride=1),
            ConvBnReLU(ch[1],ch[0], kernel_size=3, stride=1)
        )

        self.conv_out0 = nn.Sequential(
            ConvBnReLU(ch[0],ch[0], kernel_size=3, stride=1),
            ConvBnReLU(ch[0],ch[0], kernel_size=3, stride=1),
            nn.Conv2d(ch[0],ch[0], kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.conv_out1 = nn.Sequential(
            ConvBnReLU(ch[1],ch[1], kernel_size=3, stride=1),
            ConvBnReLU(ch[1],ch[1], kernel_size=3, stride=1),
            nn.Conv2d(ch[1],ch[1], kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.conv_out2 = nn.Sequential(
            ConvBnReLU(ch[2],ch[2], kernel_size=3, stride=1),
            ConvBnReLU(ch[2],ch[2], kernel_size=3, stride=1),
            nn.Conv2d(ch[2],ch[2], kernel_size=3, stride=1, padding=1, bias=False)
        )

        if self.init_context_ch > 0:
            self.context_out2 = nn.Sequential(
                ConvBnReLU(ch[2],ch[2], kernel_size=3, stride=1),
                ConvBnReLU(ch[2],ch[2], kernel_size=3, stride=1),
                nn.Conv2d(ch[2],init_context_ch, kernel_size=3, stride=1, padding=1, bias=False)
            )

    def forward(self, img):

        conv0 = self.conv0(img) # H x W
        conv1 = self.conv1(conv0) # H/2 x W/2
        conv2 = self.conv2(conv1) # H/4 x W/4

        conv1o = self.conv2up(conv2) + conv1 # H/2 x W/2
        conv0o = self.conv1up(conv1o) + conv0 # H x W

        f0 = self.conv_out0(conv0o)
        f1 = self.conv_out1(conv1o)
        f2 = self.conv_out2(conv2)

        c0 = None
        c1 = None

        if self.init_context_ch > 0:
            c2 = self.context_out2(conv2)
        else:
            c2 = None

        return [[f0,c0],[f1,c1],[f2,c2]]

class CostRegNet_v3_full(nn.Module):
    def __init__(self,feature_ch):
        super(CostRegNet_v3_full, self).__init__()

        base_ch=feature_ch

        self.input0 = ConvBnReLU3D(base_ch, base_ch, kernel_size=3, pad=1)
        self.input1 = ConvBnReLU3D(base_ch, base_ch, kernel_size=3, pad=1)

        self.conv1a = ConvBnReLU3D(base_ch, base_ch*2,stride=2, kernel_size=3, pad=1)
        self.conv1b = ConvBnReLU3D(base_ch*2, base_ch*2, kernel_size=3, pad=1)
        self.conv1c = ConvBnReLU3D(base_ch*2, base_ch*2, kernel_size=3, pad=1)
        self.conv2a = ConvBnReLU3D(base_ch*2, base_ch*4,stride=2, kernel_size=3, pad=1)
        self.conv2b = ConvBnReLU3D(base_ch*4, base_ch*4, kernel_size=3, pad=1)
        self.conv2c = ConvBnReLU3D(base_ch*4, base_ch*4, kernel_size=3, pad=1)
        self.conv3a = ConvBnReLU3D(base_ch*4, base_ch*8,stride=2, kernel_size=3, pad=1)
        self.conv3b = ConvBnReLU3D(base_ch*8, base_ch*8, kernel_size=3, pad=1)
        self.conv3c = ConvBnReLU3D(base_ch*8, base_ch*8, kernel_size=3, pad=1)

        self.conv3d = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear',align_corners=False),
            ConvBnReLU3D(base_ch*8, base_ch*4,stride=1, kernel_size=3, pad=1)
        )

        self.conv2d = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear',align_corners=False),
            ConvBnReLU3D(base_ch*4, base_ch*2,stride=1, kernel_size=3, pad=1)
        )

        self.conv1d = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear',align_corners=False),
            ConvBnReLU3D(base_ch*2, base_ch,stride=1, kernel_size=3, pad=1)
        )

        self.prob0 = nn.Conv3d(base_ch, 1, 3, stride=1, padding=1)

    def forward(self, x, coord=None):

        input0 = self.input1(self.input0(x))

        conv1c = self.conv1c(self.conv1b(self.conv1a(input0)))
        conv2c = self.conv2c(self.conv2b(self.conv2a(conv1c)))
        conv3c = self.conv3c(self.conv3b(self.conv3a(conv2c)))

        conv3d = conv2c+self.conv3d(conv3c)
        conv2d = conv1c+self.conv2d(conv3d)
        conv1d = input0+self.conv1d(conv2d)

        prob = self.prob0(conv1d)

        return prob

class CostRegNet_sparse_v4(nn.Module):
    def __init__(self,feature_ch):
        super(CostRegNet_sparse_v4, self).__init__()

        base_ch=feature_ch

        self.input = nn.Sequential(
            ConvBnReLU3DSparseFactorize(base_ch, base_ch, kernel_size=3),
            ConvBnReLU3DSparseFactorize(base_ch, base_ch, kernel_size=3),
            ConvBnReLU3DSparseFactorize(base_ch, base_ch, kernel_size=3)
        )

        self.conv1up = nn.Sequential(
            ConvBnReLU3DSparse(base_ch, base_ch*2,stride=2, kernel_size=2),
            ConvBnReLU3DSparseFactorize(base_ch*2, base_ch*2, kernel_size=3),
            ConvBnReLU3DSparseFactorize(base_ch*2, base_ch*2, kernel_size=3)
        )
        self.conv2up = nn.Sequential(
            ConvBnReLU3DSparse(base_ch*2, base_ch*4,stride=2, kernel_size=2),
            ConvBnReLU3DSparseFactorize(base_ch*4, base_ch*4, kernel_size=3),
            ConvBnReLU3DSparseFactorize(base_ch*4, base_ch*4, kernel_size=3)
        )
        self.conv3up = nn.Sequential(
            ConvBnReLU3DSparse(base_ch*4, base_ch*8,stride=2, kernel_size=2),
            ConvBnReLU3DSparseFactorize(base_ch*8, base_ch*8, kernel_size=3),
            ConvBnReLU3DSparseFactorize(base_ch*8, base_ch*8, kernel_size=3)
        )

        self.conv3down = nn.Sequential(
            ConvBnReLU3DSparse(base_ch*8, base_ch*4, kernel_size=2, stride=2, bias=False, transposed=True),
            ConvBnReLU3DSparseFactorize(base_ch*4, base_ch*4, kernel_size=3),
            ConvBnReLU3DSparseFactorize(base_ch*4, base_ch*4, kernel_size=3)
        )

        self.conv2down = nn.Sequential(
            ConvBnReLU3DSparse(base_ch*4, base_ch*2, kernel_size=2, stride=2, bias=False, transposed=True),
            ConvBnReLU3DSparseFactorize(base_ch*2, base_ch*2, kernel_size=3),
            ConvBnReLU3DSparseFactorize(base_ch*2, base_ch*2, kernel_size=3)
        )

        self.conv1down = nn.Sequential(
            ConvBnReLU3DSparse(base_ch*2, base_ch, kernel_size=2, stride=2, bias=False, transposed=True),
            ConvBnReLU3DSparseFactorize(base_ch, base_ch, kernel_size=3),
            ConvBnReLU3DSparseFactorize(base_ch, base_ch, kernel_size=3)
        )

        self.prob = nn.Sequential(
            ConvBnReLU3DSparseFactorize(base_ch, base_ch, kernel_size=3),
            ConvBnReLU3DSparseFactorize(base_ch, base_ch, kernel_size=3),
            spnn.Conv3d(base_ch, base_ch, (1,1,3), stride=1, bias=False),
            spnn.Conv3d(base_ch, base_ch, (1,3,1), stride=1, bias=False),
            spnn.Conv3d(base_ch, base_ch, (3,1,1), stride=1, bias=False),
            spnn.Conv3d(base_ch, 1, 1, stride=1, bias=False)
        )

    def forward(self, cost_volume, hypo_coords, mode='train'):

        # Convert cost volume and depth hypothesis to sparse feature and coordinates
        B,CH,D,H,W = cost_volume.shape
        # feats
        feats = cost_volume.permute(0,2,3,4,1).reshape(B*D*H*W,CH)
        # coords
        coords_z = hypo_coords.permute(0,2,3,4,1).reshape(B*D*H*W)
        coords_b, plain_coords_z, coords_h, coords_w = torch.where(torch.ones_like(hypo_coords.squeeze(1)))
        coords = torch.stack((coords_h,coords_w,coords_z,coords_b),dim=1).int()

        if mode == 'test': 
            del coords_h,coords_w,coords_z,coords_b,plain_coords_z
            torch.cuda.empty_cache()

        # Make sparse feature
        x = SparseTensor(coords=coords, feats=feats)

        conv0 = self.input(x)

        if mode == 'test': 
            del x
            torch.cuda.empty_cache()

        conv1up = self.conv1up(conv0) # 1/2
        conv2up = self.conv2up(conv1up) # 1/4
        conv3up = self.conv3up(conv2up) # 1/8
        conv3down = conv2up+self.conv3down(conv3up) # 1/4

        if mode == 'test': 
            del conv3up
            del conv2up
            torch.cuda.empty_cache()

        conv2down = conv1up+self.conv2down(conv3down) # 1/2
        
        if mode == 'test': 
            del conv1up
            del conv3down
            torch.cuda.empty_cache()

        conv1down = conv0+self.conv1down(conv2down) # 1/1

        if mode == 'test': 
            del conv0
            del conv2down
            torch.cuda.empty_cache()

        prob = self.prob(conv1down)

        if mode == 'test': 
            del conv1down
            torch.cuda.empty_cache()

        # Convert back into dense volume
        est_prob = prob.F
        est_prob = est_prob.reshape((B,D,H,W,1))
        est_prob = est_prob.permute(0,4,1,2,3)

        return est_prob

class ViewAggregationNet_small(nn.Module):
    def __init__(self,va_feature_ch):
        super(ViewAggregationNet_small, self).__init__()

        base_ch = int(va_feature_ch/2)

        self.input0 = ConvBnReLU3D(va_feature_ch, va_feature_ch, kernel_size=3, pad=1)
        self.input1 = ConvBnReLU3D(va_feature_ch, base_ch, kernel_size=3, pad=1)

        self.conv1a = ConvBnReLU3D(base_ch, base_ch*2,stride=2, kernel_size=3, pad=1)
        self.conv1b = ConvBnReLU3D(base_ch*2, base_ch*2, kernel_size=3, pad=1)
        self.conv1c = ConvBnReLU3D(base_ch*2, base_ch*2, kernel_size=3, pad=1)

        self.conv1d = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear',align_corners=False),
            ConvBnReLU3D(base_ch*2, base_ch,stride=1, kernel_size=3, pad=1)
        )

        self.conv0 = nn.Sequential(
            ConvBnReLU3D(base_ch, base_ch, kernel_size=3, pad=1),
            nn.Conv3d(base_ch, 1, kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.sigmoid = torch.sigmoid

    def forward(self, x):

        input0 = self.input1(self.input0(x))

        conv1c = self.conv1c(self.conv1b(self.conv1a(input0)))

        conv1d = input0+self.conv1d(conv1c)

        conv0 = self.conv0(conv1d)

        sig = self.sigmoid(conv0).squeeze(1)

        vis_weight, _ = torch.max(sig,dim=1)

        vis_threshold = 0.05

        vis_weight[vis_weight<vis_threshold] = 0

        return vis_weight # [B,H,W]


class network(nn.Module):
    def __init__(self, args):
        super(network, self).__init__()
        if args.nscale == 4:
            self.feature_extractor = FeaturePyramid_FPN_4level(args.feature_ch)
            self.cost_reg_net_init = CostRegNet_v3_full(args.gwc_groups[3])
            self.cost_reg_net_refine_2 = CostRegNet_sparse_v4(args.gwc_groups[2])
            self.cost_reg_net_refine_1 = CostRegNet_sparse_v4(args.gwc_groups[1])
            self.cost_reg_net_refine_0 = CostRegNet_sparse_v4(args.gwc_groups[0])
            self.va_net_init = ViewAggregationNet_small(args.gwc_groups[3])
        elif args.nscale == 3:
            self.feature_extractor = FeaturePyramid_FPN_3level(args.feature_ch,args.init_context_ch)
            self.cost_reg_net_init = CostRegNet_v3_full(args.gwc_groups[2]+args.init_context_ch)
            self.cost_reg_net_refine_1 = CostRegNet_sparse_v4(args.gwc_groups[1])
            self.cost_reg_net_refine_0 = CostRegNet_sparse_v4(args.gwc_groups[0])
            self.va_net_init = ViewAggregationNet_small(args.gwc_groups[2])

        self.args = args

    def forward(self, ref_img, src_imgs, ref_in, src_in, ref_ex, src_ex, \
                    depth_min, depth_max,train_levels):

        outputs = {}

        self.args.actual_nsrc = src_imgs.shape[1]

        # Extract features
        features = []
        ref_feature = self.feature_extractor(ref_img)
        features.append(ref_feature)
        for idx in range(src_imgs.shape[1]):
            tmp_src_feature = self.feature_extractor(src_imgs[:,idx])
            features.append(tmp_src_feature)

        ## Scaling intrinsics for the feature pyramid:
        ref_in_multiscales = conditionIntrinsics(ref_in,ref_img.shape,[feature[0].shape for feature in features[0]])
        src_in_multiscales = []
        for i in range(self.args.actual_nsrc):
            src_in_multiscales.append(conditionIntrinsics(src_in[:,i],ref_img.shape, [feature[0].shape for feature in features[i]]))
        src_in_multiscales = torch.stack(src_in_multiscales).permute(1,0,2,3,4) # [B, nSrc, nScale, 3, 3]

        ## Iterative estimation
        hypos = [None]*self.args.nscale
        hypo_coords = [None]*self.args.nscale
        intervals = [None]*self.args.nscale
        prob_grids = [None]*self.args.nscale
        global_probs = [None]*self.args.nscale
        for level in reversed(range(self.args.nscale)):

            if level not in train_levels:
                continue

            ########## Init global search ##########
            if level == self.args.nscale-1:

                #### Make depth hypothesis ####
                B, CH, H, W = features[0][level][0].shape
                init_d = self.args.target_d[-1]
                depth_hypos, hypo_intervals, level_hypo_coord = calculate_depth_hypothesis_init(
                    self.args,
                    ref_in_multiscales[:,level],
                    src_in_multiscales[:,:,level],
                    ref_ex,
                    src_ex,
                    depth_min, 
                    depth_max, 
                    H, W,
                    init_d
                )
                if self.args.random_init_planes_offset > 0:
                    depth_hypos = depth_hypos[:,:,:self.args.target_d[-1],:,:]
                    hypo_intervals = hypo_intervals[:,:self.args.target_d[-1],:,:]
                    level_hypo_coord = level_hypo_coord[:,:,:self.args.target_d[-1],:,:]
                hypos[level] = depth_hypos
                hypo_coords[level] = level_hypo_coord
                intervals[level] = hypo_intervals.unsqueeze(1)

                #### Build cost volume ####
                cost_volume, vis_weights = proj_cost(
                    self.args,
                    features,
                    level,
                    ref_in_multiscales[:,level,:,:], 
                    src_in_multiscales[:,:,level,:,:],
                    ref_ex, 
                    src_ex[:,:],
                    hypos[level],
                    self.args.gwc_groups,
                    self.va_net_init
                )

                #### Cost aggregation ####
                occ_grid = self.cost_reg_net_init(cost_volume,hypo_coords[level])
                if self.args.mode == "test":
                    del cost_volume
                    torch.cuda.empty_cache()
                occ_grid = torch.softmax(occ_grid,dim=2)
                prob_grids[level] = occ_grid
                global_probs[level] = prob_grids[level]
                # print("maximum vRAM for level {} (MB):".format(level))
                # print(torch.cuda.max_memory_allocated()/1000000)
                # torch.cuda.reset_max_memory_allocated()

            ########## Depth refinement levels ##########
            else:
                #### Make depth hypothesis ####
                with torch.no_grad():
                    # Do top_k selection
                    selected_prob, selected_idx = torch.topk(prob_grids[level+1],k=int(self.args.target_d[level]/2),dim=2)
                    selected_hypos = torch.gather(hypos[level+1],dim=2,index=selected_idx)
                    selected_intervals = torch.gather(intervals[level+1],dim=2,index=selected_idx)
                    selected_intervals = selected_intervals/2
                    selected_coords = torch.gather(hypo_coords[level+1],dim=2,index=selected_idx)
                    # subdivide hypos
                    upper_new_hypos = selected_hypos+selected_intervals/2
                    lower_new_hypos = selected_hypos-selected_intervals/2
                    new_hypos = torch.cat((upper_new_hypos,lower_new_hypos),dim=2)
                    new_hypos = torch.repeat_interleave(new_hypos,2,dim=3)
                    new_hypos = torch.repeat_interleave(new_hypos,2,dim=4)
                    hypos[level] = new_hypos
                    # subdivide coords
                    upper_new_coords = selected_coords*2+1
                    lower_new_coords = selected_coords*2
                    new_coords = torch.cat((upper_new_coords,lower_new_coords),dim=2)
                    new_coords = torch.repeat_interleave(new_coords,2,dim=3)
                    new_coords = torch.repeat_interleave(new_coords,2,dim=4)
                    hypo_coords[level] = new_coords
                    # subdivide intervals
                    selected_intervals = torch.cat((selected_intervals,selected_intervals),dim=2) # Dx2
                    selected_intervals = torch.repeat_interleave(selected_intervals,2,dim=3) # Hx2
                    selected_intervals = torch.repeat_interleave(selected_intervals,2,dim=4) # Wx2
                    intervals[level] = selected_intervals

                #### Build cost volume ####
                cost_volume, vis_weights = proj_cost(
                    self.args,
                    features,
                    level,
                    ref_in_multiscales[:,level,:,:], 
                    src_in_multiscales[:,:,level,:,:],
                    ref_ex, 
                    src_ex[:,:],
                    hypos[level],
                    self.args.gwc_groups,
                    vis_weights=vis_weights
                )

                #### Cost aggregation ####
                if level == 0:
                    occ_grid = self.cost_reg_net_refine_0(cost_volume,hypo_coords[level],mode=self.args.mode)
                elif level == 1:
                    occ_grid = self.cost_reg_net_refine_1(cost_volume,hypo_coords[level],mode=self.args.mode)
                elif level == 2:
                    occ_grid = self.cost_reg_net_refine_2(cost_volume,hypo_coords[level],mode=self.args.mode)

                if self.args.mode == "test":
                    del cost_volume
                    torch.cuda.empty_cache()

                occ_grid = torch.softmax(occ_grid,dim=2)
                prob_grids[level] = occ_grid
                global_probs[level] = prob_grids[level]

                # print("maximum vRAM for level {} (MB):".format(level))
                # print(torch.cuda.max_memory_allocated()/1000000)
                # torch.cuda.reset_max_memory_allocated()

        ## Return
        outputs["hypos"] = hypos
        outputs["hypo_coords"] = hypo_coords
        outputs["intervals"] = intervals
        outputs["prob_grids"] = prob_grids
        outputs["global_probs"] = global_probs

        return outputs

# Loss
def sL1_loss(depth_est, depth_gt, mask):
    return F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')

################################ Submodules ##################################
class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        # self.bn = nn.SyncBatchNorm(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

class ConvBnReLU3DSparse(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, transposed=False, bias=False):
        super(ConvBnReLU3DSparse, self).__init__()
        self.conv = spnn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, bias=bias, transposed=transposed)
        self.bn = spnn.BatchNorm(out_channels)
        self.relu = spnn.ReLU(True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
        
class ConvBnReLU3DSparseFactorize(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, transposed=False, bias=False):
        super(ConvBnReLU3DSparseFactorize, self).__init__()
        self.conv1 = spnn.Conv3d(in_channels, out_channels, kernel_size=(1,1,kernel_size), stride=stride, bias=bias, transposed=transposed)
        self.conv2 = spnn.Conv3d(in_channels, out_channels, kernel_size=(1,kernel_size,1), stride=stride, bias=bias, transposed=transposed)
        self.conv3 = spnn.Conv3d(in_channels, out_channels, kernel_size=(kernel_size,1,1), stride=stride, bias=bias, transposed=transposed)
        self.bn = spnn.BatchNorm(out_channels)
        self.relu = spnn.ReLU(True)

    def forward(self, x):
        return self.relu(self.bn(self.conv3(self.conv2(self.conv1(x)))))

class ConvBn3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))
