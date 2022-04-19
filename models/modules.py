# Modules used in the depth estimation network
# From: https://github.com/JiayuYANG/CVP-MVSNet
# by: Jiayu Yang

import numpy as np
np.seterr(all='raise')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

def conditionIntrinsics(intrinsics,img_shape,fp_shapes):
    # Calculate downsample ratio for each level of feture pyramid
    down_ratios = []
    for fp_shape in fp_shapes:
        down_ratios.append(img_shape[2]/fp_shape[2])

    # condition intrinsics
    intrinsics_out = []
    for down_ratio in down_ratios:
        intrinsics_tmp = intrinsics.clone()
        intrinsics_tmp[:, :2, :] = intrinsics_tmp[:, :2, :] / down_ratio
        intrinsics_out.append(intrinsics_tmp)

    return torch.stack(intrinsics_out).permute(1,0,2,3) # [B, nScale, 3, 3]

def calculate_depth_hypothesis_init(args, ref_in,src_in,ref_ex,src_ex,depth_min, depth_max, img_height, img_width, nhypothesis_init):

    mode = args.init_search_mode

    batchSize = ref_in.shape[0]
    depth_range = depth_max-depth_min

    if mode == 'uniform':
        depth_hypos = torch.zeros((batchSize,nhypothesis_init),device=ref_in.device)
        for b in range(0,batchSize):
            depth_hypos[b] = torch.linspace(depth_min[b],depth_max[b],steps=nhypothesis_init,device=ref_in.device)
        depth_hypos = depth_hypos.unsqueeze(2).unsqueeze(3).repeat(1,1,img_height,img_width)

        # Make coordinate for depth hypothesis, to be used by sparse convolution.
        depth_hypo_coords = torch.zeros((batchSize,nhypothesis_init),device=ref_in.device)
        for b in range(0,batchSize):
            depth_hypo_coords[b] = torch.linspace(0,nhypothesis_init-1,steps=nhypothesis_init,device=ref_in.device)
        depth_hypo_coords = depth_hypo_coords.unsqueeze(2).unsqueeze(3).repeat(1,1,img_height,img_width)

    # Calculate hypothesis interval
    hypo_intervals = depth_hypos[:,1:]-depth_hypos[:,:-1]
    hypo_intervals = torch.cat((hypo_intervals,hypo_intervals[:,-1].unsqueeze(1)),dim=1)

    return depth_hypos.unsqueeze(1), hypo_intervals, depth_hypo_coords.unsqueeze(1)

def homography_warping(src_feature, ref_in, src_in, ref_ex, src_ex, depth_hypos):

    batch, channels = src_feature.shape[0], src_feature.shape[1]
    num_depth = depth_hypos.shape[1]
    height, width = src_feature.shape[2], src_feature.shape[3]

    with torch.no_grad():
        src_proj = torch.matmul(src_in,src_ex[:,0:3,:])
        ref_proj = torch.matmul(ref_in,ref_ex[:,0:3,:])
        last = torch.tensor([[[0,0,0,1.0]]],dtype=src_proj.dtype).repeat(len(src_in),1,1).cuda()
        src_proj = torch.cat((src_proj,last),1)
        ref_proj = torch.cat((ref_proj,last),1)

        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_feature.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_feature.device)],
                               indexing='ij')
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_hypos.view(batch, 1, num_depth,1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    grid = grid.type(src_feature.dtype)

    warped_src_fea = F.grid_sample(src_feature, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros',align_corners=False)
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea

def proj_cost(args,features,level,ref_in,src_in,ref_ex,src_ex,depth_hypos,gwc_groups,va_net=None,vis_weights=None):

    depth_hypos = depth_hypos.squeeze(1)

    B,fCH,H,W = features[0][level][0].shape
    num_depth = depth_hypos.shape[1]
    nSrc = len(features)-1

    vis_weight_list = []

    if args.costmetric == "gwc_weighted_sum":

        ref_volume = features[0][level][0].unsqueeze(2).repeat(1,1,num_depth,1,1)
        

        cost_volume = None
        reweight_sum = None

        for src in range(nSrc):

            with torch.no_grad():
                with autocast(enabled=False):
                    src_proj = torch.matmul(src_in[:,src,:,:],src_ex[:,src,0:3,:])
                    ref_proj = torch.matmul(ref_in,ref_ex[:,0:3,:])
                    last = torch.tensor([[[0,0,0,1.0]]]).repeat(len(src_in),1,1).cuda()
                    src_proj = torch.cat((src_proj,last),1)
                    ref_proj = torch.cat((ref_proj,last),1)

                    proj = torch.matmul(src_proj, torch.inverse(ref_proj))
                    rot = proj[:, :3, :3]  # [B,3,3]
                    trans = proj[:, :3, 3:4]  # [B,3,1]

                    y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=ref_volume.device),
                                        torch.arange(0, W, dtype=torch.float32, device=ref_volume.device)],
                                        indexing='ij')
                    y, x = y.contiguous(), x.contiguous()
                    y, x = y.view(H * W), x.view(H * W)
                    xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
                    xyz = torch.unsqueeze(xyz, 0).repeat(B, 1, 1)  # [B, 3, H*W]
                    rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]

                    rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_hypos.view(B, 1, num_depth,H*W)  # [B, 3, Ndepth, H*W]
                    proj_xyz = rot_depth_xyz + trans.view(B, 3, 1, 1)  # [B, 3, Ndepth, H*W]
                    proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
                    proj_x_normalized = proj_xy[:, 0, :, :] / ((W - 1) / 2) - 1
                    proj_y_normalized = proj_xy[:, 1, :, :] / ((H - 1) / 2) - 1
                    proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
                    grid = proj_xy

            
            grid = grid.type(ref_volume.dtype)
            src_feature = features[src+1][level][0]
            warped_src_fea = F.grid_sample(src_feature, grid.view(B, num_depth * H, W, 2), mode='bilinear',
                                        padding_mode='zeros',align_corners=False)

            warped_src_fea = warped_src_fea.view(B, fCH, num_depth, H, W)

            two_view_cost_volume = groupwise_correlation(warped_src_fea, ref_volume, gwc_groups[level]) #B,C,D,H,W

            # Estimate visability weight for init level
            if va_net is not None:
                B,C,D,H,W = warped_src_fea.shape
                reweight = va_net(two_view_cost_volume) #B, H, W
                vis_weight_list.append(reweight)
                reweight = reweight.unsqueeze(1).unsqueeze(2) #B, 1, 1, H, W
                two_view_cost_volume = reweight*two_view_cost_volume

            # Use estimated visability weights for refine levels
            elif vis_weights is not None:
                reweight = vis_weights[src].unsqueeze(1)
                if reweight.shape[2] < two_view_cost_volume.shape[3]:
                    reweight = F.interpolate(reweight,scale_factor=2,mode='bilinear',align_corners=False)
                vis_weight_list.append(reweight.squeeze(1))
                reweight = reweight.unsqueeze(2)
                two_view_cost_volume = reweight*two_view_cost_volume

            if cost_volume == None:
                cost_volume = two_view_cost_volume
                reweight_sum = reweight
            else:
                cost_volume = cost_volume + two_view_cost_volume
                reweight_sum = reweight_sum + reweight

            if args.mode=="test":
                del src_feature
                del two_view_cost_volume
                del warped_src_fea
                del reweight
                torch.cuda.empty_cache()

        cost_volume = cost_volume/(reweight_sum+0.00001)

        if features[0][level][1] is not None:
            ref_context_volume = features[0][level][1].unsqueeze(2).repeat(1,1,num_depth,1,1)
            cost_volume = torch.cat((cost_volume, ref_context_volume),dim=1)

    else:
        print("Error! Invalid cost metric!")
        pdb.set_trace()

    return cost_volume, vis_weight_list

def groupwise_correlation(v1, v2, num_groups):
    B, C, D, H, W = v1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost_volume = (v1 * v2).view([B, num_groups, channels_per_group,D, H, W]).mean(dim=2)
    assert cost_volume.shape == (B, num_groups, D, H, W)
    return cost_volume

def getWhiteMask(ref_img, nscale):
    # downsampling image
    nScale = nscale
    nBatch = ref_img.shape[0]
    width = ref_img.shape[2]
    height = ref_img.shape[3]

    imgPyramid = []
    maskPyramid = []
    down2nearest = nn.Upsample(scale_factor=0.5, mode='nearest')
    for scale in range(nScale):
        # Add current scale of image into the pyramid
        imgPyramid.append(ref_img)
        # Generate mask
        mask_R = ref_img[:,0,:,:]==1
        mask_G = ref_img[:,1,:,:]==1
        mask_B = ref_img[:,2,:,:]==1

        mask_RGB = mask_R & mask_G & mask_B
        mask_not_white = torch.bitwise_not(mask_RGB)
        maskPyramid.append(mask_not_white)

        showMasks = 0
        if showMasks == 1:
            plt.figure()
            plt.imshow(ref_img[10,:,:,:].permute(1,2,0).data.cpu().numpy())
            plt.figure()
            plt.imshow(mask_not_white[10,:,:].data.cpu().numpy())
            plt.show()
            pdb.set_trace()
        # Downsample the image using nearest method 
        ref_img = down2nearest(ref_img)

    return maskPyramid