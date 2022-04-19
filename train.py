# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os,sys,time,logging,datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import dtu_generic
from dataset import blendedmvs
from models import net
from models.modules import getWhiteMask
from utils import *
from argsParser import getArgsParser
import torch.utils
import torch.utils.checkpoint

# DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# AMP
from torch.cuda.amp import autocast, GradScaler

# Ignore pytorch futurewarnings
import warnings
warnings.filterwarnings('ignore')

# Arg parser
parser = getArgsParser()
args = parser.parse_args()

# Setup torch
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.benchmark=True

# Setup DDP
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')
device = torch.device("cuda", local_rank)

# Checkpoint directory
if dist.get_rank() == 0:
    if not os.path.exists(args.logckptdir+args.info.replace(" ","_")):
        try:
            os.makedirs(args.logckptdir+args.info.replace(" ","_"))
        except OSError as error:
            print("Log directory exists.")

settings_str = "\n******************** Settings ********************\n"
line_width = 30
for k,v in vars(args).items(): 
    settings_str += '{0}: {1}\n'.format(k,v)
print(settings_str)
print("**************************************************\n")

# Summary writter
sw_path = args.logckptdir+args.info.replace(" ","_")

# Dataset
if args.dataset == 'dtu':
    train_dataset = dtu_generic.MVSDataset(args)
elif args.dataset == 'blendedmvs':
    train_dataset = blendedmvs.MVSDataset(args)
elif args.dataset == 'blendedmvspp':
    train_dataset = blendedmvs.MVSDataset(args)

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, args.batch_size, num_workers=4, drop_last=False, sampler=train_sampler, pin_memory=True)

# Network
model = net.network(args)
print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
model = model.to(device)
model.train()

# Loss
if args.loss_function == 'BCE':
    BCELoss = torch.nn.BCELoss(reduction='none').to(local_rank)
elif args.loss_function == 'KL':
    KLDivLoss = torch.nn.KLDivLoss(size_average=None, reduce=None,reduction='none',log_target=False).to(local_rank)
regression_loss = net.sL1_loss

optimizer = optim.Adam(model.parameters(), lr = args.lr, betas=(0.9, 0.999), weight_decay=args.wd)

# Load ckpt
if dist.get_rank() == 0:
    if (args.mode == "train" and args.resume) or (args.mode == "test" and not args.loadckpt):
        print("Resuming or testing...")
        saved_models = [fn for fn in os.listdir(sw_path) if fn.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))

        # use the latest checkpoint file
        loadckpt = os.path.join(sw_path, saved_models[-1])
        print("Resuming "+loadckpt)
        state_dict = torch.load(loadckpt)
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1
    elif args.loadckpt:
        # load checkpoint file specified by args.loadckpt
        print("loading model {}".format(args.loadckpt))
        state_dict = torch.load(args.loadckpt)
        model.load_state_dict(state_dict['model'],strict=False)

# load network parameters
start_epoch = 0

# DDP
model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

# AMP
if args.amp:
    print("##### AMP Enabled #####")
    scaler = GradScaler()

# Start training
print("start at epoch {}".format(start_epoch))

# main function
def train():
    milestones = [int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma,
                                                        last_epoch=start_epoch - 1)

    last_loss = None
    this_loss = None
    for epoch_idx in range(start_epoch, args.epochs):
        print('Epoch {}:'.format(epoch_idx))
        global_step = len(train_loader) * epoch_idx
        train_loader.sampler.set_epoch(epoch_idx)

        if last_loss is None:
            last_loss = 999999
        else:
            last_loss = this_loss
        this_loss = []

        ii=0
        optimizer.zero_grad()

        # Set levels for training
        train_levels = []
        for level in range(args.nscale):
            if global_step >= args.activate_level_itr[level]:
                train_levels.append(level)
        print("train_levels",train_levels)

        for batch_idx, sample in enumerate(train_loader):

            if batch_idx > 30:
                break

            start_time = time.time()
            global_step = len(train_loader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0

            loss = train_sample(sample,train_levels)

            if loss == -1:
                continue
            this_loss.append(loss)

            if ii%1 == 0:
                print('Epoch {}/{}, Iter {}/{}, train loss = {:.8f}, time = {:.3f}'.format(epoch_idx, args.epochs, batch_idx,
                                                                                        len(train_loader), loss,
                                                                                        time.time() - start_time))
        
            ii+=1
            if ii%100 == 0:
                if dist.get_rank() == 0:
                    torch.save({
                        'epoch': epoch_idx,
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict()},
                        "{}/model_{:0>6}.ckpt".format(args.logckptdir+args.info.replace(" ","_"), epoch_idx))
                    print("partial model_{:0>6}.ckpt saved".format(epoch_idx))

        # checkpoint
        if dist.get_rank() == 0:
            if (epoch_idx + 1) % args.save_freq == 0:
                torch.save({
                    'epoch': epoch_idx,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    "{}/model_{:0>6}.ckpt".format(args.logckptdir+args.info.replace(" ","_"), epoch_idx))
                print("model_{:0>6}.ckpt saved".format(epoch_idx))
        this_loss = np.mean(this_loss)
        print("Epoch loss: {:.5f} --> {:.5f}".format(last_loss, this_loss))

        lr_scheduler.step()

def train_sample(sample, train_levels=None):

    optimizer.zero_grad()

    if train_levels == None:
        train_levels = list(reversed(range(args.nscale)))

    sample_cuda = tocuda(sample)
    ref_depth = sample_cuda["ref_depth"]

    with autocast(enabled=args.amp):
        outputs = model(\
            sample_cuda["ref_img"].float(), \
            sample_cuda["src_imgs"].float(), \
            sample_cuda["ref_intrinsics"], \
            sample_cuda["src_intrinsics"], \
            sample_cuda["ref_extrinsics"], \
            sample_cuda["src_extrinsics"], \
            sample_cuda["depth_min"], \
            sample_cuda["depth_max"], \
            train_levels)

    hypos = outputs["hypos"]
    hypo_coords = outputs["hypo_coords"]
    intervals = outputs["intervals"]
    global_probs = outputs["global_probs"] 
    prob_grids = outputs["prob_grids"]

    loss = []
    loss_level_weights = [1,1,1,1]

    # Calculate edge mask
    down_gt = F.interpolate(ref_depth.unsqueeze(1),scale_factor=0.5,mode='bilinear',align_corners=False,recompute_scale_factor=False)
    down_up_gt = F.interpolate(down_gt,scale_factor=2,mode='bilinear',align_corners=False,recompute_scale_factor=False)
    res = torch.abs(ref_depth.unsqueeze(1)-down_up_gt)
    high_frequency_mask = res>(0.001*(sample_cuda["depth_max"]-sample_cuda["depth_min"])[:,None,None,None])
    valid_gt_mask = (-F.max_pool2d(-ref_depth.unsqueeze(1),kernel_size=5,stride=1,padding=2))>sample_cuda["depth_min"][:,None,None,None]
    high_frequency_mask = high_frequency_mask * valid_gt_mask

    # Compute white mask
    ref_img = sample_cuda["ref_img"]
    if args.dataset == 'dtu':
        white_desk_mask = sample_cuda["ref_img"].sum(dim=1) < 3

    for level in reversed(range(args.nscale)):

        if level not in train_levels:
            continue

        if level ==0:
            # Apply softargmax depth regression for subpixel depth estimation on final level.
            B,_,D,H,W = prob_grids[level].shape

            final_prob = prob_grids[level]
            final_hypo = hypos[level]
            regressed_depth = torch.sum(final_prob*final_hypo,dim=2)
            gt_depth = ref_depth.unsqueeze(1)

            mask = (-F.max_pool2d(-ref_depth.unsqueeze(1),kernel_size=5,stride=1,padding=2))>sample_cuda["depth_min"][:,None,None,None]

            if args.dataset == 'dtu_ours':
                mask = mask * white_desk_mask.unsqueeze(1)

            tmp_loss = F.smooth_l1_loss(regressed_depth[mask], gt_depth[mask], reduction='none')

            tmp_high_frequency_mask = high_frequency_mask[mask]
            tmp_high_frequency_weight = tmp_high_frequency_mask.float().mean()
            weight = (1-tmp_high_frequency_weight)*tmp_high_frequency_mask + (tmp_high_frequency_weight)*(~tmp_high_frequency_mask)
            if args.final_edge_mask > 0:
                tmp_loss *= weight
            tmp_loss *= args.final_weight
            loss.append(tmp_loss.mean())

            if args.final_continue > 0:
                continue

        B,_,D,H,W = prob_grids[level].shape

        # Create gt labels
        unfold_kernel_size = int(2**level)
        assert unfold_kernel_size%2 == 0 or unfold_kernel_size == 1
        unfolded_patch_depth = torch.nn.functional.unfold(ref_depth.unsqueeze(1),unfold_kernel_size,dilation=1,padding=0,stride=unfold_kernel_size)
        unfolded_patch_depth = unfolded_patch_depth.reshape(B,1,unfold_kernel_size**2,H,W)
        # valid gt depth mask
        mask = (unfolded_patch_depth>sample_cuda["depth_min"].view((B,1,1,1,1))).all(dim=2)
        mask *= (unfolded_patch_depth<sample_cuda["depth_max"].view((B,1,1,1,1))).all(dim=2)
        # Apply white area mask
        if args.dataset == 'dtu':
            down_white_mask = (-F.max_pool2d(-white_desk_mask.float(),kernel_size=unfold_kernel_size).unsqueeze(1)).bool()
            mask *= down_white_mask
        
        # Approximate depth distribution from depth observations
        gt_occ_grid = torch.zeros_like(hypos[level])
        if args.gt_prob_mode == "hard":
            for pixel in range(unfolded_patch_depth.shape[2]):
                selected_depth = unfolded_patch_depth[:,:,pixel]
                distance_to_hypo = abs(hypos[level]-selected_depth.unsqueeze(2))
                occupied_mask = distance_to_hypo<=(intervals[level]/2)
                gt_occ_grid[occupied_mask]+=1
            gt_occ_grid = gt_occ_grid/gt_occ_grid.sum(dim=2,keepdim=True)
            gt_occ_grid[torch.isnan(gt_occ_grid)] = 0
        elif args.gt_prob_mode == "soft":
            for pixel in range(unfolded_patch_depth.shape[2]):
                selected_depth = unfolded_patch_depth[:,:,pixel]
                distance_to_hypo = abs(hypos[level]-selected_depth.unsqueeze(2))
                distance_to_hypo /= intervals[level]
                mask = distance_to_hypo>1
                weights = 1-distance_to_hypo
                weights[mask] = 0
                gt_occ_grid+=weights
            gt_occ_grid = gt_occ_grid/gt_occ_grid.sum(dim=2,keepdim=True)
            gt_occ_grid[torch.isnan(gt_occ_grid)] = 0

        covered_mask = gt_occ_grid.sum(dim=2,keepdim=True) > 0
        occ_hypos_count = (gt_occ_grid>0).sum(dim=2,keepdim=True).repeat(1,1,D,1,1)
        edge_weight = occ_hypos_count
        final_mask = mask.unsqueeze(2) * covered_mask

        # Choose loss
        if args.loss_function == 'BCE':
            est = torch.masked_select(prob_grids[level],final_mask)
            gt = torch.masked_select(gt_occ_grid,final_mask)
            tmp_loss = BCELoss(est,gt)
            edge_weight = torch.masked_select(edge_weight,final_mask)
            # Apply edge weight
            tmp_loss = tmp_loss * edge_weight
            # class balance
            num_positive = (gt>0).sum()
            num_negative = (gt==0).sum()
            num_total = gt.shape[0]
            alpha_positive = num_negative/float(num_total)
            alpha_negative = num_positive/float(num_total)
            weight = alpha_positive*(gt>0) + alpha_negative*(gt==0)
            tmp_loss = weight*tmp_loss
            tmp_loss = tmp_loss.mean()
            tmp_loss = loss_level_weights[level]*tmp_loss
            loss.append(tmp_loss)
        elif args.loss_function == 'KL':
            est = torch.masked_select(prob_grids[level],final_mask)
            gt = torch.masked_select(gt_occ_grid,final_mask)
            tmp_loss = KLDivLoss(est.log(),gt)
            edge_weight = torch.masked_select(edge_weight,final_mask)
            # Apply edge weight
            tmp_loss = tmp_loss * edge_weight
            tmp_loss = tmp_loss.mean()
            tmp_loss = loss_level_weights[level]*tmp_loss
            loss.append(tmp_loss)

    loss = torch.stack(loss).mean()

    if args.amp:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    return loss.data.cpu().item()


if __name__ == '__main__':
    train()
