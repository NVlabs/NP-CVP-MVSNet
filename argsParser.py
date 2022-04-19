# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse

def getArgsParser():
    parser = argparse.ArgumentParser(description='NP-CVP-MVSNet')
    
    # General settings
    parser.add_argument('--info', default='None', help='Info about current run')
    parser.add_argument('--mode', default='train', help='train or test ro validation', choices=['train', 'test', 'val'])
    
    # Data settings
    parser.add_argument('--dataset', default='dtu', help='select datareader')
    parser.add_argument('--dataset_root', help='path to dataset root')
    parser.add_argument('--imgsize', type=int, default=128, help='image size parameter for training, to be deprecated.')
    parser.add_argument('--depth_h', type=int, default=1184, help='height of depth map to be estimate')
    parser.add_argument('--depth_w', type=int, default=1600, help='width of depth map to be estimate')
    parser.add_argument('--vselection', default='next', help='view selection', choices=['next', 'mvsnet'])
    parser.add_argument('--nsrc', type=int, default=1, help='number of src views per ref view')
    parser.add_argument('--nbadsrc', type=int, default=0, help='number of src views that are intended to be far away from current view')
    parser.add_argument('--random_src', type=int, default=0, help='number of src views that are randomly sampled from all possible views.')
    parser.add_argument('--nscale', type=int, default=5, help='number of scales to use')
    parser.add_argument('--gtdepth', type=int, default=1, help='require ground truth depth')
    parser.add_argument('--refine_gwc_groups', type=int, default=4, help='number of groups for gwc')
    parser.add_argument('--min_depth', type=float, default=-1, help='override min depth')
    parser.add_argument('--max_depth', type=float, default=-1, help='override max depth')
    parser.add_argument('--random_light', type=int, default=0, help='use random light index for src views')
    parser.add_argument('--random_init_planes_offset', type=int, default=0, help='random offset for number of init planes')
    
    # Network settings
    parser.add_argument('--init_context_ch', type=int, default=0, help='size for init context feature')
    parser.add_argument('--feature_ch', nargs='+', type=int, default=[8,16,32,64], help='feature channel size for each level')
    parser.add_argument('--va_feature_ch', type=int, default=4, help='view aggregation feature channel size')
    parser.add_argument('--gwc_groups', nargs='+', type=int, default=[2,4,4,8], help='number of gwc groups for each level')
    parser.add_argument('--costmetric', default='l1', help='error metric for building cost volume', choices=['variance','weighted_variance','gwc_weighted_max','gwc_weighted_sum'])

    # Geometry settings
    parser.add_argument('--init_search_mode', type=str, default='uniform', help='depth searching mode for depth init', choices=['uniform'])
    parser.add_argument('--target_d', nargs='+', type=int, default=[8,16,32,96], help='number of depth hypothesis for each level')

    # Training settings
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--amp", dest='amp', action='store_true', help='Enable AMP training')
    parser.add_argument('--epochs', type=int, default=16, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lrepochs', type=str, default="10,12,14:2", help='epoch ids to downscale lr and the downscale rate')
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=1, help='batchsize')
    parser.add_argument('--summary_freq', type=int, default=1, help='print and summary frequency')
    parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
    parser.add_argument('--loss_function', default='BCE', help='which loss function to use')
    parser.add_argument('--activate_level_itr', nargs='+', type=int, default=[0,0,0,0], help='Activate training of each level at certain epoch')
    parser.add_argument('--final_edge_mask', type=int, default=1, help='apply edge weight on final estimation')
    parser.add_argument('--final_weight', type=float, default=0.1, help='loss weight for final depth estimation')
    parser.add_argument('--final_continue', type=int, default=0, help='skip probability supervision on final scale')
    parser.add_argument('--gt_prob_mode', type=str, default="soft", help='how to generate gt probability distribution')

    # Checkpoint settings
    parser.add_argument('--loadckpt', type=str, default='', help='load a specific checkpoint')
    parser.add_argument('--logckptdir', default='./checkpoints/debug', help='the directory to save checkpoints/logs')
    parser.add_argument('--resume', type=int, default=0, help='continue to train the model')
    parser.add_argument('--ckptloadmode', default='whole', help='ckpt loading mode', choices=['whole', 'partial'])
    
    # Evaluation settings 
    parser.add_argument('--outdir', default='./outputs/debug/', help='the directory to save depth outputs')
    parser.add_argument('--eval_read_gt_depth', type=int, default=0)
    parser.add_argument('--eval_visualizeDepth', type=int, default=0)
    parser.add_argument('--eval_prob_filtering', type=int, default=0)
    parser.add_argument('--eval_prob_threshold', type=float, default=0.99)
    parser.add_argument('--eval_shuffle', type=int, default=0)
    parser.add_argument('--eval_precision', type=int, default=32)
    parser.add_argument('--eval_scan_skip', type=int, default=0)
    parser.add_argument('--eval_view_skip', type=int, default=0)
    parser.add_argument('--eval_only_this_scene', type=int, default=-1)
    parser.add_argument('--eval_enable_dataparallel', type=int, default=0)

    return parser