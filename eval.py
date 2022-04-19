# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os,sys,time,argparse,datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import net
from models.modules import *
from utils import *
from PIL import Image
from argsParser import getArgsParser
from multiprocessing import Pool
from multiprocessing import cpu_count


cudnn.benchmark = True
# torch.backends.cudnn.deterministic=True
# torch.backends.cudnn.enabled = False

# Arg parser
parser = getArgsParser()
args = parser.parse_args()
assert args.mode == "test"

# dataset 
if args.dataset=="dtu":
    from dataset import dtu_generic as chosen_dataset
elif args.dataset=="eth3d":
    from dataset import dtu_eth3d_hires as chosen_dataset
elif args.dataset=="tanks":
    from dataset import dtu_tanks as chosen_dataset

settings_str = "All settings:\n"
line_width = 30
for k,v in vars(args).items(): 
    settings_str += '{0}: {1}\n'.format(k,v)
print(settings_str)

# run MVS model to save depth maps and confidence maps
def save_depth():
    # dataset, dataloader
    test_dataset = chosen_dataset.MVSDataset(args)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=args.eval_shuffle, num_workers=32, drop_last=False)

    # model
    model = net.network(args)
    if args.eval_precision == 16:
        print("************ Using half precision on eval ************")
        model = model.half()
    if args.eval_enable_dataparallel:
        model = nn.DataParallel(model) 
    model.cuda()
    model.eval()
    
    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    if len(args.loadckpt) > 0:
        state_dict = torch.load(args.loadckpt)
        if args.ckptloadmode == 'whole':
            model.load_state_dict(state_dict['model'])
        else: # load partial state_dict
            pretrained_dict = torch.load(args.loadckpt)
            model_dict = model.state_dict()
            own_state = model.state_dict()
            print("Loading partial model parameters...")
            for name, param in pretrained_dict["model"].items():
                if name not in own_state:
                    print("Skiped:"+name)
                    continue
                if isinstance(param, nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                own_state[name].copy_(param)
                print("Loading:"+name)
    else:
        print("EMPTY CKPT\nEMPTY CKPT\nEMPTY CKPT\nEMPTY CKPT\nEMPTY CKPT\n")

    train_levels = list(range(args.nscale))

    with torch.no_grad():
        RMSEs = []
        ii = 0
        
        for batch_idx, sample in enumerate(test_loader):

            sample_cuda = tocuda(sample)
            mask = sample["ref_depth_mask"]

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            start_time = time.time()

            if args.eval_precision == 32:
                outputs = model(\
                    sample_cuda["ref_img"].float(), \
                    sample_cuda["src_imgs"].float(), \
                    sample_cuda["ref_intrinsics"], \
                    sample_cuda["src_intrinsics"], \
                    sample_cuda["ref_extrinsics"], \
                    sample_cuda["src_extrinsics"], \
                    sample_cuda["depth_min"], \
                    sample_cuda["depth_max"],
                    train_levels)
            elif args.eval_precision == 16:
                outputs = model(\
                    sample_cuda["ref_img"].half(), \
                    sample_cuda["src_imgs"].half(), \
                    sample_cuda["ref_intrinsics"], \
                    sample_cuda["src_intrinsics"], \
                    sample_cuda["ref_extrinsics"], \
                    sample_cuda["src_extrinsics"], \
                    sample_cuda["depth_min"].half(), \
                    sample_cuda["depth_max"].half(),
                    train_levels)

            tmp_time = time.time()

            hypos = outputs["hypos"]
            hypo_coords = outputs["hypo_coords"]
            intervals = outputs["intervals"]
            global_probs = outputs["global_probs"] 
            prob_grids = outputs["prob_grids"]

            # Calculate confidence
            init_prob = prob_grids[-1].float()
            maximum_prob, max_prob_idx = init_prob.max(dim=2)
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(init_prob, pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
            max_sum4_prob, _ = prob_volume_sum4.max(dim=1)
            max_sum4_prob = torch.nn.functional.interpolate(max_sum4_prob.unsqueeze(1),scale_factor=8,mode='bilinear',align_corners=False).squeeze(1)

            # Final depth regression
            B,_,D,H,W = prob_grids[0].shape

            final_prob = prob_grids[0].float()
            final_hypo = hypos[0].float()
            regressed_depth = torch.sum(final_prob*final_hypo,dim=2)
            final_depth = regressed_depth[:,0].data.cpu().numpy()

            print('Iter {}/{}, time = {:.3f}, mem = {}'.format(
                batch_idx,
                len(test_loader),time.time() - start_time,
                int(torch.cuda.max_memory_allocated()/1000000)
            ))

            for sample_idx in range(B):
                filename = sample["filename"][sample_idx]
                # save depth maps and confidence maps
                depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
                confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                # save depth maps
                save_pfm(depth_filename, final_depth[sample_idx])
                write_depth_img(depth_filename+".png", final_depth[sample_idx])
                # Save prob maps
                save_pfm(confidence_filename, max_sum4_prob[sample_idx].squeeze().data.cpu().numpy())
                write_depth_img(confidence_filename+".png", max_sum4_prob[sample_idx].squeeze().data.cpu().numpy())

            del sample
            del outputs
            torch.cuda.empty_cache()

def save_pfm(filename, image, scale=1):

    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()

def write_depth_img(filename,depth):

    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    depth_min = depth.min()
    depth_max = depth.max()

    depth_normalized = (depth-depth_min)/(depth_max-depth_min)
    depth_normalized = depth_normalized*255

    image = Image.fromarray(depth_normalized).convert("L")
    image.save(filename)
    return 1

if __name__ == '__main__':
    save_depth()