# Data io utilities for the dataloader
# From https://github.com/JiayuYANG/CVP-MVSNet
# by: Jiayu Yang
# date: 2019-07-31

import numpy as np
import re
import sys
from PIL import Image
import os, errno

def readScanList(scal_list_file,mode):
    print("Reading scan list...")
    scan_list_f = open(scal_list_file, "r")
    scan_list = scan_list_f.read()
    scan_list = scan_list.split()
    scan_list_f.close()
    print("Done, Using following scans for "+mode+":\n"+str(scan_list))
    return scan_list

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

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

def read_cam_file(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    depth_min = float(lines[11].split()[0])
    depth_interval = float(lines[11].split()[1])
    depth_max = depth_min+(256*depth_interval)
    return intrinsics, extrinsics, depth_min, depth_max

def read_cam_file_with_size(filename,imgsize):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    if imgsize != 1200:
        if imgsize not in [128,256,512,1024]:
            intrinsice_down_factor = 1200/imgsize
            intrinsics[:2, :] /= intrinsice_down_factor

    depth_min = float(lines[11].split()[0])
    depth_interval = float(lines[11].split()[1])
    depth_max = depth_min+(256*depth_interval)
    return intrinsics, extrinsics, depth_min, depth_max

def write_cam(filename, intrinsic, extrinsic, depth_min, depth_max):
    with open(filename, 'w') as f:
        f.write('extrinsic\n')
        for j in range(4):
            for k in range(4):
                f.write(str(extrinsic[j, k]) + ' ')
            f.write('\n')
        f.write('\nintrinsic\n')
        for j in range(3):
            for k in range(3):
                f.write(str(intrinsic[j, k]) + ' ')
            f.write('\n')
        f.write('\n%f %f\n' % (depth_min,depth_max))

def read_img_with_size(filename,imgsize):
    img = Image.open(filename)

    if imgsize != 1200: # input image does not match image size we want
        if imgsize in [128,256,512,1024]:
            new_size = [int(imgsize*(5/4)),imgsize]
            img = img.resize((new_size),Image.BILINEAR)
        else:
            new_size = [int(imgsize*(4/3)),imgsize]
            img = img.resize((new_size),Image.BILINEAR)

    # scale 0~255 to 0~1
    img = np.array(img, dtype=np.float32) / 255.

    if img.shape[0] == 1200:
        img = img[:1152,:,:]

    if img.shape[0] == 600:
        img = img[:592,:,:]

    if img.shape[0] == 300:
        img = img[:288,:,:]

    if img.shape[0] == 150:
        img = img[:144,:,:]

    # print(img.shape)

    return img


def write_img(filename,image):

    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    image.save(filename)
    return 1

def read_mask(filename):
    mask = Image.open(filename)
    mask = np.array(mask, dtype=np.bool)
    return mask

def write_mask(filename,mask):

    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
                
    mask = Image.fromarray(mask)
    mask.save(filename)
    return 1

def read_depth(filename):
    # read pfm depth file
    return np.array(read_pfm(filename)[0], dtype=np.float32)

def read_depth_with_size(filename,imgsize):
    # read pfm depth file
    depth = np.array(read_pfm(filename)[0], dtype=np.float32)

    if imgsize != 1200:
        if imgsize in [128,256,512,1024]:
            depth = Image.fromarray(depth)
            new_size = [int(imgsize*(5/4)),imgsize]
            depth = depth.resize((new_size),Image.BILINEAR)
            depth = np.array(depth, dtype=np.float32)
        else:
            depth = Image.fromarray(depth)
            new_size = [int(imgsize*(4/3)),imgsize]
            depth = depth.resize((new_size),Image.BILINEAR)
            depth = np.array(depth, dtype=np.float32)

    return depth

