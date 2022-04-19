# Fetching file path and name for dataloader on our DTU dataset. 
# From https://github.com/JiayuYANG/CVP-MVSNet
# by: Jiayu Yang
# date: 2019-08-01

import os

# DTU:
def getScanListFile(data_root,mode):
    scan_list_file = data_root+"scan_list_"+mode+".txt"
    return scan_list_file

def getPairListFile(data_root,mode,selection="next"):
    if selection == "next":
        pair_list_file = data_root+"Cameras/pair_next.txt"
    elif selection == "mvsnet":
        pair_list_file = data_root+"Cameras/pair.txt"
    else:
        pair_list_file = None
    return pair_list_file

def getDepthFile(data_root,mode,scan,view):
    depth_name = "depth_map_"+str(view).zfill(4)+".pfm"
    scan_path = "Depths/"+scan+"_train/"
    depth_file = os.path.join(data_root,scan_path,depth_name)
    return depth_file

def getImageFile(data_root,mode,scan,view,light):
    image_name = "rect_"+str(view+1).zfill(3)+"_"+str(light)+"_r5000.png"
    scan_path = "Rectified/"+scan+"_train/"
    image_file = os.path.join(data_root,scan_path,image_name)
    return image_file

def getCameraFile(data_root,mode,view):
    cam_name = str(view).zfill(8)+"_cam.txt"
    cam_path = "Cameras/"
    cam_file = os.path.join(data_root,cam_path,cam_name)
    return cam_file