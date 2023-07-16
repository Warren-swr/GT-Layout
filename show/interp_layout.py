"""
    a test script for box-shape free generation
"""

import os
import sys
import shutil
import numpy as np
import torch
import utils
import vis_utils_layout as vis_utils
from data_layout import LayoutDataset, Tree
import model_layout as model

sys.setrecursionlimit(5000) # this code uses recursion a lot for code simplicity

root_dir = '/home/weiran/Projects/RvNN-Layout/GT-Layout/magazine/logs/magazine_2.5K/'
data_dir = '/home/weiran/Projects/RvNN-Layout/data/magazine-ours/magazine_2.5K/'

num_interp = 11
shape_id1 = '3_layout'
shape_id2 = '963_layout'

# load train config
conf = torch.load(root_dir + 'conf.pth')

# load object category information
Tree.load_category_info(conf.category)

# set up device
device = torch.device(conf.device)
print(f'Using device: {conf.device}')

# check if eval results already exist. If so, delete it. 
out_dir = root_dir + 'interp/interped_%s_%s' % (shape_id1, shape_id2)

if os.path.exists(out_dir):
    shutil.rmtree(out_dir)

# create a new directory to store eval results
os.mkdir(out_dir)

# create models
# we disable probabilistic because we do not need to jitter the decoded z during inference
encoder = model.RecursiveEncoder(conf, variational=True, probabilistic=False)
decoder = model.RecursiveDecoder(conf)

print('Loading ckpt net_encoder.pth')
data_to_restore = torch.load(root_dir + '/ckpts/net_encoder.pth')
encoder.load_state_dict(data_to_restore, strict=True)
print('DONE\n')
print('Loading ckpt net_decoder.pth')
data_to_restore = torch.load(root_dir + '/ckpts/net_decoder.pth')
decoder.load_state_dict(data_to_restore, strict=True)
print('DONE\n')

# send to device
encoder.to(device)
decoder.to(device)

# set models to evaluation mode
encoder.eval()
decoder.eval()

# globally interpolate shapes
with torch.no_grad():

    # load the two shapes as the inputs
    obj1 = LayoutDataset.load_object(os.path.join(data_dir, shape_id1 + '.json'))
    obj1.to(device)
    obj2 = LayoutDataset.load_object(os.path.join(data_dir, shape_id2 + '.json'))
    obj2.to(device)

    # store interpolated results for visuals
    obj_arr_outs = []
    obj_rel_outs = []
    obj_abs_outs = []

    # STUDENT CODE START
    # feed through the encoder to get two codes z1 and z2
    z1 = encoder.encode_structure(obj1)
    z2 = encoder.encode_structure(obj2)

# create a forloop looping 0, 1, 2, ..., num_interp - 1, num_interp
# interpolate the feature so that the first feature is exactly z1 and the last is exactly z2
for i in range(num_interp+1):
    alpha = i / num_interp
    code = (1 - alpha) * z1 + alpha * z2
    
    # infer through the decoder to get the iterpolate output
    # set maximal tree depth to conf.max_tree_depth
    obj_arr = decoder.decode_structure(z=code, max_depth=conf.max_tree_depth)

    utils.arr_layout(obj_arr.root, isRoot=True)
    utils.arr_layout(obj_arr.root, isRoot=True)
    obj_arr.get_arrbox()
    
    # add to the list obj_outs
    obj_arr_outs.append(obj_arr)

obj_names = []
for i in range(num_interp+1):
    obj_names.append('interp-%d'%i)

    # output the hierarchy
    with open(os.path.join(out_dir, 'step-%d.txt'%i), 'w') as fout:
        fout.write(str(obj_arr_outs)+'\n\n')

# output the assembled box-shape
vis_utils.draw_partnet_objects(obj_arr_outs, object_names=obj_names, \
        out_fn=os.path.join(out_dir, 'interp_figs_arr.png'), figsize=(66, 6), \
        leafs_only=False, sem_colors_filename='./part_colors_magazine.txt')