import os
import sys
import shutil
import numpy as np
import torch
import utils
import vis_utils_layout as vis_utils
from data_layout import LayoutDataset, Tree
import model_layout as model
import tqdm

sys.setrecursionlimit(5000) 

num_gen = 300
exp = 'magazine_2.5K'
category = 'magazine'
checkpoint = ''
device = 'cuda:0'

path = '/home/weiran/Projects/RvNN-Layout/GT-Layout/' + category + '/logs/' + exp

# load train config
conf = torch.load(path + '/conf.pth')

# load object category information
Tree.load_category_info(conf.category)

# set up device
print(f'Using device: {conf.device}')

gen_dir = path + '/generation/'
gen_dir_arr = path + '/generation-arr/'

# create models
encoder = model.RecursiveEncoder(conf, variational=True, probabilistic=False)
decoder = model.RecursiveDecoder(conf)

# load the pretrained models
print('Loading ckpt pretrained_encoder.pth')
data_to_restore = torch.load(path + '/ckpts/' + checkpoint + 'net_encoder.pth')
encoder.load_state_dict(data_to_restore, strict=True)
print('DONE\n')
print('Loading ckpt pretrained_decoder.pth')
data_to_restore = torch.load(path + '/ckpts/' + checkpoint + 'net_decoder.pth')
decoder.load_state_dict(data_to_restore, strict=True)
print('DONE\n')

# send to device
encoder.to(device)
decoder.to(device)

# set models to evaluation mode
encoder.eval()
decoder.eval()

if os.path.exists(gen_dir):
    shutil.rmtree(gen_dir)
os.mkdir(gen_dir)

if os.path.exists(gen_dir_arr):
    shutil.rmtree(gen_dir_arr)
os.mkdir(gen_dir_arr)

# generate shapes
with torch.no_grad():
    for i in tqdm.tqdm(range(num_gen)):
        # get a Gaussian noise
        code = torch.randn(1, conf.feature_size).to(device)
        
        # infer through the model to get the generated hierarchy
        # set maximal tree depth to conf.max_tree_depth
        obj_arr = decoder.decode_structure(z=code, max_depth=conf.max_tree_depth)

        obj_arr.get_arrbox()
        
        # output the hierarchy
        with open(os.path.join(gen_dir, 'gen-%03d.txt'%i), 'w') as fout:
            fout.write(str(obj_arr)+'\n\n')

        # output the assembled box-shape
        vis_utils.draw_partnet_objects([obj_arr],\
                object_names=['GENERATION'], \
                figsize=(5, 5), out_fn=os.path.join(gen_dir, 'gen-%03d.png'%i),\
                leafs_only=True,sem_colors_filename='./part_colors_magazine.txt')
        
        vis_utils.draw_partnet_objects([obj_arr], \
            object_names=['GENERATION'], leafs_only=True, \
            sem_colors_filename='./part_colors_magazine.txt', figsize=(5, 5), \
            out_fn=os.path.join(gen_dir, 'gen-%03d.svg'%i))
        
        
        utils.arr_layout(obj_arr.root, isRoot=True)
        utils.arr_layout(obj_arr.root, isRoot=True)
        obj_arr.get_arrbox()
        
        # output the hierarchy
        with open(os.path.join(gen_dir_arr, 'gen-%03d.txt'%i), 'w') as fout:
            fout.write(str(obj_arr)+'\n\n')

        # output the assembled box-shape
        vis_utils.draw_partnet_objects([obj_arr],\
                object_names=['GENERATION'], \
                figsize=(5, 5), out_fn=os.path.join(gen_dir_arr, 'gen-%03d.png'%i),\
                leafs_only=True,sem_colors_filename='./part_colors_magazine.txt')
        
        vis_utils.draw_partnet_objects([obj_arr], \
            object_names=['GENERATION'], leafs_only=True, \
            sem_colors_filename='./part_colors_magazine.txt', figsize=(5, 5), \
            out_fn=os.path.join(gen_dir_arr, 'gen-%03d.svg'%i))