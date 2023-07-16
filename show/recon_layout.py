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

num_recon = 200
exp = 'magazine_2.5K'
category = 'magazine'
checkpoint = ''
device = 'cuda:0'

path = '/home/weiran/Projects/RvNN-Layout/GT-Layout/' + category + '/logs/' + exp
data_path = '/home/weiran/Projects/RvNN-Layout/data/' + category + '-ours/' + exp

# load train config
conf = torch.load(path + '/conf.pth')

# load object category information
Tree.load_category_info(conf.category)

# set up device
print(f'Using device: {conf.device}')

recon_dir = path + '/recon/'

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

# read test.txt
with open(data_path + '/test.txt', 'r') as fin:
    data_list = [l.rstrip() for l in fin.readlines()]

if os.path.exists(recon_dir):
    shutil.rmtree(recon_dir)
os.mkdir(recon_dir)

# reconstruct shapes
with torch.no_grad():
    for i in tqdm.tqdm(range(num_recon)):
        # load the gt data as the input
        obj = LayoutDataset.load_object(data_path + '/' + data_list[i] + '.json')
        obj.to(device)

        # feed through the encoder to get a code z
        root_code_and_kld = encoder.encode_structure(obj)
        root_code = root_code_and_kld[:, :conf.feature_size]

        # infer through the decoder to get the reconstructed output
        # set maximal tree depth to conf.max_tree_depth
        obj_arr = decoder.decode_structure(z=root_code, max_depth=conf.max_tree_depth)
        obj_arr.get_arrbox()
        obj.get_arrbox()

        # output the hierarchy
        with open(os.path.join(recon_dir, data_list[i] + '_GT.txt'), 'w') as fout:
            fout.write(str(obj)+'\n\n')
            
        with open(os.path.join(recon_dir, data_list[i] + '_PRED.txt'), 'w') as fout:
            fout.write(str(obj_arr)+'\n\n')

        # output the assembled box-shape
        vis_utils.draw_partnet_objects([obj], \
            object_names=['GT'], leafs_only=True, \
            sem_colors_filename='./part_colors_magazine.txt', figsize=(5, 5), \
            out_fn=os.path.join(recon_dir, data_list[i] + '_GT.png'))
        
        vis_utils.draw_partnet_objects([obj], \
            object_names=['GT'], leafs_only=True, \
            sem_colors_filename='./part_colors_magazine.txt', figsize=(5, 5), \
            out_fn=os.path.join(recon_dir, data_list[i] + '_GT.svg'))
        
        vis_utils.draw_partnet_objects([obj_arr], \
            object_names=['PRED'], leafs_only=True, \
            sem_colors_filename='./part_colors_magazine.txt', figsize=(5, 5), \
            out_fn=os.path.join(recon_dir, data_list[i] + '_PRED.png'))

        vis_utils.draw_partnet_objects([obj_arr], \
            object_names=['PRED'], leafs_only=True, \
            sem_colors_filename='./part_colors_magazine.txt', figsize=(5, 5), \
            out_fn=os.path.join(recon_dir, data_list[i] + '_PRED.svg'))