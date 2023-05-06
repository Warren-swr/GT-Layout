"""
    some helper utility functions
"""

import os
import sys
import torch
import numpy as np


def save_checkpoint(models, model_names, dirname, epoch=None, prepend_epoch=False, optimizers=None, optimizer_names=None):
    if len(models) != len(model_names) or (optimizers is not None and len(optimizers) != len(optimizer_names)):
        raise ValueError('Number of models, model names, or optimizers does not match.')

    for model, model_name in zip(models, model_names):
        filename = f'net_{model_name}.pth'
        if prepend_epoch:
            filename = f'{epoch}_' + filename
        torch.save(model.state_dict(), os.path.join(dirname, filename))

    if optimizers is not None:
        filename = 'checkpt.pth'
        if prepend_epoch:
            filename = f'{epoch}_' + filename
        checkpt = {'epoch': epoch}
        for opt, optimizer_name in zip(optimizers, optimizer_names):
            checkpt[f'opt_{optimizer_name}'] = opt.state_dict()
        torch.save(checkpt, os.path.join(dirname, filename))

def optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

# out shape: (label_count, in shape)
def one_hot(inp, label_count):
    out = torch.zeros(label_count, inp.numel(), dtype=torch.uint8, device=inp.device)
    out[inp.view(-1), torch.arange(out.shape[1])] = 1
    out = out.view((label_count,) + inp.shape)
    return out

def collate_feats(b):
    return list(zip(*b))

def load_pts(fn):
    with open(fn, 'r') as fin:
        lines = [item.rstrip() for item in fin]
        pts = np.array([[float(line.split()[0]), float(line.split()[1]), float(line.split()[2])] for line in lines], dtype=np.float32)
        return pts

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]
    
    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)
    
    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

# pc is N x 3, feat is B x 10-dim
def transform_pc_batch(pc, feat, anchor=False):
    batch_size = feat.size(0)
    num_point = pc.size(0)
    pc = pc.repeat(batch_size, 1, 1)
    center = feat[:, :3].unsqueeze(dim=1).repeat(1, num_point, 1)
    shape = feat[:, 3:6].unsqueeze(dim=1).repeat(1, num_point, 1)
    quat = feat[:, 6:].unsqueeze(dim=1).repeat(1, num_point, 1)
    if not anchor:
        pc = pc * shape
    pc = qrot(quat.view(-1, 4), pc.view(-1, 3)).view(batch_size, num_point, 3)
    if not anchor:
        pc = pc + center
    return pc

def get_surface_reweighting_batch(xyz, cube_num_point):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    assert cube_num_point % 6 == 0, 'ERROR: cube_num_point %d must be dividable by 6!' % cube_num_point
    np = cube_num_point // 6
    out = torch.cat([(x*y).unsqueeze(dim=1).repeat(1, np*2), \
                     (y*z).unsqueeze(dim=1).repeat(1, np*2), \
                     (x*z).unsqueeze(dim=1).repeat(1, np*2)], dim=1)
    out = out / (out.sum(dim=1).unsqueeze(dim=1) + 1e-12)
    return out


'''
post process
'''

def arr_v(node, isRoot=False):
    total = 0.0
    if node.children is not None:
        for child in node.children:
            total += child.box[0,1] + child.box[0,3]
        for child in node.children:
            if isRoot == False or total > 1:
                child.box[0,1] /= total
                child.box[0,3] /= total
        for index, child in enumerate(node.children):
            if index != 0 and child.box[0,1] < 0.02:
                child.box[0,3] -= 0.02 - child.box[0,0]
                child.box[0,1] = 0.02


def arr_h(node, isRoot=False):
    total = 0.0
    if node.children is not None:
        for child in node.children:
            total += child.box[0,0] + child.box[0,2]
        for child in node.children:
            if isRoot == False or total > 1:
                child.box[0,0] /= total
                child.box[0,2] /= total
        for index, child in enumerate(node.children):
            if index != 0 and child.box[0,0] < 0.02:
                child.box[0,2] -= 0.02 - child.box[0,0]
                child.box[0,0] = 0.02
            

def arr_s(node, isRoot=False):
    if node.children is not None:
        for child in node.children:
            total_h = 0.0
            total_v = 0.0
            total_h += child.box[0,0] + child.box[0,2]
            total_v += child.box[0,1] + child.box[0,3]
            
            if total_h > 1:
                child.box[0,0] /= total_h
                child.box[0,2] /= total_h
            
            if total_v > 1:
                child.box[0,1] /= total_v
                child.box[0,3] /= total_v


def arr_layout(node, isRoot=False):
    if node.children is not None:
        if node.label == 'vertical_branch':
            arr_v(node, isRoot=isRoot)
        elif node.label == 'horizontal_branch':
            arr_h(node, isRoot=isRoot)
        elif node.label == 'stack_branch':
            arr_s(node, isRoot=isRoot)
        for index, child in enumerate(node.children):
            if child.box[0,2]  > 0.95:
                child.box[0,2] = 1.0
                child.box[0,0] = 0.0
            if child.box[0,3] > 0.95:
                child.box[0,3] = 1.0
                child.box[0,1] = 0.0
                
            if child.box[0,2] + child.box[0,0] > 0.95:
                child.box[0,2] = 1.0 - child.box[0,0]
            if child.box[0,3] + child.box[0,1] > 0.95:
                child.box[0,3] = 1.0 - child.box[0,1]
            
            if child.box[0,2] + child.box[0,0] < 0.05:
                child.box[0,0] = 0.0
            if child.box[0,3] + child.box[0,1] < 0.05:
                child.box[0,1] = 0.0
            
            arr_layout(child)