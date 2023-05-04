import sys
import os
import json
import torch
import numpy as np
from torch.utils import data
# from pyquaternion import Quaternion
from collections import namedtuple
from utils import one_hot

class Tree(object):

    part_name2id = dict()
    part_id2name = dict()
    part_name2cids = dict()
    part_non_leaf_sem_names = []
    num_sem = None
    root_sem = None

    @ staticmethod
    def load_category_info(cat):
        with open('part_semantics_'+cat+'.txt', 'r') as fin:
            for l in fin.readlines():
                x, y, _ = l.rstrip().split()
                x = int(x)
                Tree.part_name2id[y] = x
                Tree.part_id2name[x] = y
                Tree.part_name2cids[y] = []
                if x == 1:
                    Tree.part_name2cids[y] = ['4','5','6','7', '8', '9']
                if x == 2:
                    Tree.part_name2cids[y] = ['4','5','6','7', '8', '9']
                if x == 3:
                    Tree.part_name2cids[y] = ['4','5','6','7', '8', '9']
        # Tree.num_sem = len(Tree.part_name2id) + 1
        Tree.part_name2cids['branch'] = ['1','2','3']
        Tree.num_sem = 10
        for k in Tree.part_name2cids:
            Tree.part_name2cids[k] = np.array(Tree.part_name2cids[k], dtype=np.int32)
            if len(Tree.part_name2cids[k]) > 0:
                Tree.part_non_leaf_sem_names.append(k)
        Tree.root_sem = Tree.part_id2name[2]
        # 根结点暂时只竖直分布 Vertical-Branch
    
    # store a part node in the tree
    class Node(object):

        def __init__(self, part_id=0, is_leaf=False, box=None, absbox=None, relbox=None, arrange=None, size=None, label=None, children=None, full_label=None, device=None):
            self.is_leaf = is_leaf          # store True if the part is a leaf node
            self.part_id = part_id          # part_id in result_after_merging.json of PartNet
            self.box = box                  # box parameter for all nodes
            self.absbox = absbox
            self.relbox = relbox
            self.arrange = arrange  # proportion of child nodes
            self.arrange_tensor = None
            self.size = size
            self.label = label              # node semantic label at the current level
            self.full_label = full_label    # node semantic label from root (separated by slash)
            self.children = [] if children is None else children
                                            # all of its children nodes; each entry is a Node instance
            self.device = device
        
        def get_semantic_id(self):
            return Tree.part_name2id[self.full_label]
        
        def get_node_id(self):
            if self.is_leaf:
                return 0
            else:
                return Tree.part_name2id[self.full_label]
            
        def get_semantic_one_hot(self):
            out = np.zeros((1, Tree.num_sem), dtype=np.float32)
            out[0, Tree.part_name2id[self.full_label]] = 1
            return torch.tensor(out, dtype=torch.float32).to(device=self.box.device)

        def get_arrange_tensor(self):
            arrange = torch.tensor(self.arrange, dtype=torch.float32).to(device=self.box.device)
            size = torch.tensor(self.size, dtype=torch.float32).to(device=self.box.device)
            return torch.cat([arrange, size], dim=0).to(device=self.box.device).unsqueeze(dim=0)

        def get_box_quat(self):
            box = self.box.cpu().numpy().squeeze()
            center = box[:2]
            # center = np.append(center, 0)
            size = box[2:4]

            box_quat = np.hstack([center, size]).astype(np.float32)
            return torch.from_numpy(box_quat).view(1, -1).to(device=self.box.device)
        
        def get_box(self):
            # absbox = self.absbox.cpu().numpy().squeeze()
            # box = self.box.cpu().numpy().squeeze()
            # size = absbox[2:4]
            # x = box[0] * absbox[2] / box[2]
            # y = box[1] * absbox[3] / box[3]
            # box_quat = np.hstack([size]).astype(np.float32)
            return self.box.view(1, -1).to(device=self.box.device)


        def get_boxfeat(self):
            # box = self.box.cpu().numpy().squeeze()
            box = self.get_box()
            
            # print(self.device)
            semantic = self.get_semantic_one_hot().view(1,-1)
            # box.to(semantic.device)
            # print(box.device)
            out = torch.cat([box, semantic], dim=1).to(device=self.box.device)
            
            return out
 
        def set_from_box_quat(self, box_quat):
            box_quat = box_quat.cpu().detach().numpy().squeeze()
            center = box_quat[:2]
            size = box_quat[2:4]

            box = np.hstack([center, size]).astype(np.float32)
            self.box = torch.from_numpy(box).view(1, -1)
        
        def set_from_box(self, box_quat, father_label):
            box = box_quat.cpu().detach().numpy().squeeze()
            if father_label == 'Vertical-Branch':
                box = np.hstack([0, box]).astype(np.float32)
            else:
                box = np.hstack([box, 0]).astype(np.float32)

            self.box = torch.from_numpy(box).view(1, -1)

        def set_from_boxfeat(self, box_quat):
            box_quat = box_quat.cpu().detach().numpy().squeeze()
            center = box_quat[:2]
            size = box_quat[2:]
            box = np.hstack([center, size]).astype(np.float32)
            self.box = torch.from_numpy(box).view(1, -1)
        
        def get_absbox(self, father_box, arrange=None, size=None, num=0):
            father_box = father_box.cpu().detach().numpy().squeeze()
            # size = self.size.cpu().detach().numpy().squeeze()
            
            x = father_box[0] + arrange[2*num] * father_box[2]
            y = father_box[1] + arrange[2*num+1] * father_box[3]
            w = size[0] * father_box[2]
            h = size[1] * father_box[3]
            
            # if w < 0.05:
            #     w = father_box[2]
            # if h < 0.05:
            #     h = father_box[3]
            box = np.hstack([x, y, w, h]).astype(np.float32)
            self.absbox = torch.from_numpy(box).view(1, -1)
        
        def get_absbox_all(self, arrange, size, father_box=None, box=None, num=0):
            # if box is not None:
            #     box = box.cpu().detach().numpy().squeeze()

            if arrange is not None:
                arrange = arrange.cpu().detach().numpy().squeeze()

            if size is not None:
                size = size.cpu().detach().numpy().squeeze()
            
            if father_box is None:
                # self.absbox = self.box
                # self.absbox = torch.tensor([0.0, 0.0, 0.8, 1.0]).to(device=self.device)
                self.absbox = box
            else:
                self.get_absbox(father_box, arrange=arrange, size=size, num=num)

            for i in range(len(self.children)):
                self.children[i].get_absbox_all(father_box=self.absbox, arrange=self.arrange, size=self.size, num=i)


        def get_relbox(self, father_box, pri_box=None, num=0, father_label=None):
            father_box = father_box.cpu().detach().numpy().squeeze()
            rel_box = self.relbox.cpu().detach().numpy().squeeze()
            
            x = father_box[0]
            y = father_box[1]

            if pri_box is not None:
                pri_box = pri_box.cpu().detach().numpy().squeeze()
                if father_label == 'Vertical-Branch':
                    y = pri_box[1] + pri_box[3]
                elif father_label == 'Horizontal-Branch':
                    x = pri_box[0] + pri_box[2]
            
            # print(pri_box)
            w = rel_box[0]
            h = rel_box[1]
            
            # if w < 0.05:
            #     w = father_box[2]
            # if h < 0.05:
            #     h = father_box[3]
            box = np.hstack([x, y, w, h]).astype(np.float32)
            self.absbox = torch.from_numpy(box).view(1, -1)


        def get_absbox_rel(self, father_box=None, box=None, pri_box=None, num=0, father_label=None):
            if father_box is None:
                # self.absbox = self.box
                xy = torch.tensor([[0.0, 0.0]]).to(device=box.device)
                self.absbox = torch.cat([xy, box], dim=1).squeeze()
                # print(self.absbox)
            else:
                self.get_relbox(father_box, num=num, pri_box=pri_box, father_label=father_label)

            for i in range(len(self.children)):
                if i == 0:
                    pri_box = None
                else:
                    pri_box = self.children[i-1].absbox
                self.children[i].get_absbox_rel(father_box=self.absbox, num=i, pri_box=pri_box, father_label=self.label)
        

        def get_arrbox(self, father_box, pri_box=None, num=0, father_label=None):
            father_box = father_box.cpu().detach().numpy().squeeze()
            rel_box = self.box.cpu().detach().numpy().squeeze()
            
            x = father_box[0] + rel_box[0] * father_box[2]
            y = father_box[1] + rel_box[1] * father_box[3]

            if pri_box is not None:
                pri_box = pri_box.cpu().detach().numpy().squeeze()
                if father_label == 'vertical_branch':
                    # x = x + rel_box[0] * father_box[2]
                    y = pri_box[1] + pri_box[3] + rel_box[1] * father_box[3]
                    w = rel_box[2] * father_box[2]
                    h = rel_box[3] * father_box[3]
                elif father_label == 'horizontal_branch':
                    x = pri_box[0] + pri_box[2] + rel_box[0] * father_box[2]
                    # y = y + rel_box[1] * father_box[3]
                    w = rel_box[2] * father_box[2]
                    h = rel_box[3] * father_box[3]
                elif father_label == 'stack_branch':
                    # x = x + rel_box[0] * father_box[2]
                    # y = y + rel_box[1] * father_box[3]
                    w = rel_box[2] * father_box[2]
                    h = rel_box[3] * father_box[3]
            else:
                if father_label == 'vertical_branch':
                    w = rel_box[2] * father_box[2]
                    h = rel_box[3] * father_box[3]
                elif father_label == 'horizontal_branch':
                    w = rel_box[2] * father_box[2]
                    h = rel_box[3] * father_box[3]
                elif father_label == 'stack_branch':
                    # x = x + rel_box[0] * father_box[2]
                    # y = y + rel_box[1] * father_box[3]
                    w = rel_box[2] * father_box[2]
                    h = rel_box[3] * father_box[3]

            # if w < 0.05:
            #     w = father_box[2]
            # if h < 0.05:
            #     h = father_box[3]
            box = np.hstack([x, y, w, h]).astype(np.float32)
            self.absbox = torch.from_numpy(box).view(1, -1)
        

        def get_arrbox_all(self, father_box=None, box=None, pri_box=None, num=0, father_label=None):
            if father_box is None:
                self.absbox = self.box
                # print(self.absbox)
            else:
                self.get_arrbox(father_box, num=num, pri_box=pri_box, father_label=father_label)

            for i in range(len(self.children)):
                if i == 0:
                    pri_box = None
                else:
                    pri_box = self.children[i-1].absbox
                self.children[i].get_arrbox_all(father_box=self.absbox, num=i, pri_box=pri_box, father_label=self.label)


        def get_arrange(self):
            self.arrange = []
            self.size = []
            x = torch.tensor(0.).to(device=self.box.device)
            y = torch.tensor(0.).to(device=self.box.device)
            for child in self.children:
                # if self.label == 'Vertical-Branch':
                #     y += child.box[3]
                # else:
                #     x += child.box[2]
                self.arrange.extend(torch.tensor([child.box[0], child.box[1]]))
                self.size.extend(torch.tensor([child.box[2], child.box[3]]))
            
            while len(self.arrange) < 10:
                if self.label == 'Vertical-Branch':
                    self.arrange.extend(torch.tensor([0.0, 1.0]))
                    self.size.extend(torch.tensor([0.0, 0.0]))
                else:
                    self.arrange.extend(torch.tensor([1.0, 0.0]))
                    self.size.extend(torch.tensor([0.0, 0.0]))
            
            for child_node in self.children:
                child_node.get_arrange()

        def to(self, device):
            if self.box is not None:
                self.box = self.box.to(device)
            for child_node in self.children:
                child_node.to(device)
            return self
        
        def _to_str(self, level, pid, detailed=True):
            box = self.absbox.cpu().detach().numpy().reshape(-1).tolist()
            out_str = ''
            # if self.is_leaf:
            #     # out_str = str(self.get_semantic_id()) + ' ' + str(' '.join(str(i) for i in box)) + '\n' 
            #     out_str = str(self.get_semantic_id()) + ' ' + str(' '.join(f"{i:.5f}" for i in box)) + '\n'

            # if len(self.children) > 0:
            #     for idx, child in enumerate(self.children):
            #         out_str += child._to_str(level+1, idx)

            # return out_str

            out_str = '  |'*(level-1) + '  ├'*(level > 0) + str(pid) + ' ' + self.label + \
                    (' [LEAF] ' if self.is_leaf else '    ') + '{' + str(self.part_id) + '}' + \
                    ' ' + str(self.get_semantic_id())
            if detailed:
                out_str += ' Box('+str(box)+')\n'
            else:
                out_str += '\n'

            if len(self.children) > 0:
                for idx, child in enumerate(self.children):
                    out_str += child._to_str(level+1, idx)

            return out_str
        
        def __str__(self):
            return self._to_str(0, 0)

        def depth_first_traversal(self):
            nodes = []

            stack = [self]
            while len(stack) > 0:
                node = stack.pop()
                nodes.append(node)

                stack.extend(reversed(node.children))

            return nodes

        # run a DFS over the tree and gather all information
        # return lists of Node.box, Node.part_id, Node.full_label for all nodes
        # when leafs_only=True, only return the leaf-level node information
        # otherwise, return all node information
        def get_part_hierarchy(self, leafs_only=False, show_mode=False):
            part_boxes = []
            part_ids = []
            part_sems = []

            nodes = list(reversed(self.depth_first_traversal()))

            box_index_offset = 0
            for node in nodes:
                child_count = 0
                box_idx = {}
                for i, child in enumerate(node.children):
                    if leafs_only and not child.is_leaf:
                        continue
                    
                    if show_mode:
                        part_boxes.append(child.absbox)
                    else:
                        part_boxes.append(child.box)
                    part_ids.append(child.part_id)
                    part_sems.append(child.full_label)

                    box_idx[i] = child_count+box_index_offset
                    child_count += 1

                box_index_offset += child_count

            return part_boxes, part_ids, part_sems

    # functions for class Tree
    def __init__(self, root, device=None):
        self.root = root
        self.device = device

    def to(self, device):
        self.root = self.root.to(device)
        return self

    def __str__(self):
        return str(self.root)

    def get_part_hierarchy(self, leafs_only=False, show_mode=False):
        return self.root.get_part_hierarchy(leafs_only=leafs_only, show_mode=show_mode)

    def get_arrange(self):
        return self.root.get_arrange()
    
    def get_absbox(self):
        return self.root.get_absbox_all(arrange=self.root.arrange, size=self.root.size, box=self.root.relbox)
    
    def get_arrbox(self):
        return self.root.get_arrbox_all(box=self.root.box)
    
    def get_relbox(self):
        return self.root.get_absbox_rel(box=self.root.relbox)


# extend torch.data.Dataset class for LayoutDataset
class LayoutDataset(data.Dataset):

    def __init__(self, root, object_list, data_features):
        self.root = root
        self.data_features = data_features

        with open(os.path.join(self.root, object_list), 'r') as f:
            self.object_names = [item.rstrip() for item in f.readlines()]

    def __getitem__(self, index):
        if 'object' in self.data_features:
            obj = self.load_object(os.path.join(self.root, self.object_names[index]+'.json'))

        data_feats = ()
        for feat in self.data_features:
            if feat == 'object':
                data_feats = data_feats + (obj,)
            elif feat == 'name':
                data_feats = data_feats + (self.object_names[index],)
            else:
                assert False, 'ERROR: unknow feat type %s!' % feat
                
        return data_feats

    def __len__(self):
        return len(self.object_names)

    def get_from_anno_id(self, anno_id):
        obj = self.load_object(os.path.join(self.root, anno_id+'.json'))
        return obj

    @staticmethod
    def load_object(fn):
        with open(fn, 'r') as f:
            root_json = json.load(f)

        # create a virtual parent node of the root node and add it to the stack
        StackElement = namedtuple('StackElement', ['node_json', 'parent', 'parent_child_idx'])
        stack = [StackElement(node_json=root_json, parent=None, parent_child_idx=None)]

        root = None
        # traverse the tree, converting each node json to a Node instance
        while len(stack) > 0:
            stack_elm = stack.pop()

            parent = stack_elm.parent
            parent_child_idx = stack_elm.parent_child_idx
            node_json = stack_elm.node_json

            node = Tree.Node(
                part_id=node_json['id'],
                is_leaf=('children' not in node_json),
                label=node_json['label'])

            if 'box' in node_json:
                node.box = torch.from_numpy(np.array(node_json['box'])).to(dtype=torch.float32)
            
            if 'abs_box' in node_json:
                node.absbox = torch.from_numpy(np.array(node_json['abs_box'])).to(dtype=torch.float32)

            if ('box' in node_json) and ('abs_box' in node_json):
                node.relbox = node.get_box()

            if 'children' in node_json:
                for ci, child in enumerate(node_json['children']):
                    stack.append(StackElement(node_json=node_json['children'][ci], parent=node, parent_child_idx=ci))

            if parent is None:
                root = node
                root.full_label = root.label
            else:
                if len(parent.children) <= parent_child_idx:
                    parent.children.extend([None] * (parent_child_idx+1-len(parent.children)))
                parent.children[parent_child_idx] = node
                # node.full_label = parent.full_label + '/' + node.label
                node.full_label = node.label

        obj = Tree(root=root)
        obj.get_arrange()

        return obj

    @staticmethod
    def save_object(obj, fn):

        # create a virtual parent node of the root node and add it to the stack
        StackElement = namedtuple('StackElement', ['node', 'parent_json', 'parent_child_idx'])
        stack = [StackElement(node=obj.root, parent_json=None, parent_child_idx=None)]

        obj_json = None

        # traverse the tree, converting child nodes of each node to json
        while len(stack) > 0:
            stack_elm = stack.pop()

            parent_json = stack_elm.parent_json
            parent_child_idx = stack_elm.parent_child_idx
            node = stack_elm.node

            node_json = {
                'id': node.part_id,
                'label': f'{node.label if node.label is not None else ""}'}

            if node.box is not None:
                node_json['box'] = node.box.cpu().numpy().reshape(-1).tolist()

            if len(node.children) > 0:
                node_json['children'] = []
            for child in node.children:
                node_json['children'].append(None)
                stack.append(StackElement(node=child, parent_json=node_json, parent_child_idx=len(node_json['children'])-1))

            if parent_json is None:
                obj_json = node_json
            else:
                parent_json['children'][parent_child_idx] = node_json

        with open(fn, 'w') as f:
            json.dump(obj_json, f)