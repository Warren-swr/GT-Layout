"""
    Recursive Layout Generation model
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from data_layout import Tree
from utils import load_pts

VAE_adjust = False
pos_emb = False


#########################################################################################
## Encoder
#########################################################################################


class Sampler(nn.Module):

    def __init__(self, feature_size, hidden_size, probabilistic=True):
        super(Sampler, self).__init__()
        self.probabilistic = probabilistic
        
        if VAE_adjust:
            print("add layer norm and a layer")
            self.mlp_pre = nn.Linear(feature_size, hidden_size)
            self.ln = nn.LayerNorm(hidden_size)
            
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2mu = nn.Linear(hidden_size, feature_size)
        self.mlp2var = nn.Linear(hidden_size, feature_size)

    def forward(self, x):
        if VAE_adjust:
            x = x + self.mlp_pre(self.ln(x))
        
        encode = torch.relu(self.mlp1(x))
        mu = self.mlp2mu(encode)

        if self.probabilistic:
            logvar = self.mlp2var(encode)
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)

            kld = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
            return torch.cat([eps.mul(std).add_(mu), kld], 1)
        else:
            return mu


class BoxEncoder(nn.Module):

    def __init__(self, hidden_size, box_feat_num):
        super(BoxEncoder, self).__init__()
        self.encoder = nn.Linear(box_feat_num, hidden_size)
    
    def forward(self, box_input):
        box_vector = torch.relu(self.encoder(box_input))

        return box_vector


class SemEncoder(nn.Module):

    def __init__(self, hidden_size):
        super(SemEncoder, self).__init__()
        self.encoder = nn.Linear(Tree.num_sem, hidden_size)
    
    def forward(self, sem_input):
        sem_vector = torch.relu(self.encoder(sem_input))

        return sem_vector


class LeafEncoder(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(LeafEncoder, self).__init__()
        self.encoder = nn.Linear(hidden_size*2, feature_size)
    
    def forward(self, box_vector, sem_vector):
        # input = torch.cat([box_vector, sem_vector], dim=1)
        # leaf_vector = torch.relu(self.encoder(input))
        leaf_vector = box_vector + sem_vector

        return leaf_vector
        

class BranchEncoder(nn.Module):

    def __init__(self, feature_size, arrange_size, hidden_size, max_child_num):
        super(BranchEncoder, self).__init__()

        # self.child_op = nn.Linear(feature_size, hidden_size)
        self.node_op = nn.Linear(max_child_num*hidden_size, hidden_size)
        # self.arrange_op = nn.Linear(arrange_size, hidden_size)
        self.parent_op = nn.Linear(hidden_size*2, feature_size)
        if pos_emb == True:
            self.pos_emb = nn.Parameter(torch.zeros(1, 5, hidden_size))

    def forward(self, child_feats, child_exists, box_vector):
        # use the child_op linear layer to extract per-child features
        # child_feats = torch.relu(self.child_op(child_feats))

        if pos_emb == True:
            child_feats = child_feats + self.pos_emb[:, :child_feats.size(1), :]

        # zero non-existent children
        child_feats = child_feats * child_exists
        child_feats = child_feats.view(1, -1)

        # use the node_op linear layer to summarize a parent node feature
        parent_feat = torch.relu(self.node_op(child_feats)) + box_vector

        # feat_sum = torch.cat([parent_feat, box_vector], dim=1)
        # parent_feat = torch.relu(self.parent_op(feat_sum))

        return parent_feat


class RecursiveEncoder(nn.Module):

    def __init__(self, config, variational=False, probabilistic=True):
        super(RecursiveEncoder, self).__init__()
        self.conf = config

        self.box_encoder = BoxEncoder(hidden_size=config.hidden_size, box_feat_num=4)
        self.sem_encoder = SemEncoder(hidden_size=config.hidden_size)
        self.leaf_encoder = LeafEncoder(
            hidden_size=config.hidden_size, feature_size=config.feature_size)
        self.vertical_encoder = BranchEncoder(
            feature_size=config.feature_size, hidden_size=config.hidden_size,
            max_child_num=config.max_child_num, arrange_size=config.max_child_num*4)
        self.horizontal_encoder = BranchEncoder(
            feature_size=config.feature_size, hidden_size=config.hidden_size,
            max_child_num=config.max_child_num, arrange_size=config.max_child_num*4)
        self.stack_encoder = BranchEncoder(
            feature_size=config.feature_size, hidden_size=config.hidden_size,
            max_child_num=config.max_child_num, arrange_size=config.max_child_num*4)

        if variational:
            self.sample_encoder = Sampler(feature_size=config.feature_size,
                hidden_size=config.hidden_size, probabilistic=probabilistic)
    
    def encode_node(self, node):
        if node.is_leaf:
            box = node.get_box()
            # print(box)
            box_vector = self.box_encoder(box)
            sem_vector = self.sem_encoder(node.get_semantic_one_hot())
            leaf_feat = self.leaf_encoder(box_vector, sem_vector)
            return leaf_feat
        else:
            # get features of all children
            child_feats = []

            for child in node.children:
                cur_child_feat = torch.cat([self.encode_node(child)], dim=1)
                child_feats.append(cur_child_feat.unsqueeze(dim=1))

            child_feats = torch.cat(child_feats, dim=1)

            if child_feats.shape[1] > self.conf.max_child_num:
                print(child_feats.shape[1])
                raise ValueError('Node has too many children.')

            # pad with zeros
            if child_feats.shape[1] < self.conf.max_child_num:
                padding = child_feats.new_zeros(child_feats.shape[0], self.conf.max_child_num - child_feats.shape[1], child_feats.shape[2])
                child_feats = torch.cat([child_feats, padding], dim=1)

            # 1 if the child exists, 0 if it is padded
            child_exists = child_feats.new_zeros(child_feats.shape[0], self.conf.max_child_num, 1)
            child_exists[:, :len(node.children), :] = 1
            
            # get box feature of current branch
            box = node.get_box()
            box_vector = self.box_encoder(box)

            # get feature of current node (parent of the children)
            if node.label == 'vertical_branch':
                return self.vertical_encoder(child_feats, child_exists, box_vector)
            elif node.label == 'horizontal_branch':
                return self.horizontal_encoder(child_feats, child_exists, box_vector)
            elif node.label == 'stack_branch':
                return self.stack_encoder(child_feats, child_exists, box_vector)
            else:
                print('ERROR')

    def encode_structure(self, obj):
        root_latent = self.encode_node(obj.root)
        # print(root_latent.shape)
        return self.sample_encoder(root_latent)


#########################################################################################
## Decoder
#########################################################################################


class LeafClassifier(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(LeafClassifier, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, 4)

    def forward(self, input_feature):
        output = torch.relu(self.mlp1(input_feature))
        output = self.mlp2(output)
        return output


class SampleDecoder(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(SampleDecoder, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, feature_size)
        '''KL loss 偏高'''
        if VAE_adjust:
            self.mlp3 = nn.Linear(hidden_size, feature_size)

    def forward(self, input_feature):
        output = torch.relu(self.mlp1(input_feature))
        output = torch.relu(self.mlp2(output))
        if VAE_adjust:
            output = torch.relu(self.mlp3(output))
        return output


class LeafDecoder(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(LeafDecoder, self).__init__()
        self.decoder = nn.Linear(feature_size, hidden_size)
    
    def forward(self, input_feature):
        return torch.relu(self.decoder(input_feature))


class BoxDecoder(nn.Module):

    def __init__(self, hidden_size):
        super(BoxDecoder, self).__init__()
        self.xy = nn.Linear(hidden_size, 2)
        self.size = nn.Linear(hidden_size, 2)
    
    def forward(self, input_feature):
        xy = torch.sigmoid(self.xy(input_feature))
        size = torch.sigmoid(self.size(input_feature))
        return torch.cat([xy, size], dim=1)


class SemDecoder(nn.Module):

    def __init__(self, hidden_size):
        super(SemDecoder, self).__init__()
        self.decoder = nn.Linear(hidden_size, Tree.num_sem)
    
    def forward(self, input_feature):
        # 参考分类网络，看看输出的时候需要不需要用激活函数
        sem_logits = self.decoder(input_feature)
        return sem_logits


class BranchDecoder(nn.Module):

    def __init__(self, feature_size, hidden_size, max_child_num):
        super(BranchDecoder, self).__init__()

        self.max_child_num = max_child_num
        self.hidden_size = hidden_size

        self.mlp_parent_1 = nn.Linear(feature_size, hidden_size*max_child_num)
        self.mlp_parent_2 = nn.Linear(hidden_size*max_child_num, hidden_size*max_child_num)
        self.mlp_exists = nn.Linear(hidden_size, 1)
        self.mlp_arrange = nn.Linear(hidden_size, max_child_num*4)
        self.mlp_sem = nn.Linear(hidden_size, Tree.num_sem)
        self.mlp_child = nn.Linear(hidden_size, feature_size)
        if pos_emb == True:
            self.pos_emb = nn.Parameter(torch.zeros(1, 5, hidden_size))

    def forward(self, parent_feature):
        batch_size = parent_feature.shape[0]
        feat_size = parent_feature.shape[1]
        
        # use the mlp_parent linear layer to get the children node features
        parent_feature = torch.relu(self.mlp_parent_1(parent_feature))
        # parent_feature = torch.relu(self.mlp_parent_2(parent_feature))

        child_feats = parent_feature.view(batch_size, self.max_child_num, self.hidden_size)
        
        if pos_emb == True:
            child_feats = child_feats + self.pos_emb[:, :child_feats.size(1), :]

        # use the mlp_exists linear layer to predict children node existence (output logits, i.e. no sigmoid)
        child_exists_logits = self.mlp_exists(child_feats.view(batch_size*self.max_child_num, self.hidden_size))
        child_exists_logits = child_exists_logits.view(batch_size, self.max_child_num, 1)
        
        # use the mlp_sem linear layer to predict children node semantics (output logits, i.e. no sigmoid)
        # child_sem_logits = self.mlp_sem(child_feats.view(-1, self.hidden_size))
        # child_sem_logits = child_sem_logits.view(batch_size, self.max_child_num, Tree.num_sem)

        # use the mlp_child linear layer to further evolve the children node features
        child_feats = self.mlp_child(child_feats.view(-1, self.hidden_size))
        child_feats = child_feats.view(batch_size, self.max_child_num, feat_size)
        child_feats = torch.relu(child_feats)
        
        return child_feats, child_exists_logits


class RecursiveDecoder(nn.Module):

    def __init__(self, config):
        super(RecursiveDecoder, self).__init__()

        self.conf = config

        self.box_decoder = BoxDecoder(config.hidden_size)
        self.sem_decoder = SemDecoder(config.hidden_size)
        self.leaf_decoder = LeafDecoder(config.feature_size, config.hidden_size)

        self.vertical_decoder = BranchDecoder(
                feature_size=config.feature_size, 
                hidden_size=config.hidden_size, 
                max_child_num=config.max_child_num)
        
        self.horizontal_decoder = BranchDecoder (
                feature_size=config.feature_size, 
                hidden_size=config.hidden_size, 
                max_child_num=config.max_child_num)


        self.stack_decoder = BranchDecoder (
                feature_size=config.feature_size, 
                hidden_size=config.hidden_size, 
                max_child_num=config.max_child_num)
        self.sample_decoder = SampleDecoder(config.feature_size, config.hidden_size)
        self.leaf_classifier = LeafClassifier(config.feature_size, config.hidden_size)

        self.bceLoss = nn.BCEWithLogitsLoss(reduction='none')
        self.semCELoss = nn.CrossEntropyLoss(reduction='none')
        self.mseLoss = nn.MSELoss(reduction='sum')

    def boxLossEstimator(self, box_feature, gt_box_feature):
        loss = box_feature.sub(gt_box_feature).pow(2).mean(dim=1)
        return loss
    
    def chamferLossEstimator(self, box_feature, gt_box_feature):
        loss = box_feature.sub(gt_box_feature).pow(2).mean(dim=1)
        return loss
    
    def isLeafLossEstimator(self, is_leaf_logit, gt_is_leaf):
        return self.bceLoss(is_leaf_logit, gt_is_leaf).view(-1)

    # decode a root code into a tree structure
    def decode_structure(self, z, max_depth):
        root_latent = self.sample_decoder(z)
        root = self.decode_node(root_latent, max_depth, None, is_root=True)
        obj = Tree(root=root, device=self.conf.device)
        return obj

    # decode a part node (inference only)
    def decode_node(self, node_latent, max_depth, full_label, is_leaf=False, is_root=False):
        if node_latent.shape[0] != 1:
            raise ValueError('Node decoding does not support batch_size > 1.')
        
        # use maximum depth to avoid potential infinite recursion
        if max_depth < 1:
            is_leaf = True

        node_classifier_logit = self.leaf_classifier(node_latent)
        node_classifier_logit = node_classifier_logit.cpu().detach().numpy().squeeze()
        idx = np.argmax(node_classifier_logit)
        node_is_leaf = (idx == 0)

        # decode the current part box
        node_feat = self.leaf_decoder(node_latent)
        box = self.box_decoder(node_feat)
        box_semantic_logit = self.sem_decoder(node_feat)
        # print(box)
        # exit()
    
        if node_is_leaf or is_leaf:
            if full_label == None:
                full_label = 'vertical_branch'
            box_semantic_logit = box_semantic_logit.cpu().detach().numpy().squeeze()
            idx = np.argmax(box_semantic_logit[Tree.part_name2cids[full_label]])
            idx = Tree.part_name2cids[full_label][idx]
            child_full_label = Tree.part_id2name[idx]

            ret = Tree.Node(is_leaf=True, full_label=child_full_label, label = child_full_label, device=self.conf.device, box=box)
            return ret
        else:
            father_label = full_label
            full_label = Tree.part_id2name[idx]
            if full_label == 'vertical_branch':
                child_feats, child_exists_logit = self.vertical_decoder(node_latent)
            elif full_label == 'horizontal_branch':
                child_feats, child_exists_logit = self.horizontal_decoder(node_latent)
            elif full_label == 'stack_branch':
                child_feats, child_exists_logit = self.stack_decoder(node_latent)
            else:
                print('ERROR')
            
            # child_sem_logits = child_sem_logits.cpu().detach().numpy().squeeze()

            # children
            child_nodes = []
            for ci in range(child_feats.shape[1]):
                if torch.sigmoid(child_exists_logit[:, ci, :]).item() > 0.5:
                    # print(full_label)
                    # print(child_sem_logits[ci])
                    # idx = np.argmax(child_sem_logits[ci, Tree.part_name2cids[full_label]])
                    # idx = Tree.part_name2cids[full_label][idx]
                    # child_full_label = Tree.part_id2name[idx]
                    child_nodes.append(self.decode_node(child_feats[:, ci, :], max_depth-1, full_label))

            ret = Tree.Node(is_leaf=False, children=child_nodes, full_label=full_label, label=full_label, device=self.conf.device, box=box)
            return ret

    # use gt structure, compute the reconstruction losses
    def structure_recon_loss(self, z, gt_tree):
        root_latent = self.sample_decoder(z)
        losses = self.node_recon_loss(root_latent, gt_tree.root)
        return losses

    # use gt structure, compute the reconstruction losses (used during training)
    def node_recon_loss(self, node_latent, gt_node):
        if gt_node.is_leaf:
            node_feat = self.leaf_decoder(node_latent)

            box = self.box_decoder(node_feat)
            box_gt = gt_node.get_box()
            box_loss = self.mseLoss(box, box_gt)

            node_logit = self.leaf_classifier(node_latent)
            sem = gt_node.get_node_id()
            sem_gt = torch.tensor([sem], dtype=torch.int64, device=node_logit.device)
            is_leaf_loss = self.semCELoss(node_logit, sem_gt)
            
            box_sem = self.sem_decoder(node_feat)
            leaf_sem_gt = gt_node.get_semantic_id()
            leaf_sem_gt = torch.tensor([leaf_sem_gt], dtype=torch.int64, device=box_sem.device)
            sem_loss = self.semCELoss(box_sem, leaf_sem_gt)

            return {'box': box_loss, 'leaf': is_leaf_loss, 'arrange': torch.zeros_like(box_loss), 'exists': torch.zeros_like(box_loss), 'semantic': sem_loss}
        else:
            if gt_node.label == 'vertical_branch':
                child_feats, child_exists_logits = self.vertical_decoder(node_latent)
            elif gt_node.label == 'horizontal_branch':
                child_feats, child_exists_logits = self.horizontal_decoder(node_latent)
            elif gt_node.label == 'stack_branch':
                child_feats, child_exists_logits = self.stack_decoder(node_latent)
            else:
                print('ERROR')
            
            # generate box prediction for each child
            feature_len = node_latent.size(1)
            
            node_logit = self.leaf_classifier(node_latent)
            sem = gt_node.get_node_id()
            sem_gt = torch.tensor([sem], dtype=torch.int64, device=node_logit.device)
            is_leaf_loss = self.semCELoss(node_logit, sem_gt)

            # train the current node box to gt
            node_feat = self.leaf_decoder(node_latent)
            box = self.box_decoder(node_feat)
            box_gt = gt_node.get_box()
            box_loss = self.mseLoss(box, box_gt)

            # gather information
            child_sem_gt_labels = []
            child_sem_pred_logits = []
            child_exists_gt = torch.zeros_like(child_exists_logits)

            for i in range(len(gt_node.children)):
                child_sem_gt_labels.append(gt_node.children[i].get_semantic_id())
                # child_sem_pred_logits.append(child_sem_logits[i, :].view(1, -1))
                child_exists_gt[:, i, :] = 1
                
            # train semantic labels
            # child_sem_pred_logits = torch.cat(child_sem_pred_logits, dim=0)
            # child_sem_gt_labels = torch.tensor(child_sem_gt_labels, dtype=torch.int64, \
            #         device=child_sem_pred_logits.device)
            # semantic_loss = self.semCELoss(child_sem_pred_logits, child_sem_gt_labels)
            # semantic_loss = semantic_loss.sum()

            # train exist scores
            child_exists_loss = F.binary_cross_entropy_with_logits(input=child_exists_logits, target=child_exists_gt, reduction='none')
            child_exists_loss = child_exists_loss.sum()

            semantic_loss = torch.zeros_like(is_leaf_loss)
            
            # calculate children + aggregate losses
            for i in range(len(gt_node.children)):
                child_losses = self.node_recon_loss(child_feats[:, i, :], gt_node.children[i])
                box_loss = box_loss + child_losses['box']
                is_leaf_loss = is_leaf_loss + child_losses['leaf']
                arrange_loss = torch.zeros_like(is_leaf_loss) + child_losses['arrange']
                child_exists_loss = child_exists_loss + child_losses['exists']
                semantic_loss = semantic_loss + child_losses['semantic']

            return {'box': box_loss, 'leaf': is_leaf_loss, 'arrange': arrange_loss, 'exists': child_exists_loss, 'semantic': semantic_loss}