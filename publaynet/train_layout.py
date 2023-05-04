"""
    the main trainer script for the Layout
"""

import os
import time
import sys
import shutil
import random
from time import strftime
from argparse import ArgumentParser
import numpy as np
import torch
import torch.utils.data
from data_layout import LayoutDataset, Tree
import model_layout as model
import utils
import vis_utils_layout as vis_utils
from manager_torch import GPUManager
from tqdm import tqdm

# Use 1-4 CPU threads to train.
# Don't use too many CPU threads, which will slow down the training.
torch.set_num_threads(4)


def train(conf):

    # check if training run already exists. If so, delete it.
    while os.path.exists(conf.exp_name):
        if conf.exp_name[-1] == '1':
            num = 2
            while os.path.exists(conf.exp_name):
                conf.exp_name = conf.exp_name[:-1] + str(num)
                num = num + 1
        else:
            conf.exp_name = conf.exp_name + '_1'
    if os.path.exists(conf.exp_name):
        shutil.rmtree(conf.exp_name)

    # create directories for this run
    os.makedirs(conf.exp_name)
    os.makedirs(os.path.join(conf.exp_name, 'ckpts'))
    os.makedirs(os.path.join(conf.exp_name, 'val_visu'))

    # file log
    flog = open(os.path.join(conf.exp_name, 'train_log.txt'), 'w')

    # set training device
    gm=GPUManager()
    d = gm.auto_choice()
    # d= (d-1)%3
    device = torch.device('cuda:' + str(d))
    print(f'Using device: {conf.device}')
    flog.write(f'Using device: {device}\n')

    # log the object category information
    print(f'Object Category: {conf.category}')
    flog.write(f'Object Category: {conf.category}\n')

    # control randomness
    if conf.seed < 0:
        conf.seed = random.randint(1, 10000)
    print("Random Seed: %d" % (conf.seed))
    flog.write(f'Random Seed: {conf.seed}\n')
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)

    # save config
    torch.save(conf, os.path.join(conf.exp_name, 'conf.pth'))

    # create 
    encoder = model.RecursiveEncoder(conf, variational=True, probabilistic=not conf.non_variational)
    decoder = model.RecursiveDecoder(conf)
    models = [encoder, decoder]
    model_names = ['encoder', 'decoder']

    # load pertrained models
    if conf.pretrained_model is not None:
        print('Loading pretrained net_encoder')
        data_to_restore = torch.load(conf.pretrained_model + 'net_encoder.pth')
        encoder.load_state_dict(data_to_restore, strict=True)
        print('DONE\n')
        print('Loading pretrained net_decoder')
        data_to_restore = torch.load(conf.pretrained_model + 'net_decoder.pth')
        decoder.load_state_dict(data_to_restore, strict=True)
        print('DONE\n')


    # create optimizers
    encoder_opt = torch.optim.Adam(encoder.parameters(), lr=conf.lr)
    decoder_opt = torch.optim.Adam(decoder.parameters(), lr=conf.lr)
    optimizers = [encoder_opt, decoder_opt]
    optimizer_names = ['encoder', 'decoder']

    # learning rate scheduler
    encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_opt, \
            step_size=conf.lr_decay_every, gamma=conf.lr_decay_by)
    decoder_scheduler = torch.optim.lr_scheduler.StepLR(decoder_opt, \
            step_size=conf.lr_decay_every, gamma=conf.lr_decay_by)

    # create training and validation datasets and data loaders
    data_features = ['object', 'name']
    train_dataset = LayoutDataset(conf.data_path, conf.train_dataset, data_features)
    valdt_dataset = LayoutDataset(conf.data_path, conf.val_dataset, data_features)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.batch_size, \
            shuffle=True, collate_fn=utils.collate_feats)
    valdt_dataloader = torch.utils.data.DataLoader(valdt_dataset, batch_size=conf.batch_size, \
            shuffle=True, collate_fn=utils.collate_feats)

    # create logs
    if not conf.no_console_log:
        header = '|   Time     Epoch   Data     Iter     %  |   Box    Leaf   Arr   Exist    Sem    KL  |  Total |'
    if not conf.no_tb_log:
        # https://github.com/lanpa/tensorboard-pytorch
        from tensorboardX import SummaryWriter
        train_writer = SummaryWriter(os.path.join(conf.exp_name, 'train'))
        valdt_writer = SummaryWriter(os.path.join(conf.exp_name, 'val'))

    # send parameters to device
    for m in models:
        m.to(device)
    for o in optimizers:
        utils.optimizer_to_device(o, device)

    # start training
    print("Starting training ...... ")
    flog.write('Starting training ......\n')

    start_time = time.time()

    last_checkpoint_step = None
    last_train_console_log_step, last_valdt_console_log_step = None, None
    train_num_batch, valdt_num_batch = len(train_dataloader), len(valdt_dataloader)

    # train for every epoch
    for epoch in range(conf.epochs):
        if not conf.no_console_log:
            print(f'training run {conf.exp_name}')
            # print(f'data path {conf.data_path}')
            flog.write(f'training run {conf.exp_name}\n')
            # print(header)
            flog.write(header+'\n')

        # train_batches = enumerate(train_dataloader, 0)
        train_batches = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        valdt_batches = enumerate(valdt_dataloader, 0)

        train_fraction_done, valdt_fraction_done = 0.0, 0.0
        valdt_batch_ind = -1

        # train for every batch
        for train_batch_ind, batch in train_batches:
            train_fraction_done = (train_batch_ind + 1) / train_num_batch
            train_step = epoch * train_num_batch + train_batch_ind

            log_console = not conf.no_console_log and (last_train_console_log_step is None or \
                    train_step - last_train_console_log_step >= conf.console_log_interval)
            if log_console:
                last_train_console_log_step = train_step

            # set models to training mode
            for m in models:
                m.train()

            # forward pass (including logging)
            total_loss, losses = forward(
                batch=batch, data_features=data_features, encoder=encoder, decoder=decoder, device=device, conf=conf,
                is_valdt=False, step=train_step, epoch=epoch, batch_ind=train_batch_ind, num_batch=train_num_batch, start_time=start_time,
                log_console=log_console, log_tb=not conf.no_tb_log, tb_writer=train_writer,
                lr=encoder_opt.param_groups[0]['lr'], flog=flog)

            # optimize one step
            encoder_opt.zero_grad()
            decoder_opt.zero_grad()
            total_loss.backward()
            encoder_opt.step()
            decoder_opt.step()
            encoder_scheduler.step()
            decoder_scheduler.step()

            # save checkpoint
            with torch.no_grad():
                if last_checkpoint_step is None or \
                        train_step - last_checkpoint_step >= conf.checkpoint_interval:
                    print("Saving checkpoint ...... ", end='', flush=True)
                    flog.write("Saving checkpoint ...... ")
                    utils.save_checkpoint(
                        models=models, model_names=model_names, dirname=os.path.join(conf.exp_name, 'ckpts'),
                        epoch=epoch, prepend_epoch=True, optimizers=optimizers, optimizer_names=model_names)
                    print("DONE")
                    flog.write("DONE\n")
                    last_checkpoint_step = train_step

            # validate one batch
            while valdt_fraction_done <= train_fraction_done and valdt_batch_ind+1 < valdt_num_batch:
                valdt_batch_ind, batch = next(valdt_batches)

                valdt_fraction_done = (valdt_batch_ind + 1) / valdt_num_batch
                valdt_step = (epoch + valdt_fraction_done) * train_num_batch - 1

                log_console = not conf.no_console_log and (last_valdt_console_log_step is None or \
                        valdt_step - last_valdt_console_log_step >= conf.console_log_interval)
                if log_console:
                    last_valdt_console_log_step = valdt_step

                # set models to evaluation mode
                for m in models:
                    m.eval()

                with torch.no_grad():
                    # forward pass (including logging)
                    total_loss, losses = forward(
                        batch=batch, data_features=data_features, encoder=encoder, decoder=decoder, device=device, conf=conf,
                        is_valdt=True, step=valdt_step, epoch=epoch, batch_ind=valdt_batch_ind, num_batch=valdt_num_batch, start_time=start_time,
                        log_console=log_console, log_tb=not conf.no_tb_log, tb_writer=valdt_writer,
                        lr=encoder_opt.param_groups[0]['lr'], flog=flog)
            
            train_batches.set_description(f"epoch{epoch+1} : box{losses['box'].item():5.3f} le{losses['leaf'].item():5.3f} ex{losses['exists'].item():5.3f} sem{losses['semantic'].item():5.3f} kl{losses['kldiv'].item():5.3f} T{total_loss.item():5.3f}")

    # save the final models
    print("Saving final checkpoint ...... ", end='', flush=True)
    flog.write("Saving final checkpoint ...... ")
    utils.save_checkpoint(
        models=models, model_names=model_names, dirname=os.path.join(conf.exp_name, 'ckpts'),
        epoch=epoch, prepend_epoch=False, optimizers=optimizers, optimizer_names=optimizer_names)
    print("DONE")
    flog.write("DONE\n")

    flog.close()


def forward(batch, data_features, encoder, decoder, device, conf,
            is_valdt=False, step=None, epoch=None, batch_ind=0, num_batch=1, start_time=0,
            log_console=False, log_tb=False, tb_writer=None, lr=None, flog=None):
    objects = batch[data_features.index('object')]
    names = batch[data_features.index('name')]

    losses = {
        'box': torch.zeros(1, device=device),
        'leaf': torch.zeros(1, device=device),
        'arrange': torch.zeros(1, device=device),
        'exists': torch.zeros(1, device=device),
        'semantic': torch.zeros(1, device=device),
        'kldiv': torch.zeros(1, device=device)}

    # process every data in the batch individually
    # since different tree has different structure, which is hard to be batched
    for obj_id in range(len(objects)):
        obj = objects[obj_id]
        name = names[obj_id]

        obj.to(device)

        # encode object to get root code
        root_code = encoder.encode_structure(obj=obj)

        # get kldiv loss
        if not conf.non_variational:
            root_code, obj_kldiv_loss = torch.chunk(root_code, 2, 1)
            obj_kldiv_loss = -obj_kldiv_loss.sum() # negative kldiv, sum over feature dimensions
            losses['kldiv'] = losses['kldiv'] + obj_kldiv_loss

        # decode root code to get reconstruction loss
        obj_losses = decoder.structure_recon_loss(z=root_code, gt_tree=obj)
        for loss_name, loss in obj_losses.items():
            losses[loss_name] = losses[loss_name] + loss

        # visu
        if is_valdt and (epoch + 1) % conf.num_epoch_every_visu == 0:
            visu_dir = os.path.join(conf.exp_name, 'val_visu')
            out_dir = os.path.join(visu_dir, 'epoch-%03d' % epoch)
            
            if batch_ind == 0 and obj_id == 0:
                to_visu = False
                os.mkdir(out_dir)

            with torch.no_grad():
                obj_rel = decoder.decode_structure(z=root_code, max_depth=conf.max_tree_depth)
                obj_rel.get_arrbox()

                with open(os.path.join(out_dir, 'data-%03d.txt'%obj_id), 'w') as fout:
                    fout.write(name+'\n\n')
                    fout.write('GT Hierarchy:\n')
                    fout.write(str(obj)+'\n')
                    fout.write('\n'+str(obj.root.arrange)+'\n\n')
                    fout.write('PRED Hierarchy:\n')
                    fout.write(str(obj_rel)+'\n\n')
                    fout.write('\n'+str(obj_rel.root.arrange)+'\n')
                vis_utils.draw_partnet_objects([obj, obj_rel],\
                        object_names=['GT', 'PRED'], \
                        out_fn=os.path.join(out_dir, 'data-%03d.png'%obj_id), \
                        leafs_only=True, sem_colors_filename='./part_colors_magazine.txt',figsize=(10, 6))
    for loss_name in losses.keys():
        losses[loss_name] = losses[loss_name] / len(objects)

    if epoch <= 50:
        loss_weight_kldiv = conf.loss_weight_kldiv * (float(epoch) / 50.0)
    else:
        loss_weight_kldiv = conf.loss_weight_kldiv

    losses['box'] *= conf.loss_weight_box
    losses['leaf'] *= conf.loss_weight_leaf
    losses['arrange'] *= conf.loss_weight_box
    losses['exists'] *= conf.loss_weight_exists
    losses['semantic'] *= conf.loss_weight_semantic
    losses['kldiv'] *= loss_weight_kldiv

    total_loss = 0
    for loss in losses.values():
        total_loss += loss
    
    with torch.no_grad():
        # log to console
        if log_console:
            # print(
            #     f'''|{strftime("%H:%M:%S", time.gmtime(time.time()-start_time)):>9s} '''
            #     f'''{epoch:>4.0f}/{conf.epochs:<4.0f} '''
            #     f'''{'val' if is_valdt else 'train':^5s} '''
            #     f'''{batch_ind:>4.0f}/{num_batch:<4.0f} '''
            #     f'''{100. * (1+batch_ind+num_batch*epoch) / (num_batch*conf.epochs):>3.0f}% | '''
            #     f'''{losses['box'].item():>6.2f} '''
            #     f'''{losses['leaf'].item():>6.2f} '''
            #     f'''{losses['arrange'].item():>6.2f} '''
            #     f'''{losses['exists'].item():>6.2f} '''
            #     f'''{losses['semantic'].item():>7.2f} '''
            #     f'''{losses['kldiv'].item():>5.2f} | '''
            #     f'''{total_loss.item():>6.2f} |''')
            flog.write(
                f'''|{strftime("%H:%M:%S", time.gmtime(time.time()-start_time)):>9s} '''
                f'''{epoch:>4.0f}/{conf.epochs:<4.0f} '''
                f'''{'val' if is_valdt else 'train':^5s} '''
                f'''{batch_ind:>4.0f}/{num_batch:<4.0f} '''
                f'''{100. * (1+batch_ind+num_batch*epoch) / (num_batch*conf.epochs):>3.0f}% | '''
                f'''{losses['box'].item():>5.2f} '''
                f'''{losses['leaf'].item():>6.2f} '''
                f'''{losses['exists'].item():>6.2f} '''
                f'''{losses['semantic'].item():>7.2f} '''
                f'''{losses['kldiv'].item():>5.2f} | '''
                f'''{total_loss.item():>6.2f} |\n''')
            flog.flush()

        # log to tensorboard
        if log_tb and tb_writer is not None:
            tb_writer.add_scalar('loss', total_loss.item(), step)
            tb_writer.add_scalar('lr', lr, step)
            tb_writer.add_scalar('box_loss', losses['box'].item(), step)
            tb_writer.add_scalar('leaf_loss', losses['leaf'].item(), step)
            tb_writer.add_scalar('arrange_loss', losses['arrange'].item(), step)
            tb_writer.add_scalar('exists_loss', losses['exists'].item(), step)
            tb_writer.add_scalar('semantic_loss', losses['semantic'].item(), step)
            tb_writer.add_scalar('kldiv_loss', losses['kldiv'].item(), step)

    return total_loss, losses

if __name__ == '__main__':
    sys.setrecursionlimit(5000) # this code uses recursion a lot for code simplicity

    parser = ArgumentParser()
    
    exp = 'publay-12W'
    
    parser.add_argument('--exp_name', type=str, default='./logs/' + exp, help='name of the training run')
    parser.add_argument('--category', type=str, default='magazine', help='object category')
    parser.add_argument('--device', type=str, default='cuda:1', help='cpu or cuda:x for using cuda on GPU number x')
    parser.add_argument('--seed', type=int, default=3124256514, help='random seed (for reproducibility)')
    parser.add_argument('--data_path', type=str, default='/home/weiran/Projects/RvNN-Layout/data/publaynet-ours/' + exp)
    parser.add_argument('--train_dataset', type=str, default='train.txt', help='file name for the list of object names for training')
    parser.add_argument('--val_dataset', type=str, default='val.txt', help='file name for the list of object names for validation')
    parser.add_argument('--pretrained_model', type=str, default=None)

    # model hyperparameters
    parser.add_argument('--feature_size', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--max_tree_depth', type=int, default=8, help='maximum depth of generated object trees')
    parser.add_argument('--max_child_num', type=int, default=5, help='maximum number of children per parent')

    # training parameters
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--lr_decay_by', type=float, default=0.9)
    parser.add_argument('--lr_decay_every', type=float, default=1500)
    parser.add_argument('--non_variational', action='store_true', default=False, help='make the variational autoencoder non-variational')

    # loss weights
    parser.add_argument('--loss_weight_kldiv', type=float, default=0.2, help='weight for the kl divergence loss')
    parser.add_argument('--loss_weight_box', type=float, default=10.0, help='weight for the box reconstruction loss')
    parser.add_argument('--loss_weight_leaf', type=float, default=5.0, help='weight for the "node is leaf" reconstruction loss')
    parser.add_argument('--loss_weight_exists', type=float, default=5.0, help='weight for the "node exists" reconstruction loss')
    parser.add_argument('--loss_weight_semantic', type=float, default=7.5, help='weight for the semantic reconstruction loss')

    # logging
    parser.add_argument('--no_tb_log', action='store_true', default=False)
    parser.add_argument('--no_console_log', action='store_true', default=False)
    parser.add_argument('--console_log_interval', type=int, default=5, help='number of optimization steps beween console log prints')
    parser.add_argument('--checkpoint_interval', type=int, default=1000, help='number of optimization steps beween checkpoints')
    parser.add_argument('--num_epoch_every_visu', type=int, default=5, help='number of epochs between every visu')
    config = parser.parse_args()

    Tree.load_category_info(config.category)
    train(config)

