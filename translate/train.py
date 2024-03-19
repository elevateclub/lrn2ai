import argparse
import math
import os
import time

import numpy as np
import torch

from models import Transformer, ScheduledOptim

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_pkl', default='train.pkl')
    parser.add_argument('-train-path', default=None)
    parser.add_argument('-val_path', default=None)

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=2048)
    
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-warmup', '--n_warmup_steps', type=int, default=4000)
    parser.add_argument('-lr_mul', type=float, default=2.0)
    parser.add_argument('-seed', type=int, default=None)

    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')
    parser.add_argument('-scale_emb_or_prj', type=str, default='prj')

    parser.add_argument('-output_dir', type=str, default='out')
    parser.add_argument('-use_tb', action='store_true')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = False
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    
    if opt.batch_size < 2048 and opt.n_warmup_steps <= 4000:
        print("[Warning] The warmup steps may not be enough.")

    device = torch.device('cuda' if opt.cuda else 'cpu')

    # Load data
    if all((opt.train_path, opt.val_path)):
        training_data, validation_data = prepare_dataloaders_from_bpe_files(opt, device)
    elif opt.data_pkl:
        training_data, validation_data = prepare_dataloaders(opt, device)
    else:
        raise Exception("No valid path")

    print(opt)

    transformer = Transformer(
    ).to(device)

    optimizer = ScheduledOptim(
        torch.optim.Adam([torch.nn.parameter.Parameter()], betas=(0.9, 0.98), eps=1e-09), # TODO pass real params
        opt.lr_mul, opt.d_model, opt.n_warmup_steps)

    train(transformer, training_data, validation_data, optimizer, device, opt)

def prepare_dataloaders_from_bpe_files(opt, dev):
    return (0, 0)

def prepare_dataloaders(opt, dev):
    return (0, 0)

def train(model, training_data, validation_data, optimizer, device, opt):
    if opt.use_tb:
        print("[Info] Use Tensorboard")
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=os.path.join(opt.output_dir, "tensorboard"))
    
    log_train_file = os.path.join(opt.output_dir, "train.log")
    log_valid_file = os.path.join(opt.output_dir, "valid.log")

    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        log_tf.write('epoch,loss,ppl,accuracy\n')
        log_vf.write('epoch,loss,ppl,accuracy\n')

    def print_performances(header, ppl, accu, start_time, lr):
        print(' - {header:12} ppl: {ppl:8.5f}, accuracy: {accu:3.3f} %, lr: {lr:8.5f}, elapse: {elapse:3.3f} min'.format(
            header=header, ppl=ppl, accu=100*accu, elapse=(time.time()-start_time)/60, lr=lr))

    valid_losses = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        # train
        start = time.time()
        train_loss, train_accu = train_epoch(model, training_data, optimizer, opt, device, smoothing=opt.label_smoothing)
        train_ppl = math.exp(min(train_loss, 100))
        lr = optimizer._optimizer.param_groups[0]['lr']
        print_performances('training', train_ppl, train_accu, start, lr)

        # eval
        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device, opt)
        valid_ppl = math.exp(min(valid_loss, 100))
        print_performances('validation', valid_ppl, valid_accu, start, lr)

        valid_losses += [train_loss]

        checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': model.state_dict()}

        if opt.save_mode == 'all':
            model_name = 'model_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
            torch.save(checkpoint, model_name)
        elif opt.save_mode == 'best':
            model_name = 'model.chkpt'
            if valid_loss <= min(valid_losses):
                torch.save(checkpoint, os.path.join(opt.output_dir, model_name))
                print('  - [Info] The checkpoint file has been updated')
        
        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write('{epoch},{loss:8.5f},{ppl:8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=train_loss, ppl=train_ppl, accu=100*train_accu))
            log_vf.write('{epoch},{loss:8.5f},{ppl:8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=valid_loss, ppl=valid_ppl, accu=100*valid_accu))
        
        if opt.use_tb:
            tb_writer.addscalars('ppl', {'train': train_ppl, 'val': valid_ppl}, epoch_i)
            tb_writer.addscalars('accuracy', {'train': train_accu*100, 'val': valid_accu*100}, epoch_i)
            tb_writer.addscalars('learning_rate', lr, epoch_i)

def train_epoch(model, training_data, optimizer, opt, device, smoothing):
    return (0, 0)

def eval_epoch(model, validation_data, device, opt):
    return (0, 0)

if __name__ == "__main__":
    main()