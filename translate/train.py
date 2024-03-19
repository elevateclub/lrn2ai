import argparse
import os

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
    parser.add_argument('-warmup', type=int, default=4000)
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

    optimizer = ScheduledOptim()

    train(transformer, training_data, validation_data, optimizer, device, opt)

def prepare_dataloaders_from_bpe_files(opt, dev):
    return (0, 0)

def prepare_dataloaders(opt, dev):
    return (0, 0)

def train(model, training_data, validation_data, optimizer, device, opt):
    pass

if __name__ == "__main__":
    main()