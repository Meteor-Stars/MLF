import argparse
import os
import time
import torch
from exp.exp_main_public import Exp_Main
import random
import numpy as np
import pandas as pd


def get_file_info(directory):
    file_info_list = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            parent_dir = os.path.basename(os.path.dirname(file_path))
            grandparent_dir = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
            file_info_list.append((grandparent_dir, parent_dir, filename))
    return file_info_list


def main():
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--use_multi_scale', action='store_true', help='using mult-scale')
    parser.add_argument('--prob_forecasting', action='store_true', help='using probabilistic forecasting')
    parser.add_argument('--scales', default=[16, 8, 4, 2, 1], help='scales in mult-scale')
    parser.add_argument('--scale_factor', type=int, default=2, help='scale factor for upsample')

    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=4, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    parser.add_argument('--reconstruct_loss', action='store_true',
                        help='whether to use reconstruction loss for patch squeeze', default=False)
    parser.add_argument('--LWI', action='store_true',
                        help='Learnable Weighted-average Integration', default=False)
    parser.add_argument('--MAP', action='store_true',
                        help='Multi-period self-Adaptive Patching', default=True)
    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=3, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
    args = parser.parse_args()

    args.speed_mode = True
    # data_type_all = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather','national_illness', 'exchange_rate','traffic','electricity']
    data_type = 'weather'
    pred_len_ = 720
    seed = 1986  # 2021 2023
    args.gpu = 0
    fix_seed = seed
    torch.manual_seed(fix_seed)
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    args.mode = 'long-term'
    args.data_type = data_type
    args.train_only = False
    args.loss = 'mse'
    args.model = 'MLF'
    args.target = 'OT'
    args.root_path = './dataset/Public_Datasets/'

    args.fixed_patch_num = 64
    args.MAP_alpha = 2
    ###equal patch
    args.MAP = True  # Multi-period self-Adaptive patching
    args.embed_dim = 3
    if args.data_type == 'ETTh1':
        args.data_path = 'ETTh1.csv'
        args.data = 'ETTh1'
        args.model_id = 'ETTh1'
        args.data_real = args.data
        args.enc_in = 7
        args.n_heads = 4
        args.d_model = 64 * 2
        args.d_ff = 128
        args.dropout = 0.3
        args.fc_dropout = 0.3
        args.head_dropout = 0
        args.activation_tag = False
        args.D_norm = True
        args.revin_norm = False
        args.embed_dim = 3
        args.redundancy_scaling = True
        if pred_len_ == 192 or pred_len_ == 96:  #
            args.scal_all = [128, 384, 512, 768, 1024]
            # [patch_length L=int(n^s/N)*alpha, stride K=int(n^s/N)]
            args.patchLen_stride_all = []
            args.equal_patch_len = []
            for i, period_s in enumerate(args.scal_all):
                args.patchLen_stride_all.append(
                    [int(period_s / args.fixed_patch_num) * args.MAP_alpha, int(period_s / args.fixed_patch_num)])
                args.equal_patch_len.append(int(period_s / args.fixed_patch_num) * args.MAP_alpha)
            if not args.MAP:
                # using fixed patch length
                args.patchLen_stride_all = [args.patchLen_stride_all[-1] for _ in range(len(args.scal_all))]
            else:
                # using self-adaptive patch length and stride
                pass
            args.max_patch_len = args.patchLen_stride_all[-1][0]
            args.patch_squeeze = True
            args.squeeze_factor = [4 for _ in range(len(args.scal_all))]

        elif pred_len_ == 336:
            args.scal_all = [128, 512]
            # [patch_length L=int(n^s/N)*alpha, stride K=int(n^s/N)]
            args.patchLen_stride_all = []
            args.equal_patch_len = []
            for i, period_s in enumerate(args.scal_all):
                args.patchLen_stride_all.append(
                    [int(period_s / args.fixed_patch_num) * args.MAP_alpha, int(period_s / args.fixed_patch_num)])
                args.equal_patch_len.append(int(period_s / args.fixed_patch_num) * args.MAP_alpha)
            if not args.MAP:
                # using fixed patch length
                args.patchLen_stride_all = [args.patchLen_stride_all[-1] for _ in range(len(args.scal_all))]
            else:
                # using self-adaptive patch length and stride
                pass
            args.max_patch_len = args.patchLen_stride_all[-1][0]
            args.patch_squeeze = True
            args.squeeze_factor = [2 for _ in range(len(args.scal_all))]

        elif pred_len_ == 720:
            # args.D_norm = False
            # args.revin_norm = True

            args.scal_all = [128, 384]
            # [patch_length L=int(n^s/N)*alpha, stride K=int(n^s/N)]
            args.patchLen_stride_all = []
            args.equal_patch_len = []
            for i, period_s in enumerate(args.scal_all):
                args.patchLen_stride_all.append(
                    [int(period_s / args.fixed_patch_num) * args.MAP_alpha, int(period_s / args.fixed_patch_num)])
                args.equal_patch_len.append(int(period_s / args.fixed_patch_num) * args.MAP_alpha)
            if not args.MAP:
                # using fixed patch length
                args.patchLen_stride_all = [args.patchLen_stride_all[-1] for _ in range(len(args.scal_all))]
            else:
                # using self-adaptive patch length and stride
                pass
            args.max_patch_len = args.patchLen_stride_all[-1][0]
            args.patch_squeeze = True
            args.squeeze_factor = [2 for _ in range(len(args.scal_all))]
        args.max_patch_len = args.patchLen_stride_all[-1][0]

    elif args.data_type == 'ETTh2':
        args.data_path = 'ETTh2.csv'
        args.data = 'ETTh2'
        args.data_real = args.data
        args.model_id = 'ETTh2'
        args.enc_in = 7
        args.n_heads = 4
        args.d_model = 64 * 2
        args.d_ff = 128
        args.dropout = 0.3
        args.fc_dropout = 0.3
        args.head_dropout = 0
        args.activation_tag = False
        args.redundancy_scaling = True
        args.embed_dim = 3
        args.D_norm = False
        args.revin_norm = True
        if pred_len_ == 96 or pred_len_ == 192 or pred_len_ == 336:
            args.scal_all = [128, 384, 512, 768, 1024]
            # [patch_length L=int(n^s/N)*alpha, stride K=int(n^s/N)]
            args.patchLen_stride_all = []
            args.equal_patch_len = []
            for i, period_s in enumerate(args.scal_all):
                args.patchLen_stride_all.append(
                    [int(period_s / args.fixed_patch_num) * args.MAP_alpha, int(period_s / args.fixed_patch_num)])
                args.equal_patch_len.append(int(period_s / args.fixed_patch_num) * args.MAP_alpha)
            if not args.MAP:
                # using fixed patch length
                args.patchLen_stride_all = [args.patchLen_stride_all[-1] for _ in range(len(args.scal_all))]
            else:
                # using self-adaptive patch length and stride
                pass
            args.max_patch_len = args.patchLen_stride_all[-1][0]
            args.patch_squeeze = True
            args.squeeze_factor = [4 for _ in range(len(args.scal_all))]

        elif pred_len_ == 720:
            args.LWI = True
            args.scal_all = [128, 512]
            args.scal_all = [128, 384, 512]
            # [patch_length L=int(n^s/N)*alpha, stride K=int(n^s/N)]
            args.patchLen_stride_all = []
            args.equal_patch_len = []
            for i, period_s in enumerate(args.scal_all):
                args.patchLen_stride_all.append(
                    [int(period_s / args.fixed_patch_num) * args.MAP_alpha, int(period_s / args.fixed_patch_num)])
                args.equal_patch_len.append(int(period_s / args.fixed_patch_num) * args.MAP_alpha)
            if not args.MAP:
                # using fixed patch length
                args.patchLen_stride_all = [args.patchLen_stride_all[-1] for _ in range(len(args.scal_all))]
            else:
                # using self-adaptive patch length and stride
                pass
            # args.max_patch_len = args.patchLen_stride_all[-1][0]
            args.patch_squeeze = True
            args.squeeze_factor = [2 for _ in range(len(args.scal_all))]
        args.max_patch_len = args.patchLen_stride_all[-1][0]

    elif args.data_type == 'ETTm1':
        args.data_path = 'ETTm1.csv'
        args.data = 'ETTm1'
        args.model_id = 'ETTm1'
        args.data_real = 'ETTm1'
        args.enc_in = 7
        args.n_heads = 16
        args.d_model = 64 * 2
        args.d_ff = 256
        args.dropout = 0.2
        args.fc_dropout = 0.2
        args.head_dropout = 0
        args.D_norm = False
        args.revin_norm = True
        args.redundancy_scaling = True
        args.activation_tag = False
        args.embed_dim = 3
        args.patch_squeeze = True

        if pred_len_ == 96 or pred_len_ == 192:
            args.scal_all = [128, 384, 512, 768]
            # [patch_length L=int(n^s/N)*alpha, stride K=int(n^s/N)]
            args.patchLen_stride_all = []
            args.equal_patch_len = []
            for i, period_s in enumerate(args.scal_all):
                args.patchLen_stride_all.append(
                    [int(period_s / args.fixed_patch_num) * args.MAP_alpha, int(period_s / args.fixed_patch_num)])
                args.equal_patch_len.append(int(period_s / args.fixed_patch_num) * args.MAP_alpha)
            if not args.MAP:
                # using fixed patch length
                args.patchLen_stride_all = [args.patchLen_stride_all[-1] for _ in range(len(args.scal_all))]
            else:
                # using self-adaptive patch length and stride
                pass
            args.max_patch_len = args.patchLen_stride_all[-1][0]
            args.patch_squeeze = True
            args.squeeze_factor = [4 for _ in range(len(args.scal_all))]
        elif pred_len_ == 336:
            args.scal_all = [128, 384, 512, 768, 1024]
            # [patch_length L=int(n^s/N)*alpha, stride K=int(n^s/N)]
            args.patchLen_stride_all = []
            args.equal_patch_len = []
            for i, period_s in enumerate(args.scal_all):
                args.patchLen_stride_all.append(
                    [int(period_s / args.fixed_patch_num) * args.MAP_alpha, int(period_s / args.fixed_patch_num)])
                args.equal_patch_len.append(int(period_s / args.fixed_patch_num) * args.MAP_alpha)
            if not args.MAP:
                # using fixed patch length
                args.patchLen_stride_all = [args.patchLen_stride_all[-1] for _ in range(len(args.scal_all))]
            else:
                # using self-adaptive patch length and stride
                pass
            args.max_patch_len = args.patchLen_stride_all[-1][0]
            args.equal_patch_len_or = [4, 12, 16, 24, 32]
            args.patch_squeeze = True
            args.squeeze_factor = [8 for _ in range(len(args.scal_all))]
        elif pred_len_ == 720:
            args.LWI = True

            # args.scal_all = [128, 384, 512, 768, 1024]
            args.scal_all = [128, 384, 512, 1024]
            # args.scal_all = [128, 512,  1024]
            # [patch_length L=int(n^s/N)*alpha, stride K=int(n^s/N)]
            args.patchLen_stride_all = []
            args.equal_patch_len = []
            for i, period_s in enumerate(args.scal_all):
                args.patchLen_stride_all.append(
                    [int(period_s / args.fixed_patch_num) * args.MAP_alpha, int(period_s / args.fixed_patch_num)])
                args.equal_patch_len.append(int(period_s / args.fixed_patch_num) * args.MAP_alpha)
            if not args.MAP:
                # using fixed patch length
                args.patchLen_stride_all = [args.patchLen_stride_all[-1] for _ in range(len(args.scal_all))]
            else:
                # using self-adaptive patch length and stride
                pass
            args.max_patch_len = args.patchLen_stride_all[-1][0]
            # args.equal_patch_len_or = [4, 12, 16, 24, 32]
            args.patch_squeeze = True
            args.squeeze_factor = [8 for _ in range(len(args.scal_all))]

        args.max_patch_len = args.patchLen_stride_all[-1][0]

    elif args.data_type == 'ETTm2':
        args.data_path = 'ETTm2.csv'
        args.data = 'ETTm2'
        args.data_real = 'ETTm2'
        args.model_id = 'ETTm2'
        args.enc_in = 7
        args.e_layers = 1  # 1
        args.n_heads = 16
        args.d_model = 64
        args.d_ff = 256
        args.dropout = 0.2
        args.fc_dropout = 0.2
        args.head_dropout = 0
        args.D_norm = False
        args.revin_norm = True
        args.redundancy_scaling = True
        args.activation_tag = False
        args.LWI = False
        args.embed_dim = 3
        if pred_len_ == 96:
            args.scal_all = [128, 384, 512, 768]
            # [patch_length L=int(n^s/N)*alpha, stride K=int(n^s/N)]
            args.patchLen_stride_all = []
            args.equal_patch_len = []
            for i, period_s in enumerate(args.scal_all):
                args.patchLen_stride_all.append(
                    [int(period_s / args.fixed_patch_num) * args.MAP_alpha, int(period_s / args.fixed_patch_num)])
                args.equal_patch_len.append(int(period_s / args.fixed_patch_num) * args.MAP_alpha)
            if not args.MAP:
                # using fixed patch length
                args.patchLen_stride_all = [args.patchLen_stride_all[-1] for _ in range(len(args.scal_all))]
            else:
                # using self-adaptive patch length and stride
                pass
            args.max_patch_len = args.patchLen_stride_all[-1][0]
            args.equal_patch_len_or = [4, 12, 16, 24]
            args.patch_squeeze = True
            args.squeeze_factor = [4 for _ in range(len(args.scal_all))]
        elif pred_len_ == 192:
            args.scal_all = [128, 384, 512, 768, 1024]
            # [patch_length L=int(n^s/N)*alpha, stride K=int(n^s/N)]
            args.patchLen_stride_all = []
            args.equal_patch_len = []
            for i, period_s in enumerate(args.scal_all):
                args.patchLen_stride_all.append(
                    [int(period_s / args.fixed_patch_num) * args.MAP_alpha, int(period_s / args.fixed_patch_num)])
                args.equal_patch_len.append(int(period_s / args.fixed_patch_num) * args.MAP_alpha)
            if not args.MAP:
                # using fixed patch length
                args.patchLen_stride_all = [args.patchLen_stride_all[-1] for _ in range(len(args.scal_all))]
            else:
                # using self-adaptive patch length and stride
                pass
            args.max_patch_len = args.patchLen_stride_all[-1][0]
            args.equal_patch_len_or = [4, 12, 16, 24, 32]

            args.patch_squeeze = True
            args.squeeze_factor = [4 for _ in range(len(args.scal_all))]
        elif pred_len_ == 336 or pred_len_ == 720:
            args.scal_all = [128, 384, 512, 768, 1024]
            # [patch_length L=int(n^s/N)*alpha, stride K=int(n^s/N)]
            args.patchLen_stride_all = []
            args.equal_patch_len = []
            for i, period_s in enumerate(args.scal_all):
                args.patchLen_stride_all.append(
                    [int(period_s / args.fixed_patch_num) * args.MAP_alpha, int(period_s / args.fixed_patch_num)])
                args.equal_patch_len.append(int(period_s / args.fixed_patch_num) * args.MAP_alpha)
            if not args.MAP:
                # using fixed patch length
                args.patchLen_stride_all = [args.patchLen_stride_all[-1] for _ in range(len(args.scal_all))]
            else:
                # using self-adaptive patch length and stride
                pass
            args.max_patch_len = args.patchLen_stride_all[-1][0]
            args.equal_patch_len_or = [4, 12, 16, 24, 32]
            args.patch_squeeze = True
            args.squeeze_factor = [8 for _ in range(len(args.scal_all))]
        args.max_patch_len = args.patchLen_stride_all[-1][0]


    elif args.data_type == 'weather':
        args.LWI = True
        args.data_path = 'weather.csv'
        args.model_id = 'weather'
        args.data = 'custom'
        args.data_real = 'weather'
        args.enc_in = 21
        args.n_heads = 16
        args.d_model = 128
        args.d_ff = 256
        args.dropout = 0.2
        args.fc_dropout = 0.2
        args.head_dropout = 0
        args.redundancy_scaling = True
        args.activation_tag = True
        args.patch_squeeze = True
        args.D_norm = False
        args.revin_norm = True

        args.embed_dim = 3

        if pred_len_ == 96 or pred_len_ == 192 or pred_len_ == 336:

            args.scal_all = [128, 384, 512, 768]
            # [patch_length L=int(n^s/N)*alpha, stride K=int(n^s/N)]
            args.patchLen_stride_all = []
            args.equal_patch_len = []
            for i, period_s in enumerate(args.scal_all):
                args.patchLen_stride_all.append(
                    [int(period_s / args.fixed_patch_num) * args.MAP_alpha, int(period_s / args.fixed_patch_num)])
                args.equal_patch_len.append(int(period_s / args.fixed_patch_num) * args.MAP_alpha)
            if not args.MAP:
                # using fixed patch length
                args.patchLen_stride_all = [args.patchLen_stride_all[-1] for _ in range(len(args.scal_all))]
            else:
                # using self-adaptive patch length and stride
                pass

            args.max_patch_len = args.patchLen_stride_all[-1][0]
            args.max_patch_len = args.patchLen_stride_all[-1][0]
            args.squeeze_factor = [4 for _ in range(len(args.scal_all))]

        elif pred_len_ == 720:
            args.scal_all = [128, 512, 1024, 2048]
            # [patch_length L=int(n^s/N)*alpha, stride K=int(n^s/N)]
            args.patchLen_stride_all = []
            args.equal_patch_len = []
            for i, period_s in enumerate(args.scal_all):
                args.patchLen_stride_all.append(
                    [int(period_s / args.fixed_patch_num) * args.MAP_alpha, int(period_s / args.fixed_patch_num)])
                args.equal_patch_len.append(int(period_s / args.fixed_patch_num) * args.MAP_alpha)
            if not args.MAP:
                # using fixed patch length
                args.patchLen_stride_all = [args.patchLen_stride_all[-1] for _ in range(len(args.scal_all))]
            else:
                # using self-adaptive patch length and stride
                pass
            args.squeeze_factor = [8 for _ in range(len(args.scal_all))]
        args.max_patch_len = args.patchLen_stride_all[-1][0]
    elif args.data_type == 'traffic':
        args.data_path = 'traffic.csv'
        args.data_real='traffic'
        args.data = 'custom'
        args.model_id = 'traffic'
        args.enc_in = 862
        args.e_layers = 1  # 1
        # args.n_heads = 16
        args.n_heads = 8
        args.d_model = 128
        # args.d_model = 64
        args.d_ff = 256
        args.dropout = 0.2
        args.fc_dropout = 0.2
        args.head_dropout = 0
        args.batch_size=24//2
        # args.batch_size=24
        args.learning_rate=0.0001
        args.e_layers = 3

        args.redundancy_scaling = True
        args.activation_tag = True
        args.patch_squeeze = True
        args.D_norm = False
        args.revin_norm = True
        args.scal_all =[512, 1024, 1536]
        args.LWI = False

        args.patchLen_stride_all = []
        args.equal_patch_len = []
        for i, period_s in enumerate(args.scal_all):
            args.patchLen_stride_all.append(
                [int(period_s / args.fixed_patch_num) * args.MAP_alpha, int(period_s / args.fixed_patch_num)])
            args.equal_patch_len.append(int(period_s / args.fixed_patch_num) * args.MAP_alpha)
        if not args.MAP:
            # using fixed patch length
            args.patchLen_stride_all = [args.patchLen_stride_all[-1] for _ in range(len(args.scal_all))]
        else:
            # using self-adaptive patch length and stride
            pass
        args.squeeze_factor = [4 for _ in range(len(args.scal_all))]
        args.max_patch_len = args.patchLen_stride_all[-1][0]
    elif args.data_type == 'electricity':
        args.data_path = 'electricity.csv'
        args.data = 'custom'
        args.model_id = 'electricity'
        args.data_real='electricity'
        args.enc_in = 321
        args.e_layers = 1  # 1
        # args.n_heads = 16
        args.n_heads = 8
        args.d_model = 128
        args.d_ff = 256
        args.dropout = 0.2
        args.fc_dropout = 0.2
        args.head_dropout = 0
        args.batch_size=32
        args.learning_rate=0.0001
        args.e_layers = 3

        args.redundancy_scaling = True
        args.activation_tag = True
        args.patch_squeeze = True
        args.D_norm = False
        args.revin_norm = True
        args.scal_all =[512, 1024, 1536]
        args.LWI = False

        args.patchLen_stride_all = []
        args.equal_patch_len = []
        for i, period_s in enumerate(args.scal_all):
            args.patchLen_stride_all.append(
                [int(period_s / args.fixed_patch_num) * args.MAP_alpha, int(period_s / args.fixed_patch_num)])
            args.equal_patch_len.append(int(period_s / args.fixed_patch_num) * args.MAP_alpha)
        if not args.MAP:
            # using fixed patch length
            args.patchLen_stride_all = [args.patchLen_stride_all[-1] for _ in range(len(args.scal_all))]
        else:
            # using self-adaptive patch length and stride
            pass
        args.squeeze_factor = [4 for _ in range(len(args.scal_all))]
        args.max_patch_len = args.patchLen_stride_all[-1][0]
    elif args.data_type == 'national_illness':
        args.data_path = 'national_illness.csv'
        args.data = 'custom'
        args.model_id = 'national_illness'
        args.data_real='national_illness'
        args.enc_in = 7
        args.e_layers = 1  # 1
        args.n_heads = 4
        args.d_model = 16
        args.d_ff = 128
        args.dropout = 0.3
        args.fc_dropout = 0.3
        args.head_dropout = 0
        args.batch_size=16
        args.learning_rate=0.0025
        args.e_layers = 3

        args.redundancy_scaling = True
        args.activation_tag = True
        args.patch_squeeze = True
        args.D_norm = False
        args.revin_norm = True
        args.scal_all=[[384,512, 1024]]
        args.LWI = False

        args.patchLen_stride_all = []
        args.equal_patch_len = []
        for i, period_s in enumerate(args.scal_all):
            args.patchLen_stride_all.append(
                [int(period_s / args.fixed_patch_num) * args.MAP_alpha, int(period_s / args.fixed_patch_num)])
            args.equal_patch_len.append(int(period_s / args.fixed_patch_num) * args.MAP_alpha)
        if not args.MAP:
            # using fixed patch length
            args.patchLen_stride_all = [args.patchLen_stride_all[-1] for _ in range(len(args.scal_all))]
        else:
            # using self-adaptive patch length and stride
            pass
        args.squeeze_factor = [4 for _ in range(len(args.scal_all))]
        args.max_patch_len = args.patchLen_stride_all[-1][0]
    elif args.data_type == 'exchange_rate':
        args.data_path = 'exchange_rate.csv'
        args.data = 'custom'
        args.model_id = 'exchange_rate'
        args.data_real='exchange_rate'
        args.enc_in = 8
        args.e_layers = 1  # 1
        args.n_heads = 4
        args.d_model = 128
        args.d_ff = 128
        args.dropout = 0.3
        args.fc_dropout = 0.3
        args.head_dropout = 0
        args.batch_size=32
        args.learning_rate=0.0025
        args.e_layers = 3

        args.redundancy_scaling = True
        args.activation_tag = True
        args.patch_squeeze = True
        args.D_norm = False
        args.revin_norm = True
        args.scal_all=[[384,512, 1024]]
        args.LWI = False

        args.patchLen_stride_all = []
        args.equal_patch_len = []
        for i, period_s in enumerate(args.scal_all):
            args.patchLen_stride_all.append(
                [int(period_s / args.fixed_patch_num) * args.MAP_alpha, int(period_s / args.fixed_patch_num)])
            args.equal_patch_len.append(int(period_s / args.fixed_patch_num) * args.MAP_alpha)
        if not args.MAP:
            # using fixed patch length
            args.patchLen_stride_all = [args.patchLen_stride_all[-1] for _ in range(len(args.scal_all))]
        else:
            # using self-adaptive patch length and stride
            pass
        args.squeeze_factor = [4 for _ in range(len(args.scal_all))]
        args.max_patch_len = args.patchLen_stride_all[-1][0]
    args.learning_rate = 1e-4
    args.batch_size = 32 * 4
    args.is_training = True
    args.model_id = 0
    args.patch_pad = True
    args.shared_num = 3

    args.seq_len = args.scal_all[-1]
    args.context_window = None
    args.threshold_patch_num = 7  # When the number of patches is  less than or equal to this value, no patch squeeze is performed
    args.explore_fund_memory = False
    args.pred_len = pred_len_
    args.state = 'train'
    args.checkpoints = './checkpoints_MLF_longterm/' + args.data_real + '/' + args.model + '/' + 'random_seed_' + str(
        seed)

    extra = 'MultiPeriod'
    for period in args.scal_all:
        extra = extra + '_' + str(period)
    args.individual = 0
    args.d_layers = 1
    args.factor = 3
    args.label_len = 0
    args.is_training = True
    args.only_test = False
    args.train_epochs = 10


    args.record = True

    args.itr = 1
    args.e_layers = 3
    args.script_id = '0_'  # model checkpoint id
    args.device = 'cuda:' + str(args.gpu)
    print('Args in experiment:')
    # extra
    print(vars(args))
    Exp = Exp_Main
    args.is_training = True
    if args.is_training:
        for ii in range(args.itr):
            setting = f'{args.data_real}_{args.model}_{extra}_pl{args.pred_len}'
            args.save_path = os.path.join(args.checkpoints, setting)
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            exp = Exp(args)  # set experiments

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)

            # if args.do_predict:
            #     print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            #     exp.predict(setting, True)

            torch.cuda.empty_cache()



if __name__ == "__main__":
    main()
