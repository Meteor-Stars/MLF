import argparse
import os
import time
import torch
# from exp.exp_main import Exp_Main
from exp.exp_main_public import Exp_Main
import random
import numpy as np


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
    fix_seed = 2021
    # fix_seed = 1995
    torch.manual_seed(fix_seed)
    random.seed(fix_seed)

    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # PatchTST
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
    # parser.add_argument('--model', type=str, required=True, default='Autoformer',
    #         help='model name, options: [Autoformer, Informer, Transformer, Reformer, FEDformer] and their MS versions: [AutoformerMS, InformerMS, etc]')

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

    # supplementary config for FiLM model
    parser.add_argument('--modes1', type=int, default=64, help='modes to be selected random 64')
    parser.add_argument('--mode_type', type=int, default=0)

    # supplementary config for FEDformer model
    parser.add_argument('--version', type=str, default='Wavelets',
                        help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
    parser.add_argument('--mode_select', type=str, default='low',
                        help='for FEDformer, there are two mode selection method, options: [random, low]')
    parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
    parser.add_argument('--L', type=int, default=3, help='ignore level')
    parser.add_argument('--base', type=str, default='legendre', help='mwt base')
    parser.add_argument('--cross_activation', type=str, default='tanh',
                        help='mwt cross atention activation function tanh or softmax')

    # supplementary config for Reformer model
    parser.add_argument('--bucket_size', type=int, default=4, help='for Reformer')
    parser.add_argument('--n_hashes', type=int, default=4, help='for Reformer')
    parser.add_argument('--film_ours', default=True, action='store_true')
    parser.add_argument('--ab', type=int, default=2, help='ablation version')
    parser.add_argument('--ratio', type=float, default=0.5, help='dropout')
    parser.add_argument('--film_version', type=int, default=0, help='compression')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
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

    ##Pyraformer
    # Architecture selection.
    parser.add_argument('-model', type=str, default='Pyraformer')
    parser.add_argument('-decoder', type=str, default='FC')  # selection: [FC, attention]

    # Common Model parameters.
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=512)
    parser.add_argument('-d_k', type=int, default=128)
    parser.add_argument('-d_v', type=int, default=128)
    parser.add_argument('-d_bottleneck', type=int, default=128)
    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layer', type=int, default=4)

    # Pyraformer parameters.
    parser.add_argument('-window_size', type=str,
                        default=[4, 4, 4])  # The number of children of a parent node.
    parser.add_argument('-inner_size', type=int, default=3)  # The number of ajacent nodes.
    # CSCM structure. selection: [Bottleneck_Construct, Conv_Construct, MaxPooling_Construct, AvgPooling_Construct]
    parser.add_argument('-CSCM', type=str, default='Bottleneck_Construct')
    parser.add_argument('-truncate', action='store_true',
                        default=False)  # Whether to remove coarse-scale nodes from the attention structure
    parser.add_argument('-use_tvm', action='store_true', default=False)  # Whether to use TVM.

    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    args = parser.parse_args()

    seed = 1986  ## 2021 2023 1995 2015 2022
    fix_seed = seed
    torch.manual_seed(fix_seed)
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    args.data_type = 'electricity'  # 'ETTh1' 'electricity' 'ETTm1' 'exchange_rate' 'national_illness' 'ETTm2' 'ETTh2'
    args.model = 'NHits'  # NHits FiLM
    args.seq_len = 120
    args.pred_len = 1  #

    args.loss = 'mse'
    args.use_multi_gpu = False
    args.train_epochs = 10
    args.itr = 1
    args.gpu = 0
    args.devices = '1'
    args.script_id = ''
    args.train_only = False
    args.d_model = 512

    args.dived = True

    args.loss = 'mse'

    args.dir_id = '_1'

    args.checkpoints = './checkpoints/' + args.data_type + '/' + args.model + '/' + 'random_seed_' + str(
        fix_seed)

    args.preprocess_data = True
    args.embed_id_size = 64
    args.learning_rate = 1e-4

    args.batch_size = 8
    args.cal_scaler = False

    args.d_model = 128
    args.d_ff = 128
    args.n_heads = 4
    args.train_epochs = 30
    args.wmape = False
    c = 320
    args.enc_in = 320
    args.seq_len = 5
    args.enc_in = c
    args.dec_in = c
    args.c_out = c
    # args.n_heads = 4
    args.pred_len = 1

    args.label_len = 5

    args.pre_single_grand = True
    args.is_training = True
    args.debug = True
    args.only_test = False
    args.preict = False
    args.train_epochs = 15
    args.wmape = False
    args.data = 'custom'
    args.D_norm = True
    tag = ''
    args.itr = 1
    args.is_training = True
    args.debug = True
    args.only_test = False
    args.preict = False
    args.train_epochs = 15
    args.wmape = False
    args.data = 'ETTh1'
    args.D_norm = False
    args.record = True
    args.root_path = './dataset/Public_Datasets/'
    args.batch_size = 128

    if args.data_type == 'ETTh1':
        args.data_path = 'ETTh1.csv'
        args.data = 'ETTh1'
        c = 7
        args.enc_in = c
        args.dec_in = c
        args.c_out = c
        args.n_heads = 4
        args.d_model = 128
        args.d_ff = 128
        args.pred_len = pred_len_
    elif args.data_type == 'exchange_rate':
        args.data = 'custom'
        args.data = 'exchange_rate'
        args.data_path = 'exchange_rate.csv'
        c = 8
        args.enc_in = c
        args.dec_in = c
        args.c_out = c
    elif args.data_type == 'ETTm2':
        args.data = 'ETTm2'
        args.data_path = 'ETTm2.csv'
        c = 7
        args.enc_in = c
        args.dec_in = c
        args.c_out = c
    elif args.data_type == 'ETTm1':
        args.data = 'ETTm1'
        args.data_path = 'ETTm1.csv'
        c = 7
        args.enc_in = c
        args.dec_in = c
        args.c_out = c
    elif args.data_type == 'ETTh2':
        args.data_path = 'ETTh2.csv'
        args.data = 'ETTh2'
        c = 7
        args.enc_in = c
        args.dec_in = c
        args.c_out = c
    elif args.data_type == 'weather':
        args.batch_size = 64
        args.data_path = 'weather.csv'
        args.data = 'custom'
        c = 21
        args.enc_in = c
        args.dec_in = c
        args.c_out = c
    elif args.data_type == 'national_illness':
        args.data_path = 'national_illness.csv'
        args.data = 'national_illness'
        c = 7
        args.enc_in = c
        args.dec_in = c
        args.c_out = c
        args.batch_size = 16
    elif args.data_type == 'electricity':
        args.data_path = 'electricity.csv'
        args.data = 'custom'
        args.model_id = 'electricity'
        c = 321
        args.enc_in = c
        args.dec_in = c
        args.c_out = c
        args.batch_size = 16
    args.learning_rate = 1e-4
    args.batch_size = 128

    if args.data_type == 'national_illness':
        args.batch_size = 32
    if args.data_type == 'electricity':
        args.batch_size = 16

    if args.label_len > args.seq_len:
        args.label_len = args.seq_len
    print('Args in experiment:')
    print(args)

    if args.data_type == 'electricity' or args.data_type == 'national_illness':
        args.batch_size = 16
    Exp = Exp_Main
    if args.is_training:
        for ii in range(args.itr):
            if tag != '':
                setting = f'{args.data_type}_{args.model}_{args.seq_len}_{args.pred_len}_{args.loss}'
            else:
                setting = f'{args.data_type}_{args.model}_{args.seq_len}_{args.pred_len}_{args.loss}'
            args.save_path = os.path.join(args.checkpoints, setting)
            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
            best_model_path = args.save_path + '/' + args.script_id + 'checkpoint.pth'
            os.remove(best_model_path)

    else:
        ii = 0

        if tag != '':
            setting = f'{args.data_type}_{args.model}_{args.seq_len}_{args.pred_len}_{args.loss}_{tag}'
        else:
            setting = f'{args.data_type}_{args.model}_{args.seq_len}_{args.pred_len}_{args.loss}'
        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()