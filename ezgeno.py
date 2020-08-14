import time
import argparse
import warnings

from utils import *
from trainer import ezGenoTrainer,epigenomeTrainer
from network import ezGenoModel,epigenomeModel
from dataset import *
from epigenomeDataset import *

warnings.simplefilter('once', UserWarning)

def main():
    parser = argparse.ArgumentParser("ezGeno")


    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam', 'adagrad'])
    parser.add_argument('--supernet_learning_rate', type=float, default=0.01, help='init learning rate')
    parser.add_argument('--supernet_epochs', type=int, default=100, help='num of supernet training epochs')
    parser.add_argument('--controller_learning_rate', type=float, default=0.001, help='init learning rate')
    parser.add_argument('--controller_optimizer', type=str, default='adam', choices=['sgd', 'adam', 'adagrad'])
    parser.add_argument('--cstep', type=int, default=2000,help='num of training controller steps')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='init learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')

    parser.add_argument('--train_pos_data_path', type=str, default="../SUZ12/SUZ12_positive_augmentation_includeOrig_training.fa", help='training positive data path')
    parser.add_argument('--train_neg_data_path', type=str, default="../SUZ12/SUZ12_negative_dinuclShuffle_augmentation_includeOrig_training.fa", help='training negative data path')
    parser.add_argument('--test_pos_data_path', type=str, default="../SUZ12/SUZ12_positive_test.fa", help='testing positive data path')
    parser.add_argument('--test_neg_data_path', type=str, default="../SUZ12/SUZ12_negative_test.fa", help='testing negative data path')

    parser.add_argument('--train_dNase_path', type=str, default="../dNase/h1hesc_dnase.training.score", help='training score data path')
    parser.add_argument('--train_seq_path', type=str, default="../dNase/h1hesc_dnase.training_input_seq", help='training seq data path')
    parser.add_argument('--train_label_path', type=str, default="../dNase/h1hesc_dnase.training_label", help='training seq label path')
    parser.add_argument('--test_dNase_path', type=str, default="../dNase/h1hesc_dnase.validation.score", help='testing positive data path')
    parser.add_argument('--test_seq_path', type=str, default="../dNase/h1hesc_dnase.validation_input_seq", help='testing negative data path')
    parser.add_argument('--test_label_path', type=str, default="../dNase/h1hesc_dnase.validation_label", help='testing negative data path')

    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--feature_dim', type=int, default=128)
    parser.add_argument('--conv_filter_size_list', type=list, default=[3,7,11,15,19])
    parser.add_argument('--task', type=str, default='TFBind', choices=['TFBind', 'epigenome'])
    parser.add_argument('--dNase_layers', type=int, default=6)
    parser.add_argument('--dNase_feature_dim', type=int, default=64)
    parser.add_argument('--dNase_conv_filter_size_list', type=list, default=[3,7,11])

    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--load', type=str, default="model.t7", help='model to load')
    parser.add_argument('--save', type=str, default="model.t7", help='model to save')


    args, unparsed = parser.parse_known_args()
    print(args)

    print("Task:",args.task)
    if args.task == 'TFBind':
        train_loader, valid_loader, test_loader = prepare_all_data(args.train_pos_data_path,args.train_neg_data_path,args.test_pos_data_path,args.test_neg_data_path, args.batch_size, train_supernet=True)
        trainer = ezGenoTrainer(args)
    elif args.task == 'epigenome':
        train_loader, valid_loader, test_loader = prepare_all_epigenome_data(args.train_seq_path,args.train_dNase_path,args.train_label_path,args.test_seq_path,args.test_dNase_path,args.test_label_path, args.batch_size, train_supernet=True)        
        trainer = epigenomeTrainer(args)
    if args.eval:
        trainer.test(trainer.subnet,test_loader)
    else:
        trainer.train(train_loader, valid_loader, enable_stage_1=True, enable_stage_2=True, enable_stage_3=True)
        trainer.test(trainer.subnet,test_loader)

if __name__ == '__main__':
    start_time = time.time()
    main() 
    end_time = time.time()
    duration = end_time - start_time
    print("total time: %.3fs"%(duration))