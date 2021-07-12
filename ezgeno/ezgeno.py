import time
import argparse
import warnings

from utils import set_seed
from dataset import prepare_all_data
from trainer import ezGenoTrainer

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

    parser.add_argument('--trainFileList', type=str ,default="../../ezGeno_exp/v2/SUZ12.training.sequence", help='training file list path')
    parser.add_argument('--testFileList', type=str ,default="../../ezGeno_exp/v2/SUZ12.testing.sequence", help='testing file list path')
    parser.add_argument('--trainLabel', type=str, default="../../ezGeno_exp/v2/SUZ12.training.label", help='testing negative data path')
    parser.add_argument('--testLabel', type=str, default="../../ezGeno_exp/v2/SUZ12.testing.label", help='testing negative data path')
    
    parser.add_argument('--layers', type=int ,nargs='+')
    parser.add_argument('--feature_dim', type=int, nargs='+')
    parser.add_argument('--conv_filter_size_list', type=str)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--load', type=str, default="model.t7", help='model to load')
    parser.add_argument('--save', type=str, default="model.t7", help='model to save')
    parser.add_argument('--seed', help='seed number', type=int, default=0)

    args, _ = parser.parse_known_args()
    print(args)
    set_seed(args.seed)

    if args.eval:
        test_loader, data_source = prepare_all_data(args.trainFileList, args.trainLabel, args.testFileList, args.testLabel, args.batch_size, args.num_workers, args.eval, train_supernet=True)
        trainer = ezGenoTrainer(args, data_source)
        print("loading model and predicting testing sequence")
        trainer.test(trainer.subnet,test_loader)
    else:
        train_loader, valid_loader, test_loader, data_source = prepare_all_data(args.trainFileList, args.trainLabel, args.testFileList, args.testLabel, args.batch_size, args.num_workers, args.eval, train_supernet=True)
        trainer = ezGenoTrainer(args, data_source)
        trainer.train(train_loader, valid_loader, test_loader, enable_stage_1=True, enable_stage_2=True, enable_stage_3=True)
        trainer.test(trainer.subnet, test_loader)

if __name__ == '__main__':
    start_time = time.time()
    main() 
    end_time = time.time()
    duration = end_time - start_time
    print("total time: %.3fs"%(duration))
