import os
import torch
import argparse
from torch.backends import cudnn
from models.OURNet_water import build_net
from train import _train
from eval import _eval


def main(args):

    cudnn.benchmark = False


    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    model = build_net(args.model_name)

    device = torch.device('cpu')
    model.to(device)

    if args.mode == 'train':
        _train(model, args)
    elif args.mode == 'test':
        _eval(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument('--model_name', default='OURNet_water', choices=['OURNet_water'], type=str)
    parser.add_argument('--mode', default='test', choices=['train', 'test'], type=str)


    parser.add_argument('--data_dir', type=str,
        default='/home/gxy/PycharmProjects/Model_main_water/datasets/U45/test/input')


    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=1000)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--valid_freq', type=int, default=100)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--lr_steps', type=list, default=[(x + 1) * 500 for x in range(1000 // 500)])


    parser.add_argument('--test_model', type=str,
        default='/home/gxy/PycharmProjects/Model_main_water/results/OURNet_water/model_1000.pkl')
    parser.add_argument('--save_image', type=bool, default=True, choices=[True, False])
    parser.add_argument('--result_txt', type=str,
        default='/home/gxy/PycharmProjects/Model_main_water/results/OURNet_water/U45_results_2025_10_30.txt')

    args = parser.parse_args()
    args.model_save_dir = os.path.join('results/', args.model_name, 'weights/')
    args.result_dir = os.path.join('results/', args.model_name, 'U45_test_visual_1030/')
    print(args)

    main(args)
