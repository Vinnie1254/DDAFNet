import os
import torch
import argparse
from torch.backends import cudnn
from models.OURNet_water import build_net
from train import _train
from eval import _eval


def main(args):

    cudnn.benchmark = True

    if not os.path.exists('results/'):
        os.makedirs(args.model_save_dir)
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = build_net(args.model_name)

    if torch.cuda.is_available():
        model.cuda()
    if args.mode == 'train':
        _train(model, args)

    elif args.mode == 'test':
        _eval(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument('--model_name', default='OURNet_water', choices=['OURNet_water'],
                        type=str)
    parser.add_argument('--data_dir', type=str, default='/home/gxy/PycharmProjects/Model_main_water/datasets/LSUI')
    parser.add_argument('--mode', default='test', choices=['train', 'test'], type=str)

    # Train
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
    parser.add_argument('--lr_steps', type=list, default=[(x+1) * 500 for x in range(1000//500)])


    # Test
    parser.add_argument('--test_model', type=str, default='/home/gxy/PycharmProjects/Model_main_water/results/OURNet_water/LSUI_weights_1020/model_1001.pkl')
    parser.add_argument('--save_image', type=bool, default=True, choices=[True, False]) # 选择保存图像
    parser.add_argument('--result_txt', type=str, default='/home/gxy/PycharmProjects/Model_main_water/results/OURNet_water/LSUI_results_2025_10_20.txt')

    args = parser.parse_args()
    args.model_save_dir = os.path.join('results/', args.model_name, 'weights/') # 用于保存模型权重文件的目录路径
    args.result_dir = os.path.join('results/', args.model_name, '/home/gxy/PycharmProjects/Model_main_water/results/OURNet_water/LSUI_test_visual_1020')
    print(args)
    main(args)




