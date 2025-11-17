import os
import argparse


def move(src, dst):
    if not os.path.exists(dst):
        os.mkdir(dst)
    if not os.path.exists(os.path.join(dst, 'input')):
        os.mkdir(os.path.join(dst, 'input'))
    if not os.path.exists(os.path.join(dst, 'target')):
        os.mkdir(os.path.join(dst, 'target'))

    folders = os.listdir(src)
    cnt = 0
    for f in folders:
        image_names = os.listdir(os.path.join(src, f, 'input'))

        for i in image_names:
            os.rename(os.path.join(src, f, 'input', i), os.path.join(dst, 'input', f + '_' + i))
            os.rename(os.path.join(src, f, 'target', i), os.path.join(dst, 'target', f + '_' + i))
            cnt += 1
    print('%d images are moved' % cnt)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Directories
    parser.add_argument('--root_src', default='dataset/datasets_Large', type=str)
    parser.add_argument('--root_dst', default='dataset/datasets', type=str)

    args = parser.parse_args()

    if not os.path.exists(args.root_dst):
        os.mkdir(args.root_dst)

    move(os.path.join(args.root_src, 'train'), os.path.join(args.root_dst, 'train'))
    move(os.path.join(args.root_src, 'test'), os.path.join(args.root_dst, 'test'))
