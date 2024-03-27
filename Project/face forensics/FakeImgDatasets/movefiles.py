import os
import pandas as pd
# importing shutil module
import shutil
from tqdm import tqdm
import argparse


def mvfile(image_path, split, args):
    # print(args.dataset)
    if args.dataset == 'original':
        dfx = pd.read_csv(read_fldr[split])
        dfx['0'] = dfx['0'].astype(str).apply(lambda i: '{0:0>3}'.format(i))
        dfx['1'] = dfx['1'].astype(str).apply(lambda i: '{0:0>3}'.format(i))
        df = pd.DataFrame()
        df['flname'] = pd.concat([dfx['0'], dfx['1']], ignore_index=True)
        df['flname'].astype(str)
        # print(df.shape)
    else:
        df = pd.read_csv(read_fldr[split])
        df['flname'] = df['0'].astype(str).apply(lambda i: '{0:0>3}'.format(i)) + '_' + df['1'].astype(str).apply(
            lambda i: '{0:0>3}'.format(i))
        df['flname'].astype(str)

    # # path
    # path = 'manipulated_sequences/Face2Face/c23/images'
    # print("Before moving file:")
    # # Destination path
    # destination = src + folder + '/c23/' + split + '/'
    destination = os.path.join(args.out_path, split)

    if not os.path.exists(destination):
        print(destination)
        os.makedirs(destination)
    i = 0
    for file in tqdm(list(df['flname'])):

        sourcex = os.path.join(image_path, file)
        # print(os.path.exists(r'./original_sequences/youtube/c23/images/429_404'))
        # print(sourcex)
        if os.path.exists(sourcex):
            # print('h')
            for f in os.listdir(sourcex):
                # print(f)
                if f.endswith('.png'):
                    fl = os.path.join(sourcex, f'{file}_{i}.png')
                    os.rename(os.path.join(sourcex, f), fl)
                    dest = shutil.move(fl, destination)
                    # print(dest)
                    i += 1
            # print(file)


if __name__ == '__main__':

    DATASET_PATHS = {
        'original': 'original_sequences/youtube',
        'Deepfakes': 'manipulated_sequences/Deepfakes',
        # 'Face2Face': 'manipulated_sequences/Face2Face',
        # 'FaceSwap': 'manipulated_sequences/FaceSwap'
    }
    """Extracts all videos of a specified method and compression in the
    FaceForensics++ file structure"""
    compression = 'c23'
    data_path = './'
    read_fldr = {'validation': 'val_split.csv', 'train': 'train_split.csv', 'test': 'test_split.csv'}
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # p.add_argument('--data_path', type=str)
    p.add_argument('--dataset', '-d', type=str,
                   choices=list(DATASET_PATHS.keys()) + ['all'],
                   default='all')
    #    p.add_argument('--compression', '-c', type=str, choices=COMPRESSION,
    #                   default='c0')
    p.add_argument('--split', '-s', type=str,
                   choices=list(read_fldr.keys()) + ['all'],
                   default='all')
    args = p.parse_args()
    if args.dataset == 'all':
        for dataset in DATASET_PATHS.keys():
            images_path = os.path.join(data_path, DATASET_PATHS[dataset], compression, 'images')
            args.out_path = os.path.join(data_path, DATASET_PATHS[dataset], compression)
            args.dataset = dataset
            if args.split == 'all':
                for split in read_fldr.keys():
                    mvfile(images_path, split, args)
            else:
                mvfile(images_path, args.split, args)
    else:
        dataset = args.dataset
        images_path = os.path.join(data_path, DATASET_PATHS[dataset], compression, 'images')
        args.out_path = os.path.join(data_path, DATASET_PATHS[dataset], compression)
        if args.split == 'all':
            for split in read_fldr.keys():
                mvfile(images_path, split, args)
        else:
            mvfile(images_path, args.split, args)