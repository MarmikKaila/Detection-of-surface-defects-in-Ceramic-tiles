#-----------------------------------------------------------------------#
#            Surface Defect Detection of Magnetic Tile - CLI            #
#-----------------------------------------------------------------------#
# Runnable equivalent of SurfaceDefectDetection_MagneticTile.ipynb.      #
# The notebook export (surfacedefectdetection_magnetictile.py) cannot be #
# executed directly: it contains Colab shell magics and its imports are  #
# commented out inside a %%capture cell.                                 #
#                                                                        #
# Usage:                                                                 #
#   python main.py                       # full run, 200 epochs          #
#   python main.py --epochs 10           # short run                     #
#   SDD_DEVICE=cpu python main.py        # force a device                #
#-----------------------------------------------------------------------#
import argparse
import os
import random

import matplotlib
matplotlib.use('Agg')          # headless: write figures to disk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from PIL import Image
from torchsummary import summary

from dataset import SurfaceDefectDetectionDataset, partitioning
from device import DEVICE
from inference import get_inference_performance_metrics, plot_prediction_results
from loss import TverskyLoss, WeightedBCELoss
from train import train_2D
from unet import UNet_2D

# plot_prediction_results opens one figure per test image without closing
# them, which trips matplotlib's default 20-figure warning.
plt.rcParams['figure.max_open_warning'] = 0


#-----------------------------------------------------------------------#
#                               set_seed                                #
#-----------------------------------------------------------------------#
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


#-----------------------------------------------------------------------#
#                             parse_args                                #
#      Defaults mirror the "Set the parameters" notebook section        #
#-----------------------------------------------------------------------#
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs', type=int, default=200,
                   help='number of training epochs (notebook default: 200)')
    p.add_argument('--batch-size', type=int, default=32,
                   help='batch size for the train and validation sets')
    p.add_argument('--test-batch-size', type=int, default=1,
                   help='batch size for the test set')
    p.add_argument('--num-workers', type=int, default=0,
                   help='DataLoader worker processes (must stay 0: the '
                        'Dataset moves masks onto the accelerator)')
    p.add_argument('--optimizer', choices=['Adam', 'SGD'], default='Adam')
    p.add_argument('--criterion', choices=['TverskyLoss', 'WeightedBCE'],
                   default='TverskyLoss')
    p.add_argument('--threshold', type=float, default=0.5,
                   help='probability threshold used to binarize predictions')
    p.add_argument('--seed', type=int, default=51)
    p.add_argument('--skip-summary', action='store_true',
                   help='skip the torchsummary layer table')
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    split_ratio = [0.70, 0.10, 0.20]   # (train, val, test)

    print(f'Device: {DEVICE}')
    if DEVICE.type == 'cpu':
        print('Training on CPU - this will be slow.')

    #-------------------------------------------------------------------#
    #                             Dataset                               #
    #-------------------------------------------------------------------#
    if not os.path.isdir('data'):
        raise SystemExit(
            "Missing 'data/' directory. Download the dataset first:\n"
            "  curl -sSL -o data.zip https://github.com/abin24/"
            "Magnetic-tile-defect-datasets./archive/refs/heads/master.zip\n"
            "  unzip -q data.zip && mv Magnetic-tile-defect-datasets.-master data"
        )

    partition = partitioning(split_ratio)

    surface_defect_dataset = {}
    for p in ['train', 'val', 'test']:
        surface_defect_dataset[p] = SurfaceDefectDetectionDataset(partition[p], p)

    #-------------------------------------------------------------------#
    #                     Some stats about the dataset                  #
    #-------------------------------------------------------------------#
    H, W = [], []
    for path in partition['train']:
        image = Image.open(path)
        W.append(image.size[0])
        H.append(image.size[1])
    print("maximum height:", max(H), "\tmaximum width:", max(W),
          "\tminimum height:", min(H), "\tminimum width:", min(W))

    print('Length of train dataset: ', len(surface_defect_dataset['train']))
    print('Length of validation dataset: ', len(surface_defect_dataset['val']))
    print('Length of test dataset: ', len(surface_defect_dataset['test']))

    # find the weight of positive and negative pixels
    positive_weight = 0
    negative_weight = 0
    total_pixels = 0
    for _, target in surface_defect_dataset['train']:
        positive_weight += ((target.cpu().numpy()) >= args.threshold).sum()
        negative_weight += ((target.cpu().numpy()) < args.threshold).sum()
        total_pixels += (224 * 224)
    positive_weight /= total_pixels
    negative_weight /= total_pixels
    print('positive weight = ', positive_weight,
          '\tnegative weight = ', negative_weight)

    #-------------------------------------------------------------------#
    #                        Batch and load data                        #
    #-------------------------------------------------------------------#
    loaders = {}
    loaders['train'] = torch.utils.data.DataLoader(
        surface_defect_dataset['train'], batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers)
    loaders['val'] = torch.utils.data.DataLoader(
        surface_defect_dataset['val'], batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers)
    loaders['test'] = torch.utils.data.DataLoader(
        surface_defect_dataset['test'], batch_size=args.test_batch_size,
        shuffle=False, num_workers=args.num_workers)

    #-------------------------------------------------------------------#
    #                      Obtain model architecture                    #
    #-------------------------------------------------------------------#
    model = UNet_2D(1, 1, 32, 0.2)

    if not args.skip_summary:
        # torchsummary builds its probe tensor on CPU unless CUDA is present,
        # so run the summary before moving the model onto the device.
        summary(model, (1, 224, 224), batch_size=args.batch_size, device='cpu')

    model = model.to(DEVICE)

    #-------------------------------------------------------------------#
    #                 Specify the loss function and optimizer           #
    #-------------------------------------------------------------------#
    if args.criterion == 'WeightedBCE':
        weight = np.array([negative_weight, positive_weight])
        weight = torch.from_numpy(weight)
        criterion = WeightedBCELoss(weights=weight)
    else:
        criterion = TverskyLoss(1e-10, 0.3, .7)

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.002)

    # NOTE: the notebook builds this OneCycleLR scheduler, but train_2D never
    # calls scheduler.step(), so the learning rate stays constant. Kept here to
    # match the original code path.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.02, steps_per_epoch=len(loaders['train']),
        epochs=args.epochs)

    #-------------------------------------------------------------------#
    #                     Train and validate the model                  #
    #-------------------------------------------------------------------#
    model = train_2D(args.epochs, loaders, model, optimizer, criterion,
                     DEVICE.type != 'cpu', 'model.pt')

    # plot the variation of train and validation losses vs n_epochs
    loss = pd.read_csv('loss_epoch.csv', header=0, index_col=False)
    plt.plot(loss['epoch'], loss['Training Loss'], 'r',
             loss['epoch'], loss['Validation Loss'], 'g')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.legend(labels=['Train', 'Valid'])
    plt.savefig('loss_epoch.png')
    plt.clf()

    #-------------------------------------------------------------------#
    #                        Load a trained model                       #
    #-------------------------------------------------------------------#
    # load the model that got the minimum validation loss
    model.load_state_dict(torch.load('model.pt', map_location=DEVICE))

    #-------------------------------------------------------------------#
    #                        Generate predictions                       #
    #-------------------------------------------------------------------#
    plot_prediction_results(model, DEVICE.type != 'cpu', loaders['test'],
                            args.threshold)

    df = get_inference_performance_metrics(model, DEVICE.type != 'cpu',
                                           loaders['test'], args.threshold)
    print('\n=== Inference performance metrics ===')
    print(df.describe())


if __name__ == '__main__':
    main()
