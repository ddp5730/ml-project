# train.py
# 2/1/22
# Dan Popp
#
# This script will train an EfficientNet model on Rareplanes to use as a baseline for a CNN comparison
import argparse
import copy
import os
import sys
import time

import timm
import torch
from timm.data import create_transform
from timm.scheduler import CosineLRScheduler
from torch import nn, optim
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets, models
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

from mlp import MLP


def eval_setup(pretrained_path, args):
    """
    This function will test transfer learning using the provided PETS data
    """

    batch_size = args.batch_size
    test = test
    model = args.model

    # Load dataset
    eval_dataset = get_datasets(DATA_ROOT_2018)

    sampler = SequentialSampler(eval_dataset)

    dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, sampler=sampler,
                                             num_workers=20)
    class_names = eval_dataset.classes.copy()
    num_classes = len(class_names)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize Model
    model = MLP(79, num_classes)

    model.load_state_dict(torch.load(pretrained_path))

    model = model.to(device)

    eval_model(model, dataloader, device)


def eval_model(model, dataloader, device):
    model.eval()  # Set model to evaluate mode
    start_test = True

    # Iterate over data.
    iterator = tqdm(dataloader, file=sys.stdout)
    for idx, (inputs, labels) in enumerate(iterator):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        # statistics
        if start_test:
            all_preds = preds.float().cpu()
            all_labels = labels.float()
            start_test = False
        else:
            all_preds = torch.cat((all_preds, preds.float().cpu()), 0)
            all_labels = torch.cat((all_labels, labels.float()), 0)

    all_labels = all_labels.detach().cpu().numpy()
    all_preds = all_preds.detach().cpu().numpy()
    top1_acc = accuracy_score(all_labels, all_preds)
    val_f1_score = f1_score(all_labels, all_preds,
                            average='macro')

    print('Top-1 Acc: {:.4f} F1 Score: {:.4f}'.format(top1_acc, val_f1_score))
    return val_f1_score, top1_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--dataset-dir', type=str, required=True)
    parser.add_argument('--test-dir', type=str, default='test')
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--model', type=str, default='resnet50')
    args = parser.parse_args()

    path = args.model_path
    if not os.path.exists(path):
        print('Path is invalid', file=sys.stderr)
        exit(1)

    eval_setup(path, args)
    print('Done')


if __name__ == '__main__':
    main()
