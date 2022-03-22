import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from timm.scheduler import CosineLRScheduler
from torch import nn, optim
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from load_data import get_datasets

DATA_ROOT_2018 = '/home/poppfd/data/CIC-IDS2018/Processed_Traffic_Data_for_ML_Algorithms/'


class MLP(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.layer1 = nn.Linear(num_features, 100)
        self.layer2 = nn.Linear(100, 200)
        self.layer3 = nn.Linear(200, 100)
        self.fc = nn.Linear(100, num_classes)

        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.act(self.layer1(x))
        x = self.act(self.layer2(x))
        x = self.act(self.layer3(x))
        x = self.fc(x)
        x = self.softmax(x)

        return x


def train_mlp(name, args):
    """
    This function will test transfer learning using the provided PETS data
    """

    train = 'train'
    test = 'test'

    batch_size = args.batch_size
    eval_batch_freq = -1
    num_epochs = args.num_epochs
    warmup_epochs = args.warmup_epochs
    learning_rate = args.learning_rate
    min_lr = args.min_lr
    warmup_lr = args.warmup_lr

    # Load dataset
    dataset_train, dataset_test = get_datasets(DATA_ROOT_2018)
    datasets = {train: dataset_train, test: dataset_test}

    samplers = {}
    samplers[train] = RandomSampler(datasets[train])
    samplers[test] = SequentialSampler(datasets[test])

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x],
                                                  batch_size=batch_size if x == train else eval_batch_size,
                                                  sampler=samplers[x],
                                                  num_workers=20)
                   for x in [train, test]}
    dataset_sizes = {x: len(datasets[x]) for x in [train, test]}
    class_names = datasets[train].classes.copy()
    num_classes = len(class_names)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize Model
    model = MLP(79, num_classes)

    for param in model.parameters():
        param.requires_grad = True

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RAdam(model.parameters(), lr=learning_rate)

    n_iter_per_epoch = len(dataloaders[train])
    num_steps = int(num_epochs * n_iter_per_epoch)
    warmup_steps = int(warmup_epochs * n_iter_per_epoch)
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_steps,
        cycle_mul=1.,
        lr_min=min_lr,
        warmup_lr_init=warmup_lr,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )

    out_dir = os.path.join('./output/', name)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    with open(os.path.join(out_dir, 'config.txt'), 'w') as file:
        file.write('Config for run: %s\n' % name)
        file.write('NUM_EPOCHS: %d\n' % num_epochs)
        file.write('WARMUP_EPOCHS: %d\n' % warmup_epochs)
        file.write('LR: %e\n' % learning_rate)
        file.write('MIN_LR: %e\n' % min_lr)
        file.write('WARMUP_LR: %e\n' % warmup_lr)
        file.write('BATCH_SIZE: %d\n' % batch_size)

    model_ft = train_model(model, criterion, optimizer,
                           lr_scheduler, dataloaders, device, eval_batch_freq, out_dir, train, test,
                           num_epochs=num_epochs)


def train_model(model, criterion, optimizer, scheduler, dataloaders, device, eval_batch_freq, out_dir, train, test,
                num_epochs=25):
    writer = SummaryWriter(log_dir=os.path.join(out_dir, 'tensorboard_logs'))

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0
    best_acc = 0.0
    eval_num = 1

    validation_accuracies = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in [train, test]:
            if phase == train:
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            start_test = True

            running_loss = 0.0

            # Iterate over data.
            iterator = tqdm(dataloaders[phase], file=sys.stdout)
            for idx, (inputs, labels) in enumerate(iterator):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == train):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == train:
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                if start_test:
                    all_preds = preds.float().cpu()
                    all_labels = labels.float()
                    start_test = False
                else:
                    all_preds = torch.cat((all_preds, preds.float().cpu()), 0)
                    all_labels = torch.cat((all_labels, labels.float()), 0)

                if phase == train:
                    num_steps = len(dataloaders[train])
                    scheduler.step_update(epoch * num_steps + idx)

                if phase == train and eval_batch_freq > 0:
                    if (idx + 1) % eval_batch_freq == 0:
                        # Evaluate the model every set number of batches
                        model_f1, model_acc = eval.eval_model(model, dataloaders[test], device)
                        validation_accuracies.append(model_acc)
                        if model_f1 > best_f1:
                            best_f1 = model_f1
                            best_model_wts = copy.deepcopy(model.state_dict())
                        torch.save(model.state_dict(), os.path.join(out_dir, 'model_eval_%d.pth' % eval_num))
                        if model_acc > best_acc:
                            best_acc = model_acc
                        eval_num += 1

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            all_labels = all_labels.detach().cpu().numpy()
            all_preds = all_preds.detach().cpu().numpy()
            top1_acc = accuracy_score(all_labels, all_preds)
            val_f1_score = f1_score(all_labels, all_preds,
                                    average='macro')

            if phase == test:
                validation_accuracies.append(top1_acc)

                lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('Learning Rate', scalar_value=lr, global_step=epoch)
                writer.add_scalar('Training Loss', scalar_value=epoch_loss, global_step=epoch)
                writer.add_scalar('Validation Top-1 Acc', scalar_value=top1_acc, global_step=epoch)
                writer.add_scalar('Validation F1 Score', scalar_value=val_f1_score, global_step=epoch)

            print('{} Loss: {:.4f} Top-1 Acc: {:.4f} F1 Score: {:.4f}'.format(
                phase, epoch_loss, top1_acc, val_f1_score))

            # deep copy the model
            if phase == test:
                if val_f1_score > best_f1:
                    best_f1 = val_f1_score
                if top1_acc > best_acc:
                    best_acc = top1_acc

                if len(validation_accuracies) < 10:
                    end = len(validation_accuracies)
                else:
                    end = 10
                top_10 = np.flip(np.argsort(np.asarray(validation_accuracies)))[:end]
                for i in range(top_10.size):
                    epoch_num = top_10[i]
                    epoch_accuracy = validation_accuracies[epoch_num]
                    print('Rank %d: Eval Num: %d, Acc1: %.3f%%' % (i + 1, epoch_num + 1, epoch_accuracy))

                torch.save(model.state_dict(), os.path.join(out_dir, 'model_eval_%d.pth' % eval_num))
                eval_num += 1

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val F1: {:4f}'.format(best_f1))
    print('Best Accuracy: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help='Name for the run')
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--eval-batch-size', type=int, required=True)
    parser.add_argument('--num-epochs', type=int, required=True)
    parser.add_argument('--warmup-epochs', type=int, required=True)
    parser.add_argument('--learning-rate', type=float, required=True)
    parser.add_argument('--min-lr', type=float, required=True)
    parser.add_argument('--warmup-lr', type=float, required=True)

    args = parser.parse_args()

    train_mlp(args.name, args)


if __name__ == '__main__':
    main()
