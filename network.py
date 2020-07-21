from datetime import datetime

import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import os
import copy
import math
from pathlib import Path
# from tqdm.notebook import trange, tqdm
from itertools import islice
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
# import itertools
# import pixiedust
import random
from torch.utils import data

SAMPLE_SIZE = 8
NUM_EPOCHS = 15
BATCH_SIZE = 2

print(torch.__version__)
plt.ion()  # interactive mode
torch.cuda.is_available()

# try:
#     from google.colab import drive
#
#     drive.mount('/content/gdrive')
#     LABS_DIR = Path('/content/gdrive/My Drive/Labs')
# except:
#     LABS_DIR = Path('C:/Labs/')

DATA_DIR = 'Data'

#### sanity check for the images
# classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
# for emotion in classes:
#     print("Class =",emotion)
#     !ls $DATA_DIR\VAL\$emotion | wc -l


# Data augmentation and normalization for training
# for validatin we use normalization and resize (for train we also change the angle and size of the images)
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.96, 1.0), ratio=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

### Data Loader ###
''' The function takes the data loader and a parameter  '''


def create_train_val_slice(image_datasets, sample_size=None, val_same_as_train=False):
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    if not sample_size:  # return the whole data
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                                      shuffle=True, num_workers=0)
                       for x in ['train', 'val']}
        return dataloaders, dataset_sizes

    sample_n = {x: random.sample(list(range(dataset_sizes[x])), sample_size)
                for x in ['train', 'val']}

    image_datasets_reduced = {x: torch.utils.data.Subset(image_datasets[x], sample_n[x])
                              for x in ['train', 'val']}

    # clone the image_datasets_reduced[train] for the val
    if val_same_as_train:
        image_datasets_reduced['val'] = list(image_datasets_reduced['train'])
        image_datasets_reduced['train'] = image_datasets_reduced['val']  # copy the train to val (so the tranformations won't occur again)

    dataset_sizes = {x: len(image_datasets_reduced[x]) for x in ['train', 'val']}

    dataloaders_reduced = {x: torch.utils.data.DataLoader(image_datasets_reduced[x], batch_size=BATCH_SIZE,
                                                          shuffle=True, num_workers=0) for x in ['train', 'val']}
    return dataloaders_reduced, dataset_sizes


image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

class_names = image_datasets['train'].classes

my_data, dataset_sizes = create_train_val_slice(image_datasets, sample_size=SAMPLE_SIZE)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Classes: ", class_names)
print(f'Train image size: {dataset_sizes["train"]}')
print(f'Validation image size: {dataset_sizes["val"]}')


# def imshow(inp, title=None):
#     """Imshow for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # pause a bit so that plots are updated


# # Get a batch of training data
# inputs, classes = next(iter(my_data['train']))
# # Make a grid from batch
# sample_train_images = torchvision.utils.make_grid(inputs)
# #imshow(sample_train_images, title=classes)
# print(f"classes={classes}")
# imshow(sample_train_images, title=[class_names[i] for i in classes])

def freeze_layers_grad(model, total_freeze_layers=7):
    # Parameters of newly constructed modules have requires_grad=True by default
    layer = 0
    for child in model.children():
        layer += 1
        # freezes layers 1-6 in the total 10 layers of Resnet50
        if layer < total_freeze_layers:
            for param in child.parameters():
                param.requires_grad = False

def train_model(data, model, label_criterion, optimizer, scheduler, num_epochs=2, writer=None):
    since = time.time()

    print("Starting epochs")
    for epoch in range(1, num_epochs + 1):
        print(f'Epoch: {epoch} of {num_epochs}')
        model.train()  # Set model to training mode
        running_test_loss = 0.0
        running_corrects = 0.0

        for i, (inputs, labels) in enumerate(data['train']):
            # data['train'] contains (input,labels) for every batch (so i=[1...NUM OF BATCHES]
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                label_loss = label_criterion(outputs, labels)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            batch_loss = label_loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            if writer is not None:  # save train label_loss for each batch
                x_axis = 1000 * (epoch + i / (dataset_sizes['train'] // BATCH_SIZE))
                writer.add_scalar('batch label_loss', batch_loss / BATCH_SIZE, x_axis)

        if scheduler is not None:
            scheduler.step()  # scheduler step is performed per-epoch in the training phase

        train_acc = running_corrects / dataset_sizes['train']
        if writer is not None:  # save epoch accuracy
            x_axis = epoch
            writer.add_scalar('accuracy-train',
                              train_acc,
                              x_axis)

        epoch_loss, epoch_acc = eval_model(label_criterion, data, model, optimizer)

        if writer is not None:  # save epoch accuracy
            x_axis = epoch
            writer.add_scalar('accuracy-val',
                              epoch_acc,
                              x_axis)

    # TODO check stop condition by overfit
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # return the last trained model
    return model

def train_model(data, model, label_criterion, optimizer, scheduler, num_epochs=2, writer=None):
    since = time.time()

    print("Starting epochs")
    for epoch in range(1, num_epochs + 1):
        print(f'Epoch: {epoch} of {num_epochs}')
        model.train()  # Set model to training mode
        running_test_loss = 0.0
        running_corrects = 0.0

        for i, (inputs, labels) in enumerate(data['train']):
            # data['train'] contains (input,labels) for every batch (so i=[1...NUM OF BATCHES]
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                label_loss = label_criterion(outputs, labels)

                # backward + optimize only if in training phase
                label_loss.backward()
                optimizer.step()

            batch_loss = label_loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            if writer is not None:  # save train label_loss for each batch
                x_axis = 1000 * (epoch + i / (dataset_sizes['train'] // BATCH_SIZE))
                writer.add_scalar('batch label_loss', batch_loss / BATCH_SIZE, x_axis)

        if scheduler is not None:
            scheduler.step()  # scheduler step is performed per-epoch in the training phase

        train_acc = running_corrects / dataset_sizes['train']
        if writer is not None:  # save epoch accuracy
            x_axis = epoch
            writer.add_scalar('accuracy-train',
                              train_acc,
                              x_axis)

        epoch_loss, epoch_acc = eval_model(label_criterion, data, model, optimizer)

        if writer is not None:  # save epoch accuracy
            x_axis = epoch
            writer.add_scalar('accuracy-val',
                              epoch_acc,
                              x_axis)

    # TODO check stop condition by overfit
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # return the last trained model
    return model


def eval_model(criterion, data, model, optimizer):
    model.eval()  # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0

    for i, (inputs, labels) in enumerate(data['val']):
        # data['val'] contains (input,labels) for every batch (so i=[1...NUM OF BATCHES]

        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        # statistics - sum loss and accuracy on all batches
        running_loss += loss.item() * inputs.size(0)  # item.loss() is the average loss of the batch
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_sizes['val']
    epoch_acc = running_corrects.double() / dataset_sizes['val']
    print(f'Test Loss: {epoch_loss:.4f} TestAcc: {epoch_acc:.4f}')
    return epoch_loss, epoch_acc


def get_model():
    model_conv = torchvision.models.resnet18(pretrained=True)
    # model_conv = torchvision.models.resnet50(pretrained=True)
    # model_conv = torchvision.models.resnet101(pretrained=True)

    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Sequential()
    classifier = nn.Linear(num_ftrs, len(class_names))
    model_conv = model_conv.to(device)
    return model_conv


# Tensorboard Stuff
def run_experiment(data, lr_initial, gamma, step_size, weight_decay, num_of_epochs):
    """
    Gets all hyper parameters and creates the relevant optimizer and scheduler according to those params

    """
    model_net = get_model()
    criterion = nn.CrossEntropyLoss()  # softmax+log
    optimizer = optim.Adam(model_net.parameters(), lr=lr_initial, weight_decay=weight_decay)

    # decay LR by a factor of gamma every step_size epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    # check optimizer and schedular due to ADAM

    experiment_name = datetime.now().strftime("%Y%m%d-%H%M%S") + f'_lr_{lr_initial}_st_{step_size}_gma_{gamma}_wDK_{weight_decay}'
    print("Experiment name: ", experiment_name)

    writer = SummaryWriter('runs/' + experiment_name)
    trained_model = train_model(data,
                                model_net,
                                criterion,
                                optimizer,
                                exp_lr_scheduler,
                                num_epochs=num_of_epochs,
                                writer=writer)
    return trained_model


def main():
    for lr in [0.0005, 0.0001]:
        for scheduler_step_size in [5, 7, 9]:
            for scheduler_gamma in [0.1, 0.3, 0.5]:
                for weight_decay in [0.01, 0.1]:
                    model_conv = run_experiment(my_data, lr, scheduler_gamma, scheduler_step_size, weight_decay,
                                                NUM_EPOCHS)

        # torch.save({'model_state_dict': model_conv.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'best_val_loss': best_val_loss,
        #             'best_val_accuracy': best_val_acc,
        #             'scheduler_state_dict': exp_lr_scheduler.state_dict(),
        #             }, CHECK_POINT_PATH)
        # model_conv.eval()

        # x = 'train'
        # d = datasets.ImageFolder(os.path.join(DATA_DIR, x))
        # cnt = Counter([])
        # for i, (image, category) in enumerate(d):
        #     cnt.update({(image_datasets['train'].classes)[category]: 1})
        # print(cnt)
        # image_datasets['train'].classes[0]

if __name__ == '__main__':
    main()

