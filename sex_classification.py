""" The following network will be trained to classify
between facial images of men and women. It will help to increase the tagged faces by sex.
The network is a binary classification network, thus, it has just one exit"""

from datetime import datetime
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import utils
import time
from torch.optim import lr_scheduler
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
from pathlib import Path

print(torch.__version__)
torch.cuda.is_available()

DATA_DIR = Path('C:/DataDividedByGender/')

# Data augmentation and normalization for training
# for validation we use normalization and resize (for train we also change the angle and size of the images)
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomResizedCrop(224, scale=(0.96, 1.04), ratio=(0.92, 1.08)),
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

BATCH_SIZE = 16
SAMPLE_SIZE = None

''' The function takes the data loader and a parameter  '''


def create_train_val_slice(image_datasets, sample_size=None, val_same_as_train=False):
    img_dataset = image_datasets  # reminder - this is a *generator* of the dataset

    # clone the image_datasets_reduced[train] generator for the val
    if val_same_as_train:
        img_dataset['val'] = list(img_dataset['train'])

    dataset_sizes = {x: len(img_dataset[x]) for x in ['train', 'val']}

    if sample_size:  # return the whole data set
        sample_n = {x: random.sample(list(range(dataset_sizes[x])), sample_size) for x in ['train', 'val']}
        img_dataset = {x: torch.utils.data.Subset(img_dataset[x], sample_n[x]) for x in ['train', 'val']}
        dataset_sizes = {x: len(img_dataset[x]) for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(img_dataset[x], batch_size=BATCH_SIZE,
                                                  shuffle=True, num_workers=0) for x in ['train', 'val']}

    return dataloaders, dataset_sizes

image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

class_names = image_datasets['train'].classes

my_data, dataset_sizes =  create_train_val_slice(image_datasets,sample_size=SAMPLE_SIZE,val_same_as_train=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Classes: ", class_names)
print(f'Train image size: {dataset_sizes["train"]}')
print(f'Validation image size: {dataset_sizes["val"]}')


def get_model():
    model_conv = torchvision.models.resnet18(pretrained=True)
    num_ftrs = model_conv.fc.in_features  # probably the feature exctractor
    # model_conv.fc = nn.Linear(num_ftrs, 1)
    model_conv.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid(),
    )

    utils.freeze_layers_grad(model_conv)

    model_conv = model_conv.to(device)
    return model_conv

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# # Get a batch of training data
inputs, classes = next(iter(my_data['train']))
# Make a grid from batch
sample_train_images = torchvision.utils.make_grid(inputs)
#imshow(sample_train_images, title=classes)
print(f"classes={classes}")
imshow(sample_train_images, title=[class_names[i] for i in classes])


# # Get a batch of validation data
inputs, classes = next(iter(my_data['val']))
# Make a grid from batch
sample_train_images = torchvision.utils.make_grid(inputs)
#imshow(sample_train_images, title=classes)
print(f"classes={classes}")
imshow(sample_train_images, title=[class_names[i] for i in classes])


def train_model(data, model, criterion, optimizer, scheduler, num_epochs=2, writer=None):
    since = time.time()

    train_accuracy_history = []
    train_loss_history = []

    test_accuracy_history = []
    test_loss_history = []

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
                outputs = model(inputs).squeeze()
                # _, preds = torch.max(outputs, 1)
                preds = torch.round(outputs)
                loss = criterion(outputs, labels.float())

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            #                 print("Train- Outputs:    ",outputs.tolist())
            #                 print("Train- Predictions:",preds.int().tolist())
            #                 print("Train- Labels:     ",labels.data.tolist())
            #                 print("Train- Loss:     ",loss)

            batch_loss = loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            if writer is not None:  # save train loss for each batch
                x_axis = 1000 * (epoch + i / (dataset_sizes['train'] // BATCH_SIZE))
                writer.add_scalar('batch loss', batch_loss / BATCH_SIZE, x_axis)

        if scheduler is not None:
            scheduler.step()  # scheduler step is performed per-epoch in the training phase

        train_acc = running_corrects / dataset_sizes['train']
        if writer is not None:  # save epoch accuracy
            x_axis = epoch
            writer.add_scalar('accuracy-train',
                              train_acc,
                              x_axis)

        epoch_loss, epoch_acc = eval_model(criterion, data, model, optimizer)

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
    running_corrects = 0.0

    for i, (inputs, labels) in enumerate(data['val']):
        # data['val'] contains (input,labels) for every batch (so i=[1...NUM OF BATCHES]

        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs).squeeze()
            # _, preds = torch.max(outputs, 1)
            preds = torch.round(outputs)
            loss = criterion(outputs, labels.float())

        # statistics - sum loss and accuracy on all batches
        running_loss += loss.item() * inputs.size(0)  # item.loss() is the average loss of the batch
        #         print("Eval - Outputs:    ",outputs.tolist())
        #         print("Eval - Predictions:",preds.int().tolist())
        #         print("Eval - Labels:     ",labels.data.tolist())
        running_corrects += torch.sum(preds == labels.data)
    #         print("Eval - Running corrects:",running_corrects)

    epoch_loss = float(running_loss) / dataset_sizes['val']
    print()
    epoch_acc = running_corrects.double() / dataset_sizes['val']
    print(f'Test Loss: {epoch_loss:.4f} Test Acc: {epoch_acc:.4f}')

    return epoch_loss, epoch_acc