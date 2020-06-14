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
from torchvision import datasets, models, transforms
# import itertools
# import pixiedust
import random
from torch.utils import data

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

# # Data Loader
BATCH_SIZE = 1
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

    image_datasets_reduced = {
        x: torch.utils.data.Subset(image_datasets['train' if val_same_as_train else x], sample_n[x])
        for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets_reduced[x]) for x in ['train', 'val']}

    dataloaders_reduced = {x: torch.utils.data.DataLoader(image_datasets_reduced[x], batch_size=BATCH_SIZE,
                                                          shuffle=True, num_workers=0) for x in ['train', 'val']}
    return dataloaders_reduced, dataset_sizes


image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

class_names = image_datasets['train'].classes

# sample_size = 100
# data, dataset_sizes =  create_train_val_slice(image_datasets,sample_size,True)
my_data, dataset_sizes = create_train_val_slice(image_datasets, sample_size=5)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Classes: ", class_names)
print(f'Train image size: {dataset_sizes["train"]}')
print(f'Validation image size: {dataset_sizes["val"]}')


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
# inputs, classes = next(iter(dataloaders['train']))
# # Make a grid from batch
# sample_train_images = torchvision.utils.make_grid(inputs)
# #imshow(sample_train_images, title=classes)
# print(f"classes={classes}")
# imshow(sample_train_images, title=[class_names[i] for i in classes])



def train_model(data, model, criterion, optimizer, scheduler, num_epochs=2, checkpoint=None):
    since = time.time()

    if checkpoint is None:
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = math.inf
        best_acc = 0.
    else:
        print('Val loss: {}, Val accuracy: {}'.format(checkpoint["best_val_loss"], checkpoint["best_val_accuracy"]))
        model.load_state_dict(checkpoint['model_state_dict'])
        best_model_wts = copy.deepcopy(model.state_dict())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_loss = checkpoint['best_val_loss']
        best_acc = checkpoint['best_val_accuracy']

    print("Starting epochs")
    # outer = tqdm(total=num_epochs, desc='Epoch', position=0)
    # inner = tqdm(total=(dataset_sizes['train'] // BATCH_SIZE), position=1)
    for epoch in range(num_epochs):
        # outer.update(1)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            total = dataset_sizes[phase] // BATCH_SIZE

            # Handle tqdm inner loop counter
            # inner.total = dataset_sizes[phase] // BATCH_SIZE
            # inner.reset()
            inner_total = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(data[phase]):

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                if i > 0:
                    report_str = f'[{epoch + 1}, {i}] loss: {(running_loss / i * inputs.size(0)):.3f}'
                else:
                    report_str = "first iteration"
                print(report_str)
                # Update inner tqdm, we are about to override the previous maximum, update maximum
                # inner.update(1)  # Advance the tqdm counter
                # inner.desc = f'Phase: {phase} ' + report_str

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                # print("running_corrects =", running_corrects)

            if phase == 'train' and scheduler is not None:
                scheduler.step()

            # inner.write("running_corrects=", running_corrects, " epoch: ", epoch)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # loss_str = f'Epoch: {epoch + 1} of {num_epochs}, {phase:6} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}'
            # inner.write(loss_str)

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                # inner.write('New best model found!')
                # inner.write(f'New record loss:{epoch_loss}, previous record loss: {best_loss}')
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # save the weights
                # torch.save(best_model_wts, CHECK_POINT_PATH)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f} Best val loss: {:.4f}'.format(best_acc, best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_loss, best_acc


def main():
    global model_conv
    model_conv = torchvision.models.resnet18(pretrained=True)
    # model_conv = torchvision.models.resnet50(pretrained=True)
    # model_conv = torchvision.models.resnet101(pretrained=True)
    # model_conv.eval()
    # # Train Model
    # Parameters of newly constructed modules have requires_grad=True by default
    ct = 0
    for child in model_conv.children():
        ct += 1
        # freezes layers 1-6 in the total 10 layers of Resnet50
        if ct < 7:
            for param in child.parameters():
                param.requires_grad = False
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, len(class_names))
    model_conv = model_conv.to(device)
    '''two options to write the loss. They are both equal'''
    # option 1 #
    criterion = nn.CrossEntropyLoss()
    # option 2 #
    # p = nn.functional.softmax(model_conv, dim=1)
    # # to calculate loss using probabilities you can do below
    # criterion = nn.functional.nll_loss(torch.log(p), y)
    # Observe that only parameters of final layer are being optimized
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.01, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=70, gamma=0.1)
    # CHECK_POINT_PATH = 'ModelParams' / 'checkpoint.tar'
    # !del $CHECK_POINT_PATH
    # try:
    #     checkpoint = torch.load(CHECK_POINT_PATH)
    #     print("checkpoint loaded")
    # except:
    #     checkpoint = None
    #     print("checkpoint not found")
    model_conv, best_val_loss, best_val_acc = train_model(my_data,
                                                          model_conv,
                                                          criterion,
                                                          optimizer_conv,
                                                          exp_lr_scheduler,
                                                          num_epochs=5,
                                                          checkpoint=None)
    # torch.save({'model_state_dict': model_conv.state_dict(),
    #             'optimizer_state_dict': optimizer_conv.state_dict(),
    #             'best_val_loss': best_val_loss,
    #             'best_val_accuracy': best_val_acc,
    #             'scheduler_state_dict': exp_lr_scheduler.state_dict(),
    #             }, CHECK_POINT_PATH)
    # # Test Model
    model_conv.eval()
    from pprint import pprint
    x = 'train'
    d = datasets.ImageFolder(os.path.join(DATA_DIR, x))
    cnt = Counter([])
    for i, (image, category) in enumerate(d):
        cnt.update({(image_datasets['train'].classes)[category]: 1})
    print(cnt)
    # In[17]:
    image_datasets['train'].classes[0]
    # In[ ]:


if __name__ == '__main__':
    main()



