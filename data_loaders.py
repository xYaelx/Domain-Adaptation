from torchvision import datasets, transforms
from torch.utils import data
from os import path
import random

DATA_DIR_M = 'Data/Male'
DATA_DIR_F = 'Data/Female'


class DataLoaders:
    def __init__(self, sample_size, batch_size, data_dir, val_same_as_train):
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.data_transforms = {
            # for validatin we use normalization and resize (for train we also change the angle and size of the images)
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
        self.image_dataset = {x: datasets.ImageFolder(path.join(data_dir, x), self.data_transforms[x])
                              for x in ['train', 'val']}
        self.classes = self.image_dataset['train'].classes

        self.val_same_as_train = val_same_as_train
        self.data, self.dataset_size = self.create_train_val_slice(self.image_dataset, sample_size=self.sample_size,
                                                                   val_same_as_train=self.val_same_as_train)

    def create_train_val_slice(self, image_datasets, sample_size=None, val_same_as_train=False):
        img_dataset = image_datasets  # reminder - this is a generator

        # clone the image_datasets_reduced[train] generator for the val
        if val_same_as_train:
            img_dataset['val'] = list(img_dataset['train'])
            # copy the train to val (so the tranformations won't occur again)
            # image_datasets_reduced['train'] = image_datasets_reduced['val']

        dataset_sizes = {x: len(img_dataset[x]) for x in ['train', 'val']}

        if sample_size:  # return the whole data set
            sample_n = {x: random.sample(list(range(dataset_sizes[x])), sample_size) for x in ['train', 'val']}
            img_dataset = {x: data.Subset(img_dataset[x], sample_n[x]) for x in ['train', 'val']}
            dataset_sizes = {x: len(img_dataset[x]) for x in ['train', 'val']}

        dataloaders = {x: data.DataLoader(img_dataset[x], batch_size=self.batch_size,
                                          shuffle=True, num_workers=0) for x in ['train', 'val']}
        return dataloaders, dataset_sizes
