from datetime import datetime

import torchvision
import matplotlib.pyplot as plt
from DataLoaders import DataLoaders
# import time
from pathlib import Path
# from tqdm.notebook import trange, tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from trainer import Trainer
from model import get_model

from utils import NET_ARCHICECTURE

DATA_DIR = 'Data'
DATA_DIR_M = 'Data/Male'
DATA_DIR_F = 'Data/Female'
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


class TrainingParams:
    '''
    An object that contains all parameters the model needs for training: the architecture, loss criterions, num of epochs
    '''

    def __init__(self, model, lr_initial, step_size, gamma, weight_decay, num_epochs):
        self.model = model
        self.lr = lr_initial
        self.step_size = step_size
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.label_criterion = nn.CrossEntropyLoss()  # softmax+log
        self.domain_criterion = nn.functional.binary_cross_entropy_with_logits
        self.num_epochs = num_epochs

    def __str__(self):
        return f'_lr_{self.lr}_st_{self.step_size}_gma_{self.gamma}_wDK_{self.weight_decay}'

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model_conv):
        self.__model = model_conv
        self.optimizer = optim.Adam(self.model_conv.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)


""" sanity check for the images"""
# classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
# for emotion in classes:
#     print("Class =",emotion)
#     !ls $DATA_DIR\VAL\$emotion | wc -l

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

def run_experiment(use_discriminator, domain1_dataloader, domain2_dataloader, test_dataloader, training_params,
                   architecure: NET_ARCHICECTURE):
    """
    Gets all hyper parameters and creates the relevant optimizer and scheduler according to those params

    """
    training_params.model = get_model(device, domain1_dataloader.classes, architecure)

    descriminator_description = "D" if use_discriminator else "ND"
    experiment_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + descriminator_description + str(training_params)
    print("Starting, experiment name: ", experiment_name)

    writer = SummaryWriter('runs/' + experiment_name)  # Tensorboard Stuff
    trainer = Trainer(device, domain1_dataloader, domain2_dataloader, BATCH_SIZE)

    # first train
    trained_params_model = trainer.train_model(use_discriminator, training_params, writer=writer)
    print(f"Experiment {experiment_name} - testing on the women domain")

    # then test :)
    test_acc = trainer.test(test_dataloader, trained_params_model)
    print("Finished -----------------\r\n\r\n")

    return trained_params_model, test_acc


def main():
    dataloder_male = DataLoaders(sample_size=SAMPLE_SIZE, batch_size=BATCH_SIZE // 2, data_dir=DATA_DIR_M,
                                 val_same_as_train=False)
    dataloder_female = DataLoaders(sample_size=SAMPLE_SIZE, batch_size=BATCH_SIZE // 2, data_dir=DATA_DIR_F,
                                   val_same_as_train=False)

    print("Classes: ", dataloder_male.classes)
    print(f'Train image size: {dataloder_male.dataset_size["train"]}')
    print(f'Validation image size: {dataloder_male.dataset_size["val"]}')

    for lr in [0.0005, 0.0001]:
        for scheduler_step_size in [5, 7, 9]:
            for gamma in [0.1, 0.3, 0.5]:
                for weight_decay in [0.01, 0.1]:
                    # params order: model, lr_initial, step_size, gamma, weight_decay, num_epochs)
                    training_params = TrainingParams(lr, scheduler_step_size, gamma, weight_decay, NUM_EPOCHS)

                    model_conv = run_experiment(dataloder_male, dataloder_female, lr, scheduler_gamma,
                                                scheduler_step_size, weight_decay, NUM_EPOCHS)

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
