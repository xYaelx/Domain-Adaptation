import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


class TrainingParams:
    '''
    An object that contains all parameters the model needs for training: the architecture, loss criterions, num of epochs
    '''

    def __init__(self, lr_initial, step_size, gamma, weight_decay, num_epochs):
        # self.model = None
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
