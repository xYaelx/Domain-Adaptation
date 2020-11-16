# Source: https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/utils.py
from enum import Enum

from PIL import Image

import numpy as np
import torch
from torch.autograd import Function


class NET_ARCHICECTURE(Enum):
    ONE_FC = 1
    TWO_FC = 2
    THREE_FC = 3


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad

def freeze_layers_grad(model, total_freeze_layers=7):
    # Parameters of newly constructed modules have requires_grad=True by default
    layer = 0
    for child in model.children():
        layer += 1
        # freezes layers 1-6 in the total 10 layers of Resnet50
        if layer < total_freeze_layers:
            for param in child.parameters():
                param.requires_grad = False

def loop_iterable(iterable):
    while True:
        yield from iterable

class NET_ARCHICECTURE(Enum):
    '''
    ENUM object for dropout option
    '''
    NO_FC = 0
    ONE_FC = 1
    TWO_FC = 2

class GrayscaleToRgb:
    """Convert a grayscale image to rgb"""
    def __call__(self, image):
        image = np.array(image)
        image = np.dstack([image, image, image])
        return Image.fromarray(image)


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

