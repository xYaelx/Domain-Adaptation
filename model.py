from utils import NET_ARCHICECTURE
import torch.nn as nn
from torch.autograd import Function
from torchvision import models

DROPOUT_PROB = 0.5


def get_activation(name):
    def hook(model, input, output):
        model.activation[name] = output  # .detach()

    return hook


def get_model(device, class_names, architecure: NET_ARCHICECTURE):
    assert architecure in NET_ARCHICECTURE
    model_conv = models.resnet18(pretrained=True)
    # model_conv = models.resnet50(pretrained=True)
    # model_conv = models.resnet101(pretrained=True)

    num_ftrs = model_conv.fc.in_features  # The size of feature extractor output
    num_classes = len(class_names)
    linear_model_selector = {
        NET_ARCHICECTURE.ONE_FC: lin_one_fc,
        NET_ARCHICECTURE.TWO_FC: lin_two_fc,
        NET_ARCHICECTURE.THREE_FC: lin_three_fc,
    }
    model_conv.fc = linear_model_selector[architecure](num_ftrs, num_classes)

    model_conv.avgpool.activation = {}

    model_conv.avgpool.register_forward_hook(get_activation('avgpool'))

    model_conv.discriminator = nn.Sequential(
        GradientReversal(),
        nn.Linear(num_ftrs, 50),
        nn.ReLU(),
        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    ).to(device)

    model_conv = model_conv.to(device)
    return model_conv


def lin_one_fc(num_ftrs, num_classes):
    return nn.Linear(num_ftrs, num_classes)


def lin_two_fc(num_ftrs, num_classes):
    return nn.Sequential(
        nn.Linear(num_ftrs, 64),
        nn.ReLU(),
        nn.Dropout(DROPOUT_PROB),
        nn.Linear(64, num_classes)
    )


def lin_three_fc(num_ftrs, num_classes):
    return nn.Sequential(
        nn.Linear(num_ftrs, 50),
        nn.ReLU(),
        nn.Dropout(DROPOUT_PROB),
        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Dropout(DROPOUT_PROB),
        nn.Linear(20, num_classes)
    )


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


# Source: https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/utils.py
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


class GradientReversal(nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
