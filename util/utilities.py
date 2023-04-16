"""
Utilities of Project
"""
import numpy as np
import argparse
import torch
import os
import torch.nn.utils as utils
from yaml import parse
import torch.nn as nn
import matplotlib.pyplot as plt
import random

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from models import MyResNet18


import yaml
import torch
import torch.nn as nn
from pytorch_metric_learning import losses


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AverageMeter(object):
    """
    computes and stores the average and current value
    """

    def __init__(self, start_val=0, start_count=0, start_avg=0, start_sum=0):
        self.reset()
        self.val = start_val
        self.avg = start_avg
        self.sum = start_sum
        self.count = start_count

    def reset(self):
        """
        Initialize 'value', 'sum', 'count', and 'avg' with 0.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num=1):
        """
        Update 'value', 'sum', 'count', and 'avg'.
        """
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def save_model(file_path, file_name, model, optimizer=None):
    """
    In this function, a model is saved.Usually save model after training in each epoch.
    ------------------------------------------------
    Args:
        - model (torch.nn.Module)
        - optimizer (torch.optim)
        - file_path (str): Path(Folder) for saving the model
        - file_name (str): name of the model checkpoint to save
    """
    state_dict = dict()
    state_dict["model"] = model.state_dict()

    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()
    torch.save(state_dict, os.path.join(file_path, file_name))


def load_model(ckpt_path, model, optimizer=None):
    """
    Loading a saved model and optimizer (from checkpoint)
    """
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    if (optimizer != None) & ("optimizer" in checkpoint.keys()):
        optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def load_model_params(model_name):
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
        global_params = config["default"]
        model_params = config.get(model_name, {})
        init_params = {**global_params, **model_params}

    device = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu")}
    model = MyResNet18(
        in_channels=init_params["in_channels"],
        num_classes=init_params["num_classes"],
        pretrained=init_params["pretrained"],
        embedding_reduction=init_params["embedding_reduction"],
    )
    if model_name == "resnet_ce":
        criterion = nn.CrossEntropyLoss()
        model_criterion = {"model": model, "criterion": criterion}
    elif model_name == "resnet_robust_ce":
        criterion = nn.CrossEntropyLoss()
        model_criterion = {"model": model, "criterion": criterion}

    elif model_name == "resnet_angular":
        criterion = losses.AngularLoss(alpha=init_params["alpha"])
        model_criterion = {"model": model, "criterion": criterion}

    else:
        print(
            """model name is not valid! Available models: 
              1.resnet_ce
              2.resnet_robust_ce
              3.resnet_angular
              """
        )

    params = {**init_params, **device, **model_criterion}
    return params


def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
