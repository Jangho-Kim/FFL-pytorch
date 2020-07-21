import os
import shutil
import numpy as np

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb
import numpy as np



def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
    filename = os.path.join(path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(path,'model_best.pth.tar'))

def load_checkpoint(model, checkpoint):
    m_keys = list(model.state_dict().keys())

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        c_keys = list(checkpoint['state_dict'].keys())
        not_m_keys = [i for i in c_keys if i not in m_keys]
        not_c_keys = [i for i in m_keys if i not in c_keys]
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    else:
        c_keys = list(checkpoint.keys())
        not_m_keys = [i for i in c_keys if i not in m_keys]
        not_c_keys = [i for i in m_keys if i not in c_keys]
        model.load_state_dict(checkpoint, strict=False)

    print("--------------------------------------\n LOADING PRETRAINING \n")
    print("Not in Model: ")
    print(not_m_keys)
    print("Not in Checkpoint")
    print(not_c_keys)
    print('\n\n')

def get_cifar100_dataloaders(train_batch_size, test_batch_size):
    transform_train = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[x / 255.0 for x in [129.3, 124.1, 112.4]],
                                     std=[x / 255.0 for x in [68.2, 65.4, 70.4]])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[x / 255.0 for x in [129.3, 124.1, 112.4]],
                                     std=[x / 255.0 for x in [68.2, 65.4, 70.4]])])


    trainset = torchvision.datasets.CIFAR100(root='~/data', train=True, download=True,
                                             transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR100(root='~/data', train=False, download=True,
                                            transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=4)

    subset_idx = np.random.randint(0, len(trainset), size=10000)
    valloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(subset_idx))

    return trainloader, valloader, testloader

def get_cifar100_dataloaders_disjoint(train_batch_size, test_batch_size):
    np.random.seed(0)
    transform_train = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[x / 255.0 for x in [129.3, 124.1, 112.4]],
                                     std=[x / 255.0 for x in [68.2, 65.4, 70.4]])
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[x / 255.0 for x in [129.3, 124.1, 112.4]],
                                     std=[x / 255.0 for x in [68.2, 65.4, 70.4]])])


    trainset = torchvision.datasets.CIFAR100(root='~/data', train=True, download=True,transform=transform_train)

    total_idx = np.arange(0,len(trainset))
    np.random.shuffle(total_idx)
    subset_idx = total_idx[:10000]
    _subset_idx = total_idx[~np.in1d(total_idx, subset_idx)]
    valloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(subset_idx))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(_subset_idx))

    testset = torchvision.datasets.CIFAR100(root='~/data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=4)

    return trainloader, valloader, testloader


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()
    def forward(self, pred, label):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number
        T=3

        predict = F.log_softmax(pred/T,dim=1)
        target_data = F.softmax(label/T,dim=1)
        target_data =target_data+10**(-7)
        target = Variable(target_data.data.cuda(),requires_grad=False)
        loss=T*T*((target*(target.log()-predict)).sum(1).sum()/target.size()[0])
        return loss


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
