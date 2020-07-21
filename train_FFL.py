# Change the dataset path
DATASET_PATH ='~/data'

import argparse
import json
import time
from datetime import datetime
import warnings
import os
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import random
from logger import SummaryLogger
import utils
import our_network


parser = argparse.ArgumentParser(description='Quantization finetuning for CIFAR100')
parser.add_argument('--text', default='log.txt', type=str)
parser.add_argument('--exp_name', default='cifar100/FFL_res32', type=str)
parser.add_argument('--log_time', default='1', type=str)
parser.add_argument('--lr', default='0.1', type=float)
parser.add_argument('--resume_epoch', default='0', type=int)
parser.add_argument('--epoch', default='300', type=int)
parser.add_argument('--decay_epoch', default=[150, 225], nargs="*", type=int)
parser.add_argument('--w_decay', default='1e-4', type=float)
parser.add_argument('--cu_num', default='0', type=str)
parser.add_argument('--seed', default='1', type=str)
parser.add_argument('--load_pretrained', default='models/ResNet82.pth', type=str)
parser.add_argument('--save_model', default='ckpt.t7', type=str)
parser.add_argument('--depth', type=int, default=32, help='Model depth.')
parser.add_argument('--consistency_rampup', '--consistency_rampup', default=80, type=float,
                    metavar='consistency_rampup', help='consistency_rampup ratio')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args = parser.parse_args()
print(args)


# MaualSeed ####################
torch.manual_seed(int(args.seed))

#### random Seed #####
# random.seed(random.randint(1, 10000))
# torch.manual_seed(args.manualSeed)
#####################


os.environ['CUDA_VISIBLE_DEVICES'] = args.cu_num
trainloader, valloader, testloader = utils.get_cifar100_dataloaders(128, 100)
num_classes = 100

#Other parameters
DEVICE = torch.device("cuda")
RESUME_EPOCH = args.resume_epoch
DECAY_EPOCH = args.decay_epoch
DECAY_EPOCH = [ep - RESUME_EPOCH for ep in DECAY_EPOCH]
FINAL_EPOCH = args.epoch
EXPERIMENT_NAME = args.exp_name
W_DECAY = args.w_decay
base_lr = args.lr


model = our_network.ffl_resnet(num_classes=num_classes,
            depth=args.depth,)

# if len(args.load_pretrained) > 2 :
#     path = args.load_pretrained
#     state = torch.load(path)
#     utils.load_checkpoint(model, state)


# According to CIFAR
module = our_network.Fusion_module(64, num_classes, 8)
model.to(DEVICE)
module.to(DEVICE)

# Loss and Optimizer
optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=W_DECAY, nesterov=True)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=DECAY_EPOCH, gamma=0.1)
criterion_CE = nn.CrossEntropyLoss()
criterion_kl = utils.KLLoss().cuda()
optimizer_FM = optim.SGD(module.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-5, nesterov=True)
scheduler_FM = optim.lr_scheduler.MultiStepLR(optimizer_FM, milestones=DECAY_EPOCH, gamma=0.1)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return utils.sigmoid_rampup(epoch, args.consistency_rampup)

def eval(net,module, test_flag=False):
    loader = valloader if not test_flag else testloader
    flag = 'Val.' if not test_flag else 'Test'

    epoch_start_time = time.time()
    net.eval()
    module.eval()
    val_loss = 0

    correct_sub1 = 0
    correct_sub2 = 0
    correct_fused = 0

    total = 0
    criterion_CE = nn.CrossEntropyLoss()

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs1, outputs2, fmap = model(inputs)

        ensemble_logit = (outputs1 + outputs2) / 2
        fused_logit = module(fmap[0], fmap[1])


        loss_cross = criterion_CE(outputs1, targets) + criterion_CE(outputs2, targets) + criterion_CE(fused_logit, targets)
        # MKD
        loss_kl =(criterion_kl(outputs1, fused_logit) + criterion_kl(outputs2, fused_logit) + criterion_kl(fused_logit,
                                                                                                         ensemble_logit))

        loss = loss_cross + loss_kl
        val_loss += loss.item()


        _, predicted_sub1 = torch.max(outputs1.data, 1)
        _, predicted_sub2 = torch.max(outputs2.data, 1)
        _, predicted_fused = torch.max(fused_logit.data, 1)

        total += targets.size(0)

        correct_sub1 += predicted_sub1.eq(targets.data).cpu().sum().float().item()
        correct_sub2 += predicted_sub2.eq(targets.data).cpu().sum().float().item()
        correct_fused += predicted_fused.eq(targets.data).cpu().sum().float().item()
        b_idx = batch_idx

    print('%s \t Time Taken: %.2f sec' % (flag, time.time() - epoch_start_time))
    print('Loss: %.3f | Acc sub-1: %.3f%% | Acc sub-2: %.3f%% | Acc fused: %.3f%% |' % (train_loss / (b_idx + 1), 100. * correct_sub1 / total, 100. * correct_sub2 / total, 100. * correct_fused / total))

    return val_loss / (b_idx + 1),  correct_sub1 / total,  correct_sub2 / total,  correct_fused / total

def train(model,module, epoch):
    epoch_start_time = time.time()
    print('\n EPOCH: %d' % epoch)
    model.train()
    module.train()

    train_loss = 0
    correct_sub1 = 0
    correct_sub2 = 0
    correct_fused = 0

    total = 0

    global optimizer
    global optimizer_FM


    consistency_weight = get_current_consistency_weight(epoch)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        optimizer_FM.zero_grad()



        ###################################################################################
        outputs1, outputs2, fmap = model(inputs)

        ensemble_logit = (outputs1 + outputs2) / 2
        fused_logit = module(fmap[0], fmap[1])


        loss_cross = criterion_CE(outputs1, targets) + criterion_CE(outputs2, targets) + criterion_CE(fused_logit, targets)

        # MKD
        loss_kl = consistency_weight * (
                    criterion_kl(outputs1, fused_logit) + criterion_kl(outputs2, fused_logit) + criterion_kl(fused_logit,
                                                                                                             ensemble_logit))

        loss = loss_cross + loss_kl
        ###################################################################################
        loss.backward()

        optimizer.step()
        optimizer_FM.step()
        train_loss += loss.item()

        _, predicted_sub1 = torch.max(outputs1.data, 1)
        _, predicted_sub2 = torch.max(outputs2.data, 1)
        _, predicted_fused = torch.max(fused_logit.data, 1)

        total += targets.size(0)

        correct_sub1 += predicted_sub1.eq(targets.data).cpu().sum().float().item()
        correct_sub2 += predicted_sub2.eq(targets.data).cpu().sum().float().item()
        correct_fused += predicted_fused.eq(targets.data).cpu().sum().float().item()

        b_idx = batch_idx

    print('Train s1 \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc sub-1: %.3f%% | Acc sub-2: %.3f%% | Acc fused: %.3f%% |' % (train_loss / (b_idx + 1), 100. * correct_sub1 / total, 100. * correct_sub2 / total, 100. * correct_fused / total))

    return train_loss / (b_idx + 1), correct_fused / total


if __name__ == '__main__':
    time_log = datetime.now().strftime('%m-%d %H:%M')
    if int(args.log_time) :
        folder_name = 'FFL_{}'.format(time_log)


    path = os.path.join(EXPERIMENT_NAME, folder_name)
    if not os.path.exists('ckpt/' + path):
        os.makedirs('ckpt/' + path)
    if not os.path.exists('logs/' + path):
        os.makedirs('logs/' + path)

    # Save argparse arguments as logging
    with open('logs/{}/commandline_args.txt'.format(path), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # Instantiate logger
    logger = SummaryLogger(path)



    for epoch in range(RESUME_EPOCH, FINAL_EPOCH+1):
        f = open(os.path.join("logs/" + path, 'log.txt'), "a")

        ### Train ###
        train_loss, acc = train(model,module, epoch)
        scheduler.step()
        scheduler_FM.step()

        ### Evaluate FFL  ###
        val_loss, accuracy_sub1, accuracy_sub2, accuracy_fused  = eval(model,module, test_flag=True)


        utils.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, True, 'ckpt/' + path, filename='Model_{}.pth'.format(epoch))

        utils.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': module.state_dict(),
                    'optimizer' : optimizer_FM.state_dict(),
                }, True, 'ckpt/' + path, filename='Module_{}.pth'.format(epoch))



        f.write('EPOCH {epoch} \t'
                'ACC_sub-1 : {acc_sub1:.4f} \t ACC_sub-2 : {acc_sub2:.4f}\t' 
                'ACC_fused : {acc_fused:.4f} \t \n'.format(
                    epoch=epoch, acc_sub1=accuracy_sub1, acc_sub2=accuracy_sub2, acc_fused=accuracy_fused)
                )
        f.close()

