from __future__ import print_function
import argparse
import os
import sys
sys.path.append('../')
sys.path.append('/')

import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import dataset as dt
from model import PointNetDenseCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pdb
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import count
import pandas as pd
from metrics import AverageMeter
from collections import OrderedDict
import time
parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=1, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=32)
parser.add_argument(
    '--nepoch', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, default='../../bigRed_h5_pointnet', help="dataset path")
parser.add_argument('--class_choice', type=str, default='Pedestrain', help="class_choice")
parser.add_argument('--feature_transform', default=True, help="use feature transform")
parser.add_argument('--load_model_dir', type=str, default='Sun Apr 26 07:16:44 2020/', help="retest")


#pdb.set_trace()


def convert_state_dict(state_dict):

    if not next(iter(state_dict)).startswith("module."):
        return state_dict  # abort if dict is not a DataParallel model_state

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def mIoU(y_pred, y):
    ioU = []
    class_num = [0, 1]
    for num in class_num:
        I = np.sum(np.logical_and(y_pred == num, y == num))
        U = np.sum(np.logical_or(y_pred == num, y == num))
        ioU.append(I / float(U))
    ave = np.mean(ioU)
    return (ave)



opt = parser.parse_args()
# print(opt)

PATH = opt.load_model_dir+'/best_model.pth'
my_loader = torch.load(PATH)
para_state_dict = my_loader['state_dict']
opt.num_channel = my_loader['num_channel']

if(opt.num_channel>3):
    opt.feature_transform = True
else:
    opt.feature_transform = False

#pdb.set_trace()

print('----------------------Creating Model----------------------')
classifier = PointNetDenseCls(k=2, feature_transform=opt.feature_transform, num_channel=opt.num_channel)
# classifier = torch.nn.DataParallel(classifier, device_ids=[0, 1])
print("Loading: ", PATH)

# current_epoch = int(PATH[-5])
# print("current_epoch:", current_epoch)
new_state_dict = convert_state_dict(para_state_dict)
classifier.load_state_dict(new_state_dict)  # Choose whatever GPU device number you want
classifier.cuda()

num_classes = 2
opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

test_dataset = dt.BigredDataSet(
    root=opt.dataset,
    is_train=False,
    is_validation=False,
    is_test=True,
    num_channel = opt.num_channel
)
testloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=False,
    pin_memory=True,
    drop_last=True,
    num_workers=int(opt.workers))
print("len(testdataloader):", len(testloader))

num_batch = len(testloader) / opt.batchSize

mean_miou = AverageMeter()
mean_acc = AverageMeter()
mean_time = AverageMeter()

num_classes = 2
try:
    os.makedirs(opt.outf)
except OSError:
    pass

blue = lambda x: '\033[94m' + x + '\033[0m'

print("----------------------Test----------------------")
with torch.no_grad():
    mean_miou.reset()
    mean_acc.reset()
    for j, data in tqdm(enumerate(testloader), total=len(testloader), smoothing=0.9):
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = classifier.eval()
        tic = time.perf_counter()
        pred, _ = classifier(points)
        toc = time.perf_counter()
        #print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")
        pred = pred.view(-1, num_classes)
        target = target.view(-1, 1)[:, 0]
        loss = F.nll_loss(pred, target)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        pred_np = pred_choice.cpu().data.numpy()
        target_np = target.cpu().data.numpy()
        m_iou = mIoU(pred_np, target_np)
        mean_miou.update(m_iou)
        mean_acc.update(correct.item()/float(opt.batchSize * 20000))
        mean_time.update(toc - tic)
    test_time = mean_time.avg
    val_ave_miou = mean_miou.avg
    val_ave_acc = mean_acc.avg
    print('val_miou: %f' % my_loader['Validation_ave_miou'])
    print('Test_miou: %f' % val_ave_miou)
    print('Test_acc: %f' % val_ave_acc)
    print('Test ave time(sec/frame): %f' % (test_time))
    print('Test ave time(frame/sec): %f' % (1/test_time))

        # package = dict()
        # package['state_dict'] = classifier.state_dict()
        # package['val_ave_miou'] = val_ave_miou
        # package['val_ave_acc'] = val_ave_acc
        # package['current_epoch'] = current_epoch

        # torch.save(classifier.state_dict(),'seg/xyz_intensity_label_Wed_Apr_22/validation/val_miou_%f_val_acc_%f_%d.pth' % (val_ave_miou,val_ave_acc,current_epoch))
        # if(best_value<val_ave_miou):
        #     best_value = val_ave_miou
        #     torch.save(classifier.state_dict(),'seg/xyz_intensity_label_Wed_Apr_22/validation/best_model_val_miou_%f.pth'% val_ave_miou)

    ## benchmark mIOU
    # shape_ious = []
    # for i,data in tqdm(enumerate(testdataloader, 0)):
    #     points, target = data
    #     points = points.transpose(2, 1)
    #     points, target = points.cuda(), target.cuda()
    #     classifier = classifier.eval()
    #     pred, _, _ = classifier(points)
    #     pred_choice = pred.data.max(2)[1]
    #
    #     pred_np = pred_choice.cpu().data.numpy()
    #     target_np = target.cpu().data.numpy() - 1
    #
    #     for shape_idx in range(target_np.shape[0]):
    #         parts = range(num_classes)#np.unique(target_np[shape_idx])
    #         part_ious = []
    #         for part in parts:
    #             I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
    #             U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
    #             if U == 0:
    #                 iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
    #             else:
    #                 iou = I / float(U)
    #             part_ious.append(iou)
    #         shape_ious.append(np.mean(part_ious))
    #
    # print("mIOU for class {}: {}".format(opt.class_choice, np.mean(shape_ious)))
