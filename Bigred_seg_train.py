#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: main_semseg.py
@Time: 2020/2/24 7:17 PM
"""


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import S3DIS
from model import DGCNN_semseg
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
import pdb
from torch.utils.tensorboard import SummaryWriter
import BigredDataSet as dt
from tqdm import tqdm
import time
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True
# os.environ["CUDA_VISIBLE_DEVICES"] ='0,1'

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main_semseg.py checkpoints'+'/'+args.exp_name+'/'+'main_semseg.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

#m_iou = mIoU_one_frame(seg_pred, seg)

def mIoU_one_frame(seg_pred, seg):


    seg_pred_new = seg_pred.transpose(1,2)
    pred = seg_pred.view(-1, 2)
    pred_choice = pred.data.max(1)[1]
    y_pred = pred_choice.cpu().data.numpy()

    y = seg.view(-1,).cpu().detach().numpy()

    #pdb.set_trace()
    ioU = []
    class_num = [0,1]
    for num in class_num:
        I = np.sum(np.logical_and(y_pred == num, y == num))
        U = np.sum(np.logical_or(y_pred == num, y == num))
        ioU.append(I / float(U))
    ave = np.mean(ioU)
    return(ave)


def calculate_sem_IoU(pred_np, seg_np):
    I_all = np.zeros(2)
    U_all = np.zeros(2)
    for sem_idx in range(seg_np.shape[0]):
        for sem in range(2):
            I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            I_all[sem] += I
            U_all[sem] += U
    return I_all / U_all


def train(args, io):
    # pdb.set_trace()
    num_classes =2

    train_dataset = dt.BigredDataSet(
    root=args.datadir,
    is_train=True,
    is_validation=False,
    is_test=False,
    num_channel = args.num_channel,
    test_code = False
    )

    validation_dataset = dt.BigredDataSet(
    root=args.datadir,
    is_train=False,
    is_validation=True,
    is_test=False,
    num_channel = args.num_channel,
    test_code = False

    )


    train_dataloader = DataLoader(train_dataset, 
    num_workers=32, batch_size=args.batch_size, shuffle=True, drop_last=True)

    validation_loader = DataLoader(validation_dataset, 
    num_workers=32, batch_size=args.batch_size, shuffle=False, drop_last=True)

    label_weight = torch.Tensor(train_dataset.labelweights).cuda()
    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'dgcnn':
        model = DGCNN_semseg(args,num_channel = args.num_channel).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, 20, 0.5, args.epochs)

    criterion = cal_loss
    best_test_iou = 0

    writer = SummaryWriter()
    counter_play = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []
        # for data, seg in train_loader:
        for i, data1 in tqdm(enumerate(train_dataloader), total=len(train_dataloader), smoothing=0.9):
            data, seg = data1
            data, seg = data.to(device), seg.to(device)
            # data, seg = data.cuda(), seg.cuda()

            # data = data.permute(0, 2, 1)
            data = data.transpose(2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            seg_pred = model(data)
            #pdb.set_trace()
            #pdb.set_trace()
            m_iou = mIoU_one_frame(seg_pred, seg)


            # seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            seg_pred = seg_pred.transpose(2, 1).contiguous()
            #pdb.set_trace()
            # loss = seg_pred.sum()

            loss = criterion(seg_pred.view(-1, 2), seg.view(-1,1).squeeze(),weight = label_weight)
            loss.backward()
            # pdb.set_trace()
            opt.step()
            pred = seg_pred.max(dim=2)[1]               # (batch_size, num_points)
            count += batch_size
            train_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()                  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()       # (batch_size, num_points)
            train_true_cls.append(seg_np.reshape(-1))       # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))      # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
            writer.add_scalar('training_loss',loss.item() * batch_size,counter_play)
            writer.add_scalar('training_mIoU',m_iou,counter_play)

            counter_play = counter_play + 1
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_ious = calculate_sem_IoU(train_pred_seg, train_true_seg)
        # outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % (epoch, 
                                                                                                #   train_loss*1.0/count,
                                                                                                #   train_acc,
                                                                                                #   avg_per_class_acc,
                                                                                                #   np.mean(train_ious))

        print('Epoch: %d' % epoch)
        print('Train_ave_miou: %f' % np.mean(train_ious))
        print('Train_ave_acc: %f' % train_acc)
        print('Train_ave_loss: %f' % (train_loss*1.0/count))
        # io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        with torch.no_grad():
            for j, data1 in tqdm(enumerate(validation_loader), total=len(validation_loader), smoothing=0.9):
                data, seg = data1
                data, seg = data.to(device), seg.to(device)
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                seg_pred = model(data)
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                loss = criterion(seg_pred.view(-1, 2), seg.view(-1,1).squeeze(),weight = label_weight)
                pred = seg_pred.max(dim=2)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                seg_np = seg.cpu().numpy()
                pred_np = pred.detach().cpu().numpy()
                test_true_cls.append(seg_np.reshape(-1))
                test_pred_cls.append(pred_np.reshape(-1))
                test_true_seg.append(seg_np)
                test_pred_seg.append(pred_np)
        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
        # outstr = 'Validation %d, loss: %.6f, Validation acc: %.6f, Validation avg acc: %.6f, Validation iou: %.6f' % (epoch,
        #                                                                                       test_loss*1.0/count,
        #                                                                                       test_acc,
        #                                                                                       avg_per_class_acc,
        #                                                                                       np.mean(test_ious))
        # io.cprint(outstr)
        writer.add_scalar('Validation_ave_miou', np.mean(test_ious), epoch)
        writer.add_scalar('Validation_ave_acc', test_acc, epoch)

        print('Epoch: %d' % epoch)
        print('Train_ave_miou: %f' % np.mean(test_ious))
        print('Train_ave_acc: %f' % test_acc)
        print('Train_ave_loss: %f' % (test_loss*1.0/count))

        package = dict()
        package['state_dict'] = model.state_dict()
        package['scheduler'] = scheduler
        package['optimizer'] = opt
        package['Train_ave_val_ave_miou'] = np.mean(train_ious)
        package['Train_ave_acc'] = train_acc
        package['Train_ave_loss'] = train_loss*1.0/count
        package['Validation_ave_miou'] = np.mean(test_ious)
        package['Validation_ave_acc'] = test_acc
        package['epoch'] = epoch
        package['time'] = time.ctime()
        package['num_channel'] = args.num_channel
        package['num_gpu'] = torch.cuda.device_count()


        torch.save(package,'checkpoints/%s/models/val_miou_%f_val_acc_%f_%d.pth' % (args.exp_name,np.mean(test_ious), test_acc,epoch) )

        print('Is Best? ',np.mean(test_ious) >= best_test_iou)

        if np.mean(test_ious) >= best_test_iou:
            best_test_iou = np.mean(test_ious)
            torch.save(package, 'checkpoints/%s/models/best_model.pth' % (args.exp_name))
        print('Best miou:,',best_test_iou)

        # pdb.set_trace()

def test(args, io):
    all_true_cls = []
    all_pred_cls = []
    all_true_seg = []
    all_pred_seg = []
    for test_area in range(1,7):
        test_area = str(test_area)
        if (args.test_area == 'all') or (test_area == args.test_area):
            test_loader = DataLoader(S3DIS(partition='test', num_points=args.num_points, test_area=test_area),
                                     batch_size=args.test_batch_size, shuffle=False, drop_last=False)

            device = torch.device("cuda" if args.cuda else "cpu")

            #Try to load models
            if args.model == 'dgcnn':
                model = DGCNN_semseg(args).to(device)
            else:
                raise Exception("Not implemented")

            model = nn.DataParallel(model)
            model.load_state_dict(torch.load(os.path.join(args.model_root, 'model_%s.t7' % test_area)))
            model = model.eval()
            test_acc = 0.0
            count = 0.0
            test_true_cls = []
            test_pred_cls = []
            test_true_seg = []
            test_pred_seg = []
            for data, seg in test_loader:
                data, seg = data.to(device), seg.to(device)
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                seg_pred = model(data)
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                pred = seg_pred.max(dim=2)[1]
                seg_np = seg.cpu().numpy()
                pred_np = pred.detach().cpu().numpy()
                test_true_cls.append(seg_np.reshape(-1))
                test_pred_cls.append(pred_np.reshape(-1))
                test_true_seg.append(seg_np)
                test_pred_seg.append(pred_np)
            test_true_cls = np.concatenate(test_true_cls)
            test_pred_cls = np.concatenate(test_pred_cls)
            test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
            test_true_seg = np.concatenate(test_true_seg, axis=0)
            test_pred_seg = np.concatenate(test_pred_seg, axis=0)
            test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
            outstr = 'Test :: test area: %s, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (test_area,
                                                                                                    test_acc,
                                                                                                    avg_per_class_acc,
                                                                                                    np.mean(test_ious))
            io.cprint(outstr)
            all_true_cls.append(test_true_cls)
            all_pred_cls.append(test_pred_cls)
            all_true_seg.append(test_true_seg)
            all_pred_seg.append(test_pred_seg)

    if args.test_area == 'all':
        all_true_cls = np.concatenate(all_true_cls)
        all_pred_cls = np.concatenate(all_pred_cls)
        all_acc = metrics.accuracy_score(all_true_cls, all_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(all_true_cls, all_pred_cls)
        all_true_seg = np.concatenate(all_true_seg, axis=0)
        all_pred_seg = np.concatenate(all_pred_seg, axis=0)
        all_ious = calculate_sem_IoU(all_pred_seg, all_true_seg)
        outstr = 'Overall Test :: test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (all_acc,
                                                                                         avg_per_class_acc,
                                                                                         np.mean(all_ious))
        io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--exp_name', type=str, default=str(time.ctime()), metavar='N',help='Name of the experiment')

    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn'],
                        help='Model to use, [dgcnn]')

    parser.add_argument('--batch_size', type=int, default=2, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=20000,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_root', type=str, default='', metavar='N',
                        help='Pretrained model root')
    parser.add_argument('--num_channel', type=int, default=4, metavar='N',
                        help='num_channel')
    parser.add_argument('--datadir', type=str, default='../bigRed_h5_pointnet', metavar='N',
                        help='num_channel')

    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
