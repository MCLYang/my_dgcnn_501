
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
from collections import OrderedDict
from metrics import AverageMeter

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
def convert_state_dict(state_dict):

    if not next(iter(state_dict)).startswith("module."):
        return state_dict  # abort if dict is not a DataParallel model_state

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

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


def test(args, io):
    load_dir = input('Enter the best model_dir: ')
    device = torch.device("cuda" if args.cuda else "cpu")
    temp_package = torch_load(load_dir)
    args.num_channel = temp_package['num_channel']
    model = DGCNN_semseg(args,num_channel = args.num_channel).to(device)

    temp_dict = temp_package['state_dict']
    temp_dict = convert_state_dict(temp_dict)
    model.load_state_dict(temp_dict)


    num_classes =2
    test_dataset = dt.BigredDataSet(
    root=args.datadir,
    is_train=False,
    is_validation=False,
    is_test=True,
    num_channel = args.num_channel,
    test_code = False
    )   
    file_dict = test_dataset.file_dict
    sorted_keys = np.array(sorted(file_dict.keys()))

    result_sheet = {
    'Complex':[[],[]],
    'Medium':[[],[]],
    'Simple':[[],[]],
    'multiPeople':[[],[]]
    'singlePerson':[[],[]]
    }

    for key in sorted_keys:
        tempname = file_dict[key]
        result_sheet[tempname] = [[],[]]

        # tempname = tempname[:-3]
        # difficulty,location,isSingle = tempname.split("_")


    test_dataloader = DataLoader(test_dataset, 
    num_workers=32, batch_size=args.batch_size, shuffle=False, drop_last=True)

    print("Let's use", torch.cuda.device_count(), "GPUs!")


    best_test_iou = 0
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


    mean_time = AverageMeter()


    result_sheet
    with torch.no_grad():
        for j, data1 in tqdm(enumerate(test_dataloader), total=len(test_dataloader), smoothing=0.9):
            temp_arr = sorted_keys<=j
            index_for_keys = sum(temp_arr)
            file_name = sorted_keys[index_for_keys]

            tempname = tempname[:-3]
            difficulty,location,isSingle = tempname.split("_")



            data, seg = data1
            data, seg = data.to(device), seg.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]

            tic = time.perf_counter()
            seg_pred = model(data)
            toc = time.perf_counter()

            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            pred = seg_pred.max(dim=2)[1]
            count += batch_size
            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            #0test_pred_seg
            #1test_true_seg
            result_sheet[file_name][0].append(seg_np)
            result_sheet[difficulty][0].append(seg_np)
            result_sheet[isSingle][0].append(seg_np)

            result_sheet[file_name][1].append(pred_np)
            result_sheet[difficulty][1].append(pred_np)
            result_sheet[isSingle][1].append(pred_np)


            test_true_cls.append(seg_np.reshape(-1))
            test_pred_cls.append(pred_np.reshape(-1))
            test_true_seg.append(seg_np)
            test_pred_seg.append(pred_np)
            mean_time.update(toc - tic)

    test_time = mean_time.avg
    test_true_cls = np.concatenate(test_true_cls)
    test_pred_cls = np.concatenate(test_pred_cls)
    test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)

    test_true_seg = np.concatenate(test_true_seg, axis=0)
    test_pred_seg = np.concatenate(test_pred_seg, axis=0)
    test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)


    print('val_miou: %f' % temp_package['Validation_ave_miou'])
    print('Test_miou: %f' % test_ious)
    print('Test_acc: %f' % test_acc)
    print('Test ave time(sec/frame): %f' % (test_time))
    print('Test ave time(frame/sec): %f' % (1 / test_time))

    for k in result_sheet.keys():
        test_pred_seg_temp, test_true_seg_temp = result_sheet[k]
        test_ious_temp = calculate_sem_IoU(test_pred_seg_temp, test_trutest_true_seg_tempe_seg)
        result_sheet[k] = test_ious_temp

    print(result_sheet)
    pdb.set_trace()





if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--exp_name', type=str, default=str(time.ctime()), metavar='N',help='Name of the experiment')

    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn'],
                        help='Model to use, [dgcnn]')

    parser.add_argument('--batch_size', type=int, default=1, metavar='batch_size',
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
