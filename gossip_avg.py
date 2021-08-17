#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 14:59:47 2021

@author: saketi
"""
import argparse
import os
import shutil
import time
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from vgg import *

import torch.nn.functional as F
#from torch.utils.tensorboard import SummaryWriter
from math import ceil
from random import Random

# Importing modules related to distributed processing
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.autograd import Variable
from torch.multiprocessing import spawn

###########
from gossip import GossipDataParallel
from gossip import RingGraph
from gossip import RingGraph_dynamic
from gossip import UniformMixing
from quant_train import *

#GRAPH_TOPOLOGIES = {
#	0: DDEGraph,
#	1: DBEGraph,
#	2: DDLGraph,
#	3: DBLGraph,
#	4: RingGraph,
#	5: NPDDEGraph,
#	-1: None,
#}

#MIXING_STRATEGIES = {
#	0: UniformMixing, 
#	-1: None,
#}
##########

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet', help = 'resnet or vgg or resquant' )
parser.add_argument('-depth', '--depth', default=20, type=int,
                    help='depth of the resnet model')
parser.add_argument('--normtype',   default='batchnorm', help = 'batchnorm or rangenorm or groupnorm' )
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--run_no', default=1, type=str, help='parallel run number, models saved as model_{rank}_{run_no}.th')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=130, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--pretrain_path', dest='pretrain_path', 
                    help='Path for pretrained model', default='', type=str)
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--port', dest='port',
                    help='between 3000 to 65000',default='29500' , type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=5)
parser.add_argument('--biased', dest='biased', action='store_true', 
                    help='biased compression')
parser.add_argument('--unbiased', dest='biased', action='store_false', 
                    help='biased compression')
parser.add_argument('--level', default=32, type=int, metavar='k',
                    help='quantization level 1-32')
parser.add_argument('--eta',  default=1.0, type=float,
                    metavar='AR', help='averaging rate')
parser.add_argument('--compressor', dest='fn',
                    help='Compressor function: quantize, sparsify', default='quantize', type=str)
parser.add_argument('--k', default=0.0, type=float,
                     help='compression ratio for sparsification')
parser.add_argument('--skew', default=0.0, type=float,
                     help='obelongs to [0,1] where 0= completely iid and 1=completely non-iid')
parser.add_argument('--dataset', dest='dataset',
                    help='available datasets: cifar10, cifar100', default='cifar10', type=str)
parser.add_argument('--classes', default=10, type=int, 
                    help='number of classes in the dataset')

args = parser.parse_args()
print(args)

class Partition(object):
    
    def __init__(self, data, index):
        self.data = data
        self.index = index
    
    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

def skew_sort(indices, skew, classes, class_size, seed=512):
    # skew belongs to [0,1]
    rng = Random()
    rng.seed(seed)
    class_indices = {}
    for i in range(0, classes):
        class_indices[i]=indices[0:class_size]
        indices = indices[class_size:]
    random_indices = []
    sorted_indices = []
    sorted_size    = int(skew*class_size)
    for i in range(0, classes):
        sorted_indices = sorted_indices + class_indices[i][0:sorted_size]
        random_indices = random_indices + class_indices[i][sorted_size:]
    rng.shuffle(random_indices)
    return random_indices, sorted_indices
            
    
class DataPartitioner(object):
    """ Partitions a dataset into different chunks"""
    def __init__(self, data, sizes, skew, classes, class_size, seed=512):
        
        self.data = data
        self.partitions = []
        data_len = len(data)
        labels  = [data[i][1] for i in range(0, data_len)]
        sort_index = np.argsort(np.array(labels))
        indices = sort_index.tolist()
        indices_rand, indices = skew_sort(indices, skew=skew, classes=classes, class_size=class_size)
        
        for frac in sizes:
            if skew==1:
                part_len = int(frac*data_len)
                self.partitions.append(indices[0:part_len])
                indices = indices[part_len:]
            elif skew==0:
                part_len = int(frac*data_len)
                self.partitions.append(indices_rand[0:part_len])
                indices_rand = indices_rand[part_len:] 
            else:
                part_len = int(frac*data_len*skew); 
                part_len_rand = int(frac*data_len*(1-skew))
                part_ind = indices[0:part_len]+indices_rand[0:part_len_rand]
                self.partitions.append(part_ind)
                indices = indices[part_len:]
                indices_rand = indices_rand[part_len_rand:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

    
def partition_trainDataset():
    """Partitioning dataset""" 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    if args.dataset == 'cifar10':
        classes = 10
        class_size = 5000
        dataset = datasets.CIFAR10(root='/home/min/a/saketi/Desktop/research/data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True)
    elif args.dataset == 'cifar100':
        classes = 100
        class_size = 500
        dataset = datasets.CIFAR100(root='/home/min/a/saketi/Desktop/research/data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True)
    
    size = dist.get_world_size()
    bsz = int((128) / float(size))
    
    partition_sizes = [1.0/size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes, skew=args.skew, classes=classes, class_size=class_size)
    #partition = DataPartitioner_iid(dataset, partition_sizes)

    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition, batch_size=bsz, shuffle=True,
                pin_memory=False)
    return train_set, bsz



def test_Dataset():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.dataset=='cifar10':
        dataset = datasets.CIFAR10(root='/home/min/a/saketi/Desktop/research/data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
    elif args.dataset=='cifar100':
        dataset = datasets.CIFAR100(root='/home/min/a/saketi/Desktop/research/data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))

    #size = dist.get_world_size()
    val_bsz = 100
    val_set = torch.utils.data.DataLoader(dataset, batch_size=val_bsz, shuffle=False,
                pin_memory=False)
    return val_set, val_bsz


def run(rank, size):
    #writer = SummaryWriter(comment='rank_{}'.format(rank)) 
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.backends.cudnn.deterministic = True
    
    device = torch.device("cuda:{}".format(rank%4))
	    
    global args, best_prec1
    #args = parser.parse_args()
    best_prec1 = 0
    ##############
    data_transferred = 0
    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    if 'resnet' in args.arch:
         model = resnet(num_classes=args.classes, depth=args.depth, dataset=args.dataset, norm_type=args.normtype, group_size=2).to(device)
    elif args.arch == 'vgg11':
        model = vgg11(classes=args.classes).to(device)
    elif args.arch == 'resquant':
        model = resnet_quantized(num_classes=args.classes, depth=args.depth, dataset=args.dataset).to(device)
   
    graph = RingGraph(rank, size)
    mixing = UniformMixing(graph, device)
    model = GossipDataParallel(model, 
				device_ids=[rank%4],
				rank=rank,
				world_size=size,
				graph=graph, 
				mixing=mixing,
				comm_device=device, 
                level = args.level,
                biased = args.biased,
                eta = args.eta,
                compress_ratio=args.k, 
                compress_fn = args.fn, 
                compress_op = 'top_k') 
    model.to(device)#cuda()
    # optionally resume from a checkpoint


    cudnn.benchmark = True

    train_loader, bsz_train = partition_trainDataset()
    val_loader, bsz_val     = test_Dataset()
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)#cuda()

    optimizer = optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    state_dict = torch.load('./save_temp/model_'+str(rank)+'_'+str(args.run_no)+'.th')['state_dict']
    #state_dict = {k: state_dict[k] for k, _ in zip(state_dict, range(20))}
    model.load_state_dict(state_dict) 
    prec1 = validate(val_loader, model, criterion, bsz_val,device, 0)
    
    for epoch in range(0,1):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        #train(train_loader, model, criterion, optimizer, epoch, bsz_train, writer, device)
       
        model.block()
        
        #################
        data_transferred += train(train_loader, model, criterion, optimizer, epoch, bsz_train, device)
        print('after gossip avg')
        
        prec1 = validate(val_loader, model, criterion, bsz_val,device, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)


    #############################
    print("Rank : ", rank, "Data transferred(in GB) : ", data_transferred/1.0e9, "\n")

#def train(train_loader, model, criterion, optimizer, epoch, batch_size, writer, device):
def train(train_loader, model, criterion, optimizer, epoch, batch_size, device):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    data_transferred = 0 
    # switch to train mode
    model.train()
    end = time.time()
    step = len(train_loader)*batch_size*epoch
    for i, (input, target) in enumerate(train_loader):
        #print(i)
        # measure data loading time
        data_time.update(time.time() - end)
        input_var, target_var = Variable(input).to(device), Variable(target).to(device)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        # compute gradient and do SGD step
        loss.backward()
        optimizer.zero_grad()
        _, amt_data_transfer = model.transfer_params(epoch=epoch+(1e-3*i))
        data_transferred += amt_data_transfer
        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target_var)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        

        if i % args.print_freq == 0:
            print('Rank: {0}\t'
                  'Epoch: [{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      dist.get_rank(), epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
                 
        step += batch_size 
        if i==50:
            break

    return data_transferred

def validate(val_loader, model, criterion, batch_size, device, epoch=0):
#def validate(val_loader, model, criterion, batch_size, writer, device, epoch=0):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    step = len(val_loader)*batch_size*epoch

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input_var, target_var = Variable(input).to(device), Variable(target).to(device)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target_var)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Rank: {0}\t'
                      'Test: [{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          dist.get_rank(),i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

                  
            step += batch_size
           

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def init_process(rank, size, fn, backend='nccl'):
    """Initialize distributed enviornment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank,size)

if __name__ == '__main__':
    size = 8
    
    spawn(init_process, args=(size,run), nprocs=size,join=True)
    
