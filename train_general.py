'''
Training script for ImageNet
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import hydra
from omegaconf import OmegaConf, DictConfig

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import wandb

import models.imagenet as customized_models

from utils.misc import GroupNormCreator
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import weight_regularization as wr

CIFAR_PATH = '/data/dataset/'

# Models
default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=200, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[30, 60],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default=None, type=str, metavar='PATH',
                    help='path to save checkpoint (default: None)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--use-ws', action='store_true', default=False)
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed', default=0)
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--wandb-off', action='store_true', default=False)
parser.add_argument('--checkpoint-off', action='store_true', default=False)
parser.add_argument('--notes', type=str, default=None)

#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
# Ortho reg
parser.add_argument('--reg-type', type=str, default=None)
parser.add_argument('--reg-gamma', type=float, default=1e-2)

# args = parser.parse_args()
# state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
use_cuda = torch.cuda.is_available()


def print_flare(s: str):
    print('<' + '=' * 5 + s + '=' * 5 + '>')


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_loaders(args, is_cifar):
    if is_cifar:
        print_flare('CIFAR training')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        if args.data == 'cifar10':
            dataloader = datasets.CIFAR10
            num_classes = 10
        else:
            dataloader = datasets.CIFAR100
            num_classes = 100

        trainset = dataloader(root=os.path.join(CIFAR_PATH, args.data), train=True, download=False,
                              transform=transform_train)
        
        # Setting seed to loader generator. Each worker process is seeded using seed_worker() method
        g = torch.Generator()
        g.manual_seed(args.manualSeed)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True,
                                                   num_workers=args.workers, worker_init_fn=seed_worker, generator=g)
        testset = dataloader(root=os.path.join(CIFAR_PATH, args.data), train=False, download=False,
                             transform=transform_test)
        val_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, shuffle=False,
                                                 num_workers=args.workers)
    else:
        print_flare('IMNET training')
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(traindir, transforms.Compose([
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.train_batch, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.test_batch, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        num_classes = 1000

    return train_loader, val_loader, num_classes


@hydra.main(version_base=None, config_path='conf_train', config_name='train')
def main(args: DictConfig):
    # Random seed
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    best_acc = 0
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    use_chkpt = not args.checkpoint_off
    if not use_chkpt:
        print_flare('Checkpoints are disabled!')

    if args.checkpoint is None:
        # Create path according to runtime flags
        dir_name = 'ws_' + str(args.use_ws) + '_reg_' + ('0' if args.reg_type is None else args.reg_type)
        args.checkpoint = os.path.join('checkpoint', args.data, args.arch, dir_name)
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    print(f'Log/Chkpt dir is {args.checkpoint}')

    is_cifar = args.data in ['cifar10', 'cifar100']

    use_wandb = not args.wandb_off
    if use_wandb:
        tags = [args.data if is_cifar else 'imnet']
        tags += [args.extra_tags]
        tags += ['resnet110']
        wandb.init(project='group_ortho', config=OmegaConf.to_container(args, resolve=True),
                   notes=args.notes, tags=tags)
        wandb.run.log_code(".")
    # Data loading code
    train_loader, val_loader, num_classes = get_loaders(args, is_cifar)

    # create model
    if args.arch in ['l_resnet110']:
        assert is_cifar, f'{args.arch} was created to run on CIFAR'

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    elif args.arch.startswith('l_resnext'):
        model = models.__dict__[args.arch](
                    baseWidth=args.base_width,
                    cardinality=args.cardinality,
                )
    else:
        print("=> creating model '{}'".format(args.arch))
        norm_layer = nn.BatchNorm2d if args.norm == 'BN' else GroupNormCreator(args.force_num_groups)
        model = models.__dict__[args.arch](
            num_classes=num_classes,
            norm_layer=norm_layer,
            use_ws=args.use_ws)

    weight_groups_dict = wr.get_layers_to_regularize(model, input_shape=(3, 32, 32) if is_cifar else (3, 224, 224),
                                                     regularize_all=args.regularize_all)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    elif torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule,
                                                        last_epoch=args.start_epoch - 1)

    if args.warmup and args.arch in ['l_resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * 0.1

    # Resume
    title = 'ImageNet-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(val_loader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):

        if args.warmup and args.arch in ['l_resnet110'] and epoch == 1:
            # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
            # then switch back. In this setup it will correspond for first epoch.
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

        current_lr = optimizer.param_groups[0]['lr']
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, current_lr))

        train_loss, train_acc, reg_loss, class_loss = train(train_loader, model, criterion, optimizer, epoch, use_cuda,
                                                            weight_groups_dict, args)
        test_loss, test_acc = test(val_loader, model, criterion, epoch, use_cuda)

        if use_wandb:
            wandb.log({'train_loss': train_loss, 'epoch': epoch, 'val_loss': test_loss, 'val acc': test_acc,
                       'reg_loss': reg_loss, 'classification loss': class_loss})

        # append logger file
        logger.append([current_lr, train_loss, test_loss, train_acc, test_acc])
        lr_scheduler.step()

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        if use_chkpt:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    logger.close()

    print('Best acc:')
    print(best_acc)

    if use_wandb:
        wandb.summary['best top1'] = best_acc
        wandb.finish()


def train(train_loader, model, criterion, optimizer, epoch, use_cuda, weight_groups_dict, args):
    # switch to train mode
    model.train()
    torch.set_grad_enabled(True)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    reg_loss = AverageMeter()
    task_losses = AverageMeter()
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        batch_size = inputs.size(0)
        if batch_size < args.train_batch:
            continue
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # compute output
        outputs = model(inputs)
        task_loss = criterion(outputs, targets)
        if args.reg_type is not None:
            ortho_loss = wr.group_reg_ortho_l2(model, args.reg_type[len('GSO_'):], weight_groups_dict)
            loss = task_loss + args.ortho_decay * ortho_loss
        else:
            loss = task_loss

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        task_losses.update(task_loss.item(), inputs.shape[0])
        if args.reg_type is not None:
            reg_loss.update(ortho_loss.item(), inputs.shape[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if batch_idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'top1 acc {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, batch_idx, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1))

    return losses.avg, top1.avg, reg_loss.avg, task_losses.avg


def test(val_loader, model, criterion, epoch, use_cuda):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    torch.set_grad_enabled(False)

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        with torch.no_grad():
            outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg, top1.avg


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


if __name__ == '__main__':
    main()