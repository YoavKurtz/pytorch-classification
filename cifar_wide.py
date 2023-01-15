import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import wandb

import hydra
from omegaconf import OmegaConf, DictConfig

from timm.utils import random_seed

from utils.misc import GroupNormCreator
from models.cifar.wrn import WideResNet
import weight_regularization as wr

CIFAR_PATH = '/data/dataset/'
GSO_TYPES = ['GSO_intra', 'GSO_inter']

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--norm', type=str, default='BN', help='BN/GN')
parser.add_argument('--reg-type', type=str, default=None, help='SO, GSO_intra, GSO_inter')

parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-8, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--ortho-decay', '--od', default=1e-2, type=float,
                    help='ortho weight decay')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=28, type=int,
                    help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=10, type=int,
                    help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0.3, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.set_defaults(augment=True)
parser.add_argument('--wandb-off', action='store_true', default=False)
parser.add_argument('--notes', type=str, default=None)
parser.add_argument('--extra-tags', type=str, default='')
parser.add_argument('--group-dict-path', type=str, default=None)
parser.add_argument('--use-ws', action='store_true', default=False)
parser.add_argument('--adjust-decay', action='store_true', default=False)
parser.add_argument('--manualSeed', type=int, help='manual seed', default=0)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--force_num_groups', type=int, default=None)


@hydra.main(version_base=None, config_path='conf', config_name='cifar_wide')
def main(args: DictConfig):
    best_prec1 = 0
    # args = parser.parse_args()

    if args.manualSeed is not None:
        random_seed(seed=args.manualSeed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    assert args.use_ws is False, "not supported yet!"
    use_wandb = not args.wandb_off
    if use_wandb:
        tags = [args.dataset, 'wide resnet', 'repr']
        if not args.adjust_decay:
            tags += ['no decay adj']
        if args.extra_tags != '':
            tags += [args.extra_tags]

        wandb.init(project='group_ortho', config=OmegaConf.to_container(args, resolve=True),
                   notes=args.notes, tags=tags)
        wandb.run.log_code(".")

    dir_name = 'ws_' + str(args.use_ws) + '_reg_' + ('0' if args.reg_type is None else args.reg_type) +\
               '_norm_' + args.norm + '_decay_' + str(args.adjust_decay)
    chkpt_path = os.path.join('checkpoint', args.dataset, 'wide_resnet', dir_name)
    print(f'Saving chkpt to {chkpt_path}')

    # Data loading code
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    with torch.no_grad():
        if args.augment:
            print("Inside Augment")
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    x.unsqueeze(0), (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
    assert (args.dataset == 'cifar10' or args.dataset == 'cifar100')
    train_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()](os.path.join(CIFAR_PATH, args.dataset), train=True, download=False,
                                                transform=transform_train),
        batch_size=args.batch_size, shuffle=args.shuffle_train, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()](os.path.join(CIFAR_PATH, args.dataset), train=False, transform=transform_test),
        batch_size=args.batch_size, shuffle=args.shuffle_val, **kwargs)

    # create model
    norm_layer = nn.BatchNorm2d if args.norm == 'BN' else GroupNormCreator(args.force_num_groups)
    model = WideResNet(args.layers, args.dataset == 'cifar10' and 10 or 100,
                       norm_layer, args.widen_factor, dropRate=args.droprate)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # Initial Value of Decay Values
    odecay = args.ortho_decay

    if args.norm == 'GN' and args.reg_type in GSO_TYPES:
        weight_groups_dict = wr.get_layers_to_regularize(model, (3, 32, 32), regularize_all=args.regularize_all)
    elif args.norm == 'BN' and args.reg_type in GSO_TYPES:
        # Load saved group dict and use it to set ortho groups
        assert os.path.exists(args.group_dict_path), 'Must specify path to weight-group dict'
        weight_groups_dict = torch.load(args.group_dict_path)
    else:
        weight_groups_dict = {}

    if args.random_filter_mode == 'constant':
        weight_groups_dict = wr.generate_permutation(weight_groups_dict)
    if args.group_dict_path is not None and args.norm == 'GN':
        torch.save(weight_groups_dict, args.group_dict_path)

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, nesterov=args.nesterov,
                                weight_decay=args.weight_decay)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch + 1, args)
        if args.adjust_decay:
            adjust_weight_decay_rate(optimizer, epoch+1, args)  # uncommenting since using SO/DSO reg
            odecay = adjust_ortho_decay_rate(epoch + 1, args)
        # train for one epoch
        train_loss, top1, reg_loss, class_loss = train(train_loader, model, criterion, optimizer, epoch,
                                                       odecay, weight_groups_dict, args)

        # evaluate on validation set
        test_acc, test_loss = validate(val_loader, model, criterion, epoch)

        if use_wandb:
            wandb.log({'train_loss': train_loss, 'epoch': epoch, 'val_loss': test_loss, 'val acc': test_acc,
                       'reg_loss': reg_loss, 'classification loss': class_loss})

        # remember best prec@1 and save checkpoint
        is_best = test_acc > best_prec1
        best_prec1 = max(test_acc, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, chkpt_path)
    print('Best accuracy: ', best_prec1)

    if use_wandb:
        wandb.summary['best top1'] = best_prec1
        wandb.finish()


def train(train_loader, model, criterion, optimizer, epoch, odecay, weight_groups_dict, args):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    reg_loss = AverageMeter()
    class_loss = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda()
        input = input.cuda()

        # compute output
        output = model(input)

        # Compute Loss
        oloss = wr.weights_reg(model, args.reg_type, weight_groups_dict=weight_groups_dict,
                               randomize_mode=args.random_filter_mode)
        reg_loss.update(oloss, input.shape[0])
        oloss = odecay * oloss
        loss = criterion(output, target)
        class_loss.update(loss, input.shape[0])
        loss = loss + oloss
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # del input_var, target_var, output, loss
        # torch.cuda.empty_cache()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        wd = optimizer.param_groups[0]['weight_decay']
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t '
                  'weight_decay {wd}\t '
                  'ortho lambda = {odecay}'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1, wd=wd, odecay=odecay))

    return losses.avg, top1.avg, reg_loss.avg, class_loss.avg


def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()
        # print ( input.shape)
        with torch.no_grad():
            # compute output
            output = model(input)

        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        # torch.cuda.empty_cache()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    print('Validate * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg, losses.avg


def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    if not os.path.exists(path):
        os.makedirs(path)
    filename = os.path.join(path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(path,  'model_best.pth.tar'))


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


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = args.lr * ((0.2 ** int(epoch >= 60)) * (0.2 ** int(epoch >= 120)) * (0.2 ** int(epoch >= 160)))
    # log to TensorBoard
    # if args.tensorboard:
    #     log_value('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_weight_decay_rate(optimizer, epoch, args):
    w_d = args.weight_decay

    if epoch > 20:
        w_d = 1e-4  # Paper's SO value
    elif epoch > 10:
        w_d = 1e-6

    for param_group in optimizer.param_groups:
        param_group['weight_decay'] = w_d


def adjust_ortho_decay_rate(epoch, args):
    o_d = args.ortho_decay

    if epoch > 120:
        o_d = 0.0
    elif epoch > 70:
        o_d = 1e-6 * o_d
    elif epoch > 50:
        o_d = 1e-4 * o_d
    elif epoch > 20:
        o_d = 1e-3 * o_d

    return o_d


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


if __name__ == '__main__':
    main()