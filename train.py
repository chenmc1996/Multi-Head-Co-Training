import math
from torch.optim.lr_scheduler import LambdaLR
import os
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from multi_head  import Multi_Head 
from dataset import SSL_Dataset,get_data_loader

def get_SGD(net, name='SGD', lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True, bn_wd_skip=True):

    optim = getattr(torch.optim, name)
    
    decay = []
    no_decay = []
    for name, param in net.named_parameters():
        if ('bn' in name) and bn_wd_skip:
            no_decay.append(param)
        else:
            decay.append(param)
    
    per_param_args = [{'params': decay},
                      {'params': no_decay, 'weight_decay': 0.0}]
    
    optimizer = optim(per_param_args, lr=lr,
                    momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    return optimizer
        
        
def get_cosine_schedule_with_warmup(optimizer, num_training_steps, num_cycles=7./16., num_warmup_steps=0, last_epoch=-1):
    
    def _lr_lambda(current_step):

        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr
    
    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def accuracy(output, target, topk=(1,)):
    """
    refer: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    
    with torch.no_grad():
        maxk = max(topk) #get k in top-k
        batch_size = target.size(0) #get batch size of target

        # torch.topk(input, k, dim=None, largest=True, sorted=True, out=None)
        # return: value, index
        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True) # pred: [num of batch, k]
        pred = pred.t() # pred: [k, num of batch]
        
        #[1, num of batch] -> [k, num_of_batch] : bool
        correct = pred.eq(target.view(1, -1).expand_as(pred)) 

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        #np.shape(res): [k, 1]
        return res 

def net_builder(net_name, from_name: bool, net_conf=None):
    def setattr_cls_from_kwargs(cls, kwargs):
        #if default values are in the cls,
        #overlap the value by kwargs
        for key in kwargs.keys():
            if hasattr(cls, key):
                print(f"{key} in {cls} is overlapped by kwargs: {getattr(cls,key)} -> {kwargs[key]}")
            setattr(cls, key, kwargs[key])

    if from_name:
        import torchvision.models as models
        model_name_list = sorted(name for name in models.__dict__
                                if name.islower() and not name.startswith("__")
                                and callable(models.__dict__[name]))

        if net_name not in model_name_list:
            assert Exception(f"[!] Networks\' Name is wrong, check net config, \
                               expected: {model_name_list}  \
                               received: {net_name}")
        else:
            return models.__dict__[net_name]
        
    else:
        if net_name == 'WideResNet':
            import wrn as net
            builder = getattr(net, 'build_WideResNet')()
        else:
            assert Exception("Not Implemented Error")
            
        setattr_cls_from_kwargs(builder, net_conf)
        return builder.build

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--save_name', type=str, default='Multi_Head ')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--overwrite', action='store_true')
    
    parser.add_argument('--num_train_iter', type=int, default=2**20, help='total number of training iterations')
    parser.add_argument('--num_eval_iter', type=int, default=10000, help='evaluation frequency')
    parser.add_argument('--num_labels', type=int, default=4000)
    parser.add_argument('--batch_size', type=int, default=64, help='total number of batch size of labeled data')
    parser.add_argument('--uratio', type=int, default=7, help='the ratio of unlabeled data to labeld data in each mini-batch')
    parser.add_argument('--eval_batch_size', type=int, default=1024, help='batch size of evaluation data loader (it does not affect the accuracy)')
    
    parser.add_argument('--ema_m', type=float, default=0.999, help='ema momentum for eval_model')
    parser.add_argument('--ulb_loss_ratio', type=float, default=1.0)
    
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    parser.add_argument('--net', type=str, default='WideResNet')
    parser.add_argument('--net_from_name', type=bool, default=False)
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--widen_factor', type=int, default=2)
    parser.add_argument('--leaky_slope', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)
    
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--num_workers', type=int, default=1)
     
    args = parser.parse_args()

    save_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(save_path) and not args.overwrite:
        raise Exception('already existing model: {}'.format(save_path))
    if args.resume:
        if args.load_path is None:
            raise Exception('Resume of training requires --load_path in the args')
        if os.path.abspath(save_path) == os.path.abspath(args.load_path) and not args.overwrite:
            raise Exception('Saving & Loading pathes are same. \
                            If you want over-write, give --overwrite in the argument.')
        
    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    return args

if __name__ == "__main__":
    args=get_args()

    global best_acc1
    
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True

    save_path = os.path.join(args.save_dir, args.save_name)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
    
    args.bn_momentum = 1.0 - args.ema_m
    _net_builder = net_builder(args.net, args.net_from_name, {'depth': args.depth, 'widen_factor': args.widen_factor, 'leaky_slope': args.leaky_slope, 'bn_momentum': args.bn_momentum, 'dropRate': args.dropout})
    
    model = Multi_Head(_net_builder, args.num_classes, args.ema_m, args.ulb_loss_ratio,  num_eval_iter=args.num_eval_iter)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'Number of Trainable Params: {count_parameters(model.train_model)}')
        
    optimizer = get_SGD(model.train_model, 'SGD', args.lr, args.momentum, args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.num_train_iter, num_warmup_steps=args.num_train_iter*0)
    model.set_optimizer(optimizer, scheduler)
    
    
    torch.cuda.set_device(args.gpu)
    model.train_model = model.train_model.cuda(args.gpu)
    model.eval_model = model.eval_model.cuda(args.gpu)
    
    cudnn.benchmark = True

    train_dset = SSL_Dataset(name=args.dataset, train=True, num_classes=args.num_classes, data_dir=args.data_dir)

    lb_dset, ulb_dset,ulb_dset_wo_aug = train_dset.get_ssl_dset(args.num_labels)
    
    _eval_dset = SSL_Dataset(name=args.dataset, train=False, num_classes=args.num_classes, data_dir=args.data_dir)

    eval_dset = _eval_dset.get_dset()
    
    loader_dict = {}
    dset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset, 'eval': eval_dset,'train_ulb_wo_aug':ulb_dset_wo_aug}
    
    loader_dict['train_lb'] = get_data_loader(dset_dict['train_lb'], args.batch_size, data_sampler = args.train_sampler, num_iters=args.num_train_iter, num_workers=1)
    
    loader_dict['train_ulb'] = get_data_loader(dset_dict['train_ulb'], args.batch_size*args.uratio, data_sampler = args.train_sampler, num_iters=args.num_train_iter, num_workers=4*args.num_workers)

    loader_dict['eval'] = get_data_loader(dset_dict['eval'], args.eval_batch_size, num_workers=args.num_workers)
                                          
    loader_dict['train_ulb_wo_aug'] = get_data_loader(dset_dict['train_ulb_wo_aug'], args.eval_batch_size, num_workers=args.num_workers)
    
    model.set_data_loader(loader_dict)
    
    if args.resume:
        model.load_model(args.load_path)
    
    trainer = model.train
    trainer(args)
        
    model.save_model('latest_model.pth', save_path)
        

