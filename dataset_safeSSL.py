import torch.distributed as dist
from torch.utils.data import sampler, DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from randaugment import RandAugment
from PIL import Image
import numpy as np
import copy
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler

from randaugment import RandAugment

import torchvision

used_class={10:list(range(2,8))+[0,1,8,9],100:[17, 39, 48, 43, 45, 56, 18, 69, 44, 76, 28, 22, 21,  7, 81, 16, 61, 34, 49, 63, 37, 52, 71,  4, 96, 82, 14, 58, 47, 57, 93,  2, 89, 64, 70, 85, 88, 11, 23, 24, 19, 92, 30, 20, 77,  6,  3, 95,  1, 80, 73, 78, 74,  9, 60, 55, 87, 31, 79, 36, 26, 41, 12, 40, 54, 72, 15, 35, 10, 25, 66, 91, 46, 59, 53, 38,  5, 97,  8, 84, 27, 83, 33, 98, 62, 32, 94, 65, 13, 75, 90,  0, 67, 29, 99, 42, 51, 68, 86, 50]} # classmap=np.asarray([6,6,0,1,2,3,4,5,6,6])
ratio=80

def split_ssl_data(data, target, num_labels, num_classes, index=None, include_lb_to_ulb=True,num_unlabels=None):
    data, target = np.array(data), np.array(target)
    lb_data, lbs, lb_idx, ulb_data, ulbs, ulb_idx = sample_labeled_and_unlabeled(data, target, num_labels, num_classes, index)
    return lb_data, lbs, data[ulb_idx], target[ulb_idx]
    
    
def sample_labeled_and_unlabeled(data, target, num_labels, num_classes,index=None,):
    if not index is None:
        index = np.array(index, dtype=np.int32)
        return data[index], target[index], index
    
    samples_per_class = num_labels 
    lb_data = []
    lbs = []
    lb_idx = []
    ulb_idx = []
    ulbs = []

    ulb_data = []
    if num_classes==10:
        inlist=used_class[num_classes][:8]
    else:
        inlist=used_class[num_classes][:ratio]

    for c in range(num_classes):
        if c in inlist:
            idx = np.where(target == c)[0]
            np.random.shuffle(idx)

            idx,uidx = idx[:samples_per_class],idx
            lb_idx.extend(idx)
        
            lb_data.extend(data[idx])
            lbs.extend(target[idx])

            ulb_data.extend(data[uidx])
            ulbs.extend(target[uidx])
            ulb_idx.extend(uidx)
        else:
            idx = np.where(target == c)[0]
            np.random.shuffle(idx)

            ulb_data.extend(data[idx])
            ulbs.extend(target[idx])
            ulb_idx.extend(idx)

        
    return np.array(lb_data), np.array(lbs), np.array(lb_idx), np.array(ulb_data), np.array(ulbs), np.array(ulb_idx)


def get_sampler_by_name(name):
    sampler_name_list = sorted(name for name in torch.utils.data.sampler.__dict__ 
                      if not name.startswith('_') and callable(sampler.__dict__[name]))
    try:
        if name == 'DistributedSampler':
            return torch.utils.data.distributed.DistributedSampler
        else:
            return getattr(torch.utils.data.sampler, name)
    except Exception as e:
        print(repr(e))
        print('[!] select sampler in:\t', sampler_name_list)

        
def get_data_loader(dset,
                    batch_size = None,
                    shuffle = False,
                    num_workers = 0,
                    pin_memory = True,
                    data_sampler = None,
                    replacement = True,
                    num_epochs = None,
                    num_iters = None,
                    generator = None,
                    drop_last=True,
                    distributed=False):
    
    assert batch_size is not None
        
    if data_sampler is None:
        return DataLoader(dset, batch_size=batch_size, shuffle=shuffle, 
                          num_workers=num_workers, pin_memory=True)
    
    else:
        if isinstance(data_sampler, str):
            data_sampler = get_sampler_by_name(data_sampler)
        
        if distributed:
            assert dist.is_available()
            num_replicas = dist.get_world_size()
        else:
            num_replicas = 1
        
        if (num_epochs is not None) and (num_iters is None):
            num_samples = len(dset)*num_epochs
        elif (num_epochs is None) and (num_iters is not None):
            num_samples = batch_size * num_iters * num_replicas
        else:
            num_samples = len(dset)
        
        if data_sampler.__name__ == 'RandomSampler':    
            data_sampler = data_sampler(dset, replacement, num_samples, generator)
        else:
            raise RuntimeError(f"{data_sampler.__name__} is not implemented.")
        
        if distributed:
            '''
            Different with DistributedSampler, 
            the DistribuedProxySampler does not shuffle the data (just wrapper for dist).
            '''
            data_sampler = DistributedProxySampler(data_sampler)

        batch_sampler = BatchSampler(data_sampler, batch_size, drop_last)
        return DataLoader(dset, batch_sampler=batch_sampler, 
                          num_workers=num_workers, pin_memory=True)

    
def get_onehot(num_classes, idx):
    onehot = np.zeros([num_classes], dtype=np.float32)
    onehot[idx] += 1.0
    return onehot

mean, std = {}, {}

mean['cifar10'] = [x / 255 for x in [125.3, 123.0, 113.9]]
std['cifar10'] = [x / 255 for x in [63.0, 62.1, 66.7]]

mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]
std['cifar100'] = [x / 255 for x in [68.2,  65.4,  70.4]]

mean['svhn'] = [x / 255 for x in [129.3, 124.1, 112.4]]
std['svhn'] = [x / 255 for x in [68.2,  65.4,  70.4]]

def get_transform(mean, std, train=True,name='cifar'):
    if train:
        if name=='svhn':
            return transforms.Compose([transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
                                          transforms.ToTensor(), 
                                          transforms.Normalize(mean, std),
                                          #Cutout(n_holes=1, length=8)
                                          ])
        elif name=='stl10':
             return  transforms.Compose([
                transforms.RandomCrop(96, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                ])
        elif name=='miniimagenet':
             return  transforms.Compose([
                transforms.RandomCrop(84, padding=8,padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                ])

        elif name=='miniimagenet_s':
            transform=transforms.Compose([
                transforms.RandomCrop(84, padding=8,padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                ])
            strong_transform = copy.deepcopy(transform)
            strong_transform.transforms.insert(0, RandAugment(3,5))

            class Transform3:
                def __init__(self, transform0,transform1):
                    self.transform0 = transform0
                    self.transform1 = transform1
            
                def __call__(self, inp):
                    out0 = self.transform0(inp)
                    out1 = self.transform1(inp)
                    out2 = self.transform1(inp)
                    out3 = self.transform1(inp)
                    return out0,[out1, out2,out3]
            return Transform3(transform,strong_transform)
        else:
            return transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.ToTensor(), 
                                          transforms.Normalize(mean, std),
                                          ])

    else:
        return transforms.Compose([transforms.ToTensor(), 
                                     transforms.Normalize(mean, std)])

    
class SSL_Dataset:
    def __init__(self,
                 name='cifar10',
                 train=True,
                 num_classes=10,
                 data_dir='../data'):
        self.name = name
        self.train = train
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.transform = get_transform(mean[name], std[name], train,name)
        
    def get_data(self):
        dset = getattr(torchvision.datasets, self.name.upper())

        if self.name=='svhn':
            if self.train:
                dset = dset(self.data_dir, split='train', download=True)
            else:
                dset = dset(self.data_dir, split='test', download=True)
            data, targets = dset.data, dset.labels
        elif self.name=='stl10':
            if self.train:
                dset = dset(self.data_dir, split='train', download=True)
            else:
                dset = dset(self.data_dir, split='test', download=True)
            data, targets = dset.data, dset.labels
        elif self.name=='miniimagenet':
            data, targets = dset.data, dset.labels
        else:
            dset = dset(self.data_dir, train=self.train, download=True)
            data, targets = dset.data, dset.targets
        return data, targets
    
    
    def get_dset(self, use_strong_transform=False, 
                 strong_transform=None, onehot=False):
        
        data, targets = self.get_data()
        targets = np.asarray(targets)
        num_classes = self.num_classes
        transform = self.transform
        data_dir = self.data_dir


        if num_classes==10:
            inlist=used_class[num_classes][:8]
        else:
            inlist=used_class[num_classes][:ratio]
        idxs=[]
        for c in range(num_classes):
            if c in inlist:
                idx = np.where(targets == c)[0]
                idxs.extend(idx)

        idxs=np.array(idxs)
        data=data[idxs]
        
        targets=targets[idxs]
        return BasicDataset(data, targets, num_classes, transform, 
                            use_strong_transform, strong_transform, onehot)
    
    
    def get_ssl_dset(self, num_labels, index=None, include_lb_to_ulb=True,
                            use_strong_transform=True, strong_transform=None, 
                            onehot=False,num_unlabels=None):
        """
        get_ssl_dset split training samples into labeled and unlabeled samples.
        The labeled data is balanced samples over classes.
        
        Args:
            num_labels: number of labeled data.
            index: If index of np.array is given, labeled data is not randomly sampled, but use index for sampling.
            include_lb_to_ulb: If True, consistency regularization is also computed for the labeled data.
            use_strong_transform: If True, unlabeld dataset returns weak & strong augmented image pair. 
                                  If False, unlabeled datasets returns only weak augmented image.
            strong_transform: list of strong transform (RandAugment in FixMatch)
            oenhot: If True, the target is converted into onehot vector.
            
        Returns:
            BasicDataset (for labeled data), BasicDataset (for unlabeld data)
        """
        
        num_classes = self.num_classes
        transform = self.transform
        data_dir = self.data_dir

        if self.name=='stl10':
            dset = getattr(torchvision.datasets, self.name.upper())(self.data_dir, split='train',folds=1, download=True)
            data, target = dset.data, dset.labels
            data, target = np.array(data), np.array(target)
            lb_dset = BasicDataset(data, target, num_classes, transform, False, None, onehot)

            ulb_dset_wo_aug = BasicDataset(data, target, num_classes, get_transform(mean[self.name], std[self.name], train=False,name=self.name), use_strong_transform=False) 
            d=getattr(torchvision.datasets, self.name.upper())(data_dir, split='train+unlabeled', download=True)
            ulb_dset = BasicDataset(d.data, d.labels, num_classes, 
                                   transform, use_strong_transform, strong_transform, onehot)
            return lb_dset, ulb_dset,ulb_dset_wo_aug
        else:

            data, targets = self.get_data()
            lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(data, targets, num_labels, num_classes, index, include_lb_to_ulb,num_unlabels=num_unlabels)
            lb_dset = BasicDataset(lb_data, lb_targets, num_classes, 
                                   transform, False, None, onehot)
            ulb_dset = BasicDataset(ulb_data, ulb_targets, num_classes, 
                                   transform, use_strong_transform, strong_transform, onehot)
            ulb_dset_wo_aug = BasicDataset(data, targets, num_classes, 
                                   get_transform(mean[self.name], std[self.name], train=False,name=self.name), use_strong_transform=False)
            return lb_dset, ulb_dset,ulb_dset_wo_aug


class BasicDataset(Dataset):
    def __init__(self,
                 data,
                 targets=None,
                 num_classes=None,
                 transform=None,
                 use_strong_transform=False,
                 strong_transform=None,
                 onehot=False,
                 *args, **kwargs):
        super(BasicDataset, self).__init__()
        self.data = data
        self.targets = targets
        
        self.num_classes = num_classes
        self.use_strong_transform = use_strong_transform
        self.onehot = onehot
        
        self.transform = transform
        

        if use_strong_transform:
            if strong_transform is None:
                self.strong_transform = copy.deepcopy(transform)
                self.strong_transform.transforms.insert(0, RandAugment(3,5))
        else:
            self.strong_transform = strong_transform
    
    def __getitem__(self, idx):
        if self.targets is None:
            target = None
        else:
            target_ = self.targets[idx]
            target = target_ if not self.onehot else get_onehot(self.num_classes, target_)
        img = self.data[idx]
        if self.transform is None:
            return transforms.ToTensor()(img), target
        else:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            if not self.use_strong_transform:
                img_w = self.transform(img)
                return img_w, target
            else:
                return self.transform(img), self.strong_transform(img), target
    
    def __len__(self):
        return len(self.data)
