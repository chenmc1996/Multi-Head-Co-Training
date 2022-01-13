from torch.utils.data import sampler, DataLoader, Dataset
from randaugment import RandAugment
from PIL import Image
import numpy as np
import copy
import torch
from torch.utils.data.sampler import BatchSampler
from randaugment import RandAugment
import torchvision
from torchvision import transforms


def split_ssl_data(data, target, num_labels, num_classes, index=None, include_lb_to_ulb=True):
    data, target = np.array(data), np.array(target)
    lb_data, lbs, lb_idx = sample_labeled_data(
        data, target, num_labels, num_classes, index)
    ulb_idx = np.array(sorted(list(set(range(len(data))) - set(lb_idx))))
    if include_lb_to_ulb:
        return lb_data, lbs, data, target
    else:
        return lb_data, lbs, data[ulb_idx], target[ulb_idx]


def sample_labeled_data(data, target, num_labels, num_classes, index=None):
    assert num_labels % num_classes == 0
    if not index is None:
        index = np.array(index, dtype=np.int32)
        return data[index], target[index], index

    samples_per_class = int(num_labels / num_classes)

    lb_data = []
    lbs = []
    lb_idx = []
    for c in range(num_classes):
        idx = np.where(target == c)[0]
        idx = np.random.choice(idx, samples_per_class, False)
        lb_idx.extend(idx)

        lb_data.extend(data[idx])
        lbs.extend(target[idx])
    print(f'indices of labeled data: {lb_idx}')

    return np.array(lb_data), np.array(lbs), np.array(lb_idx)


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


def get_data_loader(dset, batch_size=None, shuffle=False, num_workers=4, pin_memory=False, data_sampler=None, replacement=True, num_epochs=None, num_iters=None, generator=None, drop_last=True):
    """
    get_data_loader returns torch.utils.data.DataLoader for a Dataset.
    All arguments are comparable with those of pytorch DataLoader.
    However, if distributed, DistributedProxySampler, which is a wrapper of data_sampler, is used.

    Args
        num_epochs: total batch -> (# of batches in dset) * num_epochs 
        num_iters: total batch -> num_iters
    """

    assert batch_size is not None

    if data_sampler is None:
        return DataLoader(dset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)

    else:
        if isinstance(data_sampler, str):
            data_sampler = get_sampler_by_name(data_sampler)

        if (num_epochs is not None) and (num_iters is None):
            num_samples = len(dset)*num_epochs
        elif (num_epochs is None) and (num_iters is not None):
            num_samples = batch_size * num_iters
        else:
            num_samples = len(dset)

        if data_sampler.__name__ == 'RandomSampler':
            data_sampler = data_sampler(
                dset, replacement, num_samples, generator)
        else:
            raise RuntimeError(f"{data_sampler.__name__} is not implemented.")

        batch_sampler = BatchSampler(data_sampler, batch_size, drop_last)
        return DataLoader(dset, batch_sampler=batch_sampler,
                          num_workers=num_workers, pin_memory=True)


mean, std = {}, {}

mean['cifar10'] = [x / 255 for x in [125.3, 123.0, 113.9]]
std['cifar10'] = [x / 255 for x in [63.0, 62.1, 66.7]]

mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]
std['cifar100'] = [x / 255 for x in [68.2,  65.4,  70.4]]

mean['svhn'] = [x / 255 for x in [129.3, 124.1, 112.4]]
std['svhn'] = [x / 255 for x in [68.2,  65.4,  70.4]]


def get_transform(mean, std, train=True):
    if train:
        return transforms.Compose([transforms.RandomHorizontalFlip(),
                                   transforms.RandomCrop(32, padding=4),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std),
                                   ])
    else:
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])


class SSL_Dataset:
    def __init__(self, name='cifar10', train=True, num_classes=10, data_dir='../data'):

        self.name = name
        self.train = train
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.transform = get_transform(mean[name], std[name], train)

    def get_data(self):

        dset = getattr(torchvision.datasets, self.name.upper())
        dset = dset(self.data_dir, train=self.train, download=True)
        data, targets = dset.data, dset.targets
        return data, targets

    def get_dset(self, use_strong_transform=False,
                 strong_transform=None, onehot=False):

        data, targets = self.get_data()
        num_classes = self.num_classes
        transform = self.transform
        data_dir = self.data_dir

        return BasicDataset(data, targets, num_classes, transform,
                            use_strong_transform, strong_transform, onehot)

    def get_ssl_dset(self, num_labels, index=None, include_lb_to_ulb=True,
                     use_strong_transform=True, strong_transform=None,
                     onehot=False):

        data, targets = self.get_data()
        num_classes = self.num_classes
        transform = self.transform
        data_dir = self.data_dir

        lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(data, targets,
                                                                    num_labels, num_classes,
                                                                    index, include_lb_to_ulb)

        lb_dset = BasicDataset(lb_data, lb_targets, num_classes,
                               transform, False, None, onehot)

        ulb_dset = BasicDataset(data, targets, num_classes,
                                transform, use_strong_transform, strong_transform, onehot)

        ulb_dset_wo_aug = BasicDataset(data, targets, num_classes,
                                       get_transform(mean[self.name], std[self.name], train=False), use_strong_transform=False)

        return lb_dset, ulb_dset, ulb_dset_wo_aug


class BasicDataset(Dataset):

    def __init__(self, data, targets=None, num_classes=None, transform=None, use_strong_transform=False, strong_transform=None,  *args, **kwargs):

        super(BasicDataset, self).__init__()
        self.data = data
        self.targets = targets

        self.num_classes = num_classes
        self.use_strong_transform = use_strong_transform

        self.transform = transform

        if use_strong_transform:
            if strong_transform is None:
                self.strong_transform = copy.deepcopy(transform)
                self.strong_transform.transforms.insert(0, RandAugment(3, 5))
        else:
            self.strong_transform = strong_transform

    def __getitem__(self, idx):

        if self.targets is None:
            target = None
        else:
            target_ = self.targets[idx]
            target = target_

        img = self.data[idx]
        if self.transform is None:
            return transforms.ToTensor()(img), target
        else:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
                img_w = self.transform(img)
            if not self.use_strong_transform:
                return img_w, target
            else:
                return self.transform(img), [self.strong_transform(img) for i in range(3)], target

    def __len__(self):
        return len(self.data)
