import numpy as np
import os
from mxnet.gluon.data.vision import datasets, transforms
import mxnet.gluon.data as data

import mxnet.ndarray as nd
import matplotlib.pyplot as plt

def train_loader(path, batch_size=32, num_workers=4):
    normalize = transforms.Normalize(mean=0.5, std=0.25)
    train_transforms = transforms.Compose([
                                 transforms.Resize((96,112)),# W x H
                                 transforms.RandomFlipLeftRight(),
                                 transforms.ToTensor(),
                                 normalize,
                             ])

    def my_train_transform(img, label):
        return train_transforms(img), label

    train_dataset = datasets.ImageFolderDataset(path, transform=my_train_transform)
    num_train = len(train_dataset)
    print("number of total examples is %d" % num_train)
    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print("number of batches for train, valid and test is %d"%(len(train_loader)))
    return train_loader

def LFW_test_loader(test_dir, target_dir):
    normalize = transforms.Normalize(mean=0.5, std=0.25)
    transform = transforms.Compose([
                                 transforms.Resize((96,112)),
                                 transforms.ToTensor(),
                                 normalize,
                             ])
    def my_transform(img, label):
        return transform(img), label
    testset = datasets.ImageFolderDataset(test_dir, transform=my_transform)
    targetset = datasets.ImageFolderDataset(target_dir, transform=my_transform)

    test_loader = data.DataLoader(testset,batch_size=1)
    target_loader = data.DataLoader(targetset,batch_size=1)
    return test_loader, target_loader

def train_valid_test_loader(path, train_valid_ratio=(0.8,0.1), batch_size=32, num_workers=4):
    normalize = transforms.Normalize(mean=0.5, std=0.25)
    train_transforms = transforms.Compose([
                                 transforms.Resize((96,112)),
                                 transforms.RandomFlipLeftRight(),
                                 transforms.ToTensor(),
                                 normalize,
                             ])
    untrain_transforms = transforms.Compose([
                                 transforms.Resize((96,112)),
                                 transforms.ToTensor(),
                                 normalize,
                             ])
    def my_train_transform(img, label):
        return train_transforms(img), label
    def my_untrain_transform(img, label):
        return untrain_transforms(img), label

    train_dataset = datasets.ImageFolderDataset(path, transform=my_train_transform)
    untrain_dataset = datasets.ImageFolderDataset(path, transform=my_untrain_transform)
    num_train = len(train_dataset)
    print("number of total examples is %d" % num_train)
    indices = list(range(num_train))
    split1 = int(np.floor(train_valid_ratio[0] * num_train))
    split2 = int(np.floor(sum(train_valid_ratio) * num_train))
    train_sampler = SubsetRandomSampler(indices, 0, split1)
    valid_sampler = SubsetRandomSampler(indices, split1, split2)
    test_sampler = SubsetRandomSampler(indices, split2, -1)

    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    valid_loader = data.DataLoader(
        untrain_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
    test_loader = data.DataLoader(
        untrain_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)
    print("number of batches for train, valid and test is %d, %d, %d"%(len(train_loader), len(valid_loader), len(test_loader)))

    return train_loader, valid_loader, test_loader

def prepare_data(data_dir, label_dir, new_dir):
    with open(label_dir, 'rt') as f:
        line = f.readline()
        ii = 1
        while line:
            label = line.split(None,3)
            if len(label) == 2:
                line = f.readline()
                continue
            data1_dir = os.path.join(data_dir,"%s/%s_%04d.jpg"%(label[0],label[0],int(label[1])))
            data2_dir = os.path.join(data_dir,"%s/%s_%04d.jpg"%(label[0],label[0],int(label[2])))
            cur_dir1 = os.path.join(new_dir,"test",label[0])
            cur_dir2 = os.path.join(new_dir,"target",label[0])
            if not os.path.exists(cur_dir1):
                os.popen("mkdir "+cur_dir1)
            if not os.path.exists(cur_dir2):
                os.popen("mkdir "+cur_dir2)
            os.popen("cp "+data1_dir+" "+cur_dir1)
            os.popen("cp "+data2_dir+" "+cur_dir2)
            print label[0]
            ii += 1
            if ii == 300: break
            line = f.readline()

def prepare_data2(data_dir, label_dir, new_dir):
    with open(label_dir, 'rt') as f:
        line = f.readline()
        ii = 1
        temp = []
        while line:
            label = line.split(None,3)
            if len(label) == 2:
                line = f.readline()
                continue
            data1_dir = os.path.join(data_dir,"%s/%s_%04d.jpg"%(label[0],label[0],int(label[1])))
            data2_dir = os.path.join(data_dir,"%s/%s_%04d.jpg"%(label[0],label[0],int(label[2])))
            cur_dir1 = os.path.join(new_dir,"test")
            cur_dir2 = os.path.join(new_dir,"target")
            if not os.path.exists(cur_dir1):
                os.popen("mkdir "+cur_dir1)
            if not os.path.exists(cur_dir2):
                os.popen("mkdir "+cur_dir2)
            os.popen("cp "+data1_dir+" "+cur_dir1)
            os.popen("cp "+data2_dir+" "+cur_dir2)
            temp.append((label[0]))
            ii += 1
            if ii == 300: break
            line = f.readline()


class SubsetRandomSampler(data.Sampler):

    def __init__(self, indices, start, end):
        self.indices = np.random.permutation(len(indices))
        self.indices = self.indices[start:] if end == -1 else self.indices[start:end]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


if __name__ == "__main__":
    prepare_data2("/home1/LFW/aligned_lfw-112X96","/home1/LFW/pairs.txt",
                 "/home/hfq/model_compress/prune/1611.06440/prune_mx_face/test_lfw")