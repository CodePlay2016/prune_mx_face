import cv2
import sys
import os,gc
import numpy as np
import dataset
from prune_once import *
import argparse
from operator import itemgetter
from heapq import nsmallest
import time,pickle, threading
import matplotlib.pyplot as plt

import mxnet.gluon as gluon
import mxnet.gluon.nn as nn
import mxnet as mx
from mxnet import autograd


import models
from test_model import test_on_LFW

import gradcam
# reference: https://github.com/apache/incubator-mxnet/blob/master/example/cnn_visualization/gradcam.py

class FilterPrunner:
    def __init__(self, mmodel, ctx):
        self.model = mmodel
        self.filter_ranks = {}
        self.flag = True
        self.ctx = ctx

    def reset(self):
        del self.filter_ranks
        gc.collect()
        self.filter_ranks = {}

    def forward(self, x, label, criterion):
        self.grad_index = 0  # index of the layer
        self.activation_to_layer = {}

        activation_index = 0
        for layer, (name, module) in enumerate(self.model.features._children.items()):
            if isinstance(module, nn.Dense): break
            for slayer,(sname, smodule) in enumerate(module._children.items()):
                if isinstance(smodule, gradcam.Conv2D):
                    activation, grad = gradcam.get_conv_out_grad(self.model,x,label,criterion=criterion,
                                                                 conv_layer_name=smodule._name)
                    # self.activation_to_layer[activation_index] = slayer
                    self.compute_rank(activation_index, activation, grad, smodule)
                    activation_index += 1

    def compute_rank(self,activation_index, activation, grad, module, lamda=0.001):
        # should add abs to the oracle ranking value according to the paper
        values = mx.nd.sum((activation * grad), axis =[0,2,3])
        # Normalize the rank by the filter dimensions
        act_shape = activation.shape # [batches, channels, size, size]
        # filter_shape = module.conv.weight.shape
        values = \
            values / (act_shape[0] * act_shape[2] * act_shape[3])
        values = values.as_in_context(mx.cpu())

        # values -= (filter_shape[1]*filter_shape[2]*filter_shape[3]*act_shape[1]*act_shape[2]*act_shape[3])*lamda

        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = mx.nd.zeros(activation.shape[1],ctx=mx.cpu())

        self.filter_ranks[activation_index] += values
        self.grad_index += 1

    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].shape[0]):
                data.append(
                    (i, j, self.filter_ranks[i][j]))

        return nsmallest(num, data, itemgetter(2))

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = mx.nd.abs(self.filter_ranks[i])
            v = v / mx.nd.sqrt((v * v).sum())
            self.filter_ranks[i] = v

    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)

        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(
                filters_to_prune_per_layer[l])

        return filters_to_prune_per_layer

class PrunningFineTuner_VGG16:
    def __init__(self, train_path, model, log_dir=None, ctx=mx.cpu()):
        self.train_data_loader = dataset.train_loader(train_path,batch_size=128)
        a,self.valid_data_loader,b = dataset.train_valid_test_loader(train_path,batch_size=64)
        self.model = model
        self.ctx = ctx
        self.criterion = models.AngleLoss()
        self.log_dir = log_dir
        self.p = Printer(log_dir)
        self.model_save_path = os.path.join(self.log_dir, "model")
        self.model_saved = False
        self.device_id = 6

    def eval(self):
        return test_on_LFW(self.model,ctx=self.ctx)

    def train(self, trainer=None, epoches=10,
              save_highest=True, eval_train_acc=False, best_acc=0):
        lr,lr_decay = 1e-4, 0.95
        if trainer is None:
            optimizer = mx.optimizer.Adam(lr)
            trainer = gluon.Trainer(self.model.collect_params(), optimizer)
        # self.get_cuda_memory("before training ")
        best_loss = 1e5
        loss_list = []
        for i in range(epoches):
            self.p.log("\nEpoch: %d" % (i+1))
            start = time.time()
            self.get_cuda_memory()
            trainer = gluon.Trainer(self.model.collect_params(), mx.optimizer.Adam(lr))
            lr *= lr_decay
            self.p.log("current learning rate is {}".format(lr))
            self.model.get_feature=False # whether to drop the last layer
            train_loss, loss_list = self.train_epoch(trainer,loss_list)
            self.p.log("train loss is %.4f"%train_loss)
            train_time = time.time() - start
            self.get_cuda_memory()
            if best_loss > train_loss:
                best_acc = train_loss
                if save_highest:
                    self.model.save_params(self.model_save_path)
                    self.model_saved = True
                    self.p.log("model resaved...")
            self.p.log("train step time elaps: %.2fs, total time elaps: %.2fs" % (
                train_time, time.time()-start))
            # self.get_cuda_memory("Fine tuning cuda memory is:")
        if save_highest and self.model_saved:
            self.p.log("model reloaded...")
            self.model.load_params(self.model_save_path,ctx=self.ctx)
            self.model_saved = False
        else:
            self.model.save_params(self.model_save_path)
        self.eval()
        with open(os.path.join(self.log_dir,"loss_list.pkl"),'rb') as f:
            pickle.dump(loss_list,f)
        iter_list = np.arange(10,epoches*len(self.train_data_loader)+1,10)
        plt.plot(iter_list,loss_list)
        self.p.log("Finished fine tuning. best valid acc is %.4f" % best_acc)

    def train_batch(self, batch, label, rank_filters):
        if rank_filters:
            self.prunner.forward(batch, label,
                                 self.criterion)
        else:
            with autograd.record():
                out = self.model(batch)
                loss=self.criterion(out[0],out[1], label)
            loss.backward()
            return loss

    def train_epoch(self, trainer=None, loss_list=[], rank_filters=False):
        cumulative_train_loss = mx.nd.zeros(1, ctx=mx.cpu())
        train_samples = 0
        start = time.time()
        data_loader = self.valid_data_loader if rank_filters else self.train_data_loader
        print(len(data_loader))
        for ii, (batch, label) in enumerate(data_loader):
            if ii == len(data_loader)-1: break
            if not isinstance(ctx, list):
                batch = batch.as_in_context(ctx)
                label = label.as_in_context(ctx)
                loss = self.train_batch(batch, label, rank_filters)
            else:
                loss = mx.nd.zeros(1, ctx=mx.cpu())
                batches = gluon.utils.split_and_load(batch,ctx)
                labels = gluon.utils.split_and_load(label,ctx)
                losses = [self.train_batch(sbatch, slabel, rank_filters)\
                        for sbatch, slabel in zip(batches,labels)]
            train_samples += batch.shape[0]
            if not rank_filters:
                trainer.step(batch.shape[0])
                if isinstance(ctx, list):
                    for one_loss in losses:
                        loss += one_loss.as_in_context(mx.cpu())
                    cumulative_train_loss += loss.sum()
                else:
                    cumulative_train_loss += loss.as_in_context(mx.cpu()).sum()
            mx.ndarray.waitall()
            if ii % 100 == 0:
                # self.p.log("iter {}, time use: {}, loss: {}".format(ii,time.time()-start,
                #                                                     loss.sum().asscalar()/batch.shape[0]))
                print ii
                start = time.time()
                if ii % 10 == 0: loss_list.append(cumulative_train_loss.asscalar()/train_samples)
        return cumulative_train_loss.asscalar()/train_samples, loss_list# total loss

    def get_candidates_to_prune(self, num_filters_to_prune):
        self.prunner.reset()
        self.train_epoch(rank_filters=True)
        self.prunner.normalize_ranks_per_layer()
        return self.prunner.get_prunning_plan(num_filters_to_prune)

    def total_num_filters(self):
        filters = 0
        num_convlayers = 0
        for name, module in self.model.features._children.items():
            if isinstance(module, nn.Dense): continue
            for name_, smodule in module._children.items():
                if isinstance(smodule, gradcam.Conv2D):
                    filters = filters + smodule.conv._channels
                    num_convlayers += 1
        return filters

    def set_grad_requirment(self, status):
        for param in self.model.features.parameters():
            param.requires_grad = status

    def get_cuda_memory(self,msg=""):
        command = "nvidia-smi -q -d Memory | grep -A4 GPU |grep Free"
        res = os.popen(command).readlines()[self.device_id][8:-1]
        res += "  ||  number of params in model is "+str(
            sum(param.numel() for _,param in self.model.params))
        self.p.log(msg+res)
        return res

    def reload_model(self):
        self.model.save_params(self.model_save_path)
        # self.model = ModifiedVGG16Model(ctx=ctx)
        self.model.load_params(self.model_save_path, ctx=self.ctx)
        gc.collect()

    def prune(self):
        # Get the accuracy before prunning
        # self.test()

        number_of_filters = self.total_num_filters()
        num_filters_to_prune_per_iteration = 512
        iterations = int(float(number_of_filters) /
                         num_filters_to_prune_per_iteration)
        iterations = int(iterations * 2.0 / 3)
        self.p.log(
            r"We will prune 67% filters in " + str(iterations)+ "iterations")
        # Make sure all the layers are trainable
        # self.set_grad_requirment(True)
        for ii in range(iterations):
            self.p.log("#"*80)
            self.p.log("Prune iteration %d: " % ii)
            self.p.log("Ranking filters.. ")
            start = time.time()
            # update model to the prunner
            self.reload_model()
            self.prunner = FilterPrunner(self.model, self.ctx)
            self.get_cuda_memory()
            self.model.get_feature=False
            if ii+1:
                prune_targets = self.get_candidates_to_prune(
                    num_filters_to_prune_per_iteration)
            # if ii == 0:
            #     with open(os.path.join(self.log_dir,"prune_target01.pkl"),"wb") as f:
            #         pickle.dump(prune_targets,f)
            else:
                with open(os.path.join(self.log_dir,"prune_target01.pkl"),"rb") as f:
                    prune_targets = pickle.load(f)
            self.model.get_feature = True
            self.get_cuda_memory()
            layers_prunned = {}
            for layer_index, filter_index in prune_targets.items():
                layers_prunned[layer_index] = len(filter_index)
            self.p.log("Ranking filter use time %.2fs" % (time.time()-start))
            self.p.log("Layers that will be prunned :"+str(layers_prunned))
            start = time.time()
            self.p.log(prune_targets)
            self.prunner.reset()
            self.p.log("Prunning filters.. ")
            self.model = prune_spherenet20_conv_once(self.model, prune_targets, ctx)
            cur_acc = self.test()
            self.reload_model()
            self.prunner.reset()
            self.p.log(self.model)
            self.p.log("Pruning filter use time %.2fs" % (time.time()-start))
            message = "%.2f%s" % (
                100*float(self.total_num_filters()) / number_of_filters, "%")
            self.p.log("Filters left"+str(message))
            self.get_cuda_memory()

            self.p.log("#"*80)
            self.p.log("Fine tuning to recover from prunning iteration.")
            optimizer = mx.optimizer.Adam(0.0001)
            trainer = gluon.Trainer(self.model.collect_params(),optimizer)
            self.train(trainer, epoches=5, best_acc = cur_acc)
            cur_acc = self.eval()

        self.p.log("#"*80)
        self.p.log("Finished. Going to fine tune the model a bit more")
        self.reload_model()
        optimizer = mx.optimizer.Adam(0.0001)
        trainer = gluon.Trainer(self.model.classifier.collect_params(),optimizer)
        self.train(trainer, epoches=10, best_acc=cur_acc)
        self.eval()
        self.model.save_params(os.path.join(self.log_dir, "model_pruned"))
        return self.model

class myThread(threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

    def run(self, sbatch, slabel):
        pass

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prune", dest="prune", action="store_true")
    parser.add_argument("--no-log", dest="log", action="store_false")
    parser.add_argument("--train_path", type=str, default="/home1/CASIA-WebFace/aligned_Webface-112X96")
    parser.add_argument("--model_path", type=str,
                        default="./log/train-2018-07-02_091330/model")
    parser.add_argument("--device_id", type=int, default=5)
    parser.set_defaults(prune=True)
    parser.set_defaults(log=False)
    args = parser.parse_args()
    return args

class Printer():
    def __init__(self, log_dir, log=True):
        self.log_dir = log_dir
        self.do_log = log

    def log(self, mstr):
        print(mstr)
        if self.do_log:
            log_path = os.path.join(self.log_dir, "log.txt")
            if not os.path.exists(log_path):
                os.system("mkdir "+self.log_dir)
                os.system("touch "+log_path)
            with open(os.path.join(self.log_dir, "log.txt"), "a") as f:
                f.write(str(mstr)+"\n")

if __name__ == '__main__':
    args = get_args()

    # gpus = [0,1]
    # ctx = [mx.gpu(ii) for ii in gpus]
    ctx = mx.gpu(5)
    time_info = time.strftime('%Y-%m-%d_%H%M%S', time.localtime(time.time()))
    log_dir = os.path.abspath("./log/")
    if not args.prune:
        model = models.SphereNet20()
        model.load_params('spherenet_model', ctx=ctx)
        if args.log: log_dir = os.path.abspath("./log/train-"+time_info+"/")
    else:
        model = models.SphereNet20()
        model.load_params(args.model_path, ctx=ctx)
        if args.log: log_dir = os.path.abspath("./log/prune-"+time_info+"/")
    p = Printer(log_dir, args.log)
    p.log(ctx)
    msg = "doing fine tuning(train)" if not args.prune else "doing pruning, using model " + \
        args.model_path
    p.log(msg)
    p.log(model)

    p.log("time is :"+time_info)
    fine_tuner = PrunningFineTuner_VGG16(args.train_path, model, log_dir, ctx)
    if not args.prune:
        p.log("begin training...")
        fine_tuner.train(epoches=10)
        if args.log:
            os.system("cp finetune.py "+log_dir)
        # torch.save(model, log_dir+"model")
    else:
        model = fine_tuner.prune()
        p.log(model)
