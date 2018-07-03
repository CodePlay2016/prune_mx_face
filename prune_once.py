import mxnet as mx
from mxnet.gluon.model_zoo import vision
import mxnet.gluon.nn as nn
import time
import cv2
import sys
import gradcam
import numpy as np
from models import *
from finetune import *

def replace_layers(model, index, layer):
    new_model = mx.gluon.nn.HybridSequential()
    for ii,_ in enumerate(model):
        if ii == index:
            new_model.add(layer)
        else:
            new_model.add(model[ii])
    return new_model

def prune_spherenet20_conv_once(model, prune_targets,ctx):

    new_feature = nn.Sequential()
    res_heads = [0,3,8,17]

    last_conv_pruned = False
    prune_target = None
    index = 0
    for _, module in model.features._children.items():
        if isinstance(module, nn.Dense): continue
        for _, layer in module._children.items():
            if index in prune_targets.keys():
                new_layer = gradcam.Conv2D(channels=layer.conv._kwargs["num_filter"]-len(prune_targets[index]),
                                           kernel_size=layer.conv._kwargs["kernel"],
                                           strides=layer.conv._kwargs["stride"],
                                           padding=layer.conv._kwargs["pad"],
                                           dilation=layer.conv._kwargs["dilate"],
                                           groups=layer.conv._kwargs["num_group"])
                old_weight = layer.conv.weight._data[0]

                if last_conv_pruned:
                    filter_index = [ii for ii in range(old_weight.shape[1]) if ii not in prune_target]
                    old_weight = old_weight[:,filter_index,:,:]

                prune_target = prune_targets[index]
                filter_index = [ii for ii in range(old_weight.shape[0]) if ii not in prune_target]
                new_weight = old_weight[filter_index,:,:,:]
                new_bias   = layer.conv.bias._data[0][filter_index]
                new_layer.conv.initialize(init=myInitializer(new_weight, new_bias), ctx=ctx)
                last_conv_pruned = True
                if index == 28: # if the last conv layer
                    first_dense = model.classifier._children["1"]
                    old_weight = first_dense.weight._data[0]
                    params_per_input_channel = old_weight.shape[1] / layer.conv._channels
                    new_Dense = \
                        nn.Dense(first_dense._units, in_units=params_per_input_channel * len(filter_index))
                    new_weight_index = [jj for ii in filter_index for jj in range(ii * params_per_input_channel,
                                                                                  (ii + 1) * params_per_input_channel)]
                    new_weights = \
                        old_weight[:, new_weight_index]
                    new_bias = first_dense.bias._data[0]
                    new_Dense.initialize(init=myInitializer(new_weights, new_bias), ctx=ctx)
            elif last_conv_pruned and isinstance(layer, gradcam.Conv2D):
                new_layer = gradcam.Conv2D(channels=layer.conv._kwargs["num_filter"],
                                           kernel_size=layer.conv._kwargs["kernel"],
                                           strides=layer.conv._kwargs["stride"],
                                           padding=layer.conv._kwargs["pad"],
                                           dilation=layer.conv._kwargs["dilate"],
                                           groups=layer.conv._kwargs["num_group"])
                old_weight = layer.conv.weight._data[0]
                filter_index = [ii for ii in range(old_weight.shape[1]) if ii not in prune_target]
                new_weight = old_weight[:, filter_index, :, :]
                new_bias = layer.conv.bias._data[0]
                last_conv_pruned = False
                new_layer.conv.initialize(init=myInitializer(new_weight, new_bias), ctx=ctx)
            else:
                new_layer = layer
            new_feature.add(new_layer)
            index += 1

    if last_conv_pruned:
        new_classifier = nn.Sequential()
        for index, layer in model.classifier._children.items():
            if int(index) == 1:
                new_classifier.add(new_Dense)
            else:
                new_classifier.add(layer)
        model.classifier = new_classifier
    # del model.features
    # del model.classifier
    model.features = new_feature
    return model

def prune_resblock(block,prune_mode,last_layer_pruned):
    pass

if __name__ == '__main__':
    ctx = mx.gpu()
    model = ModifiedVGG16Model(ctx=ctx)
    model.load_params("./log/train-2018-06-14_131152/model", ctx=ctx)

    t0 = time.time()
    prune_targets = {10: [93], 17: [28, 88, 187, 238], 19: [196, 463], 21: [10, 22, 39, 55, 56, 66, 97, 105, 187, 203, 212, 215, 224, 272, 284, 295, 351, 355, 358, 387, 411, 420, 422, 425, 468, 476, 486], 24: [29, 57, 112, 137, 171, 227, 233, 236, 252, 261, 332, 365, 373, 388, 404, 411, 456, 479, 509], 26: [1, 2, 3, 9, 25, 28, 30, 35, 38, 40, 43, 44, 47, 48, 55, 58, 62, 64, 68, 71, 73, 76, 80, 88, 92, 93, 102, 104, 113, 120, 124, 128, 137, 140, 149, 154, 159, 162, 165, 167, 170, 172, 184, 186, 187, 191, 193, 199, 200, 205, 206, 207, 211, 217, 219, 222, 225, 226, 232, 233, 235, 238, 242, 243, 247, 256, 258, 261, 275, 280, 281, 283, 287, 291, 293, 302, 309, 316, 317, 318, 319, 324, 335, 336, 338, 340, 342, 343, 347, 349, 352, 356, 360, 366, 372, 381, 382, 392, 393, 394, 395, 402, 403, 404, 413, 414, 415, 416, 417, 419, 437, 438, 445, 449, 452, 460, 463, 470, 471, 472, 473, 478, 480, 483, 486, 496, 497, 503, 504, 505], 28: [0, 1, 2, 4, 5, 6, 7, 8, 9, 14, 15, 16, 21, 25, 26, 28, 29, 31, 34, 40, 41, 42, 43, 44, 45, 46, 49, 50, 51, 54, 56, 57, 58, 59, 60, 61, 64, 65, 67, 68, 69, 70, 72, 74, 75, 76, 77, 79, 80, 83, 85, 86, 88, 89, 90, 91, 92, 93, 95, 96, 97, 98, 100, 101, 102, 103, 104, 106, 107, 108, 109, 110, 111, 113, 116, 118, 119, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 133, 137, 138, 144, 145, 146, 147, 148, 150, 151, 152, 153, 154, 155, 156, 159, 161, 163, 164, 166, 167, 171, 172, 173, 178, 179, 180, 181, 183, 184, 185, 187, 190, 196, 197, 198, 199, 200, 201, 202, 204, 205, 206, 208, 209, 210, 213, 214, 217, 218, 220, 221, 222, 224, 225, 226, 227, 228, 229, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 247, 249, 251, 254, 256, 258, 259, 260, 262, 263, 265, 266, 267, 268, 269, 270, 273, 276, 277, 279, 280, 282, 283, 287, 288, 289, 292, 294, 296, 297, 299, 301, 302, 303, 304, 305, 308, 309, 310, 312, 313, 314, 315, 318, 323, 324, 325, 326, 327, 328, 329, 331, 333, 335, 338, 340, 341, 342, 343, 344, 345, 346, 347, 350, 352, 354, 357, 359, 362, 364, 365, 366, 367, 368, 370, 373, 374, 376, 377, 378, 380, 381, 383, 384, 385, 386, 387, 390, 392, 398, 399, 401, 402, 403, 404, 405, 406, 408, 409, 411, 412, 414, 415, 421, 423, 424, 425, 426, 428, 429, 430, 433, 434, 435, 436, 437, 438, 439, 440, 442, 443, 444, 445, 446, 447, 448, 449, 450, 455, 456, 458, 459, 461, 462, 463, 464, 465, 470, 471, 472, 473, 475, 476, 477, 478, 480, 481, 482, 483, 484, 486, 488, 490, 491, 494, 495, 496, 498, 499, 500, 501, 504, 505, 506, 508, 509, 511]}
    model = prune_spherenet20_conv_once(model, prune_targets,ctx)
    print "The prunning took", time.time() - t0
    t0 = time.time
