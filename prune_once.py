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

def prune_spherenet20_conv_once(model, prune_targets,ctx):

    new_feature = nn.HybridSequential()

    last_index = 0
    for name, module in model.features._children.items():
        if isinstance(module, nn.Dense):
            old_channels = model.features._children['7'].conv2.conv._channels
            old_weight = module.weight._data[0] # [out_channel, in_channel]
            params_per_input_channel = old_weight.shape[1] / old_channels
            filter_index = [ii for ii in range(old_channels) if ii not in prune_targets[last_index-1]]
            new_Dense = \
                nn.Dense(module._units, in_units=params_per_input_channel * len(filter_index))
            new_weight_index = [jj for ii in filter_index for jj in range(ii * params_per_input_channel,
                                                                          (ii + 1) * params_per_input_channel)]
            new_weights = \
                old_weight[:, new_weight_index]
            new_bias = module.bias._data[0]
            new_Dense.initialize(init=myInitializer(new_weights), ctx=ctx)
            new_Dense.bias.initialize(init=mx.init.Constant(new_bias),force_reinit=True, ctx=ctx)
            new_feature.add(new_Dense)
        else:
            new_block, last_index, prune_targets = prune_resblock(module,prune_targets,last_index,ctx)
            new_feature.add(new_block)

    model.features = new_feature
    print prune_targets
    return model

def prune_resblock(block,prune_plan,start_index,ctx):
    '''
    1) if the prune channel number of x and f(x) is different, make them to be the same
    2) ignore the mismatch of channels between two feature maps
    :param block:
    :param prune_plan:
    :param start_index:
    :param ctx:
    :return:
    '''
    new_block = models.Residual(same_shape=block.same_shape)
    last_prune_target = prune_plan[start_index - 1] if start_index else []
    prune_targets = [last_prune_target]
    for ii in range(len(block._children)//2):
        prune_targets.append(prune_plan[start_index+ii])
    if not block.same_shape: # choose the smallest prune size between the 1st and last conv
        if len(prune_targets[1]) >= len(prune_targets[-1]):
            prune_targets[1] = prune_targets[1][:len(prune_targets[-1])]
        else:
            prune_targets[-1] = prune_targets[-1][:len(prune_targets[1])]
    else:
        if len(prune_targets[0]) >= len(prune_targets[-1]):
            for target in prune_targets[0]:
                if len(prune_targets[0]) == len(prune_targets[-1]): break
                if target not in prune_targets[-1]: prune_targets[-1].append(target)
        else:
            prune_targets[-1] = prune_targets[-1][:len(prune_targets[0])]
    # prune
    i_conv = 0
    i_a = 0
    for name, layer in block._children.items():
        if isinstance(layer, gradcam.Conv2D):
            i_conv += 1
            prune_target = prune_targets[i_conv]
            new_layer = get_new_conv_layer(layer, prune_target, last_prune_target, ctx)
            last_prune_target = prune_target
        else:
            i_a += 1
            prune_target = prune_targets[i_a]
            new_layer = get_new_prelu_layer(layer,prune_target,ctx)
        new_block._children[name] = new_layer
        setattr(new_block,name,new_layer)

    # update prune_plan
    for ii in range(len(block._children)//2):
        prune_plan[start_index+ii] = sorted(prune_targets[ii+1])
    return new_block, start_index+len(block._children)//2, prune_plan

def get_new_conv_layer(layer, prune_target, last_prune_target,ctx):
    new_layer = gradcam.Conv2D(channels=layer.conv._kwargs["num_filter"] - len(prune_target),
                               kernel_size=layer.conv._kwargs["kernel"],
                               strides=layer.conv._kwargs["stride"],
                               padding=layer.conv._kwargs["pad"],
                               dilation=layer.conv._kwargs["dilate"],
                               groups=layer.conv._kwargs["num_group"])
    old_weight = layer.conv.weight._data[0]

    filter_index = [ii for ii in range(old_weight.shape[1]) if ii not in last_prune_target]
    old_weight = old_weight[:, filter_index, :, :]

    filter_index = [ii for ii in range(old_weight.shape[0]) if ii not in prune_target]
    new_weight = old_weight[filter_index, :, :, :]
    new_bias = layer.conv.bias._data[0][filter_index]
    new_layer.conv.initialize(init=myInitializer(new_weight, new_bias), ctx=ctx)
    try:
        new_layer.conv.bias.initialize(init=mx.init.Constant(new_bias),
                                        force_reinit=True, ctx=ctx)
    except Exception:
        print new_layer
    return new_layer

def get_new_prelu_layer(layer,prune_target,ctx):
    old_alpha = layer.alpha._data[0]
    filter_index = [ii for ii in range(old_alpha.shape[1]) if ii not in prune_target]
    new_layer = models.mPReLU(num_units=len(filter_index))
    new_alpha = old_alpha[:,filter_index,:,:]
    new_layer.alpha.initialize(init=mx.init.Constant(new_alpha),ctx=ctx)
    return new_layer

if __name__ == '__main__':
    ctx = mx.gpu()
    model = ModifiedVGG16Model(ctx=ctx)
    model.load_params("./log/train-2018-06-14_131152/model", ctx=ctx)

    t0 = time.time()
    prune_targets = {10: [93], 17: [28, 88, 187, 238], 19: [196, 463], 21: [10, 22, 39, 55, 56, 66, 97, 105, 187, 203, 212, 215, 224, 272, 284, 295, 351, 355, 358, 387, 411, 420, 422, 425, 468, 476, 486], 24: [29, 57, 112, 137, 171, 227, 233, 236, 252, 261, 332, 365, 373, 388, 404, 411, 456, 479, 509], 26: [1, 2, 3, 9, 25, 28, 30, 35, 38, 40, 43, 44, 47, 48, 55, 58, 62, 64, 68, 71, 73, 76, 80, 88, 92, 93, 102, 104, 113, 120, 124, 128, 137, 140, 149, 154, 159, 162, 165, 167, 170, 172, 184, 186, 187, 191, 193, 199, 200, 205, 206, 207, 211, 217, 219, 222, 225, 226, 232, 233, 235, 238, 242, 243, 247, 256, 258, 261, 275, 280, 281, 283, 287, 291, 293, 302, 309, 316, 317, 318, 319, 324, 335, 336, 338, 340, 342, 343, 347, 349, 352, 356, 360, 366, 372, 381, 382, 392, 393, 394, 395, 402, 403, 404, 413, 414, 415, 416, 417, 419, 437, 438, 445, 449, 452, 460, 463, 470, 471, 472, 473, 478, 480, 483, 486, 496, 497, 503, 504, 505], 28: [0, 1, 2, 4, 5, 6, 7, 8, 9, 14, 15, 16, 21, 25, 26, 28, 29, 31, 34, 40, 41, 42, 43, 44, 45, 46, 49, 50, 51, 54, 56, 57, 58, 59, 60, 61, 64, 65, 67, 68, 69, 70, 72, 74, 75, 76, 77, 79, 80, 83, 85, 86, 88, 89, 90, 91, 92, 93, 95, 96, 97, 98, 100, 101, 102, 103, 104, 106, 107, 108, 109, 110, 111, 113, 116, 118, 119, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 133, 137, 138, 144, 145, 146, 147, 148, 150, 151, 152, 153, 154, 155, 156, 159, 161, 163, 164, 166, 167, 171, 172, 173, 178, 179, 180, 181, 183, 184, 185, 187, 190, 196, 197, 198, 199, 200, 201, 202, 204, 205, 206, 208, 209, 210, 213, 214, 217, 218, 220, 221, 222, 224, 225, 226, 227, 228, 229, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 247, 249, 251, 254, 256, 258, 259, 260, 262, 263, 265, 266, 267, 268, 269, 270, 273, 276, 277, 279, 280, 282, 283, 287, 288, 289, 292, 294, 296, 297, 299, 301, 302, 303, 304, 305, 308, 309, 310, 312, 313, 314, 315, 318, 323, 324, 325, 326, 327, 328, 329, 331, 333, 335, 338, 340, 341, 342, 343, 344, 345, 346, 347, 350, 352, 354, 357, 359, 362, 364, 365, 366, 367, 368, 370, 373, 374, 376, 377, 378, 380, 381, 383, 384, 385, 386, 387, 390, 392, 398, 399, 401, 402, 403, 404, 405, 406, 408, 409, 411, 412, 414, 415, 421, 423, 424, 425, 426, 428, 429, 430, 433, 434, 435, 436, 437, 438, 439, 440, 442, 443, 444, 445, 446, 447, 448, 449, 450, 455, 456, 458, 459, 461, 462, 463, 464, 465, 470, 471, 472, 473, 475, 476, 477, 478, 480, 481, 482, 483, 484, 486, 488, 490, 491, 494, 495, 496, 498, 499, 500, 501, 504, 505, 506, 508, 509, 511]}
    model = prune_spherenet20_conv_once(model, prune_targets,ctx)
    print "The prunning took", time.time() - t0
    t0 = time.time
