import mxnet.gluon.nn as nn
import mxnet.gluon as gluon
from mxnet.gluon import HybridBlock
from mxnet import nd
from mxnet.gluon.model_zoo import vision
import mxnet as mx
import gradcam, pickle, dataset
import math

@mx.init.register
class myInitializer(mx.init.Initializer):
    def __init__(self, weight,bias=None):
        super(myInitializer,self).__init__()
        self.weight = weight
        self.bias = bias

    def _init_weight(self, _, arr):
        arr[:] = self.weight

    def _init_bias(self, _, arr):
        arr[:] = self.bias

class AngleLinear(HybridBlock):
    def __init__(self, units, in_units=0, initializer=None,
                 m = 4, phiflag=True, **kwargs):
        super(AngleLinear, self).__init__(**kwargs)
        self.out_features = units
        self.weight = self.params.get('weight',shape=(units,in_units),
                                      init=initializer,allow_deferred_init=True)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [ # given cos_x, calculate the cos_mx
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def hybrid_forward(self, F, x, weight, *args, **params):
        # xsize=(B,F)    F is feature len
        # w = self.weight._data[0] # size=(Classnum,F) F=in_features Classnum=out_features
        ww = F.L2Normalization(weight)
        xlen = x.square().sum(axis=1,keepdims=True).sqrt() # size=B
        wlen = ww.square().sum(axis=1,keepdims=True).sqrt() # size=Classnum

        cos_theta = F.broadcast_div(F.broadcast_div(F.dot(x,ww.transpose()), xlen.reshape((-1,1))),
                                    wlen.reshape((1,-1)).clip(-1,1)) # size=(B,Classnum)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = F.arccos(cos_theta)
            k = (self.m * theta / math.pi).floor()
            n_one = k*0.0 - 1
            phi_theta = F.broadcast_mul((n_one**k), cos_m_theta) - 2*k
        else:
            theta = cos_theta.acos()
            phi_theta = self.myphi(theta,self.m)
            phi_theta = phi_theta.clip(-1*self.m,1)

        xcos_theta = F.broadcast_mul(cos_theta, xlen.reshape((-1,1)))
        xphi_theta = F.broadcast_mul(phi_theta, xlen.reshape((-1,1)))
        output = (xcos_theta,xphi_theta)
        return output # size=(B,Classnum,2)

    def myphi(self, x, m):
        x = x * m
        return 1 - x ** 2 / math.factorial(2) + x ** 4 / math.factorial(4) - x ** 6 / math.factorial(6) + \
               x ** 8 / math.factorial(8) - x ** 9 / math.factorial(9)

class AngleLoss(gluon.loss.Loss):
    def __init__(self, weight=1, batch_axis=0, gamma=0, **kwargs):
        super(AngleLoss, self).__init__(weight, batch_axis, **kwargs)
        self.gamma   = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def hybrid_forward(self, F, xcos_theta,xphi_theta,target):
        self.it += 1

        batch_size = target.size# size = (B,classnum)
        oh_target = target.one_hot(xcos_theta.shape[1])

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))
        # because indexing is not differentiable in mxnet, we must do this
        output =  xcos_theta - oh_target * xcos_theta[range(0,batch_size),target].reshape(-1,1)*(1.0+0)/(1+self.lamb) +\
                                oh_target * xphi_theta[range(0,batch_size),target].reshape(-1,1)*(1.0+0)/(1+self.lamb)

        loss = F.softmax_cross_entropy(output, F.cast(target,'float32')) # (B,Classnum)
        return loss

class mPReLU(HybridBlock):
    '''
    modify
     the official prelu
    '''
    def __init__(self, num_units,initializer=None, **kwargs):
        super(mPReLU, self).__init__(**kwargs)
        self.num_units = num_units
        with self.name_scope():
            self.alpha = self.params.get('alpha', shape=(1,num_units,1,1), init=initializer)

    def hybrid_forward(self, F, x, alpha):
        return F.maximum(x,0) + F.minimum(F.broadcast_mul(alpha,x),0)

class Residual(HybridBlock):
    def __init__(self, channels=(64,64), same_shape=True, use_custom_relu=True, **kwargs):
        '''
        :param channels: (conv1channel, conv2channel) for conv0channel equals conv2channel
        :param same_shape:
        :param kwargs:
        '''
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        if not same_shape:
            self.conv0 = gradcam.Conv2D(channels[1], kernel_size=3,# if no need for activations' gradient: change this to nn.Conv2D
                                    padding=1, strides=2)
            self.a0 = mPReLU(channels[1]) if use_custom_relu else nn.LeakyReLU(0.1)
        self.conv1 = gradcam.Conv2D(channels[0], kernel_size=3,
                                    padding=1)
        self.a1 = mPReLU(channels[0]) if use_custom_relu else nn.LeakyReLU(0.1)
        self.conv2 = gradcam.Conv2D(channels[1], kernel_size=3,
                                    padding=1)
        self.a2 = mPReLU(channels[1]) if use_custom_relu else nn.LeakyReLU(0.1)

    def hybrid_forward(self, F, x, *args, **kwargs):
        if not self.same_shape:
            x = self.a0(self.conv0(x))
        out = self.a1(self.conv1(x))
        out = self.a2(self.conv2(out))
        return out + x

class SphereNet20(HybridBlock):
    # http://ethereon.github.io/netscope/#/gist/20f6ddf70a35dec5019a539a502bccc5
    default_params = {
        0: [64]*2,
        1: [128]*2,
        2: [128]*2,
        3: [256]*2,
        4: [256]*2,
        5: [256]*2,
        6: [256]*2,
        7: [512]*2
    }
    def __init__(self, num_classes=10574,archi_dict=None, verbose=False,
                 use_custom_relu=False, **kwargs):
        super(SphereNet20, self).__init__(**kwargs)
        self.verbose = verbose
        self.num_classes=num_classes
        self.use_custom_relu=use_custom_relu
        self.get_feature = True
        self.archi_dict = archi_dict if archi_dict else self.default_params
        # add name_scope on the outermost Sequential
        with self.name_scope():
            # block 1
            self.features = nn.HybridSequential()
            b1 = Residual(self.archi_dict[0], same_shape=False,
                          use_custom_relu=use_custom_relu)

            # block 2
            b2_1 = Residual(self.archi_dict[1], same_shape=False,
                          use_custom_relu=use_custom_relu)
            b2_2 = Residual(self.archi_dict[2],
                          use_custom_relu=use_custom_relu)

            # block3
            b3_1 = Residual(self.archi_dict[3], same_shape=False,
                          use_custom_relu=use_custom_relu)
            b3_2 = Residual(self.archi_dict[4],
                          use_custom_relu=use_custom_relu)
            b3_3 = Residual(self.archi_dict[5],
                          use_custom_relu=use_custom_relu)
            b3_4 = Residual(self.archi_dict[6],
                          use_custom_relu=use_custom_relu)

            # block 4
            b4 = Residual(self.archi_dict[7], same_shape=False,
                          use_custom_relu=use_custom_relu)
            f5 = nn.Dense(512)
            self.features.add(b1,b2_1,b2_2,b3_1,b3_2,b3_3,b3_4,b4,f5)

            f6 = AngleLinear(in_units=512, units=num_classes)
            self.classifier = f6

    def hybrid_forward(self, F,x,  *args, **kwargs):
        out = x
        for i, b in enumerate(self.features):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s'%(i+1, out.shape))
        if not self.get_feature:
            out = self.classifier(out)
        return out

    def initialize_from(self,pkl_path,ctx,*args):
        with open(pkl_path,"rb") as f:
            params = pickle.load(f)
        same_layer = map(str,[0,1,3,7])
        block_index = 0
        res_index = 1
        for name, resblock in self.features._children.items():
            if name == "8":
                resblock.initialize(init=myInitializer(params['fc5.weight']),ctx=ctx)
                resblock.bias.initialize(init=mx.init.Constant(params['fc5.bias']),force_reinit=True,ctx=ctx)
            else:
                if name in same_layer:
                    block_index += 1
                    res_index = 1
                    resblock.conv0.initialize(init=myInitializer(params['conv%d_%d.weight'%(block_index,res_index)]),ctx=ctx)
                    resblock.conv0.conv.bias.initialize(init=mx.init.Constant(params['conv%d_%d.bias'%(block_index,res_index)]),
                                                                     force_reinit=True,ctx=ctx)
                    print 'conv%d_%d.weight'%(block_index,res_index)
                    resblock.a0.initialize(init=mx.init.Constant(params['relu%d_%d'%(block_index,res_index)][0].reshape([1,-1,1,1])),
                                    force_reinit=True, ctx=ctx)
                    res_index += 1
                resblock.conv1.initialize(init=myInitializer(params['conv%d_%d.weight'%(block_index,res_index)],
                                                             params['conv%d_%d.bias'%(block_index,res_index)]),ctx=ctx)
                resblock.a1.initialize(init=mx.init.Constant(params['relu%d_%d' % (block_index, res_index)][0].reshape([1,-1,1,1])),
                                       force_reinit=True, ctx=ctx)
                res_index += 1
                resblock.conv2.initialize(init=myInitializer(params['conv%d_%d.weight'%(block_index,res_index)],
                                                             params['conv%d_%d.bias' % (block_index, res_index)]),ctx=ctx)
                resblock.a2.initialize(init=mx.init.Constant(params['relu%d_%d' % (block_index, res_index)][0].reshape([1,-1,1,1])),
                                       force_reinit=True,  ctx=ctx)
                res_index += 1

        weight = nd.L2Normalization(
            nd.random.uniform(-1, 1, shape=(self.num_classes, 512)))  # the weight need to be specially initialize
        self.classifier.initialize(init=mx.init.Constant(weight),ctx=ctx)

def init_model(pkl_path):
    mnet = SphereNet20(use_custom_relu=False)
    mnet.initialize_from(pkl_path, mx.gpu())
    train_data_loader, valid_data_loader, test_data_loader \
        = dataset.train_valid_test_loader("../pytorch-pruning/train", batch_size=16)
    mnet.feature=False
    for batch, label in valid_data_loader:
        batch = batch.as_in_context(mx.gpu())
        label = label.as_in_context(mx.gpu())
        mnet.get_feature=False
        out = mnet(batch)
        criterion = AngleLoss()
        loss = criterion(out[0],out[1],label)
        break
    mnet.save_params("./spherenet_model2")
    return mnet

def load_model():
    mnet = SphereNet20()
    mnet.load_params("./spherenet_model")
    return mnet


if __name__ == "__main__":
    pkl_path = "/home/hfq/model_compress/sphereface_model/caffemodel.pkl"
    mnet=init_model(pkl_path) #
    # mnet = SphereNet20()
    # mnet.load_params('spherenet_model',ctx=mx.gpu())
    print(mnet)
