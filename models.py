import mxnet.gluon.nn as nn
import mxnet.gluon as gluon
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

class ModifiedVGG16Model(gluon.Block):
    def __init__(self, ctx=mx.cpu(), pretrained=False, prune=True):
        super(ModifiedVGG16Model, self).__init__()
        self.prune = prune
        self.conv_output = {}

        model = vision.vgg16(pretrained=pretrained, ctx=ctx)
        self.features = nn.Sequential()

        for ii, layer in enumerate(model.features):

            if isinstance(layer, nn.Conv2D):
                fsize = layer.weight.shape[2]
                new_layer = gradcam.Conv2D(layer._channels,(fsize,fsize))
                if pretrained:
                    # new_layer.conv = layer
                    new_layer.initialize(init=myInitializer(layer.weight._data[0],
                                                            layer.bias._data[0]),ctx=ctx)
                self.features.add(new_layer)
            elif ii<31: self.features.add(layer)

        # for param in self.features.parameters():
        #     param.requires_grad = False
        self.classifier = nn.Sequential()
        self.classifier.add(nn.Dropout(0.5))
        self.classifier.add(nn.Dense(4096,activation="relu"))
        self.classifier.add(nn.Dropout(0.5))
        self.classifier.add(nn.Dense(4096,activation="relu"))
        self.classifier.add(nn.LeakyReLU(0))
        self.classifier.add(nn.Dense(2))
        if pretrained:
            self.classifier.initialize(mx.init.Xavier(), ctx=ctx)


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def set_prune(self, prune):
        self.prune = prune



class AngleLinear(nn.Block):
    def __init__(self, units, in_units=0,
                 m = 4, phiflag=True, **kwargs):
        super(AngleLinear, self).__init__(**kwargs)
        self.out_features = units
        weight = nd.L2Normalization(nd.random.uniform(-1,1,shape=(units,in_units)))# the weight need to be specially initialize
        self.weight = self.params.get('weight',shape=(units,in_units),
                                      init=mx.init.Constant(weight),allow_deferred_init=True)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [ # m-order taylor expansion
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight._data[0] # size=(Classnum,F) F=in_features Classnum=out_features

        ww = nd.L2Normalization(w)
        xlen = x.square().sum(1,keepdims=True).sqrt() # size=B
        wlen = ww.square().sum(1,keepdims=True).sqrt() # size=Classnum

        cos_theta = nd.dot(x,ww.T) # size=(B,Classnum)
        cos_theta = cos_theta / xlen.reshape(-1,1) / wlen.reshape(1,-1)
        cos_theta = cos_theta.clip(-1,1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = nd.arccos(cos_theta)
            k = (self.m*theta/math.pi).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k
        else:
            theta = cos_theta.acos()
            phi_theta = self.myphi(theta,self.m)
            phi_theta = phi_theta.clip(-1*self.m,1)

        cos_theta = cos_theta * xlen.reshape(-1,1)
        phi_theta = phi_theta * xlen.reshape(-1,1)
        output = (cos_theta,phi_theta)
        return output # size=(B,Classnum,2)

    def myphi(self, x, m):
        x = x * m
        return 1 - x ** 2 / math.factorial(2) + x ** 4 / math.factorial(4) - x ** 6 / math.factorial(6) + \
               x ** 8 / math.factorial(8) - x ** 9 / math.factorial(9)

class AngleLoss(nn.Block):
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma   = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.byte()
        index = Variable(index)

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        logpt = F.log_softmax(output)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss

class mPReLU(nn.HybridBlock):
    '''
    modify
     the official prelu
    '''
    def __init__(self, num_units,initializer=None, **kwargs):
        super(mPReLU, self).__init__(**kwargs)
        self.num_units = num_units
        with self.name_scope():
            self.alpha = self.params.get('alpha', shape=(1,num_units,1,1), init=initializer,
                                         grad_req='write')

    def hybrid_forward(self, F, x, alpha):
        return mx.nd.maximum(x,0)+ mx.nd.minimum(alpha*x,0)

class Residual(nn.Block):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        if not same_shape:
            self.conv0 = gradcam.Conv2D(channels, kernel_size=3,
                                    padding=1, strides=2)
            self.a0 = mPReLU(channels)
        self.conv1 = gradcam.Conv2D(channels, kernel_size=3,
                                    padding=1)
        self.a1 = mPReLU(channels)
        self.conv2 = gradcam.Conv2D(channels, kernel_size=3,
                                    padding=1)
        self.a2 = mPReLU(channels)

    def forward(self, x):
        if not self.same_shape:
            x = self.a0(self.conv0(x))
        out = self.a1(self.conv1(x))
        out = self.a2(self.conv2(out))
        return out + x


class SphereNet20(nn.Block):
    # http://ethereon.github.io/netscope/#/gist/20f6ddf70a35dec5019a539a502bccc5
    def __init__(self, num_classes=10574, verbose=False, **kwargs):
        super(SphereNet20, self).__init__(**kwargs)
        self.verbose = verbose
        self.feature = True
        # add name_scope on the outermost Sequential
        with self.name_scope():
            # block 1
            self.net = nn.Sequential()
            b1 = Residual(64, same_shape=False)

            # block 2
            b2_1 = Residual(128, same_shape=False)
            b2_2 = Residual(128)

            # block3
            b3_1 = Residual(256, same_shape=False)
            b3_2 = Residual(256)
            b3_3 = Residual(256)
            b3_4 = Residual(256)

            # block 4
            b4 = Residual(512, same_shape=False)
            f5 = nn.Dense(512)
            f6 = AngleLinear(in_units=512,units=num_classes)
            self.net.add(b1,b2_1,b2_2,b3_1,b3_2,b3_3,b3_4,b4,f5,f6)

    def forward(self, x):
        out = x
        for i, b in enumerate(self.net):
            if self.feature and i==9: break
            out = b(out)
            if self.verbose:
                print('Block %d output: %s'%(i+1, out.shape))
        return out
    def initialize_(self,ctx,*args):
        with open("/home/hfq/model_compress/sphereface_model/caffemodel.pkl","rb") as f:
            params = pickle.load(f)
        same_layer = map(str,[0,1,3,7])
        block_index = 0
        res_index = 1
        for name, resblock in self.net._children.items():
            if name == "8":
                resblock.initialize(init=myInitializer(params['fc5.weight']),ctx=ctx)
                resblock.bias.initialize(init=mx.init.Constant(params['fc5.bias']),force_reinit=True,ctx=ctx)
            elif name == "9":
                resblock.initialize(ctx=ctx)
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
                print 'conv%d_%d.weight'%(block_index,res_index)
                res_index += 1
                resblock.conv2.initialize(init=myInitializer(params['conv%d_%d.weight'%(block_index,res_index)],
                                                             params['conv%d_%d.bias' % (block_index, res_index)]),ctx=ctx)
                resblock.a2.initialize(init=mx.init.Constant(params['relu%d_%d' % (block_index, res_index)][0].reshape([1,-1,1,1])),
                                       force_reinit=True,  ctx=ctx)
                print 'conv%d_%d.weight'%(block_index,res_index)
                res_index += 1
        # print self.net

def init_model():
    mnet = SphereNet20()
    mnet.initialize_(mx.gpu())
    train_data_loader, valid_data_loader, test_data_loader \
        = dataset.train_valid_test_loader("../pytorch-pruning/train", batch_size=16)
    mnet.feature=False
    for batch, label in valid_data_loader:
        batch = batch.as_in_context(mx.gpu())
        out = mnet(batch)
        break
    mnet.save_params("./spherenet_model")
    print
    return mnet

def load_model():
    mnet = SphereNet20(100)
    mnet.load_params("./spherenet_model")
    return mnet


if __name__ == "__main__":
    mnet=init_model()
    print(mnet)
    print(mnet)