import mxnet as mx
import mxnet.gluon as gluon
from mxnet.gluon import HybridBlock
import models


class Residual(HybridBlock):
    def __init__(self, channels=(64,64), same_shape=True, **kwargs):
        '''
        :param channels: (conv1channel, conv2channel) for conv0channel equals conv2channel
        :param same_shape: whether this block keeps the feature map shape
        '''
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        if not same_shape:
            self.conv0 = gluon.nn.Conv2D(channels[1], kernel_size=(3,3), strides=(2,2))
            self.a0 = models.mPReLU(channels[1])
        self.conv1 = gluon.nn.Conv2D(channels[0],kernel_size=(3,3))
        self.a1 = models.mPReLU(channels[0])
        self.conv2 = gluon.nn.Conv2D(channels[1],kernel_size=(3,3))
        self.a2 = models.mPReLU(channels[1])

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
        self.archi_dict = archi_dict if archi_dict else self.default_params
        # add name_scope on the outermost Sequential
        with self.name_scope():
            # block 1
            self.features = gluon.nn.HybridSequential()
            b1 = Residual(3, self.archi_dict[0], exclude_first_conv=True, same_shape=False)

            # block 2
            b2_1 = Residual(self.archi_dict[1], same_shape=False)
            b2_2 = Residual(self.archi_dict[2])

            # block3
            b3_1 = Residual(self.archi_dict[3], same_shape=False)
            b3_2 = Residual(self.archi_dict[4])
            b3_3 = Residual(self.archi_dict[5])
            b3_4 = Residual(self.archi_dict[6])

            # block 4
            b4 = Residual(self.archi_dict[7], same_shape=False)
            f5 = gluon.nn.Dense(512)
            self.features.add(b1,b2_1,b2_2,b3_1,b3_2,b3_3,b3_4,b4,f5)

    def hybrid_forward(self, F,x,  *args, **kwargs):
        out = x
        for i, b in enumerate(self.features):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s'%(i+1, out.shape))
        return out

    def initialize_from(self,old_model_path,ctx,*args):
        if os.path.exists(os.path.join(old_model_path,'ModelAchi.pkl')):
            with open(os.path.join(old_model_path,'ModelAchi.pkl'), 'rb') as f:
                archi_dict = pickle.load(f)
        else: archi_dict = self.default_params
        old = models.SphereNet20(archi_dict=archi_dict,use_custom_relu=True)
        old.load_params(os.path.join(old_model_path,'model'))
        for name, block in self.features._children.items():
            old_block = old.features._children[name]
            if isinstance(block, qDense):
                block.weight.initialize(init=mx.init.Constant(old_block.weight._data[0]),force_reinit=True,ctx=ctx)
                block.bias.initialize(init=mx.init.Constant(old_block.bias._data[0]),force_reinit=True,ctx=ctx)
            else:
                if not old_block.same_shape:
                    block.conv0.weight.initialize(init=mx.init.Constant(old_block.conv0.conv.weight._data[0]),
                                                       force_reinit=True, ctx=ctx)
                    block.conv0.bias.initialize(init=mx.init.Constant(old_block.conv0.conv.bias._data[0]),
                                                        force_reinit=True,ctx=ctx)
                    block.a0.initialize(init=mx.init.Constant(old_block.a0.alpha._data[0]), force_reinit=True, ctx=ctx)
                block.conv1.weight.initialize(init=mx.init.Constant(old_block.conv1.conv.weight._data[0]),
                                                   force_reinit=True,ctx=ctx)
                block.conv1.bias.initialize(init=mx.init.Constant(old_block.conv1.conv.bias._data[0]),
                                                 force_reinit=True,ctx=ctx)
                block.a1.initialize(init=mx.init.Constant(old_block.a1.alpha._data[0]), force_reinit=True, ctx=ctx)
                block.conv2.weight.initialize(init=mx.init.Constant(old_block.conv2.conv.weight._data[0]),
                                                   force_reinit=True,ctx=ctx)
                block.conv2.bias.initialize(init=mx.init.Constant(old_block.conv2.conv.bias._data[0]),
                                                 force_reinit=True,ctx=ctx)
                block.a2.initialize(init=mx.init.Constant(old_block.a2.alpha._data[0]), force_reinit=True, ctx=ctx)