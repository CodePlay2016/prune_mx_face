import mxnet as mx
import mxnet.gluon as gluon
from mxnet.gluon import HybridBlock
import models
import test_model as tm
import pickle, os, cv2

def _infer_weight_shape(op_name, data_shape, kwargs):
    op = getattr(mx.symbol, op_name)
    sym = op(mx.symbol.var('data', shape=data_shape), **kwargs)
    return sym.infer_shape_partial()[0]

class qConv2D(HybridBlock):
    def __init__(self, channels, kernel_size=(3,3), strides=(1, 1), padding=(1, 1),
                 dilation=(1, 1), groups=1, layout='NCHW', in_channels=0, weight_initializer=None, **kwargs):
        super(qConv2D, self).__init__(**kwargs)
        self._kwargs = {
            'kernel': kernel_size, 'stride': strides, 'dilate': dilation,
            'pad': padding, 'num_filter': channels, 'num_group': groups,
            'layout': layout}

        op_name = 'Convolution'
        dshape = [0] * (len(kernel_size) + 2)
        dshape[layout.find('N')] = 1
        dshape[layout.find('C')] = in_channels
        wshapes = _infer_weight_shape(op_name, dshape, self._kwargs)
        with self.name_scope():
            self.weight = self.params.get('weight', shape=wshapes[1],
                                          init=weight_initializer)
            self.bias = self.params.get('bias', shape=wshapes[2], init='zeros')


    def hybrid_forward(self, F, x, weight, bias, **kwargs):
        xmin, xmax = (x.min(),x.max())
        wmin, wmax = (weight.min(),weight.max())
        bmin, bmax = (bias.min(),bias.max())
        qx = F.contrib.quantize(x,xmin,xmax,out_type='int8')
        qweight = F.contrib.quantize(weight,wmin,wmax,out_type='int8')
        qbias = F.contrib.quantize(bias,bmin,bmax,out_type='int8')
        out = F.contrib.quantized_conv(qx[0], qweight[0], qbias[0], xmin, xmax, wmin, wmax, bmin, bmax,
                                       self._kwargs['kernel'],self._kwargs['stride'],self._kwargs['dilate'],
                                       self._kwargs['pad'],self._kwargs['num_filter'],self._kwargs['num_group'])
        out = F.contrib.requantize(out[0], out[1], out[2])
        out = F.contrib.dequantize(out[0], out[1], out[2])
        return out

class qDense(HybridBlock):
    def __init__(self, units, flatten=True,
                 weight_initializer=None, bias_initializer='zeros',
                 in_units=0, **kwargs):
        super(qDense, self).__init__(**kwargs)
        self._flatten = flatten
        with self.name_scope():
            self._units = units
            self._in_units = in_units
            self.weight = self.params.get('weight', shape=(units, in_units),
                                          init=weight_initializer)
            self.bias = self.params.get('bias', shape=(units,),
                                            init=bias_initializer)
    def hybrid_forward(self, F, x, weight, bias):
        xmin, xmax = (x.min(), x.max())
        wmin, wmax = (weight.min(), weight.max())
        bmin, bmax = (bias.min(), bias.max())
        qx = F.contrib.quantize(x, xmin, xmax, out_type='int8')
        qw = F.contrib.quantize(weight, wmin, wmax, out_type='int8')
        qb = F.contrib.quantize(bias, bmin, bmax, out_type='int8')
        qx = F.contrib.quantize_fully_connected(qx[0],qw[0],qb[0],qx[1],qx[2],
                                                qw[1],qw[2],qb[1],qb[2],num_hidden=self._units,
                                                flatten=self._flatten)
        qx = F.contrib.requantize(qx[0],qx[1],qx[2])
        return F.contrib.dequantize(qx[0],qx[1],qx[2])

class qResidual(HybridBlock):
    def __init__(self, in_channels, channels=(64,64), exclude_first_conv=False, same_shape=True, **kwargs):
        '''
        :param channels: (conv1channel, conv2channel) for conv0channel equals conv2channel
        :param same_shape: whether this block keeps the feature map shape
        '''
        super(qResidual, self).__init__(**kwargs)
        self.same_shape = same_shape
        if not same_shape:
            if exclude_first_conv:
                self.conv0 = gluon.nn.Conv2D(channels[1], kernel_size=(3,3), strides=(2,2), padding=(1,1))
            else:
                self.conv0 = qConv2D(channels[1],in_channels=in_channels, strides=2)
            self.a0 = models.mPReLU(channels[1])
        self.conv1 = qConv2D(channels[0],in_channels=channels[1])
        self.a1 = models.mPReLU(channels[0])
        self.conv2 = qConv2D(channels[1],in_channels=channels[0])
        self.a2 = models.mPReLU(channels[1])

    def hybrid_forward(self, F, x, *args, **kwargs):
        if not self.same_shape:
            x = self.a0(self.conv0(x))
        out = self.a1(self.conv1(x))
        out = self.a2(self.conv2(out))
        return F.add(out, x)

class qSphereNet20(HybridBlock):
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
        super(qSphereNet20, self).__init__(**kwargs)
        self.verbose = verbose
        self.num_classes=num_classes
        self.use_custom_relu=use_custom_relu
        self.archi_dict = archi_dict if archi_dict else self.default_params
        # add name_scope on the outermost Sequential
        with self.name_scope():
            # block 1
            self.features = gluon.nn.HybridSequential()
            b1 = qResidual(3, self.archi_dict[0], exclude_first_conv=True, same_shape=False)

            # block 2
            b2_1 = qResidual(self.archi_dict[0][1], self.archi_dict[1], same_shape=False)
            b2_2 = qResidual(self.archi_dict[1][1], self.archi_dict[2])

            # block3
            b3_1 = qResidual(self.archi_dict[2][1], self.archi_dict[3], same_shape=False)
            b3_2 = qResidual(self.archi_dict[3][1], self.archi_dict[4])
            b3_3 = qResidual(self.archi_dict[4][1], self.archi_dict[5])
            b3_4 = qResidual(self.archi_dict[5][1], self.archi_dict[6])

            # block 4
            b4 = qResidual(self.archi_dict[6][1], self.archi_dict[7], same_shape=False)
            f5 = qDense(512, in_units=self.archi_dict[7][1]*42)
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

if __name__ == "__main__":
    new = qSphereNet20()
    new.initialize_from('./log/train-2018-07-02_091330',mx.gpu(6))
    new.hybridize(False) # the symbolic operation of <requantize> has some bug:
                                # src/core/symbolic.cc:301: Not enough argument to call operator
    tm.test_on_LFW(new, mx.gpu(6))

    print new
    mx.sym.contrib.requantize()