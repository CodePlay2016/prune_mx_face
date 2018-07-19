import mxnet as mx
import mxnet.symbol as sym
import models,dataset
import pickle,os

from mxnet.contrib.quantization import *
# reference: https://github.com/apache/incubator-mxnet/tree/master/example/quantization
def get_conv(inpt, name, num_filter, stride=(1,1), act_type='prelu'):
    act_name = name.replace('conv',act_type)
    return sym.LeakyReLU(sym.Convolution(inpt,name=name,num_filter=num_filter,
                                         stride=stride,kernel=(3,3),pad=(1,1)),
                         name=act_name,act_type=act_type,gamma=sym.var(act_name))

def get_symbol():
    data = sym.var('data')
    out1 = get_conv(data,'conv1-1',64,(2,2))
    out = get_conv(out1,'conv1-2',64)
    out = get_conv(out,'conv1-3',64) + out1

    out1 = get_conv(out,'conv2-1',128,(2,2))
    out = get_conv(out1,'conv2-2',128)
    out1 = get_conv(out,'conv2-3',128) + out1

    out = get_conv(out1,'conv2-4',128)
    out1 = get_conv(out,'conv2-5',128) + out1

    out1 = get_conv(out1,'conv3-1',256,(2,2))
    out = get_conv(out1,'conv3-2',256)
    out1 = get_conv(out,'conv3-3',256) + out1

    out = get_conv(out1, 'conv3-4',256)
    out1 = get_conv(out, 'conv3-5',256) + out1

    out = get_conv(out1, 'conv3-6',256)
    out1 = get_conv(out, 'conv3-7',256) + out1

    out = get_conv(out1, 'conv3-8',256)
    out1 = get_conv(out, 'conv3-9',256) + out1

    out1 = get_conv(out1, 'conv4-1',512,(2,2))
    out = get_conv(out1, 'conv4-2',512)
    out = get_conv(out, 'conv4-3',512) + out1

    return sym.FullyConnected(out,name='fc5',num_hidden=512)

def _save_params(fname, arg_params, aux_params, logger=None):
    if logger is not None:
        logger.info('Saving params into file at %s' % fname)
    save_dict = {('arg:%s' % k): v.as_in_context(cpu()) for k, v in arg_params.items()}
    save_dict.update({('aux:%s' % k): v.as_in_context(cpu()) for k, v in aux_params.items()})
    mx.nd.save(fname, save_dict)

def load_model(prefix):
    cur_path = os.path.dirname(os.path.realpath(__file__))
    symbol_file_path = prefix+'.json'
    symbol = mx.sym.load(symbol_file_path)

    param_file_path = prefix+'.param'
    save_dict = mx.nd.load(param_file_path)
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return symbol, arg_params, aux_params

def _convert_params(arg_params):
    new_arg_params = {}
    encoder = {0:'1-1',1:'1-2',2:'1-3',
               3:'2-1',4:'2-2',5:'2-3',6:'2-4',7:'2-5',
               8:'3-1',9:'3-2',10:'3-3',11:'3-4',12:'3-5',
               13:'3-6',14:'3-7',15:'3-8',16:'3-9',
               17:'4-1',18:'4-2',19:'4-3'}
    for name, param in arg_params.items():
        name = name.replace('spherenet200_','')
        if 'conv' in name:
            number = name.split('_')[0][4:]
            name = name.replace(number,encoder[int(number)])
        elif 'mprelu' in name:
            number = name.split('_')[0][6:]
            name = name.replace(number,encoder[int(number)])
            name = name.replace('_alpha','')
            name = name.replace('mprelu','prelu')
            param = param.reshape(-1,)
        elif 'dense' in name:
            name = name.replace('dense0','fc5')
        new_arg_params[name] = param
    return new_arg_params

def _get_quantized(model_path, ctx):
    sym_model_prefix = os.path.join(model_path, 'new_sym')
    sym, arg_params, aux_params = mx.model.load_checkpoint(sym_model_prefix, 0)
    excluded_sym_names = ['conv1-1']  # exclude the first layer
    cqsym, qarg_params, aux_params = quantize_model(sym=sym, arg_params=arg_params, aux_params=aux_params,
                                                    ctx=ctx, calib_mode='none',
                                                    excluded_sym_names=excluded_sym_names)
    cqsym.save(os.path.join(model_path, 'new_sym_quantized.json'))
    _save_params(os.path.join(model_path, 'new_sym_quantized.param'), qarg_params, aux_params)
    return cqsym, qarg_params, aux_params



def export_sym(model_path, ctx=mx.gpu()):
    if os.path.exists(os.path.join(model_path,'ModelAchi.pkl')):
        with open(os.path.join(model_path,'ModelAchi.pkl'),'rb') as f:
            archi_dict = pickle.load(f)
    else: archi_dict = None
    model = models.SphereNet20(archi_dict=archi_dict)
    model.load_params(os.path.join(model_path,'model'), ctx=ctx)

    train_data_loader, _, _ = dataset.train_valid_test_loader('/home1/CASIA-WebFace/aligned_Webface-112X96',
                                                           (0.9, 0.05), batch_size=1)
    # run forward once
    model.hybridize()
    model.get_feature = False
    for batch,label in train_data_loader:
        batch = batch.as_in_context(ctx)
        model(batch)
        break
    model.export(os.path.join(model_path,'model_symbol'))

def main_v1():
    original_model_path = "./log/train-2018-07-17_195201"
    model_path = "./log/prune-2018-07-14_003715"
    # export_sym(original_model_path)
    ctx = mx.gpu()
    sym_model_prefix = os.path.join(original_model_path, 'model_symbol')
    sym, arg_params, aux_params = mx.model.load_checkpoint(sym_model_prefix, 0)
    excluded_sym_names = ['spherenet200_conv0_fwd']  # exclude the first layer
    for name in sym.get_internals().list_outputs():
        if 'residual' in name:
            excluded_sym_names.append(name[:-7])
    cqsym, qarg_params, aux_params = quantize_model(sym=sym, arg_params=arg_params, aux_params=aux_params,
                                                    ctx=ctx, calib_mode='none',
                                                    excluded_sym_names=excluded_sym_names)
    cqnodes = cqsym.get_internals().list_outputs()
    for ii, name in enumerate(cqnodes):
        print ii, name
        if name == 'spherenet200_dense0_fwd_dequantize_output':
            cqfeatures = cqsym.get_internals()[:ii + 1]
            break
    mod = mx.mod.Module(symbol=cqfeatures, context=ctx, label_names=None)
    mod.bind(data_shapes=[('data', (32, 3, 96, 112))], for_training=False)
    mod.set_params()

if __name__ == '__main__':
    model_path = "./log/train-2018-07-02_091330"
    # export_sym(original_model_path)
    ctx = mx.gpu(2)
    # cqsym, qarg_params, aux_params = _get_quantized(model_path,ctx)
    prefix = os.path.join(model_path,'new_sym_quantized')
    cqsym, qarg_params, aux_params = load_model(prefix)

    mod2 = mx.mod.Module(symbol=cqsym, context=ctx,label_names=None)
    mod2.bind(data_shapes=[('data', (32, 3, 112, 96))],for_training=False)
    mod2.set_params(qarg_params,aux_params)

    print mod2

