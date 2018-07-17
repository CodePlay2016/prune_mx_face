import mxnet as mx
import models,dataset
import pickle,os

from mxnet.contrib.quantization import *
# reference: https://github.com/apache/incubator-mxnet/tree/master/example/quantization


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

if __name__ == '__main__':
    original_model_path = "./log/train-2018-07-02_091330"
    model_path = "./log/prune-2018-07-14_003715"
    # export_sym(original_model_path)
    ctx = mx.gpu()
    sym_model_prefix = os.path.join(original_model_path,'model_symbol')
    sym, arg_params, aux_params = mx.model.load_checkpoint(sym_model_prefix, 0)
    print sym.get_internals().list_outputs()
    cqsym, qarg_params, aux_params = quantize_model(sym=sym, arg_params=arg_params, aux_params=aux_params,
                                                    ctx=ctx,calib_mode='none',
                                                    excluded_sym_names=['spherenet200_conv0_fwd'])
    mod = mx.mod.Module(symbol=sym, context=ctx,label_names=None)
    mod.bind(data_shapes=[('data', (32, 3, 96, 112))],for_training=False)
    mod.load_params(sym_model_prefix+'-0000.params')

    print mod

