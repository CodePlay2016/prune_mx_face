import mxnet as mx
import mxnet.ndarray as nd
from mxnet.gluon.data.vision import transforms
import dataset, models
import glob, pickle, time, os
import numpy as np
from PIL import Image

mx.random.seed(128)
class MTransform(mx.gluon.nn.Block):
    '''
    normalize each image with its mean and variance value
    '''
    def __init__(self):
        super(MTransform, self).__init__()

    def forward(self, x, *args):
        return (x - x.mean()) / nd.sqrt(nd.mean(nd.power((x - x.mean()), 2)))


def cal_my_acc(test_files, target_files):
    '''
    this method is deprecated
    :param test_files:
    :param target_files:
    :return:
    '''
    mTransform = MTransform()
    normalize = transforms.Normalize(mean=0.5, std=0.5)
    transform = transforms.Compose([
        transforms.Resize((96, 112)),
        transforms.ToTensor(),
        normalize,
        # mTransform,
    ])
    model = models.SphereNet20()
    model.load_params("spherenet_model", ctx=mx.gpu())
    correct = 0
    total = 0
    target_emb = {}
    for target_file in target_files:
        target_image = transform(nd.array(Image.open(target_file))).as_in_context(mx.gpu())
        target_image = nd.expand_dims(target_image, axis=0)
        target_label = ''.join(target_file.split('/')[-1].split('.')[:-1])
        target_out = model(target_image)
        target_emb[target_label] = target_out
    test_emb = {}
    for test_file in test_files:
        test_image = Image.open(test_file)
        test_image = nd.expand_dims(transform(nd.array(test_image)), axis=0).as_in_context(mx.gpu())
        test_label = ''.join(test_file.split('/')[-1].split('.')[:-1])
        test_out = model(test_image)
        max_s = mx.nd.zeros(1, ctx=mx.gpu())
        max_label = ''
        sims = {}
        for target_label, target_out in target_emb.items():
            similarity = nd.sum(test_out * target_out) / \
                         (nd.norm(test_out) * nd.norm(target_out))
            sims[target_label] = similarity.asscalar()
            if max_s < similarity:
                max_s = similarity
                max_label = target_label
        if ''.join(max_label.split('_')[:-1]) == ''.join(test_label.split('_')[:-1]):
            correct += 1
        else:
            print test_label, max_s.asscalar(), max_label
        total += 1
        test_emb[test_label] = test_out
        # print correct, total, float(correct)/total

    return float(correct) / total, test_emb, target_emb


def KFold(n=6000, n_folds=10, shuffle=False):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[i * n / n_folds:(i + 1) * n / n_folds]
        train = list(set(base) - set(test))
        folds.append([train, test])
    return folds

def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0 * np.count_nonzero(y_true == y_predict) / len(y_true)
    return accuracy

def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold

def test_on_LFW(model,ctx=mx.gpu()):
    with open('/home1/LFW/pairs.txt', 'rt') as f:
        pairs_lines = f.readlines()[1:]
    sims = []
    model.get_feature=True
    normalize = transforms.Normalize(mean=0.5, std=0.25)
    transform = transforms.Compose([
        transforms.Resize((96, 112)),
        transforms.ToTensor(),
        normalize,
        # mTransform,
    ])
    start = time.time()
    forward_time = 0
    for i in range(6000):
        p = pairs_lines[i].replace('\n', '').split('\t')

        if 3 == len(p):
            sameflag = 1
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
        if 4 == len(p):
            sameflag = 0
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))

        img1 = nd.array(Image.open('/home1/LFW/aligned_lfw-112X96/' + name1))
        img2 = nd.array(Image.open('/home1/LFW/aligned_lfw-112X96/' + name2))
        img1 = transform(img1)
        img2 = transform(img2)
        img = nd.stack(img1, img2)

        img = img.as_in_context(ctx)
        fstart = time.time()
        output = model(img)
        forward_time += time.time() - fstart
        f1, f2 = output[0], output[1]
        cosdistance = nd.sum(f1 * f2) / (f1.norm() * f2.norm() + 1e-5)
        sims.append('{}\t{}\t{}\t{}\n'.format(name1, name2, cosdistance.asscalar(), sameflag))

    accuracy = []
    thd = []
    folds = KFold(n=6000, n_folds=10, shuffle=False)
    thresholds = np.arange(0, 1.0, 0.005)
    predicts = np.array(map(lambda line: line.strip('\n').split(), sims))

    for idx, (train, test) in enumerate(folds):
        best_thresh = find_best_threshold(thresholds, predicts[train])
        accuracy.append(eval_acc(best_thresh, predicts[test]))
        thd.append(best_thresh)
    # print time.time() - start-cost # single 1080Ti about 100s
    msg = 'LFWACC={:.4f} std={:.4f} thd={:.4f}, model forward test time:{:.4f}, total time: {:.4f}'.format(
        np.mean(accuracy), np.std(accuracy),np.mean(thd),forward_time, time.time()-start)

    return msg

def test_speed(model,batch_size=64, ctx=mx.gpu()):
    train_data_loader, _, _ = dataset.train_valid_test_loader('/home1/CASIA-WebFace/aligned_Webface-112X96',
                                                              (0.9, 0.05), batch_size=64)

    total = 0
    model.hybridize()
    for batch, label in train_data_loader:
        batch = batch.as_in_context(ctx)
        start = time.time()
        out = model(batch)
        total += time.time() - start
    print total

if __name__ == "__main__":
    # gpus = [0,1]
    # ctx = [mx.gpu(ii) for ii in gpus]
    model_path1 = "./log/prune-2018-07-13_121936"
    model_path2 = "./log/prune-2018-07-14_003715"
    ctx = mx.gpu()
    archi_dict = {
        0:[32,40],
        1:[73,74],
        2:[75,74],
        3:[108,115],
        4:[97,115],
        5:[106,115],
        6:[110,115],
        7:[143,205]
    }
    with open(os.path.join(model_path1,"ModelAchi.pkl"),"rb") as f:
        archi_dict1 = pickle.load(f)
    # with open(os.path.join(model_path1,"ModelAchi.pkl"),"wb") as f:
    #     pickle.dump(archi_dict,f)
    with open(os.path.join(model_path2,"ModelAchi.pkl"),"rb") as f:
        archi_dict2 = pickle.load(f)
    model1 = models.SphereNet20(archi_dict=archi_dict1,use_custom_relu=True)
    model1.load_params(os.path.join(model_path1,"model"), ctx=ctx)
    model2 = models.SphereNet20(archi_dict=archi_dict2,use_custom_relu=True)
    model2.load_params(os.path.join(model_path2,"model"), ctx=ctx)
    print(test_on_LFW(model1))
    print(test_on_LFW(model2))
    test_speed(model1)
    test_speed(model2)
    print 'over'