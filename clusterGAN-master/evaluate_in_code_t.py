from __future__ import print_function

try:
    import argparse
    import os
    import numpy as np
    import sys

    np.set_printoptions(threshold=sys.maxsize)

    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    from matplotlib.axes._axes import _log as matplotlib_axes_logger

    matplotlib_axes_logger.setLevel('ERROR')

    import pandas as pd

    from torch.autograd import Variable
    from torch.autograd import grad as torch_grad

    import torch
    import torchvision
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import datasets
    import torchvision.transforms as transforms
    from torchvision.utils import save_image

    from itertools import chain as ichain
    from sklearn.metrics.cluster import entropy, mutual_info_score, normalized_mutual_info_score, \
        adjusted_mutual_info_score, adjusted_rand_score
    from sklearn.metrics import accuracy_score

    from scipy.optimize import linear_sum_assignment as linear_assignment, linear_sum_assignment  # 添加as语句不用修改代码中的函数名

    from clusgan.definitions import DATASETS_DIR, RUNS_DIR
    from clusgan.models_t1 import Generator_CNN, Encoder_CNN, Discriminator_CNN
    from clusgan.datasets import get_dataloader, dataset_list
    from clusgan.utils import sample_z
    from sklearn.manifold import TSNE
    from clusgan.utils import load_model
except ImportError as e:
    print(e)
    raise ImportError


def pred2realprednum(gen_idx):
    # 这是映射列表

    lst = [8, 7, 3, 9, 4, 0, 1, 5, 2, 6]
    return lst[gen_idx]


def pred2realpred(pred):
    real_pred = torch.empty((1))
    for i, x in enumerate(pred):
        real_pred = np.append(real_pred, pred2realprednum(x))
    return torch.Tensor(real_pred[1:])


def NMI(x, y):
    return normalized_mutual_info_score(x, y, average_method='arithmetic')


def MI(x, y):
    return mutual_info_score(x, y)


def AMI(x, y):
    return adjusted_mutual_info_score(x, y, average_method='arithmetic')


def ARI(x, y):
    return adjusted_rand_score(x, y)


def purity_score(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def ACC(y_true, y_pred):
    """
       # 此代码非库代码，为网上代码
       链接
       https://blog.csdn.net/qq_42887760/article/details/105720735

       Calculate clustering accuracy. Require scikit-learn installed
       # Arguments
           y: true labels, numpy.array with shape `(n_samples,)`
           y_pred: predicted labels, numpy.array with shape `(n_samples,)`
       # Return
           accuracy, in [0,1]
       """
    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)

    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def evaluate(run_dir):
    global args
    parser = argparse.ArgumentParser(description="TSNE generation script")
    parser.add_argument("-r", "--run_dir", dest="run_dir",
                        default=run_dir,

                        help="Training run directory")
    parser.add_argument("-shape", "-–image_shape", dest="image_shape", default=(1, 28, 28), type=tuple,
                        help="size of image")
    parser.add_argument("-p", "--perplexity", dest="perplexity", default=-1, type=int, help="TSNE perplexity")
    parser.add_argument("-n", "--n_samples", dest="n_samples", default=100, type=int, help="Number of samples")
    args = parser.parse_args()
    # 参数定义

    n_sqrt_samp = 5
    n_samp = n_sqrt_samp * n_sqrt_samp
    # TSNE setup
    n_samples = args.n_samples

    x_shape = args.image_shape

    # Directory structure for this run
    run_dir = args.run_dir.rstrip("\\")
    # print(run_dir.split(os.sep))
    dataset_name = run_dir.split(os.sep)[-2]
    # print(dataset_name)
    run_name = run_dir.split(os.sep)[-1]
    # run_name = "300epoch_z30_wass_bs64_" + run_name

    # print(run_name)
    run_dir = os.path.join(RUNS_DIR, dataset_name, run_name)
    # print(run_dir)
    data_dir = os.path.join(DATASETS_DIR, dataset_name)
    imgs_dir = os.path.join(run_dir, 'images')
    models_dir = os.path.join(run_dir, 'models')

    # Latent space info
    train_df = pd.read_csv('%s/training_details.csv' % (run_dir))
    latent_dim = train_df['latent_dim'][0]
    n_c = train_df['n_classes'][0]

    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda') if cuda else torch.device('cpu')
    # Load encoder model

    discriminator = Discriminator_CNN(wass_metric=True )
    encoder = Encoder_CNN(latent_dim, n_c ).to(device)
    generator = Generator_CNN(latent_dim, n_c, x_shape).to(device)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters())
    ge_chain = ichain(generator.parameters(),
                      encoder.parameters())
    # # Adam 优化器
    optimizer_GE = torch.optim.Adam(ge_chain)
    model_list = [discriminator, encoder, generator]
    optimizer_list = [optimizer_D, optimizer_GE]
    epoch_init = load_model(models=model_list, optimizers=optimizer_list, out_dir=models_dir)

    xe_loss = torch.nn.CrossEntropyLoss()
    enc_figname = os.path.join(models_dir, encoder.name + '.pth.tar')
    # encoder.load_state_dict(torch.load(enc_figname, map_location=device))
    encoder.to(device)

    # Configure data loader
    dataloader = get_dataloader(dataset_name=dataset_name, data_dir=data_dir, batch_size=n_samples, train_set=False,num_workers=2)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Get full batch for encoding
    eval_acc = 0
    eval_loss = 0

    nmi = 0
    ari = 0
    acc = 0
    nmi_ege = 0
    ari_ege = 0
    acc_ege = 0
    with torch.no_grad():
        for data in dataloader:
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)
            c_imgs = Variable(imgs.type(Tensor), requires_grad=False)

            # Encode real images
            # t_img--> encoder == zn + zc
            output_enc_zn,   enc_zc, enc_zc_logits = encoder(c_imgs)
            # 预测结果
            pred = torch.max(enc_zc, 1)[1]
            # 指标
            nmi += NMI(pred.cpu(), labels.cpu())
            ari += ARI(pred.cpu(), labels.cpu())
            acc += ACC(pred.cpu(), labels.cpu())



    print('NMI:{:.6f}'.format(nmi / len(dataloader)))
    print('ARI:{:.6f}'.format(ari / len(dataloader)))
    print('ACC:{:.6f}'.format(acc / len(dataloader)))
    return nmi / len(dataloader), ari / len(dataloader), acc / len(dataloader)
    # print('acc(此acc无意义，需要配置映射函数后才有意义):{:.6f}'.format(eval_acc / len(dataloader)))

# if __name__ == "__main__":
#     # A = torch.Tensor([
#     #     4, 3, 2, 1
#     # ])
#     # B = torch.Tensor([
#     #     1, 2, 3, 4
#     # ])
#     # C = torch.Tensor([
#     #     4, 3, 2, 1
#     # ])
#     #
#     # # 测试
#     # print(ACC(A, B))  # 0.7058823529411765
#     # print(ACC(A, C))  # 0.7058823529411765
#     # print(accuracy_score(A, B))  # 0.7058823529411765
#
#     # print(ACC(np.array(x.tolist()), np.array(y.tolist())))
#     evaluate()
# tensorboard --logdir formal
