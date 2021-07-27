# coding=utf-8
from __future__ import absolute_import, print_function
import time
import argparse
import os
import sys
import torch.utils.data
from torch.backends import cudnn
from torch.autograd import Variable
import models
import losses
from utils import FastRandomIdentitySampler, mkdir_if_missing, logging, display
from utils.serialization import save_checkpoint, load_checkpoint
# from trainer import train
from trainer import train
from utils import orth_reg
from utils.ckptest import ckptest_I, ckptest
import DataSet
import numpy as np
import os.path as osp

cudnn.benchmark = True

# test import
# from __future__ import absolute_import, print_function
# import argparse
from Model2Feature import Model2Feature
from Model2Feature_t import Model2Feature_t
from evaluations import Recall_at_ks, pairwise_similarity
# from utils.serialization import load_checkpoint
import torch
import ast

# paint
import matplotlib
from matplotlib.ticker import MultipleLocator

matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

use_gpu = True


# draw
def plot_loss(loss, log_dir, dataset, is_log=False):
    fig, axes = plt.subplots()
    ax = axes

    loss = np.log10(loss) if is_log else np.array(loss)
    ax.plot(loss, label='net_losses')

    ylable = "Losses(log10)" if is_log else "Losses"
    ax.set_ylabel(ylable)
    ax.set_xlabel("Epoch")
    ax.legend()

    im_path = '%s/%s_loss.png' % (log_dir, dataset)
    mkdir_if_missing(os.path.dirname(im_path))
    plt.savefig(im_path)
    # print("save done " + im_path)
    plt.close()


def plot_rank(rank, log_dir, dataset, step, epoch):
    fig, axes = plt.subplots()
    epoch = [epoch * step for epoch in range(epoch // step)]
    ax = axes
    ax.plot(epoch, rank, label='R@1')
    # ax.xaxis.set_major_locator(xmajorLocator)
    ax.set_ylabel('R@1')
    ax.set_xlabel("Epoch")
    ax.legend()

    im_path = '%s/%s_R@.png' % (log_dir, dataset)
    mkdir_if_missing(os.path.dirname(im_path))
    plt.savefig(im_path)
    # print("save done " + im_path)
    plt.close()

def heat(x, log_dir, dataset, epoch):
    plt.matshow(x, cmap=plt.cm.cool, vmin=0, vmax=1)
    plt.colorbar()
    im_path = '%s/%s_%s_Heat.png' % (log_dir, dataset, epoch)
    plt.savefig(im_path,dpi =300)
    plt.close()

def hist(x1,x2 ,log_dir, dataset, epoch):
    # plt.hist(x1.cpu(),range=(0.00001, 1), alpha=0.3, density=True, bins=50, rwidth=0.8)
    # plt.hist(x2.cpu(), range=(0, 1), alpha=0.3,density=True, bins=50, rwidth=0.8)
    # im_path = '%s/%s_%s_Hist_ap.png' % (log_dir, dataset, epoch)
    # plt.savefig(im_path,dpi =200)
    # plt.close()

    plt.hist(x1.cpu(), range=(0.000001, 1), alpha=0.7, density=True, bins=100, rwidth=0.9)
    plt.hist(x2.cpu(), range=(0, 1), alpha=0.5, density=True, bins=100, rwidth=0.9)
    im_path = '%s/%s_%s_Hist.png' % (log_dir, dataset, epoch)
    plt.savefig(im_path, dpi=200)
    plt.close()
def hist_t(x1,x2 ,log_dir, dataset, epoch):
    # plt.hist(x1.cpu(),range=(0.00001, 1), alpha=0.3, density=True, bins=50, rwidth=0.8)
    # plt.hist(x2.cpu(), range=(0, 1), alpha=0.3,density=True, bins=50, rwidth=0.8)
    # im_path = '%s/%s_%s_Hist_ap.png' % (log_dir, dataset, epoch)
    # plt.savefig(im_path,dpi =200)
    # plt.close()

    plt.hist(x1.cpu(), range=(0.000001, 1), alpha=0.7, density=True, bins=100, rwidth=0.9)
    plt.hist(x2.cpu(), range=(0, 1), alpha=0.5, density=True, bins=100, rwidth=0.9)
    im_path = '%s/%s_%s_Hist_t.png' % (log_dir, dataset, epoch)
    plt.savefig(im_path, dpi=200)
    plt.close()

def hist_tt(x1,x2,x1_t,x2_t ,log_dir, dataset, epoch):

    plt.hist(x1.cpu(), range=(0.000001, 1), alpha=0.5, density=True, bins=100, rwidth=0.9)
    plt.hist(x2.cpu(), range=(0, 1), alpha=0.5, density=True, bins=100, rwidth=0.9)
    plt.hist(x1_t.cpu(), range=(0.000001, 1), alpha=0.6, density=True, bins=100, rwidth=0.8)
    plt.hist(x2_t.cpu(), range=(0, 1), alpha=0.6, density=True, bins=100, rwidth=0.8)
    im_path = '%s/%s_%s_Hist_tt.png' % (log_dir, dataset, epoch)
    plt.savefig(im_path, dpi=200)
    plt.close()


# Batch Norm Freezer : bring 2% improvement on CUB
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def main(args):
    # s_ = time.time()

    save_dir = args.save_dir
    mkdir_if_missing(save_dir)

    sys.stdout = logging.Logger(os.path.join(save_dir, 'log.txt'))
    display(args)
    start = 0

    model = models.create(args.net, pretrained=True, dim=args.dim)

    # for vgg and densenet
    if args.resume is None:
        model_dict = model.state_dict()

    else:
        # resume model
        print('load model from {}'.format(args.resume))
        chk_pt = load_checkpoint(args.resume)
        weight = chk_pt['state_dict']
        # start = chk_pt['epoch']
        start = 0
        ###have a try
        checkpoint = weight  # 获取模型参数
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in checkpoint.items() if k in model_dict.keys()}
        print("matched parameters: %d/%d" % (len(state_dict), len(model_dict)))
        ####have a try
        # model.load_state_dict(weight)
        model_dict.update(state_dict)  # 更新已经保存的参数至model_dict
        model.load_state_dict(model_dict)  # 加载模型参数

    model = torch.nn.DataParallel(model)
    model = model.cuda()

    # freeze BN
    if args.freeze_BN is True:
        print(40 * '#', '\n BatchNorm frozen')
        model.apply(set_bn_eval)
    else:
        print(40 * '#', 'BatchNorm NOT frozen')

    # Fine-tune the model: the learning rate for pre-trained parameter is 1/10
    new_param_ids = set(map(id, model.module.classifier.parameters()))

    new_params = [p for p in model.module.parameters() if
                  id(p) in new_param_ids]

    base_params = [p for p in model.module.parameters() if
                   id(p) not in new_param_ids]

    param_groups = [
        {'params': base_params, 'lr_mult': 0.0},
        {'params': new_params, 'lr_mult': 1.0}]


    print('initial model is save at %s' % save_dir)

    optimizer = torch.optim.Adam(param_groups, lr=args.lr,
                                 weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # criterion = losses.create(args.loss, margin=args.margin, alpha=args.alpha, base=args.loss_base).cuda()
    criterion = losses.create(args.loss, alpha=args.alpha, base=args.loss_base).cuda()
    # criterion = losses.create(args.loss, nb_classes=100, sz_embed=args.dim).cuda()
    # criterion = losses.create(args.loss).cuda()
    if args.loss == 'PAnchor':
        param_groups.append({'params': criterion.proxies, 'lr': float(args.lr) * 100})
    # Decor_loss = losses.create('decor').cuda()
    data = DataSet.create(args.data, ratio=args.ratio, width=args.width, origin_width=args.origin_width,
                          root=args.data_root)

    # train_loader = torch.utils.data.DataLoader(
    #     data.train, batch_size=args.batch_size,
    #     sampler=FastRandomIdentitySampler(data.train, num_instances=args.num_instances),
    #     drop_last=True, pin_memory=True, num_workers=args.nThreads)

    train_loader = torch.utils.data.DataLoader(
        data.train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.nThreads,
        drop_last=True,
        pin_memory=True
    )

    recall_list = np.zeros((args.epochs, 1))
    recall_list_t = np.zeros((args.epochs, 1))
    losses_p = []
    Rank1 = []
    Rank1_t = []

    # save the train information
    print("before train")
    # ckptest(args)
    for epoch in range(start, args.epochs):
        

        losse = train(epoch=epoch, model=model, criterion=criterion,
                      optimizer=optimizer, train_loader=train_loader, args=args)
        losse = losse.avg
        losses_p.append(losse)

        if epoch == 1:
            optimizer.param_groups[0]['lr_mul'] = 0.1

        if (epoch + 1) % args.save_step == 0 or epoch == 0:
            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            #
            save_checkpoint({
                'state_dict': state_dict,
                'epoch': (epoch + 1),
            }, is_best=False, fpath=osp.join(args.save_dir, 'ckp_ep' + str(epoch + 1) + '.pth.tar'))

            if epoch == 0:
                pass
            else:
                # test
                checkpoint = load_checkpoint(osp.join(args.save_dir, 'ckp_ep' + str(epoch + 1) + '.pth.tar'))
                # print(args.pool_feature)
                epoch = checkpoint['epoch']

                gallery_feature, gallery_labels, query_feature, query_labels = \
                    Model2Feature(data=args.data, root=args.data_root, width=args.width, net=args.net_t,
                                  checkpoint=checkpoint,
                                  dim=args.dim, batch_size=args.batch_size_t, nThreads=args.nThreads,
                                  pool_feature=args.pool_feature)

                gallery_feature_t, gallery_labels_t, query_feature_t, query_labels_t = \
                    Model2Feature_t(data=args.data, root=args.data_root, width=args.width, net=args.net_tt,
                                  checkpoint=checkpoint,
                                  dim=args.dim, batch_size=args.batch_size_t, nThreads=args.nThreads,
                                  pool_feature=args.pool_feature)

                sim_mat = pairwise_similarity(query_feature, gallery_feature)
                sim_mat_t = pairwise_similarity(query_feature_t, gallery_feature_t)
                # heat(x=sim_mat, log_dir=args.save_dir, dataset=args.data, epoch=epoch)  #plot heat map
                # print(sim_mat.view(-1).shape)


                if args.gallery_eq_query is True:
                    sim_mat = sim_mat - torch.eye(sim_mat.size(0))
                    sim_mat_t = sim_mat_t - torch.eye(sim_mat_t.size(0))

                smt = sim_mat.cuda()
                smt_t =sim_mat_t.cuda()
                # hist(x=sim_mat.view(-1), log_dir=args.save_dir, dataset=args.data, epoch=epoch)
                recall_ks = Recall_at_ks(sim_mat, query_ids=query_labels, gallery_ids=gallery_labels, data=args.data)
                recall_ks_t = Recall_at_ks(sim_mat_t, query_ids=query_labels_t, gallery_ids=gallery_labels_t, data=args.data)
                Rank1.append(recall_ks[0])
                Rank1_t.append(recall_ks[0])
                recall_list[epoch] = recall_ks[0]
                recall_list_t[epoch] = recall_ks_t[0]

                gl = torch.tensor(np.asarray(gallery_labels)).cuda()
                ql = torch.tensor(np.asarray(query_labels)).cuda()
                gl_O = gl.expand(len(gl),len(gl))
                ql_O =  ql.expand(len(ql),len(ql)).t()
                aplb_smt = (gl_O==ql_O).cuda().float()
                anlb_smt = (gl_O != ql_O).cuda().float()
                ap_sim = smt*aplb_smt
                an_sim = smt*anlb_smt

                gl_t = torch.tensor(np.asarray(gallery_labels_t)).cuda()
                ql_t = torch.tensor(np.asarray(query_labels_t)).cuda()
                gl_O_t = gl_t.expand(len(gl_t), len(gl_t))
                ql_O_t = ql_t.expand(len(ql_t), len(ql_t)).t()
                aplb_smt_t = (gl_O_t == ql_O_t).cuda().float()
                anlb_smt_t = (gl_O_t != ql_O_t).cuda().float()
                ap_sim_t = smt_t * aplb_smt_t
                an_sim_t = smt_t * anlb_smt_t
                hist(x1=ap_sim.view(-1), x2=an_sim.view(-1), log_dir=args.save_dir, dataset=args.data, epoch=epoch)
                hist_t(x1=ap_sim_t.view(-1), x2=an_sim_t.view(-1), log_dir=args.save_dir, dataset=args.data, epoch=epoch)
                hist_tt(x1=ap_sim.view(-1), x2=an_sim.view(-1), x1_t=ap_sim_t.view(-1), x2_t=an_sim_t.view(-1), log_dir=args.save_dir, dataset=args.data, epoch=epoch)

                # print('text%.4f',recall_list[epoch][0])
                print('text', np.argmax(recall_list), '%.4f'%np.max(recall_list))
                result = '  '.join(['%.4f' % k for k in recall_ks])
                print('Epoch-%d'% epoch, result)
                # print('train%.4f',recall_list_t[epoch][0])
                print('train',np.argmax(recall_list_t),'%.4f'%np.max(recall_list_t))
                result_t = '  '.join(['%.4f' % k for k in recall_ks_t])
                print('Epoch-%d'% epoch, result_t)


                # print('typeresult',type(result))

                plot_rank(rank=Rank1, log_dir=args.save_dir, dataset=args.data, step=args.save_step, epoch=epoch)
                plot_loss(loss=losses_p, log_dir=args.save_dir, dataset=args.data, is_log=True)
        # scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Metric Learning')

    # hype-parameters
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate of new parameters")
    parser.add_argument('--batch_size', '-b', default=128, type=int, metavar='N',
                        help='mini-batch size (1 = pure stochastic) Default: 256')
    parser.add_argument('--num_instances', default=8, type=int, metavar='n',
                        help=' number of samples from one class in mini-batch')
    parser.add_argument('--dim', default=512, type=int, metavar='n',
                        help='dimension of embedding space')
    parser.add_argument('--width', default=224, type=int,
                        help='width of input image')
    parser.add_argument('--origin_width', default=256, type=int,
                        help='size of origin image')
    parser.add_argument('--ratio', default=0.16, type=float,
                        help='random crop ratio for train data')

    parser.add_argument('--alpha', default=30, type=int, metavar='n',
                        help='hyper parameter in NCA and its variants')
    parser.add_argument('--beta', default=0.1, type=float, metavar='n',
                        help='hyper parameter in some deep metric loss functions')
    parser.add_argument('--orth_reg', default=0, type=float,
                        help='hyper parameter coefficient for orth-reg loss')
    parser.add_argument('-k', default=16, type=int, metavar='n',
                        help='number of neighbour points in KNN')
    parser.add_argument('--margin', default=0.09, type=float,
                        help='margin in loss function')
    parser.add_argument('--init', default='random',
                        help='the initialization way of FC layer')

    # network
    parser.add_argument('--freeze_BN', default=True, type=bool, required=False, metavar='N',
                        help='Freeze BN if True')
    parser.add_argument('--data', default='cub', required=True,
                        help='name of Data Set')
    parser.add_argument('--data_root', type=str, default=None,
                        help='path to Data Set')

    parser.add_argument('--net', default='VGG16-BN')
    parser.add_argument('--loss', default='branch', required=True,
                        help='loss for training network')
    parser.add_argument('--epochs', default=600, type=int, metavar='N',
                        help='epochs for training process')
    parser.add_argument('--save_step', default=50, type=int, metavar='N',
                        help='number of epochs to save model')

    # Resume from checkpoint
    # parser.add_argument('--resume', '-r', default='/home/jxr/proj/DMM_og/models/ckp_ep2400_bio_m0506_contt.pth.tar',
    #                     help='the path of the pre-trained model')
    parser.add_argument('--resume', '-r', default=None, help='the path of the pre-trained model')

    # train
    parser.add_argument('--print_freq', default=20, type=int,
                        help='display frequency of training')

    # basic parameter
    # parser.add_argument('--checkpoints', default='/opt/intern/users/xunwang',
    #                     help='where the trained models save')
    parser.add_argument('--save_dir', default=None,
                        help='where the trained models save')
    parser.add_argument('--nThreads', '-j', default=16, type=int, metavar='N',
                        help='number of data loading threads (default: 2)')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)

    parser.add_argument('--loss_base', type=float, default=0.75)

    # test

    # parser.add_argument('--data', type=str, default='cub')
    # parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--gallery_eq_query', '-g_eq_q', type=ast.literal_eval, default=True,
                        help='Is gallery identical with query')
    parser.add_argument('--net_t', type=str, default='BN-Inception')
    parser.add_argument('--net_tt', type=str, default='BN-Inception')
    parser.add_argument('--resume_t', '-r_t', type=str, default='model', metavar='PATH')

    parser.add_argument('--dim_t', '-d_t', type=int, default=512,
                        help='Dimension of Embedding Feather')
    # # parser.add_argument('--width', type=int, default=224,
    #                     help='width of input image')

    parser.add_argument('--batch_size_t', type=int, default=100)
    # parser.add_argument('--nThreads', '-j', default=16, type=int, metavar='N',
    #                     help='number of data loading threads (default: 2)')
    parser.add_argument('--pool_feature', type=ast.literal_eval, default=False, required=False,
                        help='if True extract feature from the last pool layer')

    parser.add_argument('--warm', default=1, type=int,help='Warmup training epochs')

    # args = parser.parse_args()

    main(parser.parse_args())




