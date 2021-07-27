# coding=utf-8
from __future__ import print_function, absolute_import
import time
from utils import AverageMeter, orth_reg
from utils.setBNeval import set_bn_eval
import torch
from torch.autograd import Variable
from torch.backends import cudnn
import models
from new_edite.dataparallel import DataParallel
from Model2Feature import Model2Feature
from evaluations import Recall_at_ks, pairwise_similarity
from utils.serialization import save_checkpoint, load_checkpoint
import os.path as osp

import numpy

cudnn.benchmark = True

def Norm(w):
    w=torch.relu(w)
    s=torch.sum(w)
    w=w/s
    w=w/torch.max(w)#w is belong to (0,1)
    return w

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

# def train(epoch, model, criterion, optimizer, train_loader, args):
def train_MMSI(epoch, model, criterion_i, criterion, criterion_m, optimizer, train_loader, valid_loader, margin, Glob, args):
    train_margin = margin

    losses = AverageMeter()
    batch_time = AverageMeter()
    accuracy = AverageMeter()
    pos_sims = AverageMeter()
    neg_sims = AverageMeter()

    end = time.time()

    freq = min(args.print_freq, len(train_loader))
    
    for i, data_ in enumerate(train_loader, 0):
        
        # ========================================================================
        # =========================== Outer Loop Start ===========================
        
        meta_net = models.create(args.net, args.data, pretrained=True, dim=args.dim)  # initial Meta-Net (Alg Flow.(1))
        meta_net = DataParallel(meta_net).cuda()
        meta_net.load_state_dict(model.state_dict())
        meta_net.apply(set_bn_eval)

        # Data augmentation images and lable from trainset
        inputs_t, labels_t = data_
        inputs_t, labels_t = to_var(inputs_t), to_var(labels_t, requires_grad=False)

        optimizer.zero_grad()
        
        # TrainSet Features from the Meta-Net have two way in outer loop.
        # On the one hand, these features are stored in the global dictionary. Alg Flow.(2)(5)
        # On the other hand, they are used for weight initialization. Alg Flow.(6)
        embed_feat_1 = meta_net(inputs_t)

        # -----------------------------------------------------------------------
        # --------------------- Bulid Semi-Global Dictionary --------------------

        # First destination for trainset features --- build the Semi-Global dictionary. Alg Flow.(2)(5)
        for j in range(0, labels_t.size(0)):
            if labels_t[j].item() in Glob:
                Glob[labels_t[j].item()] = torch.cat((Glob[labels_t[j].item()], embed_feat_1[j].view(1, embed_feat_1[j].size()[0]).detach().cpu()), 0)
                if Glob[labels_t[j].item()].size(0) > args.d_size:   # Dictionary size (number of features (images) per class)
                    Glob[labels_t[j].item()] = Glob[labels_t[j].item()][1:, ]
            else:
                Glob[labels_t[j].item()] = embed_feat_1[j].view(1, embed_feat_1[j].size()[0]).detach().cpu()

        # When the number of dictionary class is greater than initial class number for Semi-Global Dictionary,
        # we started training.(Alg Flow.(2))
        if len(list(Glob.keys()))<1000:
            continue

        for index, key in enumerate(list(Glob.keys())):
            if index == 0:
                global_feature = Glob[key]
                global_label = [key] * Glob[key].size(0)
            else:
                global_feature = torch.cat((global_feature, Glob[key]), 0)
                global_label += [key] * Glob[key].size(0)

        # Get Semi-Global_features from Semi-Global Dictionary.
        global_feature = to_var(global_feature, requires_grad=False)
        global_label = to_var(torch.from_numpy(numpy.array(global_label)), requires_grad=False)

        # -----------------------------------------------------------------------
        # ---------------------- Weight Generation Process ----------------------

        # Trainset features' another direction --- weight initialization(Alg Flow.(6))
        margin_t = to_var(torch.ones(inputs_t.size(0)).mul(train_margin), requires_grad=False)

        _, anchor_list, ap_list, an_list, slct_num = criterion_i(embed_feat_1, labels_t, margin_t)
        weight = to_var(torch.zeros(slct_num).cuda())   # Alg Flow.(6)

        # Get similarity matrix(train) in DML loss(Alg Flow.(4)), and calculate the loss and gradients (Alg Flow.(7))
        loss, inter_, dist_ap, dist_an = criterion(embed_feat_1, labels_t, margin_t, weight, ap_list, an_list)
        l_f_meta = loss
        param = meta_net.parameters()
        grads = torch.autograd.grad(l_f_meta, param, create_graph=True, allow_unused=True) 

        meta_net.update_params_t(args.lr, source_params=grads) # update the temporary parameters(Alg Flow.(8))
        meta_dict_t = meta_net.state_dict()
        meta_net.load_state_dict(meta_dict_t)

        # We use the KP sampler (P categories, K samples/category)
        # to generate a validation mini-batch from the validation set.
        image_v, labels_v = valid_loader.__next__()
        image_v, labels_v = to_var(image_v), to_var(labels_v, requires_grad=False)
        y_feat_2 = meta_net(image_v)

        # Get features from validation set
        # Alg Flow.(9)
        l_g_metas = criterion_m(y_feat_2, labels_v, global_feature, global_label, args.margin)

        if l_g_metas==0:
            continue

        # The derivative of the weight is performed after the calculation of loss.
        # Finally, the derivative generates the weight list after scaling, normalization.
        # We get a list of weights that will go to the next module.
        grad_weight = torch.autograd.grad(l_g_metas, weight, only_inputs=True, allow_unused=True)

        grad_weight = grad_weight[0]
        w_grad_2 = -grad_weight
        # Alg Flow.(12)(13)
        w3 = Norm(w_grad_2)

        # =========================== Outer Loop End ===========================
        # ======================================================================

        # =========================== Inner Loop ===============================

        embed_feat_3 = model(inputs_t)
        loss, inter_, dist_ap, dist_an = criterion(embed_feat_3, labels_t, margin_t, w3, ap_list, an_list)

        if args.orth_reg != 0:
            loss = orth_reg(net=model, loss=loss, cof=args.orth_reg)

        # Alg Flow.(14)
        loss.backward()
        # Alg Flow.(15)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.item())
        # accuracy.update(inter_)
        # pos_sims.update(dist_ap)
        # neg_sims.update(dist_an)
        # =========================== Inner Loop ===========================
        if args.data == 'product':
            # if (i + 1) % 1 == 0 or i == 0:
            state_dict = model.module.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
                'epoch': (i + 1),
            }, is_best=False, fpath=osp.join(args.save_dir, 'ckp_ep' + str(epoch + 1) + str(i + 1) + '.pth.tar'))

            # if epoch == 0:
            #     pass
            # else:
            # test
            checkpoint = load_checkpoint(
                osp.join(args.save_dir, 'ckp_ep' + str(epoch + 1) + str(i + 1) + '.pth.tar'))
            # print(args.pool_feature)
            epoch = checkpoint['epoch']

            gallery_feature, gallery_labels, query_feature, query_labels = \
                Model2Feature(data=args.data, root=args.data_root, width=args.width, net=args.net_t,
                              checkpoint=checkpoint,
                              dim=args.dim_t, batch_size=args.batch_size_t, nThreads=args.nThreads,
                              pool_feature=args.pool_feature)

            sim_mat = pairwise_similarity(query_feature, gallery_feature)
            if args.gallery_eq_query is True:
                sim_mat = sim_mat - torch.eye(sim_mat.size(0))

            recall_ks = Recall_at_ks(sim_mat, query_ids=query_labels, gallery_ids=gallery_labels,
                                     data=args.data)
            print(recall_ks[0])

        if (i + 1) % freq == 0 or (i+1) == len(train_loader):
            print('Epoch: [{0:03d}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f} \t'
                  'Accuracy {accuracy.avg:.4f} \t'
                  'Pos {pos.avg:.4f}\t'
                  'Neg {neg.avg:.4f} \t'.format
                  (epoch + 1, i + 1, len(train_loader), batch_time=batch_time,
                   loss=losses, accuracy=accuracy, pos=pos_sims, neg=neg_sims))


        if epoch == 0 and i == 0:
            print('-- HA-HA-HA-HA-AH-AH-AH-AH --')
    return losses, Glob


def train(epoch, model, criterion, optimizer, train_loader, args):
    losses = AverageMeter()
    batch_time = AverageMeter()
    accuracy = AverageMeter()
    pos_sims = AverageMeter()
    neg_sims = AverageMeter()
    losses_ap = AverageMeter()
    losses_an = AverageMeter()
    recall_list = []
    Rank1 = []

    end = time.time()
    ap_dim_list_e = list()

    freq = min(args.print_freq, len(train_loader))

    for i, data_ in enumerate(train_loader, 0):

        inputs, labels = data_

        # wrap them in Variable
        inputs = Variable(inputs).cuda()  # torch.Size([batchsize, 3, 227, 227])
        labels = Variable(labels).cuda()

        optimizer.zero_grad()

        embed_feat = model(inputs)
        loss, inter_, dist_ap, dist_an = criterion(embed_feat, labels, args.margin)

        if args.orth_reg != 0:
            loss = orth_reg(net=model, loss=loss, cof=args.orth_reg)

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.item())
        # losses_ap.update(dist_ap.item())
        losses_ap.update(dist_ap)
        # losses_an.update(dist_an.item())

        losses_an.update(dist_an)
        accuracy.update(inter_)

        if args.data == 'product':
            if (i + 1) % 2 == 0 or i == 0:
                state_dict = model.module.state_dict()
                save_checkpoint({
                    'state_dict': state_dict,
                    'epoch': (i + 1),
                }, is_best=False, fpath=osp.join(args.save_dir, 'ckp_ep' + str(epoch + 1) + str(i + 1) + '.pth.tar'))

                # if epoch == 0:
                #     pass
                # else:
                # test
                checkpoint = load_checkpoint(
                    osp.join(args.save_dir, 'ckp_ep' + str(epoch + 1) + str(i + 1) + '.pth.tar'))
                # print(args.pool_feature)
                epoch = checkpoint['epoch']

                gallery_feature, gallery_labels, query_feature, query_labels = \
                    Model2Feature(data=args.data, root=args.data_root, width=args.width, net=args.net_t,
                                  checkpoint=checkpoint,
                                  dim=args.dim_t, batch_size=args.batch_size_t, nThreads=args.nThreads,
                                  pool_feature=args.pool_feature)

                sim_mat = pairwise_similarity(query_feature, gallery_feature)
                if args.gallery_eq_query is True:
                    sim_mat = sim_mat - torch.eye(sim_mat.size(0))

                recall_ks = Recall_at_ks(sim_mat, query_ids=query_labels, gallery_ids=gallery_labels,
                                         data=args.data)
                print(recall_ks[0])

        if (i + 1) % freq == 0 or (i + 1) == len(train_loader):
            print('Epoch: [{0:03d}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f} \t'
                  'Accuracy {accuracy.avg:.4f} \t'
                  'Pos {pos.avg:.4f}\t'
                  'Neg {neg.avg:.4f} \t'.format
                  # (epoch + 1, i + 1, len(train_loader), batch_time=batch_time,
                  #  loss=losses, accuracy=accuracy, pos=dist_ap, neg=dist_an))
                  (epoch + 1, i + 1, len(train_loader), batch_time=batch_time,
                   loss=losses, accuracy=accuracy, pos=losses_ap, neg=losses_an))

        if epoch == 0 and i == 0:
            print('-- HA-HA-HA-HA-AH-AH-AH-AH --')
    # print('adle', ap_dim_list_e)
    return (losses)