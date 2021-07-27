# coding=utf-8
from __future__ import print_function, absolute_import
import time
from utils import AverageMeter, orth_reg
import  torch
from torch.autograd import Variable
from torch.backends import cudnn

cudnn.benchmark = True


def train(epoch, model, criterion, optimizer, train_loader, args):

    # if args.warm > 0:
    #
    #     unfreeze_model_param = list(model.module.classifier.parameters()) + list(criterion.parameters())
    #
    #     if epoch == 0:
    #         for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
    #             param.requires_grad = False
    #     if epoch == args.warm:
    #         for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
    #             param.requires_grad = True

    losses = AverageMeter()
    batch_time = AverageMeter()
    accuracy = AverageMeter()
    pos_sims = AverageMeter()
    neg_sims = AverageMeter()
    losses_ap = AverageMeter()
    losses_an = AverageMeter()

    end = time.time()
    ap_dim_list_e = list()

    freq = min(args.print_freq, len(train_loader))

    for i, data_ in enumerate(train_loader, 0):

        inputs, labels = data_

        # wrap them in Variable
        inputs = Variable(inputs).cuda() #torch.Size([batchsize, 3, 227, 227])
        labels = Variable(labels).cuda()

        optimizer.zero_grad()

        # embed_feat = model(inputs) #torch.Size([batchsize, 512])
        # embed_feat_x, embed_feat = model(inputs)  # torch.Size([batchsize, 512])


        embed_feat = model(inputs)
        loss, inter_, dist_ap, dist_an = criterion(embed_feat, labels, args.margin)
            # loss, inter_, dist_ap, dist_an, ap_dim_list = criterion(embed_feat, labels)

        # loss, inter_, dist_ap, dist_an, ap_dim_list = criterion(embed_feat, labels, args.margin)
        # ap_dim_list_e.append(ap_dim_list)
        # loss = criterion(embed_feat, labels, num_classes=100)

        if args.orth_reg != 0:
            loss = orth_reg(net=model, loss=loss, cof=args.orth_reg)


        loss.backward()
        # torch.nn.utils.clip_grad_value_(model.parameters(), 10)
        # if args.loss == 'PAnchor':
            # print('PA')
            # torch.nn.utils.clip_grad_value_(criterion.parameters(), 10)
        #autograd操作输出矩阵，两个列表一个是loss集合，一个标记loss属于哪个anker的集合，输出后对第一个list做autograd，得到grade提出值平方和
        #然后归一化，然后控制方差。
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.item())
        # losses_ap.update(dist_ap.item())
        losses_ap.update(dist_ap)
        # print(losses_an)

        # losses_an.update(dist_an.item())
        losses_an.update(dist_an)
        accuracy.update(inter_)
        # pos_sims.update(dist_ap)
        # neg_sims.update(dist_an)

        if (i + 1) % freq == 0 or (i+1) == len(train_loader):
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
