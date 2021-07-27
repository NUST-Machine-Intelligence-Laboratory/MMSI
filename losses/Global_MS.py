from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

# The function of this MS loss is to let the initial
# weight pass, then generate the gradient, autograd and
# generate the sample pair weight through a series of
# processes in the trainer.py.


class Global_MS_m(nn.Module):
    def __init__(self, alpha=10, beta=2, margin=0.5, hard_mining=None, **kwargs):
        super(Global_MS_m, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.hard_mining = hard_mining

    def forward(self, inputs, targets, global_inputs, global_targets, margin):
        n = inputs.size(0)

        sim_mat = torch.matmul(inputs, global_inputs.t())
        #targets = targets
        # print('sim: ', sim_mat.size())

        base = 0.5
        loss = list()
        # loss_list = list()
        c = 0
        length_ap = 0
        length_an = 0
        anchor_list = []
        ap_list = []
        an_list = []

        for i in range(n):

            pos_pair_ = torch.masked_select(sim_mat[i], global_targets==targets[i])#生成相似度矩阵anker为第i个图像的simmat

            #  move itself
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1)#剔除同样本的相似度('1')形成ap矩阵

            neg_pair_ = torch.masked_select(sim_mat[i], global_targets!=targets[i])#和选ap一样的方法选出an

            pos_pair_ = torch.sort(pos_pair_)[0]
            neg_pair_ = torch.sort(neg_pair_)[0]




            if self.hard_mining is not None:
                neg_pair = torch.relu(neg_pair_ + margin - pos_pair_[0])
                pos_pair = torch.relu(neg_pair_[-1] - pos_pair_ + margin)
                neg_pair = torch.masked_select(neg_pair, neg_pair > 0) - margin + pos_pair_[0]
                pos_pair = -torch.masked_select(pos_pair, pos_pair > 0) + margin + neg_pair_[-1]

                length_ap += len(pos_pair)
                length_an += len(neg_pair)


                if len(neg_pair) < 1 or len(pos_pair) < 1:
                    anchor_list.append(0)
                    an_list.append(0)
                    ap_list.append(0)
                    c += 1
                    continue
                # print('sum',torch.sum(torch.exp(self.alpha * (neg_pair - base))))

                pos_loss = 2.0/self.beta * torch.log(1 + torch.sum(torch.exp(-self.beta * (pos_pair - base))))
                neg_loss = 2.0/self.alpha * torch.log(1 + torch.sum(torch.exp(self.alpha * (neg_pair - base))))
                anchor_list.append(1)
                an_list.append(len(neg_pair))
                ap_list.append(len(pos_pair))

                loss.append(pos_loss + neg_loss)

            else:
                # print('hello world')
                neg_pair = neg_pair_
                pos_pair = pos_pair_

                length_ap += len(pos_pair)
                length_an += len(neg_pair)

                if len(neg_pair) < 1 or len(pos_pair) < 1:
                    anchor_list.append(0)
                    an_list.append(0)
                    ap_list.append(0)
                    c += 1
                    continue

                pos_loss = 2.0/self.beta * torch.log(1 + torch.sum(torch.exp(-self.beta * (pos_pair - base))))
                neg_loss = 2.0/self.alpha * torch.log(1 + torch.sum(torch.exp(self.alpha * (neg_pair - base))))
                anchor_list.append(1)
                an_list.append(len(neg_pair))
                ap_list.append(len(pos_pair))

                loss.append(pos_loss + neg_loss)
        # print('lss', torch.Tensor(loss),len(loss))
        # print(length_ap, length_an)
        loss = sum(loss)/n
        # print(loss)
        #append一个list，
        # loss_list=loss.append(loss)
        # print('ll',loss_list)

        prec = float(c)/n
        mean_neg_sim = torch.mean(neg_pair_).item()
        mean_pos_sim = torch.mean(pos_pair_).item()
        return loss
        # return loss,

def main():
    data_size = 32
    input_dim = 3
    output_dim = 2
    num_class = 4
    # margin = 0.5
    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    # print(x)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    inputs = x.mm(w)
    y_ = 8*list(range(num_class))
    targets = Variable(torch.IntTensor(y_))

    print(WeightLoss()(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')


