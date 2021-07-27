from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


# The function of this MS loss is generate the gradient initial weight.

class MS_ini(nn.Module):
    def __init__(self, alpha=10, beta=2, margin=0.1, hard_mining=1,**kwargs):
        super(MS_ini, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.hard_mining = hard_mining

    def forward(self, inputs, targets, margin):
        n = inputs.size(0)
        sim_mat = torch.matmul(inputs, inputs.t())
        targets = targets

        base = 0.5
        loss = []
        c = 0
        length_ap = 0
        length_an = 0
        anchor_list=[]
        ap_list=[]
        an_list=[]
        nouse_list=[]
        an_p = [[] for i in range(n)]
        ap_p = [[] for i in range(n)]
        for i in range(n):

            pos_pair_ = torch.masked_select(sim_mat[i], targets == targets[i])

            #  move itself
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1)
            neg_pair_ = torch.masked_select(sim_mat[i], targets != targets[i])

            pos_pair_ = torch.sort(pos_pair_)[0]
            neg_pair_ = torch.sort(neg_pair_)[0]

            if self.hard_mining is not None:
                neg_pair = torch.relu(neg_pair_ + margin[i] - pos_pair_[0])
                pos_pair = torch.relu(neg_pair_[-1] - pos_pair_ + margin[i])

                neg_pair = torch.masked_select(neg_pair, neg_pair > 0) - margin[i] + pos_pair_[0]
                pos_pair = -torch.masked_select(pos_pair, pos_pair > 0) + margin[i] + neg_pair_[-1]

                length_ap += len(pos_pair)
                length_an += len(neg_pair)

                if len(neg_pair) < 1 or len(pos_pair) < 1:
                    anchor_list.append(0)
                    nouse_list.append(1)
                    an_list.append(0)
                    ap_list.append(0)
                    # c += 1
                    continue

                pos_loss = 2.0 / self.beta * torch.log(1 + torch.sum(torch.exp(-self.beta * (pos_pair - base))))
                neg_loss = 2.0 / self.alpha * torch.log(1 + torch.sum(torch.exp(self.alpha * (neg_pair - base))))  # 补上加减margini
                # print('pl', pos_loss)
                # print('nl', neg_loss)
                loss.append(pos_loss + neg_loss)
                # print('loss',torch.Tensor(loss))
                anchor_list.append(1)
                nouse_list.append(0)
                an_list.append(len(neg_pair))
                ap_list.append(len(pos_pair))
            else:
                # print('hello world')
                neg_pair = neg_pair_
                pos_pair = pos_pair_
                length_ap += len(pos_pair)
                length_an += len(neg_pair)
                an_list.append(len(neg_pair))
                ap_list.append(len(pos_pair))

                pos_loss = 2.0 / self.beta * torch.log(1 + torch.sum(torch.exp(-self.beta * (pos_pair - base))))
                neg_loss = 2.0 / self.alpha * torch.log(1 + torch.sum(torch.exp(self.alpha * (neg_pair - base))))
                # print('ap',pos_pair)
                loss.append(pos_loss + neg_loss)

        loss = sum(loss) / n
        loss = loss.mean()
        prec = float(c) / n
        mean_neg_sim = torch.mean(neg_pair_).item()
        mean_pos_sim = torch.mean(pos_pair_).item()
        return loss, anchor_list, ap_list, an_list, sum(an_list)+sum(ap_list)


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
    y_ = 8 * list(range(num_class))
    targets = Variable(torch.IntTensor(y_))

    print(MetaWeightLoss()(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')


