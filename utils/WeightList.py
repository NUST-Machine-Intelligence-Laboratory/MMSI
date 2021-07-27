from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


class WeightList(nn.Module):
    def __init__(self, alpha=10, beta=2, margin=0.1, hard_mining=1,**kwargs):
        super(WeightList, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.hard_mining = hard_mining

    def forward(self, inputs, targets, margin, weight, hard_mining= 1):
        n = inputs.size(0)
        sim_mat = torch.matmul(inputs, inputs.t())
        targets = targets
        # margin = torch.ones(n).mul(margin)
        # print(margin)
        # 定义一个新margin 常委bs执委0.1

        base = 0.5
        # loss = list()
        loss = []
        c = 0
        length_ap = 0
        length_an = 0
        anchor_list=[]
        ap_list=[]
        an_list=[]
        nouse_list=[]
        an_pw = [[] for i in range(n)]
        ap_pw = [[] for i in range(n)]
        # ap_w = []
        # an_w = []
        for i in range(n):

            pos_pair_ = torch.masked_select(sim_mat[i], targets == targets[i])

            #  move itself
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1)
            neg_pair_ = torch.masked_select(sim_mat[i], targets != targets[i])

            pos_pair_ = torch.sort(pos_pair_)[0]
            neg_pair_ = torch.sort(neg_pair_)[0]

            if hard_mining ==1 :
                neg_pair = torch.relu(neg_pair_ + margin[i] - pos_pair_[0])
                pos_pair = torch.relu(neg_pair_[-1] - pos_pair_ + margin[i])

                neg_pair = torch.masked_select(neg_pair, neg_pair > 0) - margin[i] + pos_pair_[0]
                pos_pair = -torch.masked_select(pos_pair, pos_pair > 0) + margin[i] + neg_pair_[-1]
                for j in range(len(neg_pair)):
                    if neg_pair[j] != 0:
                        an_pw[i].append(1)
                    else:
                        an_pw[i].append(0)
                for k in range(len(pos_pair)):
                    if pos_pair[k] != 0:
                        ap_pw[i].append(1)
                    else:
                        ap_pw[i].append(0)
                length_ap += len(pos_pair)
                length_an += len(neg_pair)

                # print('ap1', an_p[i][:])
                # ap_p1 = [x for x in ap_p if x != []]
                # an_p1 = [x for x in an_p if x != []]



                if len(pos_pair)+len(neg_pair)<1:
                    anchor_list.append(0)
                    nouse_list.append(1)
                    an_list.append(0)
                    ap_list.append(0)
                    c+=1
                    continue
                # if len(neg_pair) < 1 or len(pos_pair) < 1:
                #   c += 1
                #    continue
                # print('lan', i,len(neg_pair))
                # print('lap', i, len(pos_pair))
                # print( 'pw',1 + torch.sum(torch.exp(-self.beta * (pos_pair * weight[i] - base ))))
                # anchor_list = (torch.Tensor(ap_p).cuda())
                pos_loss = 2.0 / self.beta * torch.log(1 + torch.sum(torch.exp(-self.beta * (pos_pair * weight[i] - base ))))  #
                neg_loss = 2.0 / self.alpha * torch.log(1 + torch.sum(torch.exp(self.alpha * (neg_pair * weight[i] - base ))))  # 补上加减margini
                # print('pl', pos_loss)
                # print('nl', neg_loss)
                loss.append(pos_loss + neg_loss)
                # print('loss',torch.Tensor(loss))
                anchor_list.append(1)
                nouse_list.append(0)
                an_list.append(len(neg_pair))
                ap_list.append(len(pos_pair))
                # print(i, 'anl', an_list)
                # print(i, 'apl', ap_list)

            else:
                # print('hello world')
                neg_pair = neg_pair_
                pos_pair = pos_pair_
                length_ap += len(pos_pair)
                length_an += len(neg_pair)
                # print('an', neg_pair)
                # print('ap', pos_pair)
                # if i == 1:
                #     print('None')
                #     print('L_an', len(neg_pair))
                #     print('L_ap', len(pos_pair))

                pos_loss = 2.0 / self.beta * torch.log(1 + torch.sum(torch.exp(-self.beta * (pos_pair - base - margin[i]))))
                neg_loss = 2.0 / self.alpha * torch.log(1 + torch.sum(torch.exp(self.alpha * (neg_pair - base + margin[i]))))
                # print('ap',pos_pair)
                loss.append(pos_loss + neg_loss)
        # ap_p1 = [x for x in ap_p if x != []]
        # an_p1 = [x for x in an_p if x != []]
        # print('lenap', length_ap, 'lenan', length_an)
        # # print('loss',torch.Tensor(loss),len(torch.Tensor(loss)))
        # # print('al',anchor_list)
        print('anl', an_list, len(an_list), sum(an_list))
        print('apl', ap_list, len(ap_list), sum(ap_list))
        # print( 'app', ap_p1, len(ap_p1))
        # print( 'anp', an_p1, len(an_p1))

        loss = sum(loss) / n
        loss = loss.mean()
        prec = float(c) / n
        mean_neg_sim = torch.mean(neg_pair_).item()
        mean_pos_sim = torch.mean(pos_pair_).item()
        return ap_list, an_list
        # return loss, prec, mean_pos_sim, mean_neg_sim


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


