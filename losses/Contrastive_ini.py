from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

# The function of this Contrasive loss is generate the gradient initial weight.

class Contrastive_ini(nn.Module):
    def __init__(self, margin=0.5, **kwargs):
        super(Contrastive_ini, self).__init__()
        self.margin = margin

    def forward(self, inputs, targets, margin):
        n = inputs.size(0)
        # Compute similarity matrix
        sim_mat = torch.matmul(inputs, inputs.t())
        targets = targets
        loss = list()
        c = 0
        ap_list = []
        an_list = []
        anchor_list =[]

        for i in range(n):
            pos_pair_ = torch.masked_select(sim_mat[i], targets == targets[i])

            #  move itself
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1)
            neg_pair_ = torch.masked_select(sim_mat[i], targets != targets[i])

            pos_pair_ = torch.sort(pos_pair_)[0]
            neg_pair_ = torch.sort(neg_pair_)[0]

            neg_pair = torch.masked_select(neg_pair_, neg_pair_ > margin[i])

            neg_loss = 0

            pos_loss = torch.sum(-pos_pair_ + 1)
            if len(neg_pair) > 0:
                neg_loss = torch.sum(neg_pair)
            an_list.append(len(neg_pair))
            ap_list.append(len(pos_pair_))
            loss.append(pos_loss + neg_loss)

        loss = sum(loss) / n
        prec = float(c) / n
        mean_neg_sim = torch.mean(neg_pair_).item()
        mean_pos_sim = torch.mean(pos_pair_).item()
        # return loss, prec, mean_pos_sim, mean_neg_sim
        return loss,anchor_list, ap_list, an_list, sum(an_list)+sum(ap_list)

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

    print(ContrastiveLoss()(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')


