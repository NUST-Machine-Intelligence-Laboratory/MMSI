from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

# The loss function in the inner loop participating in the training process.


class ContrastiveW(nn.Module):
    def __init__(self, margin=0.5, **kwargs):
        super(ContrastiveW, self).__init__()
        self.margin = margin

    def forward(self, inputs, targets, margin, weight, ap_list, an_list):
        n = inputs.size(0)
        # Compute similarity matrix
        sim_mat = torch.matmul(inputs, inputs.t())
        targets = targets
        loss = list()
        c = 0
        a = 0
        b = 0

        for i in range(n):
            pos_pair_ = torch.masked_select(sim_mat[i], targets == targets[i])

            #  move itself
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1)
            neg_pair_ = torch.masked_select(sim_mat[i], targets != targets[i])

            pos_pair_ = torch.sort(pos_pair_)[0]
            neg_pair_ = torch.sort(neg_pair_)[0]

            neg_pair = torch.masked_select(neg_pair_, neg_pair_ > margin[i])

            b = a + ap_list[i]
            c = b + an_list[i]
            w_tilde_ap = weight[a:b]
            w_tilde_an = weight[b:c]

            neg_loss = 0

            pos_loss = torch.sum(-((pos_pair_+ 1)* w_tilde_ap))
            if len(neg_pair) > 0:
                neg_loss = torch.sum(neg_pair* w_tilde_an)
            a = c
            loss.append(pos_loss + neg_loss)

        loss = sum(loss) / n
        prec = float(c) / n
        mean_neg_sim = torch.mean(neg_pair_).item()
        mean_pos_sim = torch.mean(pos_pair_).item()
        return loss, prec, mean_pos_sim, mean_neg_sim


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


