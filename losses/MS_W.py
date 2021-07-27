from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

# The loss function in the inner loop participating in the training process.

class Weight_MS(nn.Module):
    def __init__(self, alpha=10, beta=2, margin=0.1, hard_mining=1, **kwargs):
        super(Weight_MS, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.hard_mining = hard_mining

    def forward(self, inputs, targets, margin, weight, ap_list, an_list):
    
        n = inputs.size(0)
        sim_mat = torch.matmul(inputs, inputs.t())
        targets = targets
        

        base = 0.5
        # loss = list()
        loss = []
        c = 0
        length_ap = 0
        length_an = 0
        anchor_list=[]
        nouse_list=[]
        a = 0
        b = 0
        #print(sim_mat)
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
                    
                    continue
                b = a + ap_list[i]
                c = b + an_list[i]

                w_tilde_ap = weight[a:b]
                w_tilde_an = weight[b:c]
    
                pos_loss = 2.0 / self.beta * torch.log(1 + torch.sum(torch.exp(-self.beta * ((pos_pair * w_tilde_ap) - base))))  #
                neg_loss = 2.0 / self.alpha * torch.log(1 + torch.sum(torch.exp(self.alpha * ((neg_pair * w_tilde_an) - base))))

                a = c
                loss.append(pos_loss + neg_loss)
                # print('loss',torch.Tensor(loss))
                # anchor_list.append(1)
                # nouse_list.append(0)
                # an_list.append(len(neg_pair))
                # ap_list.append(len(pos_pair))

            else:
                neg_pair = neg_pair_
                pos_pair = pos_pair_
                length_ap += len(pos_pair)
                length_an += len(neg_pair)

                if len(neg_pair) < 1 or len(pos_pair) < 1:
                    # anchor_list.append(0)
                    # an_list.append(0)
                    # ap_list.append(0)
                    # c += 1
                    continue
                b = a + ap_list[i]
                c = b + an_list[i]

                w_tilde_ap = weight[a:b]
                w_tilde_an = weight[b:c]

                pos_loss = 2.0 / self.beta * torch.log(1 + torch.sum(torch.exp(-self.beta * ((pos_pair * w_tilde_ap) - base))))
                neg_loss = 2.0 / self.alpha * torch.log(1 + torch.sum(torch.exp(self.alpha * ((neg_pair * w_tilde_an) - base))))
                a = c
                loss.append(pos_loss + neg_loss)

        loss = sum(loss) / n
        prec = float(c) / n
        mean_neg_sim = torch.mean(neg_pair_).item()
        mean_pos_sim = torch.mean(pos_pair_).item()

        return loss, prec, mean_pos_sim, mean_neg_sim
        # return loss, anchor_list
        # return loss, prec, 


def main():
    data_size = 32
    input_dim = 3
    output_dim = 2
    num_class = 4
    # margin = 0.5
    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    inputs = x.mm(w)
    y_ = 8 * list(range(num_class))
    targets = Variable(torch.IntTensor(y_))

    print(MetaWeightLoss()(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')


