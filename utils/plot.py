import numpy as np
import matplotlib.pyplot as plt
from utils import mkdir_if_missing
import os

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