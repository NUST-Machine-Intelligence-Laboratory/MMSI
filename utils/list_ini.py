import numpy as np
def list_ini(args):
    # margin_t = args.trip_margin
    # margin_v = args.trip_margin_v
    wmean = 0.4
    losses_p = []
    meta_losses_p = []
    wmax_p = []
    margint_p = []
    wmean_p = []
    wmin_p = []
    Rank1 = []
    recall_list = np.zeros((args.epochs, 1))

    return wmean,losses_p,meta_losses_p,wmax_p,margint_p,wmean_p,wmin_p,Rank1,recall_list

def list_append(losses,losses_p,meta_losses_p,wmax_p,wmean_p,wmin_p,margint_p,meta_losses,wmax,margin_t,wmin):
    losses = losses.avg
    losses_p.append(losses)
    meta_losses_p.append(meta_losses)
    wmax_p.append(wmax)
    wmean_p.append(margin_t)
    wmin_p.append(wmin)
    margint_p.append(margin_t)
    return losses_p,meta_losses_p,wmax_p,wmean_p,wmin_p,margint_p