import argparse
import ast
def args():
    parser = argparse.ArgumentParser(description='Deep Metric Learning')

    # hype-parameters
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate of new parameters")
    parser.add_argument('--batch_size', '-b', default=128, type=int, metavar='N',
                        help='mini-batch size (1 = pure stochastic) Default: 256')
    parser.add_argument('--val_batch_size', default=128, type=int, metavar='N',
                        help='mini-batch size (1 = pure stochastic) Default: 256')
    parser.add_argument('--num_instances', default=8, type=int, metavar='n',
                        help=' number of samples from one class in mini-batch')
    parser.add_argument('--val_num_instances', default=8, type=int, metavar='n',
                        help=' number of samples from one class in mini-batch')
    parser.add_argument('--dim', default=512, type=int, metavar='n',
                        help='dimension of embedding space')
    parser.add_argument('--width', default=224, type=int,
                        help='width of input image')
    parser.add_argument('--origin_width', default=256, type=int,
                        help='size of origin image')
    parser.add_argument('--ratio', default=0.16, type=float,
                        help='random crop ratio for train data')

    parser.add_argument('--alpha', default=30, type=int, metavar='n',
                        help='hyper parameter in NCA and its variants')
    parser.add_argument('--beta', default=0.1, type=float, metavar='n',
                        help='hyper parameter in some deep metric loss functions')
    parser.add_argument('--orth_reg', default=0, type=float,
                        help='hyper parameter coefficient for orth-reg loss')
    parser.add_argument('-k', default=16, type=int, metavar='n',
                        help='number of neighbour points in KNN')
    parser.add_argument('--margin', default=0.09, type=float,
                        help='margin in loss function')
    parser.add_argument('--init', default='random',
                        help='the initialization way of FC layer')

    # network
    parser.add_argument('--freeze_BN', default=True, type=bool, required=False, metavar='N',
                        help='Freeze BN if True')
    parser.add_argument('--data', default='cub', required=True,
                        help='name of Data Set')
    parser.add_argument('--data_root', type=str, default=None,
                        help='path to Data Set')

    parser.add_argument('--net', default='VGG16-BN')
    parser.add_argument('--loss', default='Weight_MS', required=True,
                        help='loss for training network')
    parser.add_argument('--epochs', default=600, type=int, metavar='N',
                        help='epochs for training process')
    parser.add_argument('--save_step', default=50, type=int, metavar='N',
                        help='number of epochs to save model')

    # Resume from checkpoint
    # parser.add_argument('--resume', '-r', default='/home/jxr/proj/MMSI_v1/models/ckp_ep108_MS_m015_SOP_7874.pth.tar',
    #                     help='the path of the pre-trained model')
    parser.add_argument('--resume', '-r', default=None, help='the path of the pre-trained model')

    parser.add_argument('--resume_sop', '-r_sop', default='/home/jxr/proj/MMSI_v1/models/MS_SOP_784ckp.pth.tar',
                        help='the path of the pre-trained model')

    # train
    parser.add_argument('--print_freq', default=20, type=int,
                        help='display frequency of training')

    # basic parameter
    # parser.add_argument('--checkpoints', default='/opt/intern/users/xunwang',
    #                     help='where the trained models save')
    parser.add_argument('--save_dir', default=None,
                        help='where the trained models save')
    parser.add_argument('--nThreads', '-j', default=16, type=int, metavar='N',
                        help='number of data loading threads (default: 2)')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)

    parser.add_argument('--loss_base', type=float, default=0.75)

    # test
    parser.add_argument('--gallery_eq_query', '-g_eq_q', type=ast.literal_eval, default=True,
                        help='Is gallery identical with query')
    parser.add_argument('--net_t', type=str, default='BN_Inception_CrossE')
    parser.add_argument('--resume_t', '-r_t', type=str, default='model', metavar='PATH')

    parser.add_argument('--dim_t', '-d_t', type=int, default=512,
                        help='Dimension of Embedding Feather')

    parser.add_argument('--batch_size_t', type=int, default=100)
    parser.add_argument('--pool_feature', type=ast.literal_eval, default=False, required=False,
                        help='if True extract feature from the last pool layer')

    parser.add_argument('--margin_type', type=int, default=1)

    parser.add_argument('--MMSI', type=int, default=1)
    parser.add_argument('--d_size', type=int, default=1000)


    return parser