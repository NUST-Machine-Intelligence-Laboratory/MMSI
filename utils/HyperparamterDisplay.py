from __future__ import print_function, absolute_import


def display(args):
    #  Display information of current training
    
    print(40 * '#')

    print('Learn Rate  \t%.1e' % args.lr)
    print('Epochs  \t%05d' % args.epochs)
    print('Log Path \t%s' % args.save_dir)
    print('Network \t %s' % args.net)
    print('Data Set \t %s' % args.data)
    print('Batch Size  \t %d' % args.batch_size)
    print('Num-Instance  \t %d' % args.num_instances)
    print('Embedded Dimension \t %d' % args.dim)
    print('margin \t %f' % args.margin)
    print('Loss Function \t%s' % args.loss)
    print('Alpha \t %d' % args.alpha)
    print('Begin to fine tune %s Network' % args.net)
    if args.MMSI == 1:
        print(15 * '-', 'MMSI ON', 15 * '-')
        print('MMSI Hyperparamter:')
        print('Val Batch Size  \t %d' % args.val_batch_size)
        print('Val Num-Instance  \t %d' % args.val_num_instances)
        print('Semi-Globa Size \t %d' % args.d_size)
    else:
        print(15 * '-', 'MMSI OFF', 15 * '-')
    
    print(40*'#')
