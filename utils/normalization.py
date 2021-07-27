import torch
import math

def normalization(m,m_size):
    mm=torch.mean(m)
    print('mm', mm)

    ma=(m-mm)/math.sqrt(torch.var(m))
    mb=ma/30
    print('mb', mb)
    print('mb_mean', torch.mean(mb))
    print('math.sqrt(torch.var(mb))', math.sqrt(torch.var(mb)))
    md = m/sum(m)
    print('md',md)
    return (md)
