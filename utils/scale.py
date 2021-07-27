import torch


def scale(m):
    Min=torch.min(m)
    Max=torch.max(m)
    r=Max-Min
    # print('r',r)
    if r== 0:
        return m
    m=((m-Min)/r)*0.1


    return m
def scale2(m):

    Min=torch.min(m)
    Max=torch.max(m)
    if Max == 0:
        return m
    if Max>= 0 and Min<=0:
        m_z=(abs(m)+m)/2
        m_z=scale(m_z)
        m_f=-((-abs(m)+m)/2)
        m_f=-scale(m_f)
        return m_f+m_z
    if Max >= 0 and Min >= 0:
        return scale(m)
    if Max <= 0 and Min <= 0:
        return -scale(-m)
    return m