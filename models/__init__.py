# from .BN_Inception import BN_Inception
# from .meta_BN_Inception import meta_bn_inceptionv2, BN_Inception

from .BN_Inception import BN_Inception
from .BN_Inception_MetaAda import BN_Inception_MetaAda

__factory = {
    'BN_Inception': BN_Inception,
    'BN_Inception_MetaAda':BN_Inception_MetaAda,
}

def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a loss instance.

    Parameters
    ----------
    name : str
        the name of loss function
    """
    if name not in __factory:
        raise KeyError("Unknown network:", name)
    return __factory[name](*args, **kwargs)
