import torch
import torch.nn as nn
from models.meta_layers import MetaModule
import operator
import warnings
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
# from torch.nn.parallel.replicate import replicate
from new_edite.replicate_ import replicate
from torch.nn.parallel.parallel_apply import parallel_apply


def _check_balance(device_ids):
    imbalance_warn = """
    There is an imbalance between your GPUs. You may want to exclude GPU {} which
    has less than 75% of the memory or cores of GPU {}. You can do so by setting
    the device_ids argument to DataParallel, or by setting the CUDA_VISIBLE_DEVICES
    environment variable."""

    dev_props = [torch.cuda.get_device_properties(i) for i in device_ids]

    def warn_imbalance(get_prop):
        values = [get_prop(props) for props in dev_props]
        min_pos, min_val = min(enumerate(values), key=operator.itemgetter(1))
        max_pos, max_val = max(enumerate(values), key=operator.itemgetter(1))
        if min_val / max_val < 0.75:
            warnings.warn(imbalance_warn.format(device_ids[min_pos], device_ids[max_pos]))
            return True
        return False

    if warn_imbalance(lambda props: props.total_memory):
        return
    if warn_imbalance(lambda props: props.multi_processor_count):
        return


class DataParallel(MetaModule):
    r"""Implements data parallelism at the module level.

    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the batch
    dimension. In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards
    pass, gradients from each replica are summed into the original module.

    The batch size should be larger than the number of GPUs used.

    See also: :ref:`cuda-nn-dataparallel-instead`

    Arbitrary positional and keyword inputs are allowed to be passed into
    DataParallel EXCEPT Tensors. All tensors will be scattered on dim
    specified (default 0). Primitive types will be broadcasted, but all
    other types will be a shallow copy and can be corrupted if written to in
    the model's forward pass.

    .. warning::
        Forward and backward hooks defined on :attr:`module` and its submodules
        will be invoked ``len(device_ids)`` times, each with inputs located on
        a particular device. Particularly, the hooks are only guaranteed to be
        executed in correct order with respect to operations on corresponding
        devices. For example, it is not guaranteed that hooks set via
        :meth:`~torch.nn.Module.register_forward_pre_hook` be executed before
        `all` ``len(device_ids)`` :meth:`~torch.nn.Module.forward` calls, but
        that each such hook be executed before the corresponding
        :meth:`~torch.nn.Module.forward` call of that device.

    .. note::
        There is a subtlety in using the
        ``pack sequence -> recurrent network -> unpack sequence`` pattern in a
        :class:`~torch.nn.Module` wrapped in :class:`~torch.nn.DataParallel`.
        See :ref:`pack-rnn-unpack-with-data-parallelism` section in FAQ for
        details.


    Args:
        module: module to be parallelized
        device_ids: CUDA devices (default: all devices)
        output_device: device location of output (default: device_ids[0])

    Example::

        # >>> net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
        # >>> output = net(input_var)
    """

    # TODO: update notes/cuda.rst when this class handles 8+ GPUs well

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallel, self).__init__()

        self.val = False
        if not torch.cuda.is_available():
            self.module = module
            self.device_ids = []
            return

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]
        self.dim = dim
        self.module = module
        self.device_ids = device_ids
        self.output_device = output_device

        _check_balance(self.device_ids)

        if len(self.device_ids) == 1:
            self.module.cuda(device_ids[0])

    def forward(self, *inputs, **kwargs):
        self.val = False
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        if self.val:
            device_ids = self.device_ids[1:]
        else:
            device_ids = self.device_ids
        inputs, kwargs = self.scatter(inputs, kwargs, device_ids)
        if len(device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        # print(self.device_ids)
        # print(self.device_ids[:len(inputs)])
        replicas = self.replicate(self.module, self.device_ids)
        if self.val:
            outputs = self.parallel_apply(replicas[1:], inputs, kwargs, device_ids)
        else:
            # print(device_ids)
            # print(device_ids[:len(replicas)])
            outputs = self.parallel_apply(replicas, inputs, kwargs, device_ids)
        self.val = False
        return self.gather(outputs, self.output_device)

    def replicate(self, module, device_ids):
        return replicate(module, device_ids)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def parallel_apply(self, replicas, inputs, kwargs, device_ids):
        return parallel_apply(replicas, inputs, kwargs, device_ids)

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)
        # return outputs