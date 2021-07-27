import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter


from torch._six import container_abcs
from itertools import repeat

Buffer = False

def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def parameters(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                # #panduan是不是categry
                name_t, param_t = tgt
                # print(name_t)
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                # print('tg',type(grad))

                #tmp = param_t - lr_inner * grad
                temp = param_t - lr_inner * grad
                # print(type(temp))
                #name_t是cat就 temp改成param
                self.set_param(self, name_t, temp)
        else:

            for name, param in self.named_params(self):

                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def update_params_t(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                # #panduan是不是categry
                name_t, param_t = tgt
                # print('namet',name_t)
                if name_t == 'module.category.linear.weight':
                    self.set_param(self, name_t, param_t)
                    continue
                if name_t == 'module.category.linear.bias':
                    self.set_param(self, name_t, param_t)
                    continue

                # print(name_t)
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                # print('tg',type(grad))

                # tmp = param_t - lr_inner * grad
                temp = param_t - lr_inner * grad
                # print(type(temp))
                # name_t是cat就 temp改成param
                self.set_param(self, name_t, temp)
        else:

            for name, param in self.named_params(self):

                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(self, name, param)


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)
        self.is_bias = kwargs['bias']
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        if self.is_bias:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        # else:
        #     self.register_buffer('bias', None)

    def forward(self, x):
        if self.is_bias:
            return F.linear(x, self.weight, self.bias)
        else:
            return F.linear(x, self.weight)

    def named_leaves(self):
        if self.is_bias:
            return [('weight', self.weight), ('bias', self.bias)]
        else:
            return [('weight', self.weight)]


class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)

        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


# class MetaLinear(MetaModule):
#     # def __init__(self, *args, **kwargs):
#     #     super().__init__()
#     #     ignore = nn.Linear(*args, **kwargs)
#     #
#     #     # self.weight = to_var(ignore.weight.data, requires_grad=True)
#     #     # self.bias = to_var(ignore.bias.data, requires_grad=True)
#     #     self.weight = ignore.weight
#     #     self.bias = ignore.bias
#     #     # self.register_buffer('weight', to_var(ignore.weight.data.clone(), requires_grad=True))
#     #     # self.register_buffer('bias', to_var(ignore.bias.data.clone(), requires_grad=True))
#     #
#     #     del ignore
#     def __init__(self, in_features, out_features, bias=True):
#         super(MetaLinear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         if Buffer:
#             self.register_buffer('weight', to_var(torch.Tensor(out_features, in_features), requires_grad=True))
#         # self.weight = to_var(torch.Tensor(out_features, in_features), requires_grad=True)
#         else:
#             self.weight = Parameter(torch.Tensor(out_features, in_features))
#         if bias:
#             if Buffer:
#                 self.register_buffer('bias', to_var(torch.Tensor(out_features), requires_grad=True))
#             else:
#                 self.bias = Parameter(torch.Tensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#
#     def forward(self, x):
#         return F.linear(x, self.weight, self.bias)
#
#     def named_leaves(self):
#         return [('weight', self.weight), ('bias', self.bias)]
#
#
# class MetaConv2d(MetaModule):
#     # def __init__(self, *args, **kwargs):
#     #     super().__init__()
#     #     ignore = nn.Conv2d(*args, **kwargs)
#     #
#     #     self.stride = ignore.stride
#     #     self.padding = ignore.padding
#     #     self.dilation = ignore.dilation
#     #     self.groups = ignore.groups
#     #
#     #     # self.weight = to_var(ignore.weight.data, requires_grad=True)
#     #     self.weight = ignore.weight
#     #     # self.register_buffer('weight', to_var(ignore.weight.data.clone(), requires_grad=True))
#     #
#     #     if ignore.bias is not None:
#     #         # self.bias = to_var(ignore.bias.data, requires_grad=True)
#     #         self.bias = ignore.bias
#     #         # self.register_buffer('bias', to_var(ignore.bias.data.clone(), requires_grad=True))
#     #     else:
#     #         self.bias = None
#     #         # self.register_buffer('bias', None)
#     #
#     #     del ignore
#     __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias']
#
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True, transposed=False, output_padding=_pair(0)):
#         super(MetaConv2d, self).__init__()
#         kernel_size = _pair(kernel_size)
#         stride = _pair(stride)
#         padding = _pair(padding)
#         dilation = _pair(dilation)
#         if in_channels % groups != 0:
#             raise ValueError('in_channels must be divisible by groups')
#         if out_channels % groups != 0:
#             raise ValueError('out_channels must be divisible by groups')
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.transposed = transposed
#         self.output_padding = output_padding
#         self.groups = groups
#         if transposed:
#             if Buffer:
#                 self.register_buffer('weight', to_var(torch.Tensor(
#                     in_channels, out_channels // groups, *kernel_size), requires_grad=True))
#             else:
#                 self.weight = Parameter(torch.Tensor(
#                     in_channels, out_channels // groups, *kernel_size))
#         else:
#             if Buffer:
#                 self.register_buffer('weight', to_var(torch.Tensor(
#                     out_channels, in_channels // groups, *kernel_size), requires_grad=True))
#             else:
#                 self.weight = Parameter(torch.Tensor(
#                     out_channels, in_channels // groups, *kernel_size))
#         if bias:
#             if Buffer:
#                 self.register_buffer('bias', to_var(torch.Tensor(out_channels), requires_grad=True))
#             else:
#                 self.bias = Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)
#
#     def forward(self, x):
#         return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
#
#     def named_leaves(self):
#         return [('weight', self.weight), ('bias', self.bias)]


class MetaConvTranspose2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.ConvTranspose2d(*args, **kwargs)

        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x, output_size=None):
        output_padding = self._output_padding(x, output_size)
        return F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding,
                                  output_padding, self.groups, self.dilation)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaBatchNorm2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats


        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, x):
        # print('srm', self.running_mean)
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            self.training or not self.track_running_stats, self.momentum, self.eps)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]