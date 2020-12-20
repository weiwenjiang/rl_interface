"""
This code is from official pytorch document (https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html)
I modified optimizer to use name of the parameter for preventing prunned weights from updated by gradients
"""

import math
from collections import defaultdict
from torch._six import container_abcs
import torch
from copy import deepcopy
from itertools import chain


class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()


class NameOptimizer(object):
    r"""Base class for all optimizers.

    .. warning::
        Parameters need to be specified as collections that have a deterministic
        ordering that is consistent between runs. Examples of objects that don't
        satisfy those properties are sets and iterators over values of dictionaries.

    Arguments:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
    """

    def __init__(self, named_params, defaults):
        self.defaults = defaults

        if isinstance(named_params, torch.Tensor):#判断一个对象是否是一个已知的类型
            raise TypeError("params argument given to the optimizer should be "
                            "an iterable of Tensors or dicts, but got " +
                            torch.typename(named_params))

        self.state = defaultdict(dict)
        self.param_groups = []

        param_groups = list(named_params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    def __getstate__(self):
        return {
            'defaults': self.defaults,
            'state': self.state,
            'param_groups': self.param_groups,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        for i, group in enumerate(self.param_groups):
            format_string += '\n'
            format_string += 'Parameter Group {0}\n'.format(i)
            for key in sorted(group.keys()):
                if key != 'params':
                    format_string += '    {0}: {1}\n'.format(key, group[key])
        format_string += ')'
        return format_string

    def state_dict(self):
        r"""Returns the state of the optimizer as a :class:`dict`.

        It contains two entries:

        * state - a dict holding current optimization state. Its content
            differs between optimizer classes.
        * param_groups - a dict containing all parameter groups
        """
        # Save ids instead of Tensors
        def pack_group(group):
            packed = {k: v for k, v in group.items() if k != 'params'}
            packed['params'] = [id(p) for p in group['params']]
            return packed
        param_groups = [pack_group(g) for g in self.param_groups]
        # Remap state to use ids as keys
        packed_state = {(id(k) if isinstance(k, torch.Tensor) else k): v
                        for k, v in self.state.items()}
        return {
            'state': packed_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict):
        r"""Loads the optimizer state.

        Arguments:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = deepcopy(state_dict)
        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict['param_groups']

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of "
                             "parameter groups")
        param_lens = (len(g['params']) for g in groups)
        saved_lens = (len(g['params']) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError("loaded state dict contains a parameter group "
                             "that doesn't match the size of optimizer's group")

        # Update the state
        id_map = {old_id: p for old_id, p in
                  zip(chain(*(g['params'] for g in saved_groups)),
                      chain(*(g['params'] for g in groups)))}

        def cast(param, value):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, torch.Tensor):
                # Floating-point types are a bit special here. They are the only ones
                # that are assumed to always match the type of params.
                if param.is_floating_point():
                    value = value.to(param.dtype)
                value = value.to(param.device)
                return value
            elif isinstance(value, dict):
                return {k: cast(param, v) for k, v in value.items()}
            elif isinstance(value, container_abcs.Iterable):
                return type(value)(cast(param, v) for v in value)
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = defaultdict(dict)
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                state[param] = cast(param, v)
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group['params'] = group['params']
            return new_group
        param_groups = [
            update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({'state': state, 'param_groups': param_groups})

    def zero_grad(self):
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
        for group in self.param_groups:
            for name, p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    def step(self, closure):
        r"""Performs a single optimization step (parameter update).

        Arguments:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        raise NotImplementedError

    def add_param_group(self, param_group):
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.

        Arguments:
            param_group (dict): Specifies what Tensors should be optimized along with group
            specific optimization options.
        """
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group['params']
        if isinstance(params, torch.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            param_group['params'] = list(params)

        for name, param in param_group['params']:
            if not isinstance(param, torch.Tensor):
                raise TypeError("optimizer can only optimize Tensors, "
                                "but one of the params is " + torch.typename(param))
            if not param.is_leaf:
                raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError("parameter group didn't specify a value of required optimization parameter " +
                                 name)
            else:
                param_group.setdefault(name, default)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)


class PruneAdam(NameOptimizer):
    r"""Implements Adam algorithm.
    #使用了一个已经发表的Adam算法
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)#学习率
        betas (Tuple[float, float], optional): coefficients used for computing#系数，计算梯度均值和平方
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator分母 to improve      #为了增加数值计算的稳定性而加到分母里的项
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)#权重衰减，使用l2惩罚
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this#是否使用AMSGrad变体
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:#学习率
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:#分母加上的值，增加数据稳定性
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:#一组系数，计算梯度均值和平方
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)#其他默认参数
        super(PruneAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PruneAdam, self).__setstate__(state)
        for group in self.param_groups:#参数组
            group.setdefault('amsgrad', False)

    def step(self, closure=None):#进行单次优化 (参数更新).
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model一个重新评价模型并返回loss的闭包
                and returns the loss.返回loss值
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:#param_groups 是个list,每个元素是dict形式
            for name, p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data#梯度 gradient
                if grad.is_sparse:#如果梯度是稀疏的
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]#一个保存了当前优化状态的dict，optimizer的类别不同，state的内容也会不同。

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values梯度值的指数移动平均
                    state['exp_avg'] = torch.zeros_like(p.data)#创建维度和p.data一致的零矩阵
                    # Exponential moving average of squared gradient values梯度平方值的指数移动平均
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values上面两个参数的最大值
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq'] #为指数赋值
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']#梯度均值和平方值，一二阶距估计的衰减率,(k可以用来调整一二阶距的学习率)

                state['step'] += 1

                if group['weight_decay'] != 0:#权重衰减
                    grad.add_(group['weight_decay'], p.data)#更新梯度

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)#一阶距估计  公式中的Mt
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)#二阶距估计  公式中的Nt
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)#求序列最大值，
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])#分母上加上eps这个参数
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']#偏差校正   Mt'
                bias_correction2 = 1 - beta2 ** state['step']#偏差校正   Nt'
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)#更新参数，p.data = p.data  - step_size + exp_avg/denom

        return loss

    def prune_step(self, mask, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            mask: prunning mask to prevent weight update.剪枝mask，防止权重更新
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for name, p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)#一阶距估计
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)#二阶距估计

                # if name.split('.')[-1] == "weight":#若name根据小数点划分后，最后一项是weight
                a = name.split('.')[-1]
                b = name.split('.')[-2]
                if (a == "weight" and b == "conv1") or (a == "weight" and b == "conv2"):
                    exp_avg_sq.mul_(mask[name])#exp_avg_sq *= mask[name]

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']                           #一二阶距估计的校正
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # if name.split('.')[-1] == "weight":
                a = name.split('.')[-1]
                b = name.split('.')[-2]
                if (a == "weight" and b == "conv1") or (a == "weight" and b == "conv2"):
                    exp_avg.mul_(mask[name])#exp_avg *= mask[name]
                p.data.addcdiv_(-step_size, exp_avg, denom)#更新参数，p.data = p.data  - step_size + exp_avg/denom

        return loss
