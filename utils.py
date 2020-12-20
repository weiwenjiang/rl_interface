import torch
import torch.nn.functional as F
import numpy as np
from search_space import get_pattern


def regularized_nll_loss(args, model, output, target):
    index = 0
    loss = F.nll_loss(output, target)
    if args.l2:
        for name, param in model.named_parameters():#分离model参数
            if name.split('.')[-1] == "weight":
                loss += args.alpha * param.norm()
                index += 1
    return loss

def admm_loss(args, device, model, Z, U, output, target):#使用正则化计算loss？
    idx = 0
    loss = F.nll_loss(output, target)
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight":
            u = U[idx].to(device)
            z = Z[idx].to(device)
            loss += args.rho / 2 * (param - z + u).norm()
            if args.l2:
                loss += args.alpha * param.norm()
            idx += 1
    return loss


def initialize_Z_and_U(model):
    Z = ()
    U = ()
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight":
            Z += (param.detach().cpu().clone(),)#切断一些分支的反向传播，使得只对其中一部分参数进行调整，detach()后param并不会收到影响，对detach()后的矩阵进行clone()，会影响原矩阵
            U += (torch.zeros_like(param).cpu(),)#构造一个和param一样的矩阵，初始化为0
    return Z, U


def update_X(model):
    X = ()
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight":
            X += (param.detach().cpu().clone(),)
    return X


def update_Z(X, U, args):
    new_Z = ()
    idx = 0
    for x, u in zip(X, U):
        z = x + u
        pcen = np.percentile(abs(z), 100*args.percent[idx])
        under_threshold = abs(z) < pcen
        z.data[under_threshold] = 0
        new_Z += (z,)
        idx += 1
    return new_Z


def update_Z_l1(X, U, args):
    new_Z = ()
    delta = args.alpha / args.rho
    for x, u in zip(X, U):
        z = x + u
        new_z = z.clone()
        if (z > delta).sum() != 0:
            new_z[z > delta] = z[z > delta] - delta
        if (z < -delta).sum() != 0:
            new_z[z < -delta] = z[z < -delta] + delta
        if (abs(z) <= delta).sum() != 0:
            new_z[abs(z) <= delta] = 0
        new_Z += (new_z,)
    return new_Z


def update_U(U, X, Z):
    new_U = ()
    for u, x, z in zip(U, X, Z):
        new_u = u + x - z
        new_U += (new_u,)
    return new_U


def prune_weight(weight, device, percent):#使用阈值剪枝
    # to work with admm, we calculate percentile based on all elements instead of nonzero elements.
    weight_numpy = weight.detach().cpu().numpy()
    pcen = np.percentile(abs(weight_numpy), 100*percent)
    under_threshold = abs(weight_numpy) < pcen
    weight_numpy[under_threshold] = 0
    mask = torch.Tensor(abs(weight_numpy) >= pcen).to(device)
    return mask


def prune_l1_weight(weight, device, delta):#使用l1正则化剪枝
    weight_numpy = weight.detach().cpu().numpy()
    under_threshold = abs(weight_numpy) < delta
    weight_numpy[under_threshold] = 0
    mask = torch.Tensor(abs(weight_numpy) >= delta).to(device)
    return mask


def prune_pattern_weight(weight,device,str,layer):#使用pattern剪枝
    weight_numpy = weight.detach().cpu().numpy()
    if layer == 1:#如果是卷积层1,1 input,20 output
        for i in range(20):
            for j in range(1):
                weight_numpy[i][j] = get_pattern(str)#根据pattern对应位置赋0或1
    elif layer == 2:#如果是卷积层2,20 input,50 output
        for i in range(50):
            for j in range(20):
                weight_numpy[i][j] = get_pattern(str)
    mask = torch.Tensor(weight_numpy).to(device)
    return mask


def apply_prune(model, device, args):#根据设定的剪枝率剪枝
    # returns dictionary of non_zero_values' indices
    print("Apply Pruning based on percentile")
    dict_mask = {}
    idx = 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight":
            mask = prune_weight(param, device, args.percent[idx])
            param.data.mul_(mask)
            # param.data = torch.Tensor(weight_pruned).to(device)
            dict_mask[name] = mask
            idx += 1
    return dict_mask


def apply_l1_prune(model, device, args):#根据l1正则化剪枝
    delta = args.alpha / args.rho
    print("Apply Pruning based on l1")
    dict_mask = {}
    idx = 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight":
            mask = prune_l1_weight(param, device, delta)
            param.data.mul_(mask)
            dict_mask[name] = mask
            idx += 1
    return dict_mask


def apply_pattern_prune(model,device,str):#根据pattern剪枝
    print("Apply Pruning based on pattern")
    dict_mask = {}
    for name,param in model.named_parameters():
        a = name.split('.')[-1]
        b = name.split('.')[-2]
        if a == "weight" and b == "conv1":
            mask = prune_pattern_weight(param,device,str,1)
            param.data.mul_(mask)#model参数数据的对应位置变为0
            dict_mask[name] = mask
        elif a == "weight" and b == "conv2":
            mask = prune_pattern_weight(param,device,str,2)
            param.data.mul_(mask)
            dict_mask[name] = mask
            print(name,dict_mask[name])
    return dict_mask

def print_convergence(model, X, Z):#打印收敛情况
    idx = 0
    print("normalized norm of (weight - projection)")
    for name, _ in model.named_parameters():
        if name.split('.')[-1] == "weight":
            x, z = X[idx], Z[idx]
            print("({}): {:.4f}".format(name, (x-z).norm().item() / x.norm().item()))
            idx += 1


def print_prune(model):#打印剪枝情况
    prune_param, total_param = 0, 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight":
            print("[at weight {}]".format(name))
            print("percentage of pruned: {:.4f}%".format(100 * (abs(param) == 0).sum().item() / param.numel()))
            print("nonzero parameters after pruning: {} / {}\n".format((param != 0).sum().item(), param.numel()))
        total_param += param.numel()
        prune_param += (param != 0).sum().item()
    print("total nonzero parameters after pruning: {} / {} ({:.4f}%)".
          format(prune_param, total_param,
                 100 * (total_param - prune_param) / total_param))
