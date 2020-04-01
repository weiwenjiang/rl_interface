from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
from optimizer import PruneAdam
from model import LeNet, AlexNet
from utils import regularized_nll_loss, admm_loss, \
    initialize_Z_and_U, update_X, update_Z, update_Z_l1, update_U, \
    print_convergence, print_prune, apply_l1_prune, apply_pattern_prune
from torchvision import datasets, transforms
from tqdm import tqdm


def train(args, model, device, train_loader, test_loader, optimizer):
    for epoch in range(args.num_pre_epochs):#迭代次数
        print('Pre epoch: {}'.format(epoch + 1))
        # for pa in model.named_parameters():
        #     print(pa[0])
        model.train()#训练模型
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = regularized_nll_loss(args, model, output, target)
            loss.backward()
            optimizer.step()
        test(args, model, device, test_loader)

    Z, U = initialize_Z_and_U(model)
    for epoch in range(args.num_epochs):
        model.train()
        print('Epoch: {}'.format(epoch + 1))
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = admm_loss(args, device, model, Z, U, output, target)
            loss.backward()
            optimizer.step()
        X = update_X(model)
        Z = update_Z_l1(X, U, args) if args.l1 else update_Z(X, U, args)
        U = update_U(U, X, Z)
        print_convergence(model, X, Z)
        test(args, model, device, test_loader)


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def retrain(args, model, mask, device, train_loader, test_loader, optimizer):
    for epoch in range(args.num_re_epochs):
        print('Re epoch: {}'.format(epoch + 1))
        model.train()
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.prune_step(mask)

        test(args, model, device, test_loader)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--dataset', type=str, default="mnist", choices=["mnist", "cifar10"],#数据集
                        metavar='D', help='training dataset (mnist or cifar10)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',#训练批次数，64
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',#测试批次，1000
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--percent', type=list, default=[0.8, 0.92, 0.991, 0.93],#剪枝率，默认0.8
                        metavar='P', help='pruning percentage (default: 0.8)')
    parser.add_argument('--alpha', type=float, default=5e-4, metavar='L',#？
                        help='l2 norm weight (default: 5e-4)')
    parser.add_argument('--rho', type=float, default=1e-2, metavar='R',#基数权重，默认权重？
                        help='cardinality weight (default: 1e-2)')
    parser.add_argument('--l1', default=False, action='store_true',#l1正则化，放大特征
                        help='prune weights with l1 regularization instead of cardinality')
    parser.add_argument('--l2', default=False, action='store_true',
                        help='apply l2 regularization')#l2正则化，防止过拟合
    parser.add_argument('--num_pre_epochs', type=int, default=3, metavar='P',#预训练的迭代次数，3
                        help='number of epochs to pretrain (default: 3)')
    parser.add_argument('--num_epochs', type=int, default=10, metavar='N',#正式训练的迭代次数，3
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--num_re_epochs', type=int, default=10, metavar='R',#重新训练的迭代次数，3
                        help='number of epochs to retrain (default: 3)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',#学习率
                        help='learning rate (default: 1e-2)')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, metavar='E',#？
                        help='adam epsilon (default: 1e-8)')
    parser.add_argument('--no-cuda', action='store_true', default=False,#设备，有无GPU
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,#是否保存model
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()#是否使用cuda

    torch.manual_seed(args.seed)#为CPU设置种子，用于生成随机数

    device = torch.device("cuda" if use_cuda else "cpu")#设备是GPU还是CPU

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}#如果使用多GPU

    if args.dataset == "mnist":#mnist数据集的加载
        train_loader = torch.utils.data.DataLoader(#训练集加载
            datasets.MNIST('data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))#单通道均值和标准差，input[channel] = (input[channel] - mean[channel]) / std[channel]
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(#测试集加载
            datasets.MNIST('data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

    else:#cifar10数据集的加载
        args.percent = [0.8, 0.92, 0.93, 0.94, 0.95, 0.99, 0.99, 0.93]#剪枝率
        #迭代次数，三个阶段
        args.num_pre_epochs = 5
        args.num_epochs = 20
        args.num_re_epochs = 5
        train_loader = torch.utils.data.DataLoader(#训练集加载
            datasets.CIFAR10('data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                                                      (0.24703233, 0.24348505, 0.26158768))#三通道均值和标准差
                             ])), shuffle=True, batch_size=args.batch_size, **kwargs)

        test_loader = torch.utils.data.DataLoader(#测试集加载
            datasets.CIFAR10('data', train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                                                      (0.24703233, 0.24348505, 0.26158768))
                             ])), shuffle=True, batch_size=args.test_batch_size, **kwargs)

    model = LeNet().to(device) if args.dataset == "mnist" else AlexNet().to(device)#mnist-LeNet，cifar10-AlexNet
    optimizer = PruneAdam(model.named_parameters(), lr=args.lr, eps=args.adam_epsilon)
	#每一次迭代元素的名字及其值，学习率，？
    # train(args, model, device, train_loader, test_loader, optimizer)#调用优化器进行优化
    mask = apply_pattern_prune(model, device,'18')#选择剪枝标准,可以输入0或者其他1-9的组合
    # print(mask)
    print_prune(model)#打印剪枝后model,如何改变的模型
    test(args, model, device, test_loader)#使用剪枝后的模型进行测试
    retrain(args, model, mask, device, train_loader, test_loader, optimizer)#使用剪枝后的模型进行重新训练


if __name__ == "__main__":
    main()