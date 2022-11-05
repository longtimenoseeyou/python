import os
import numpy as np
import time
from IPython import display
from tqdm import tqdm
import torch
from torch import nn
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
from torchvision import datasets, models, transforms

argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)


def use_svg_display():
    """Use the svg format to display a plot in Jupyter.

    Defined in :numref:`sec_calculus`"""
    backend_inline.set_matplotlib_formats('svg')


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


class Timer:
    """Record multiple running times."""

    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()


class Animator:
    """For plotting data in animation.动态显示结果"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        """Defined in :numref:`sec_softmax_scratch`"""
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        # use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


class Accumulator:
    """For accumulating sums over `n` variables."""

    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def accuracy(y_hat, y):
    """Compute the number of correct predictions.

    Defined in :numref:`sec_softmax_scratch`"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1)
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))


def try_gpu():
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]
    Defined in :numref:`sec_use_gpu`"""
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset using a GPU.

    Defined in :numref:`sec_lenet`"""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = Accumulator(2)

    with torch.no_grad():
        for X, y in tqdm(data_iter):
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), size(y))
    return metric[0] / metric[1]


# 图像增强
def trans_forms():
    transform_train = transforms.Compose([
        # 在高度和宽度上将图像放大到40像素的正方形
        transforms.Resize(40),
        # 随机裁剪出一个高度和宽度均为40像素的正方形图像，
        # 生成一个面积为原始图像面积0.64到1倍的小正方形，
        # 然后将其缩放为高度和宽度均为32像素的正方形
        transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                     ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # 标准化图像的每个通道
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2023, 0.1994, 0.2010])])
    transform_test = transforms.Compose([
        transforms.Resize(40),
        transforms.RandomResizedCrop(32),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2023, 0.1994, 0.2010])])
    return transform_train, transform_test


def load_data(data_dir, batch_size=64):
    transform_train, transform_test = trans_forms()
    train_ds = datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=transform_train)

    valid_ds, test_ds = [datasets.ImageFolder(
        os.path.join(data_dir, folder),
        transform=transform_test) for folder in ['valid', 'test']]
    # 在训练期间，我们需要指定上面定义的所有图像增广操作。
    # 当验证集在超参数调整过程中用于模型评估时，不应引入图像增广的随机性。
    # 在最终预测之前，我们根据训练集和验证集组合而成的训练模型进行训练，以充分利用所有标记的数据。
    train_iter = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True, drop_last=True)

    valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,
                                             drop_last=True)

    test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                                            drop_last=False)
    return train_iter, valid_iter, test_iter


def train_batch(net, X, y, loss, trainer, devices):
    """用多GPU进行小批量训练
    Defined in :numref:`sec_image_augmentation`"""
    if isinstance(X, list):
        # 微调BERT中所需（稍后讨论）
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = accuracy(pred, y)
    return train_loss_sum, train_acc_sum


def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period=4,
          lr_decay=0.9):
    # momentum是冲量,weight_decay是权重衰减
    train_loss = []
    train_acc = []
    valid_acc = []
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    loss = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), Timer()
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], legend=legend)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        net.train()
        metric = Accumulator(3)
        for i, (features, labels) in enumerate(tqdm(train_iter)):
            timer.start()
            l, acc = train_batch(net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0])
            epoch_end = timer.stop()

        if valid_iter is not None:
            valid_acc.append(evaluate_accuracy_gpu(net, valid_iter))
            animator.add(epoch + 1, (None, None, valid_acc))
        scheduler.step()
        train_loss.append(metric[0] / metric[2])
        train_acc.append(metric[1] / metric[2])
        measures = (f'train loss {train_loss[-1]:.3f}, '
                    f'train acc {train_acc[-1]:.3f}')
        if valid_iter is not None:
            measures += f', valid acc {valid_acc[-1]:.3f}'
        print(f'\nEpoch: {epoch + 1:03d}, ' + measures + f' ,Time: {epoch_end:.4f}s')
    print(f'\n{metric[2] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
    return train_loss, train_acc, valid_acc



if __name__ == '__main__':
    dataset = './imagenet'
    devices = try_gpu()
    num_epochs, lr, wd, batch_size = 10, 0.05, 5e-4, 64
    lr_period, lr_decay = 4, 0.95  # 优化算法的学习速率将在每4个周期乘以0.9
    net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    train_iter, valid_iter, _ = load_data(dataset, batch_size)
    t_loss, t_acc, v_acc = train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay)

